"""Candidate final architecture: 0.50 * Qwen3-4B-LoRA + 0.25 * XLM-R-XL-LoRA + 0.25 * TF-IDF-LR (5-fold, soft voting).
It is the current final (final_qlt/, LB 0.6097) with XLM-R large replaced by XLM-R XL. The weights are the same, not re-tuned.
Compares the blends on the full 5-fold OOF (paired bootstrap vs final_qlt, per-fold wins) and writes the candidate's
submission. Uses saved probabilities only. Usage (from project root): python final_qxt/scripts/make_submission.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "final_qxt/submission"
tr = pd.read_csv(ROOT / "data/train.csv"); te = pd.read_csv(ROOT / "data/test.csv"); sample = pd.read_csv(ROOT / "data/sample_submission.csv")
y = tr.y.values
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y))
fold_id = np.zeros(len(y), int)
for f, (a, b) in enumerate(folds):
    fold_id[b] = f

Lo, Lt = np.zeros((len(tr), 3)), np.zeros((len(te), 3))
for f, (a, b) in enumerate(folds):
    d = np.load(ROOT / f"outputs/cache/xlm-roberta-large_fold{f}.npz", allow_pickle=True); Lo[b] = d["val"]; Lt += d["test"] / 5
T = np.load(ROOT / "outputs/cache/tfidf.npz")
P = dict(Q=(np.load(ROOT / "qwen/results/cv/oof.npy"), np.load(ROOT / "qwen/results/cv/test.npy")),
         L=(Lo, Lt), T=(T["oof"], T["test"]),
         XL=(np.load(ROOT / "xlmr_xl/results/cv/oof.npy"), np.load(ROOT / "xlmr_xl/results/cv/test.npy")))

REF, CAND = "Q+L+T (final_qlt)", "Q+XL+T (candidate)"
BLENDS = {REF: dict(Q=.50, L=.25, T=.25), CAND: dict(Q=.50, XL=.25, T=.25),
          "Q+L+XL+T": dict(Q=.40, L=.20, XL=.20, T=.20), "Q+XL": dict(Q=.50, XL=.50)}


def blend(w, i):  # i = 0: OOF, 1: test
    return sum(v * P[k][i] for k, v in w.items()) / sum(w.values())


def mf1(yt, yp):  # macro-F1 from the confusion matrix; equals sklearn's, and is fast enough for the bootstrap
    cm = np.bincount(yt * 3 + yp, minlength=9).reshape(3, 3); tp = np.diag(cm)
    p, r = tp / np.maximum(cm.sum(0), 1), tp / np.maximum(cm.sum(1), 1)
    return float(np.mean(np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)))


assert abs(mf1(y, P["Q"][0].argmax(1)) - f1_score(y, P["Q"][0].argmax(1), average="macro")) < 1e-12
ref = blend(BLENDS[REF], 0).argmax(1)
rng = np.random.default_rng(0); boot = [rng.integers(0, len(y), len(y)) for _ in range(2000)]
print(f"5-fold OOF, n={len(y)}. Single models: " + "  ".join(f"{k} {mf1(y, v[0].argmax(1)):.4f}" for k, v in P.items()))
print("Per-fold single models: " + "  ".join(f"{k} {[round(mf1(y[fold_id == f], v[0][fold_id == f].argmax(1)), 4) for f in range(5)]}"
                                            for k, v in P.items()))
res = {}
for name, w in BLENDS.items():
    p = blend(w, 0).argmax(1)
    pf = [mf1(y[fold_id == f], p[fold_id == f]) for f in range(5)]
    won = sum(mf1(y[fold_id == f], p[fold_id == f]) > mf1(y[fold_id == f], ref[fold_id == f]) for f in range(5))
    d = np.array([mf1(y[i], p[i]) - mf1(y[i], ref[i]) for i in boot])
    res[name] = dict(weights=w, oof_f1=mf1(y, p), per_fold=pf, folds_better_than_ref=int(won),
                     gain_vs_ref=float(d.mean()), ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                     p_no_gain=float((d <= 0).mean()),
                     test_agreement_with_ref=float((blend(w, 1).argmax(1) == blend(BLENDS[REF], 1).argmax(1)).mean()))
    r = res[name]
    print(f"{name:20s} OOF {r['oof_f1']:.4f}  per-fold {[round(x, 4) for x in pf]}  vs {REF}: {r['gain_vs_ref']:+.4f} "
          f"CI[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] P(no gain)={r['p_no_gain']:.3f} folds better {won}/5 "
          f"test agreement {r['test_agreement_with_ref']:.3f}")

c = res[CAND]
decision = ("SIGNIFICANT: the lower CI bound of Q+XL+T vs Q+L+T is above 0" if c["ci95"][0] > 0 else
            "NOT significant: the CI of Q+XL+T vs Q+L+T includes 0")
print("\nDecision rule (fixed before the full OOF was seen): switch only if the gain is significant. ->", decision)
Pc = blend(BLENDS[CAND], 0)
print(f"\n{CAND} OOF classification report:\n" + classification_report(y, Pc.argmax(1), digits=4))

Pt = blend(BLENDS[CAND], 1)
sub = sample[["id"]].merge(pd.DataFrame({"id": te["id"].values, "y_pred": Pt.argmax(1).astype(int)}), on="id", how="left")
assert len(sub) == len(sample) and sub["id"].tolist() == sample["id"].tolist(), "id mismatch with sample"
assert sub["y_pred"].notna().all() and set(sub["y_pred"].unique()) <= {0, 1, 2}, "bad predictions"
sub["y_pred"] = sub["y_pred"].astype(int)
sub.to_csv(OUT / "submission_final_qxt.csv", index=False)
np.save(OUT / "test_probs_final_qxt.npy", Pt)
old = pd.read_csv(ROOT / "final_qlt/submission/submission_final_qlt.csv")
json.dump(dict(weights=BLENDS[CAND], cv_macro_f1=c["oof_f1"], per_class=f1_score(y, Pc.argmax(1), average=None).tolist(),
               pred_dist=sub["y_pred"].value_counts(normalize=True).sort_index().to_dict(), comparison=res, decision=decision),
          open(OUT / "cv_score.json", "w"), indent=1)
print("saved", OUT / "submission_final_qxt.csv", "rows", len(sub), "dist",
      sub["y_pred"].value_counts(normalize=True).sort_index().round(3).to_dict(),
      f"| agreement with final_qlt submission {(sub.y_pred.values == old.y_pred.values).mean():.3f}")
