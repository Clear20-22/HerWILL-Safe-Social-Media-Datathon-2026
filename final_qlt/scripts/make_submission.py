"""Final architecture: 0.50 * Qwen3-4B-LoRA + 0.25 * XLM-R-large + 0.25 * TF-IDF-LR (all 5-fold, soft voting).
Chosen on 5-fold OOF: it matches the 5-component blend (0.6131 vs 0.6117, difference not significant) with fewer parts.
Uses saved probabilities only. Usage (from project root): python final_qlt/scripts/make_submission.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "final_qlt/submission"
W = {"qwen": 0.50, "xlmr_large": 0.25, "tfidf": 0.25}
tr = pd.read_csv(ROOT / "data/train.csv"); te = pd.read_csv(ROOT / "data/test.csv"); sample = pd.read_csv(ROOT / "data/sample_submission.csv")
y = tr.y.values
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y))

Lo, Lt = np.zeros((len(tr), 3)), np.zeros((len(te), 3))
for f, (a, b) in enumerate(folds):
    d = np.load(ROOT / f"outputs/cache/xlm-roberta-large_fold{f}.npz", allow_pickle=True); Lo[b] = d["val"]; Lt += d["test"] / 5
T = np.load(ROOT / "outputs/cache/tfidf.npz"); To, Tt = T["oof"], T["test"]
Qo, Qt = np.load(ROOT / "qwen/results/cv/oof.npy"), np.load(ROOT / "qwen/results/cv/test.npy")

P = W["qwen"] * Qo + W["xlmr_large"] * Lo + W["tfidf"] * To
Pt = W["qwen"] * Qt + W["xlmr_large"] * Lt + W["tfidf"] * Tt
cv = f1_score(y, P.argmax(1), average="macro")
for n, o in [("qwen", Qo), ("xlmr_large", Lo), ("tfidf", To), ("FINAL", P)]:
    print(f"{n:11s} OOF macro-F1 = {f1_score(y, o.argmax(1), average='macro'):.4f}")
print(classification_report(y, P.argmax(1), digits=4))

sub = sample[["id"]].merge(pd.DataFrame({"id": te["id"].values, "y_pred": Pt.argmax(1).astype(int)}), on="id", how="left")
assert len(sub) == len(sample) and sub["id"].tolist() == sample["id"].tolist(), "id mismatch with sample"
assert sub["y_pred"].notna().all() and set(sub["y_pred"].unique()) <= {0, 1, 2}, "bad predictions"
sub["y_pred"] = sub["y_pred"].astype(int)
sub.to_csv(OUT / "submission_final_qlt.csv", index=False)
np.save(OUT / "test_probs_final_qlt.npy", Pt)
json.dump(dict(weights=W, cv_macro_f1=float(cv), per_class=f1_score(y, P.argmax(1), average=None).tolist(),
               pred_dist=sub["y_pred"].value_counts(normalize=True).sort_index().to_dict()), open(OUT / "cv_score.json", "w"), indent=1)
print("saved", OUT / "submission_final_qlt.csv", "rows", len(sub), "dist", sub["y_pred"].value_counts(normalize=True).sort_index().round(3).to_dict())
