"""Fold-0 model comparison with paired-bootstrap CIs, ensembles, and class-prior tuning (honest via split-half)."""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "experiments/results/transformers"
tr = pd.read_pickle(ROOT / "experiments/results/train_feats.pkl").reset_index(drop=True)
y_all = tr.y.values
a, b = next(StratifiedKFold(5, shuffle=True, random_state=42).split(tr.text, y_all))
y = y_all[b]; script = tr.script.values[b]

LICENSE_OK = {"xlm-roberta-base", "xlm-roberta-large", "microsoft/mdeberta-v3-base", "google/muril-base-cased", "tfidf-lr"}
P = {}
for f in sorted(R.glob("*.json")):
    m = json.load(open(f))["model"]
    P[m] = np.load(R / f"{m.replace('/', '__')}_fold0_probs.npy")
P["tfidf-lr"] = np.load(ROOT / "experiments/results/oof_lr.npy")[b]  # same StratifiedKFold seed -> same fold-0 rows

mf1 = lambda yt, p: f1_score(yt, p, average="macro")
rng = np.random.default_rng(0)
BOOT = [rng.integers(0, len(y), len(y)) for _ in range(1000)]


def boot_ci(pred):
    s = np.array([mf1(y[i], pred[i]) for i in BOOT])
    return np.percentile(s, [2.5, 97.5])


print("=== single models (fold 0, n=%d) ===" % len(y))
rows = []
for m, p in P.items():
    pr = p.argmax(1); lo, hi = boot_ci(pr)
    rows.append(dict(model=m, license_ok=m in LICENSE_OK, macro_f1=mf1(y, pr), ci95=f"[{lo:.3f},{hi:.3f}]",
                     **{f"f1_c{k}": v for k, v in enumerate(f1_score(y, pr, average=None))},
                     **{s: mf1(y[script == s], pr[script == s]) for s in ["latin", "bangla"]}))
print(pd.DataFrame(rows).sort_values("macro_f1", ascending=False).round(4).to_string(index=False))

best_single = max((m for m in P if m in LICENSE_OK and m != "tfidf-lr"), key=lambda m: mf1(y, P[m].argmax(1)))
print("\nbest license-OK single:", best_single)


def paired(pa, pb):
    d = np.array([mf1(y[i], pa[i]) - mf1(y[i], pb[i]) for i in BOOT])
    return d.mean(), np.percentile(d, [2.5, 97.5]), (d <= 0).mean()


print("\n=== paired bootstrap vs best license-OK single (diff, 95% CI, P(diff<=0)) ===")
base = P[best_single].argmax(1)
for m in P:
    if m == best_single:
        continue
    mu, ci, pv = paired(P[m].argmax(1), base)
    print(f"{m:40s} {mu:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] p={pv:.3f}")

print("\n=== equal-weight ensembles (license-OK models only) ===")
ok = [m for m in P if m in LICENSE_OK]
ens = []
for k in range(2, len(ok) + 1):
    for c in itertools.combinations(ok, k):
        pr = np.mean([P[m] for m in c], 0).argmax(1)
        ens.append((mf1(y, pr), c, pr))
ens.sort(key=lambda t: -t[0])
for f, c, _ in ens[:10]:
    print(f"{f:.4f}  {' + '.join(c)}")
f, c, pr = ens[0]
mu, ci, pv = paired(pr, base)
print(f"\nbest ensemble vs {best_single}: {mu:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] P(no gain)={pv:.3f}")
for s in ["latin", "bangla", "mixed"]:
    mk = script == s
    print(f"  {s}: ensemble {mf1(y[mk], pr[mk]):.4f} vs single {mf1(y[mk], base[mk]):.4f}")

print("\n=== class-prior reweighting, honest split-half (tune on half A, eval on half B, and swap) ===")
Pe = np.mean([P[m] for m in c], 0)
half = np.arange(len(y)) % 2 == 0
grid = [(1, w1, w2) for w1 in np.arange(0.7, 1.5, 0.05) for w2 in np.arange(0.7, 2.0, 0.05)]
gains = []
for tune, ev in [(half, ~half), (~half, half)]:
    wbest = max(grid, key=lambda w: mf1(y[tune], (Pe[tune] * w).argmax(1)))
    g = mf1(y[ev], (Pe[ev] * wbest).argmax(1)) - mf1(y[ev], Pe[ev].argmax(1))
    gains.append(g); print("weights", np.round(wbest, 2), f"held-out gain {g:+.4f}")
print(f"mean held-out gain from prior tuning: {np.mean(gains):+.4f}")
