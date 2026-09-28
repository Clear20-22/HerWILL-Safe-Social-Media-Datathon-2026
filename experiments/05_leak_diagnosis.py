"""DIAGNOSIS ONLY (never used for submission): how high can text-model + id-neighbour leak go?
Explains leaderboard scores far above honest text-only CV."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
tr = pd.read_pickle(ROOT / "experiments/results/train_feats.pkl").reset_index(drop=True)
y = tr.y.values
a, b = next(StratifiedKFold(5, shuffle=True, random_state=42).split(tr.text, y))
R = ROOT / "experiments/results/transformers"
P = np.mean([np.load(R / f"{m}_fold0_probs.npy") for m in
             ["xlm-roberta-large", "microsoft__mdeberta-v3-base", "xlm-roberta-base"]] +
            [np.load(ROOT / "experiments/results/oof_lr.npy")[b]], 0)
f1 = lambda t, p: f1_score(t, p, average="macro")
print(f"text-only ensemble (fold 0): {f1(y[b], P.argmax(1)):.4f}")

# neighbours = labelled rows NOT in fold 0 (mimics test rows, whose neighbours are all train rows)
known = pd.Series(y[a], index=tr.id.values[a]).sort_index()
kid, ky = known.index.values, known.values
feats = []
for i in tr.id.values[b]:
    pos = np.searchsorted(kid, i); row = []
    for k in (1, 2, 4, 8, 16):
        lo, hi = max(0, pos - k), min(len(kid), pos + k)
        row += list(np.bincount(ky[lo:hi], minlength=3) / (hi - lo))
    feats.append(row)
N = np.array(feats)
print(f"id-neighbour only (k=2 majority): {f1(y[b], N[:, 3:6].argmax(1)):.4f}")
X = np.hstack([np.log(P + 1e-6), N])
cvp = cross_val_predict(LogisticRegression(max_iter=2000, class_weight="balanced"), X, y[b], cv=5)
print(f"text + id-leak stacked (honest 5-fold within fold 0): {f1(y[b], cvp):.4f}")
