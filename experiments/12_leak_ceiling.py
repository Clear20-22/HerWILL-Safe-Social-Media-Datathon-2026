"""DIAGNOSIS ONLY, never used for any submission: how far can text + id-neighbour features go?
Explains leaderboard scores far above honest text-only CV. For each validation fold, id-neighbour labels
come only from the other folds (mimicking test rows, whose neighbours are all train rows)."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

tr = pd.read_csv("data/train.csv"); y = tr.y.values; ids = tr.id.values
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y))
text = np.load("final_qlt/submission/test_probs_final_qlt.npy") if False else None
Lo = np.zeros((len(y), 3))
for f, (a, b) in enumerate(folds):
    Lo[b] = np.load(f"outputs/cache/xlm-roberta-large_fold{f}.npz", allow_pickle=True)["val"]
P = 0.5 * np.load("qwen/results/cv/oof.npy") + 0.25 * Lo + 0.25 * np.load("outputs/cache/tfidf.npz")["oof"]
f1 = lambda p: f1_score(y, p, average="macro")
print(f"text-only final blend OOF: {f1(P.argmax(1)):.4f}")

N = np.zeros((len(y), 5 * 3 + 2))
for a, b in folds:
    order = np.argsort(ids[a]); kid, ky = ids[a][order], y[a][order]
    for j, i in zip(b, ids[b]):
        pos = np.searchsorted(kid, i); row = []
        for k in (1, 2, 4, 8, 16):
            lo, hi = max(0, pos - k), min(len(kid), pos + k)
            row += list(np.bincount(ky[lo:hi], minlength=3) / (hi - lo))
        gap_l = i - kid[pos - 1] if pos > 0 else 999; gap_r = kid[pos] - i if pos < len(kid) else 999
        N[j] = row + [gap_l, gap_r]
print(f"id-neighbours only (k=1 each side, majority): {f1(N[:, 0:3].argmax(1)):.4f}")
for name, X in [("id-neighbour features only (GBM)", N), ("text blend + id features (GBM)", np.hstack([np.log(P + 1e-6), N]))]:
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0),
                          X, y, cv=StratifiedKFold(5, shuffle=True, random_state=1))
    print(f"{name}: {f1(p):.4f}")
