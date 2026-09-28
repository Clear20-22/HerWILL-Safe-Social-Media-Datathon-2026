"""Fold 0: does replacing / adding XLM-R-large+tricks change the final ensemble? Uses final-run caches."""
import glob
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
tr = pd.read_csv("data/train.csv"); y = tr.y.values
a, b = next(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y)); yt = y[b]
c = lambda t: np.load(f"outputs/cache/{t}_fold0.npz", allow_pickle=True)["val"]
L, M, B = c("xlm-roberta-large"), c("microsoft__mdeberta-v3-base"), c("xlm-roberta-base")
T = np.load("outputs/cache/tfidf.npz")["oof"][b]
tricks = {p.split("seed")[1].split("_")[0]: np.load(p) for p in sorted(glob.glob("experiments/results/tricks/large_tricks_seed*_fold0.npy"))}
f = lambda P: f1_score(yt, P.argmax(1), average="macro")
rng = np.random.default_rng(0); BOOT = [rng.integers(0, len(yt), len(yt)) for _ in range(1000)]
def gain(P, P0):
    p, p0 = P.argmax(1), P0.argmax(1)
    d = np.array([f1_score(yt[i], p[i], average="macro") - f1_score(yt[i], p0[i], average="macro") for i in BOOT])
    return f"{d.mean():+.4f} CI[{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}] P(no gain)={(d<=0).mean():.3f}"
base = (L + M + B + T) / 4
print(f"current ensemble (fold 0):            {f(base):.4f}")
for s, K in tricks.items():
    print(f"[seed {s}] tricks alone: {f(K):.4f}")
    rep = (K + M + B + T) / 4; add = (L + K + M + B + T) / 5
    print(f"  replace plain large -> tricks:      {f(rep):.4f}  gain {gain(rep, base)}")
    print(f"  add tricks as 5th model:            {f(add):.4f}  gain {gain(add, base)}")
