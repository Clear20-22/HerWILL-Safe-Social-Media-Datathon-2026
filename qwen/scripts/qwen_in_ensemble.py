"""Fold 0: does adding a Qwen LoRA model improve the final 4-model ensemble? Uses final-run caches.
Usage (from project root): python qwen/scripts/qwen_in_ensemble.py [probs.npy ...]  (default: every qwen/results/*_fold0.npy)"""
import glob
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

tr = pd.read_csv("data/train.csv"); y = tr.y.values
a, b = next(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y)); yt = y[b]
_bn, _lat = r"[ঀ-৿]", r"[A-Za-z]"
nb, nl = tr.text.str.count(_bn).values[b], tr.text.str.count(_lat).values[b]
script = np.where(nb + nl == 0, "other", np.where(nb / np.maximum(nb + nl, 1) > .9, "bangla",
                  np.where(nb / np.maximum(nb + nl, 1) < .1, "latin", "mixed")))
c = lambda t: np.load(f"outputs/cache/{t}_fold0.npz", allow_pickle=True)["val"]
L, M, B = c("xlm-roberta-large"), c("microsoft__mdeberta-v3-base"), c("xlm-roberta-base")
T = np.load("outputs/cache/tfidf.npz")["oof"][b]
base = (L + M + B + T) / 4

rng = np.random.default_rng(0); BOOT = [rng.integers(0, len(yt), len(yt)) for _ in range(1000)]
f = lambda P, m=slice(None): f1_score(yt[m], P[m].argmax(1), average="macro")


def gain(P, P0):
    p, p0 = P.argmax(1), P0.argmax(1)
    d = np.array([f1_score(yt[i], p[i], average="macro") - f1_score(yt[i], p0[i], average="macro") for i in BOOT])
    return f"{d.mean():+.4f} CI[{np.percentile(d, 2.5):+.4f},{np.percentile(d, 97.5):+.4f}] P(no gain)={(d <= 0).mean():.3f}"


def row(name, P):
    parts = " ".join(f"{s}={f(P, script == s):.4f}" for s in ["latin", "bangla"])
    pc = f1_score(yt, P.argmax(1), average=None).round(3).tolist()
    print(f"{name:34s} macroF1={f(P):.4f}  per-class={pc}  {parts}")


row("current ensemble (4 models)", base)
row("xlm-roberta-large alone", L)
for p in (sys.argv[1:] or sorted(glob.glob("qwen/results/*_fold0.npy"))):
    Q = np.load(p); tag = p.split("/")[-1].replace("_fold0.npy", "")
    print(f"\n=== {tag}")
    row("  alone", Q)
    for w in (1, 2):
        E = (L + M + B + T + w * Q) / (4 + w)
        row(f"  ensemble + qwen (weight {w})", E)
        print(f"  {'':32s} gain vs current: {gain(E, base)}")
    E = (L + M + T + Q) / 4
    row("  swap xlmr-base -> qwen", E)
    print(f"  {'':32s} gain vs current: {gain(E, base)}")
