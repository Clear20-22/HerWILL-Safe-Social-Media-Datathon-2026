"""Fold-0 test of two text-only ideas: (1) exact-duplicate label lookup, (2) TF-IDF kNN label features + stacking."""
import re, unicodedata
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

tr = pd.read_pickle("experiments/results/train_feats.pkl").reset_index(drop=True); y = tr.y.values
a, b = next(StratifiedKFold(5, shuffle=True, random_state=42).split(tr.text, y)); yt = y[b]
R = "experiments/results/transformers/"
P = np.mean([np.load(R + m + "_fold0_probs.npy") for m in ["xlm-roberta-large", "microsoft__mdeberta-v3-base", "xlm-roberta-base"]]
            + [np.load("experiments/results/oof_lr.npy")[b]], 0)
f = lambda p: f1_score(yt, p, average="macro")
rng = np.random.default_rng(0); BOOT = [rng.integers(0, len(yt), len(yt)) for _ in range(1000)]
def gain(p, p0):
    d = np.array([f1_score(yt[i], p[i], average="macro") - f1_score(yt[i], p0[i], average="macro") for i in BOOT])
    return f"{d.mean():+.4f} CI[{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}] P(no gain)={(d<=0).mean():.3f}"
p0 = P.argmax(1); print(f"baseline ensemble: {f(p0):.4f}")

# (1) exact duplicate lookup (normalized text), majority label from the training part only
norm = lambda s: re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s)).lower()).strip()
key = tr.text.map(norm).values
maj = pd.Series(y[a]).groupby(key[a]).agg(lambda s: s.value_counts().idxmax())
hit = np.array([k in maj.index for k in key[b]])
p1 = p0.copy(); p1[hit] = maj.loc[key[b][hit]].values
print(f"(1) dup lookup: {hit.sum()} val rows ({hit.mean()*100:.2f}%) | lookup acc on them={np.mean(p1[hit]==yt[hit]):.3f} "
      f"vs model acc={np.mean(p0[hit]==yt[hit]):.3f} | macroF1={f(p1):.4f} gain {gain(p1, p0)}")

# (2) kNN label features from char TF-IDF cosine neighbours in the training part
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=300_000, sublinear_tf=True)
Xa = vec.fit_transform(tr.text.values[a]); Xb = vec.transform(tr.text.values[b])
S = (Xb @ Xa.T).toarray() if Xb.shape[0] * Xa.shape[0] < 5e8 else None
feats = []
for k in (5, 20, 50):
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    w = np.take_along_axis(S, idx, 1)
    lab = y[a][idx]
    feats.append(np.stack([(w * (lab == c)).sum(1) / (w.sum(1) + 1e-9) for c in range(3)], 1))
feats.append(S.max(1, keepdims=True))
K = np.hstack(feats)
print(f"(2) kNN(k=20) alone: {f(K[:, 3:6].argmax(1)):.4f}")
X = np.hstack([np.log(P + 1e-6), K])
cvp = cross_val_predict(LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0), X, yt, cv=5)
cv0 = cross_val_predict(LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0), np.log(P + 1e-6), yt, cv=5)
print(f"    stack(ensemble only) {f(cv0):.4f} | stack(ensemble + kNN) {f(cvp):.4f} | gain vs stack-only {gain(cvp, cv0)} | vs argmax {gain(cvp, p0)}")
