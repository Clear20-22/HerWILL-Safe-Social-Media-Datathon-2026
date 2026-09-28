"""ID-leak check + classical baselines with stratified 5-fold CV (macro F1)."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
tr = pd.read_pickle(ROOT / "experiments/results/train_feats.pkl").reset_index(drop=True)
te = pd.read_pickle(ROOT / "experiments/results/test_feats.pkl")


def sec(t):
    print("\n" + "=" * 80 + f"\n{t}\n" + "=" * 80)


sec("A. ID STRUCTURE (is id = row index of concatenated source datasets?)")
allid = pd.concat([tr[["id", "script"]].assign(y=tr.y, split="train"), te[["id", "script"]].assign(y=np.nan, split="test")])
allid = allid.sort_values("id").reset_index(drop=True)
print("ids contiguous 0..N-1:", allid.id.min(), allid.id.max(), len(allid), allid.id.is_unique)
# script runs along id
runs = (allid.script != allid.script.shift()).sum()
print(f"script changes along sorted id: {runs} (random expectation ~{int(len(allid)*2*0.577*0.423)})")
s = allid[allid.split == "train"].reset_index(drop=True)
same_next = (s.y.values[1:] == s.y.values[:-1]).mean()
p = s.y.value_counts(normalize=True)
print(f"P(label == next-by-id label) = {same_next:.3f} vs chance {float((p**2).sum()):.3f}")
# coarse segments
allid["seg"] = allid.id // 2000
print(pd.crosstab(allid.seg, allid.script, normalize="index").round(2).join(
    allid[allid.split == "train"].groupby("seg").y.value_counts(normalize=True).unstack().round(2).add_prefix("y")).to_string())

# KNN-by-id predictor (leak strength): predict label of nearest train id neighbours
sid = s.id.values; sy = s.y.values.astype(int)
pred = []
for i in range(len(s)):
    nb = [j for j in (i - 2, i - 1, i + 1, i + 2) if 0 <= j < len(s)]
    pred.append(np.bincount(sy[nb], minlength=3).argmax())
print("macro-F1 using only id-neighbour labels:", round(f1_score(sy, pred, average="macro"), 4))

sec("B. TF-IDF BASELINES, 5-fold stratified CV")
X_text = tr.text.values; y = tr.y.values
skf = StratifiedKFold(5, shuffle=True, random_state=42)


def cv(make, name):
    oof = np.zeros((len(tr), 3)); preds = np.zeros(len(tr), int)
    for f, (a, b) in enumerate(skf.split(X_text, y)):
        wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=300_000, sublinear_tf=True,
                             token_pattern=r"(?u)[ঀ-৿]+|\b\w+\b")
        cv_ = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=500_000, sublinear_tf=True)
        Xa = hstack([wv.fit_transform(X_text[a]), cv_.fit_transform(X_text[a])]).tocsr()
        Xb = hstack([wv.transform(X_text[b]), cv_.transform(X_text[b])]).tocsr()
        m = make(); m.fit(Xa, y[a])
        preds[b] = m.predict(Xb)
        if hasattr(m, "predict_proba"):
            oof[b] = m.predict_proba(Xb)
    f1 = f1_score(y, preds, average="macro")
    print(f"\n### {name}: macro-F1 = {f1:.4f}")
    print(classification_report(y, preds, digits=4))
    print("confusion (rows=true):\n", confusion_matrix(y, preds))
    for sc in ["latin", "bangla", "mixed"]:
        mk = tr.script.values == sc
        print(f"  {sc:7s} n={mk.sum():6d} macroF1={f1_score(y[mk], preds[mk], average='macro'):.4f} "
              f"perclass={np.round(f1_score(y[mk], preds[mk], average=None), 3)}")
    return oof, preds


oof_lr, _ = cv(lambda: LogisticRegression(C=4, max_iter=3000, class_weight="balanced"), "LR word+char, balanced")
cv(lambda: LinearSVC(C=0.3, class_weight="balanced"), "LinearSVC word+char, balanced")
cv(lambda: LogisticRegression(C=4, max_iter=3000), "LR word+char, unweighted")

sec("C. THRESHOLD / PRIOR TUNING ON OOF PROBS (LR balanced)")
best = (f1_score(y, oof_lr.argmax(1), average="macro"), (1, 1, 1))
for w1 in np.arange(0.6, 1.6, 0.05):
    for w2 in np.arange(0.6, 2.0, 0.05):
        f = f1_score(y, (oof_lr * [1, w1, w2]).argmax(1), average="macro")
        if f > best[0]:
            best = (f, (1, round(w1, 2), round(w2, 2)))
print("argmax F1:", round(f1_score(y, oof_lr.argmax(1), average='macro'), 4), "| best class-weighted:", round(best[0], 4), best[1])
np.save(ROOT / "experiments/results/oof_lr.npy", oof_lr)
