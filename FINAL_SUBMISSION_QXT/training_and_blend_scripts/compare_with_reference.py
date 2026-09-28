"""Compare a run of kaggle_final/herwill_final_kaggle.py with the reference artifacts it should reproduce:
the saved 5-fold probabilities of each component and the blended submission file.
Usage (from project root): python kaggle_final/scripts/compare_with_reference.py RUN_DIR [REFERENCE_SUBMISSION.csv]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(sys.argv[1])
REF_SUB = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "final_qxt/submission/submission_final_qxt.csv"
tr = pd.read_csv(ROOT / "data/train.csv"); y = tr.y.values
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y))


def xlmr_large():
    o, t = np.zeros((len(tr), 3)), 0
    for f, (a, b) in enumerate(folds):
        d = np.load(ROOT / f"outputs/cache/xlm-roberta-large_fold{f}.npz", allow_pickle=True); o[b] = d["val"]; t = t + d["test"] / 5
    return o, t


REF = {"tfidf": lambda: tuple(np.load(ROOT / "outputs/cache/tfidf.npz")[k] for k in ("oof", "test")),
       "qwen": lambda: (np.load(ROOT / "qwen/results/cv/oof.npy"), np.load(ROOT / "qwen/results/cv/test.npy")),
       "xl": lambda: (np.load(ROOT / "xlmr_xl/results/cv/oof.npy"), np.load(ROOT / "xlmr_xl/results/cv/test.npy")),
       "xlmr_large": xlmr_large}
rep = json.load(open(RUN / "cv_report.json"))
print(f"run: {RUN}  mode={rep['mode']} dtype={rep['dtype']} minutes={rep['minutes']}")
for name in rep["weights"]:
    ro, rt = REF[name]()
    no, nt = np.load(RUN / f"probs/{name}_oof.npy"), np.load(RUN / f"probs/{name}_test.npy")
    has = no.sum(1) > 0  # rows with an OOF prediction in this run
    line = (f"{name:6s} test: max|dp|={np.abs(nt - rt).max():.4f} mean|dp|={np.abs(nt - rt).mean():.5f} "
            f"argmax agree={(nt.argmax(1) == rt.argmax(1)).mean():.4f}")
    if has.any():
        line += (f" | OOF rows={has.sum()}: argmax agree={(no[has].argmax(1) == ro[has].argmax(1)).mean():.4f} "
                 f"F1 run={f1_score(y[has], no[has].argmax(1), average='macro'):.4f} "
                 f"ref={f1_score(y[has], ro[has].argmax(1), average='macro'):.4f}")
    print(line)
sub, ref = pd.read_csv(RUN / "submission.csv"), pd.read_csv(REF_SUB)
assert sub["id"].tolist() == ref["id"].head(len(sub)).tolist()
print(f"submission vs {REF_SUB.relative_to(ROOT)}: label agreement "
      f"{(sub.y_pred.values == ref.y_pred.values[:len(sub)]).mean():.4f} on {len(sub)} rows")
