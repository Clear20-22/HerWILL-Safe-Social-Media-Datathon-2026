"""Test-set prediction table of the final model (Q+XL+T): id, text, prediction, label name and class probabilities.
For our own inspection only: it contains competition text, so it must not be published or put in the code submission.
Written as UTF-8 with BOM so that Excel shows Bangla correctly.
Usage (from project root): python final_qxt/scripts/make_prediction_table.py"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "final_qxt/predictions"; OUT.mkdir(exist_ok=True)
te = pd.read_csv(ROOT / "data/test.csv")
P = np.load(ROOT / "final_qxt/submission/test_probs_final_qxt.npy")  # rows in test.csv order
sub = pd.read_csv(ROOT / "final_qxt/submission/submission_final_qxt.csv")
assert len(P) == len(te)

names = {0: "explicit", 1: "subtle", 2: "neutral"}
t = pd.DataFrame({"id": te["id"], "text": te["text"], "y_pred": P.argmax(1)})
t["label"] = t["y_pred"].map(names)
for k, n in names.items():
    t[f"p_{n}"] = P[:, k].round(4)
assert (t.merge(sub, on="id", suffixes=("", "_sub")).eval("y_pred == y_pred_sub")).all(), "differs from the submitted CSV"
t.to_csv(OUT / "test_predictions_final_qxt.csv", index=False, encoding="utf-8-sig")
print(f"saved {OUT / 'test_predictions_final_qxt.csv'}  rows={len(t)}  "
      f"labels={t['label'].value_counts().to_dict()}  | matches the submitted CSV")
