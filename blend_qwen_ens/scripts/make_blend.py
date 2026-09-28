"""Blend: Qwen3-4B LoRA (5-fold) and the Submission #1 encoder ensemble (5-fold), 1:1 probability average.
Inputs are only saved OOF/test probabilities; nothing is retrained. Usage (from project root): python blend_qwen_ens/scripts/make_blend.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "blend_qwen_ens/submission"
y = pd.read_csv(ROOT / "data/train.csv").y.values
sample = pd.read_csv(ROOT / "data/sample_submission.csv"); test = pd.read_csv(ROOT / "data/test.csv")
Q, Qt = np.load(ROOT / "qwen/results/cv/oof.npy"), np.load(ROOT / "qwen/results/cv/test.npy")
E, Et = np.load(ROOT / "outputs/oof_ensemble.npy"), np.load(ROOT / "outputs/test_ensemble.npy")
W_QWEN = 0.5  # 1:1. Weights 1:1 to 4:1 were compared on OOF; 1:1 and 2:1 tie (0.6117 / 0.6115), so the simpler 1:1 is used

P, Pt = W_QWEN * Q + (1 - W_QWEN) * E, W_QWEN * Qt + (1 - W_QWEN) * Et
cv = f1_score(y, P.argmax(1), average="macro")
print(f"OOF macro-F1: blend={cv:.4f}  qwen={f1_score(y, Q.argmax(1), average='macro'):.4f}  "
      f"submission#1={f1_score(y, E.argmax(1), average='macro'):.4f}")
print(classification_report(y, P.argmax(1), digits=4))

sub = sample[["id"]].merge(pd.DataFrame({"id": test["id"].values, "y_pred": Pt.argmax(1).astype(int)}), on="id", how="left")
assert len(sub) == len(sample) and sub["id"].tolist() == sample["id"].tolist(), "id mismatch with sample"
assert sub["y_pred"].notna().all() and set(sub["y_pred"].unique()) <= {0, 1, 2}, "bad predictions"
sub["y_pred"] = sub["y_pred"].astype(int)
sub.to_csv(OUT / "submission_qwen_plus_ensemble_1to1.csv", index=False)
json.dump(dict(cv_macro_f1=float(cv), weight_qwen=W_QWEN, per_class=f1_score(y, P.argmax(1), average=None).tolist(),
               pred_dist=sub["y_pred"].value_counts(normalize=True).sort_index().to_dict()), open(OUT / "cv_score.json", "w"), indent=1)
print("saved", OUT / "submission_qwen_plus_ensemble_1to1.csv", "rows", len(sub),
      "dist", sub["y_pred"].value_counts(normalize=True).sort_index().round(3).to_dict())
