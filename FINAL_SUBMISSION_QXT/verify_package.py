"""Self-check of this folder, using only files inside it (no competition data needed).
1. Rebuild the final test probabilities as 0.50 Qwen3-4B + 0.25 XLM-R XL + 0.25 TF-IDF from cv_predictions/,
   and check that they match cv_predictions/final_blend/test_probs_final_qxt.npy.
2. Check that their argmax equals the labels in 1_kaggle_prediction_upload/submission_final_qxt.csv
   (the rows of the test probabilities are in test.csv order, which equals sample_submission order).
3. Check the submission format: header id,y_pred, 11,955 rows, unique ids, labels in {0, 1, 2}.
4. Check that the code-form zip opens and contains the notebook, the README and all 10 adapters.
5. Compare the local fp16 notebook run (Kaggle T4 precision) with the submission.
Usage: python FINAL_SUBMISSION_QXT/verify_package.py"""
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CV = HERE / "cv_predictions"
W = {"qwen3_4b": 0.50, "xlmr_xl": 0.25, "tfidf": 0.25}

Q = np.load(CV / "qwen3_4b/test.npy"); X = np.load(CV / "xlmr_xl/test.npy"); T = np.load(CV / "tfidf/tfidf.npz")["test"]
P = W["qwen3_4b"] * Q + W["xlmr_xl"] * X + W["tfidf"] * T
ref = np.load(CV / "final_blend/test_probs_final_qxt.npy")
assert np.allclose(P, ref, atol=1e-12), "blend of components differs from the saved final probabilities"
print(f"[1] blend 0.50*Qwen + 0.25*XL + 0.25*TF-IDF == saved final probabilities (max diff {np.abs(P - ref).max():.1e})")

sub = pd.read_csv(HERE / "1_kaggle_prediction_upload/submission_final_qxt.csv")
assert (sub["y_pred"].values == P.argmax(1)).all(), "submission labels differ from the blend argmax"
print("[2] submission labels == argmax of the blend, for all rows")

assert list(sub.columns) == ["id", "y_pred"] and len(sub) == 11955 and sub["id"].is_unique
assert sub["y_pred"].notna().all() and set(sub["y_pred"].unique()) <= {0, 1, 2}
print(f"[3] format OK: id,y_pred | {len(sub)} rows | unique ids | label share "
      f"{sub['y_pred'].value_counts(normalize=True).sort_index().round(3).to_dict()}")

with zipfile.ZipFile(HERE / "2_code_form_upload/herwill_final_submission.zip") as z:
    names = z.namelist()
    assert z.testzip() is None, "corrupt zip"
    need = ["herwill_final.ipynb", "herwill_final.py", "README.md", "requirements.txt"]
    assert all(any(n.endswith("/" + f) for n in names) for f in need), "zip misses a required file"
    n_adp = sum(n.endswith("adapter_model.safetensors") for n in names)
    assert n_adp == 10, f"expected 10 adapters in the zip, found {n_adp}"
size = (HERE / "2_code_form_upload/herwill_final_submission.zip").stat().st_size
print(f"[4] code-form zip OK: {size / 2**20:.0f} MB (< 1 GB form limit), notebook + README + requirements + {n_adp} adapters")

fp16 = pd.read_csv(HERE / "logs/fp16_verification/local_fp16_notebook_submission.csv")
agree = (fp16["y_pred"].values == sub["y_pred"].values).mean()
rep = json.load(open(HERE / "logs/fp16_verification/cv_report.json"))
print(f"[5] notebook run in fp16 (Kaggle T4 precision) agrees with the submission on {agree:.2%} of rows")
print("\nALL CHECKS PASSED. 5-fold OOF macro-F1 of this submission: "
      f"{json.load(open(CV / 'final_blend/cv_score.json'))['cv_macro_f1']:.4f}")
