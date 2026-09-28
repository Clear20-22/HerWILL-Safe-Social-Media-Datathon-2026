# FINAL SUBMISSION — Q+XL+T (team ScatteredBrains)

Everything related to the final submission `submission_final_qxt.csv`. All files here are real copies, not links.
The architecture is 0.50 Qwen3-4B-Base+LoRA + 0.25 XLM-R-XL+LoRA + 0.25 TF-IDF+LogReg, with 5-fold OOF macro-F1 0.6165.

| Folder | What | Where it goes |
|---|---|---|
| `1_kaggle_prediction_upload/submission_final_qxt.csv` | final predictions (md5 f78f32bf…) | Kaggle → Submit Prediction (already submitted) |
| `2_code_form_upload/ScatteredBrains_submission.zip` | **the code-form file** (593 MB): main.py, src, configs, scripts, tests, requirements, README, reports, runs, adapters | Google Form (one file, < 1 GB) |
| `2_code_form_upload/herwill_final_submission.zip` | older package (notebook-only layout), kept for reference | not needed |
| `3_kaggle_notebook/` | the Kaggle notebook + run guide | Kaggle notebook (optional) |
| `model_artifacts/` | original LoRA adapters (fp32), 5 folds x 2 models | backup |
| `cv_predictions/` | per-model OOF/test probabilities, final blend probabilities | backup / reproduction |
| `training_and_blend_scripts/` | the exact scripts that trained and blended | reference |
| `logs/` | training, model-selection, fp16-verification and self-check logs | reference |
| `internal_only_do_not_share/` | test predictions WITH competition text | never upload or share |

Self-check using only this folder: `python FINAL_SUBMISSION_QXT/verify_package.py`
