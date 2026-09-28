# Project status / handoff (updated 2026-09-28 10:40 UTC)

Read this first when resuming. The user writes in Bangla; reply in Bangla.

**User rule (2026-09-28):** work ONLY inside this project folder. Do not read or write any other folder
(no `~/.kaggle`, `~/.cache` browsing, other projects, `/tmp`), and do not use the Kaggle API or the Kaggle token on this
machine (it belongs to another account). The user uploads to Kaggle and submits files personally. Keep the work focused.

## Competition
HerWILL Safe Social Media Datathon 2026 (Kaggle). The task is 3-class toxicity classification (0 explicit, 1 subtle, 2 neutral) of Bangla/English/Banglish text. The metric is macro-F1.
The deadline is 2026-09-28 at 6 PM, read as Bangladesh time (12:00 UTC).
Rules: no external data, no id leak, only OSI licenses that allow commercial use. Code is submitted ONCE through a Google Form (not done yet) and must be reproducible. Two final Kaggle picks are allowed.

## Submissions (each in its own folder; never delete old ones)
| File | Architecture | 5-fold OOF | Public LB |
|---|---|---|---|
| `outputs/submission.csv` (#1) | XLM-R large + mDeBERTa + XLM-R base + TF-IDF, equal weights | 0.5936 | 0.5912 |
| `qwen/submission/submission_qwen3_4b.csv` | Qwen3-4B-Base LoRA alone | 0.6016 | – |
| `xlmr_xl/submission/submission_xlmr_xl.csv` | XLM-R XL LoRA alone | 0.5915 | – |
| `blend_qwen_ens/submission/submission_qwen_plus_ensemble_1to1.csv` | Qwen 50% + the #1 ensemble 50% | 0.6117 | – |
| `final_qlt/submission/submission_final_qlt.csv` | Qwen 0.50 + XLM-R large 0.25 + TF-IDF 0.25 | 0.6131 | 0.6097 (rank 8) |
| **`final_qxt/submission/submission_final_qxt.csv`** | **Qwen 0.50 + XLM-R XL 0.25 + TF-IDF 0.25** | **0.6165** | not submitted yet |

Q+XL+T vs Q+L+T on the full OOF: +0.0034, CI [+0.0012, +0.0056], better on 3 of 5 folds (`final_qxt/logs/make_submission.txt`).
It is the FINAL architecture: significant by the pre-set rule, and fully reproducible without training
(XLM-R large weights were never saved, so Q+L+T would need about 4 h of T4 training).

Leaderboard context: ranks 3–5 (~0.655) match text + id leak (our diagnosis gives 0.681, `experiments/12_leak_ceiling.py`). Ranks 1–2 (~0.743) probably use external source data as well. The user asked for a leak-based submission and was told no (rule violation); do not build one. The user asked on 2026-09-28 about reaching ~0.70; answer given: not reachable within the rules; a second Qwen seed (~+0.003–0.006, 4.6 h) was offered but not recommended because of the deadline.

## Final notebook: `kaggle_final/` (done and verified locally)
- `herwill_final_kaggle.py` / `.ipynb`: TF-IDF trains on CPU; Qwen and XL predict from the LoRA adapters; it recomputes the 5-fold OOF; the training code runs with `TRAIN_LLM=1`. Pins transformers 4.57.6 and peft 0.21.0 (pip, Internet ON). bf16 only if compute capability ≥ 8, else fp16.
- Verified in fp16 on the 4090 (`FORCE_FP16=1`): blend OOF 0.6164 vs 0.6165, 99.82% of submission labels identical, 21 min, peak GPU 8.7 GB. Qwen's largest activation is 6,559, about 10% of the fp16 max. Details are in `kaggle_final/README.md`.
- `kaggle_final/dataset/herwill-lora-adapters/` (1.3 GB, hardlinks to `qwen/adapters` and `xlmr_xl/adapters`) is ready for the user to upload.
- First Kaggle run failed: "LoRA adapters ... not found under /kaggle/input" (cause unknown: Dataset not attached, zip, or symlinked mount). The adapter search now follows symlinks, extracts zips and lists the folders it saw; the user must re-import the notebook.

## Code submission (Google Form): `code_submission/herwill_final_submission.zip` (580 MB, built)
- The form wants ONE file ≤ 1 GB: notebook or .py, supporting scripts, requirements, README (mandatory). The CSV goes to Kaggle separately.
- `code_submission/make_package.py` builds it: notebook + .py, README (`package_files/README.md`), requirements, the adapters in fp16 (bit-identical in fp16 inference, `kaggle_final/logs/fp16_adapter_check.txt`), the training scripts and the logs. Predict and `TRAIN_LLM=1` smoke tests of the packaged code pass.
- The same zip also works as the Kaggle Dataset (the notebook extracts it), instead of the 1.3 GB folder.

## Next steps (the user does these on Kaggle and in the form; Claude does not use the Kaggle API)
1. Submit `final_qxt/submission/submission_final_qxt.csv` to see its public LB.
2. Upload the adapters (the zip above, or the 1.3 GB folder) as a private Kaggle Dataset.
3. Import the updated notebook, attach the competition data and the Dataset, choose GPU T4 and Internet ON, then Save & Run All (estimated 1–2 h). Check that its `submission.csv` matches `final_qxt` (~99.8%).
4. Final picks: Q+XL+T first. The notebook reproduces only Q+XL+T, so a second pick of Q+L+T is a hedge whose code (XLM-R large) is `src/herwill_final.py`.
5. Upload `code_submission/herwill_final_submission.zip` in the code Google Form.

## Environment
- Project venv: `.venv` (torch 2.6 cu124, transformers 4.57.6, peft 0.21). GPU: RTX 4090 24 GB, free now.
- Long jobs go in tmux so they survive laptop disconnects. Watch out: `pkill -f` or `pgrep -f` patterns can match your own shell.
- Scripts that load models need `unset TRANSFORMERS_CACHE` (it points to another project's cache); the default HF cache has Qwen and XL.

## 2026-09-28 10:40 UTC: final packaging done
- The user submitted `final_qxt/submission/submission_final_qxt.csv` to Kaggle (final file).
- The **code-form file** is `dist/ScatteredBrains_submission.zip` (593 MB, adapters included, md5 91b0a97f…). It is also copied to
  `FINAL_SUBMISSION_QXT/2_code_form_upload/`. Entry point `main.py --quick | --predict | --full`. Root README.md is new;
  the old one is `README_submission1_encoders.md`, and the old requirements are `requirements_submission1.txt`.
- Verified: unzipped into a fresh folder, `--quick` output is byte-identical (md5 f78f32bf…), pytest 8/8, `--predict` QUICK
  smoke test with the zipped adapters passes, and `id` is used only as a row key.
- `data/` has only `processed/` (ids, folds, script; no text). The raw CSVs used for verification were rebuilt from
  experiments/results/*.pkl in a temp folder and deleted.
- `reports/summary.md` does NOT claim that the organizers confirmed the leak (no `reference/organizer_reply.md` exists).
- `FINAL_SUBMISSION_QXT/` holds everything related to the final submission.
