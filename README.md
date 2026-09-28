# HerWILL Safe Social Media Datathon 2026 — team **ScatteredBrains**

This package classifies harmful Bangla / English / Banglish social-media posts into 3 classes (0 explicit, 1 subtle,
2 neutral). The metric is macro-F1.

## Final scores
| | Macro-F1 |
|---|---|
| **Final submission** (`submissions/submission_final_qxt.csv`), 5-fold OOF on all 47,817 training posts | **0.6165** |
| Previous version (XLM-R large instead of XL), public leaderboard / OOF | 0.6097 / 0.6131 |
| First version (4-encoder ensemble), public leaderboard / OOF | 0.5912 / 0.5936 |

## Approach
The final model is a soft vote of three **text-only** models. Each is trained with the same 5-fold stratified CV, and its test
probabilities are averaged over the 5 fold models:

| Weight | Model | What it contributes | OOF macro-F1 |
|---|---|---|---|
| 0.50 | **Qwen3-4B-Base + LoRA** (r=16 on all attention/MLP projections, linear head on the last token, instruction prefix) | context, sarcasm, subtle toxicity | 0.6016 |
| 0.25 | **XLM-RoBERTa-XL + LoRA** (r=16 on 216 encoder modules, standard classification head) | bidirectional multilingual encoder, errors differ from Qwen's | 0.5915 |
| 0.25 | **TF-IDF word 1–2-gram + char_wb 2–5-gram + LogisticRegression** (C=4, balanced) | spelling variants, censored slurs, Banglish | 0.5277 |

Shared setup:
* **Preprocessing:** NFKC, character repeats capped at 3, and whitespace collapsed. Emojis, casing and punctuation are kept.
* **Loss:** class-weighted cross-entropy with weights (N/n_c)^0.5.
* **Training:** AdamW, linear warm-up then decay, bf16. Every fold keeps the epoch with the best validation macro-F1.

Full tables, efficiency numbers, training curves, ablations and the ethics discussion are in **[reports/summary.md](reports/summary.md)**.

## Hardware and software
* The reported runs used an NVIDIA **RTX 4090 24 GB**, Ubuntu Linux (kernel 6.8), Python 3.13, CUDA 12.4.
* **Predict mode needs about 9 GB of GPU memory and runs in fp16 on a Kaggle T4.** Retraining the LoRA models needs a GPU with bf16 and 24 GB.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```
The pretrained weights `Qwen/Qwen3-4B-Base` (Apache-2.0, 8 GB) and `facebook/xlm-roberta-xl` (MIT, 14 GB) are downloaded
from the Hugging Face Hub on first use.

## Data
Put the three competition files, unchanged, in `data/raw/`:
```
data/raw/train.csv  data/raw/test.csv  data/raw/sample_submission.csv
```
Competition data is not redistributed in this package. `data/processed/` is already included; it holds only ids, the fold
of each training row and its writing system, and no text. You can rebuild it with `python scripts/prepare_data.py`.

## Quick reproduction (CPU, < 1 s)
```bash
python main.py --quick
```
This rebuilds `submissions/submission_final_qxt.csv` from the saved 5-fold test probabilities in `runs/`. It validates
the format (columns `id,y_pred`, 11,955 rows, ids identical to `sample_submission.csv`, integer labels in {0,1,2}), writes
`submissions/submission_final_qxt.manifest.json`, and checks that the file is byte-identical to the file submitted to
Kaggle (md5 `f78f32bf1b4a43983ff98dd2f4d32a6d`). Tests: `python -m pytest -q tests/`.

## Full reproduction
The steps, in order, with times measured on the RTX 4090:

| Command | What it does | Time |
|---|---|---|
| `python main.py --predict` | prepare folds, retrain TF-IDF, recompute Qwen and XL probabilities from the **saved LoRA adapters**, then blend | 21 min (fp16) |
| `python main.py --full` | prepare folds, train TF-IDF (2.3 min), train Qwen3-4B LoRA for 5 folds x 2 epochs (4.6 h), train XLM-R XL LoRA for 5 folds x 3 epochs (2.0 h), then blend | about 7 h |

Both modes run `src/herwill_final_kaggle.py` and write into `runs_predict/` or `runs_full/`. The `.ipynb` version of the
same code (`notebooks/herwill_final_kaggle.ipynb`) runs on Kaggle:
1. Attach the competition data and the adapters as a Dataset.
2. Choose GPU T4 and Internet ON.
3. Run all cells; it takes about 1–2 h on a T4.

A fresh `--full` run gives *similar*, not bit-identical, results, because GPU kernels are not deterministic. `--predict`
reproduces 99.8% of the submitted labels (blend OOF 0.6164 vs 0.6165).

## LoRA adapters
`adapters/qwen3_4b/fold0..4` and `adapters/xlmr_xl/fold0..4` are included, stored in fp16 (about 63 MB each). They are
bit-identical in fp16 inference (`logs/fp16_verification/`). If they are missing, retrain them with
`python main.py --full`, or with the exact training scripts `src/training/qwen_lora_cv.py` and
`src/training/xlmr_xl_cv.py`, which are kept as they ran (project-relative paths).

## Project structure
```
main.py                      entry point (--quick / --predict / --full)
configs/final_qxt.json       blend weights, paths, expected md5
src/herwill_final_kaggle.py  end-to-end pipeline: TF-IDF training, LoRA training (TRAIN_LLM=1) or prediction, blend
src/training/                the exact training / model-selection scripts behind the saved runs
src/herwill_final.py         first version: 4-encoder pipeline (also the source of the TF-IDF run)
notebooks/                   the Kaggle notebook (same code as src/herwill_final_kaggle.py)
scripts/prepare_data.py      folds, test ids and writing system -> data/processed/
scripts/make_reports.py      tables, per-run summaries and figures from runs/
tests/                       pytest: format, blend reproduction, byte-identity, no id features
data/processed/              folds.csv (id, fold, script), test_ids.csv
runs/<model>/                oof.npy, test.npy, fold*.npz (val/test probs + loss history), config.json, summary.json, log.txt, curves.png
submissions/                 final CSV + JSON manifest
adapters/                    LoRA adapters (fp16) of Qwen3-4B and XLM-R XL, 5 folds each
reports/                     summary.md and figures/
logs/                        fp16 verification and model-selection logs
```

## Design decisions
* **Text only.** The `id` column leaks labels through the row order of the source corpora. Id-neighbour features alone
  reach ≈ 0.594 macro-F1. We measured it and excluded it; `id` is used only as a row key (see `reports/summary.md` §5).
* **LoRA on a 4B decoder.** It was the largest single gain (+0.008 over the 4-encoder ensemble, significant), mostly on
  subtle toxicity. A 4B model runs on a Kaggle T4 for inference; 8B would not.
* **XLM-R XL over XLM-R large.** +0.0034 in the blend (CI [+0.0012, +0.0056]), and it is reproducible from adapters
  without 4 h of retraining.
* **Averaging instead of stacking or threshold tuning.** Both alternatives were worse on held-out data.
* **Every choice was made on 5-fold OOF with paired bootstrap CIs**, never on the public leaderboard. The public scores
  land within 0.004 of the OOF scores.
* **Licenses.** Only licenses that allow commercial use (Apache-2.0 / MIT). BanglaBERT (CC BY-NC-SA) and Twitter-XLM-R
  (no license) were evaluated but excluded. No external data.
