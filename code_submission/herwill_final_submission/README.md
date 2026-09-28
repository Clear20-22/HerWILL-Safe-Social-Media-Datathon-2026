# HerWILL Safe Social Media Datathon 2026: final code

3-class harmful-content classification of Bangla, English and Banglish social-media text
(0 = explicitly toxic, 1 = subtly toxic, 2 = neutral). The metric is macro F1.
This package reproduces our final Kaggle submission from the competition CSVs.

## Solution
A soft vote of three models, all trained on the same 5-fold stratified split (`StratifiedKFold`, shuffle, seed 42):

| Component | Weight | 5-fold OOF macro-F1 |
|---|---|---|
| Qwen3-4B-Base + LoRA (decoder LLM used as a classifier) | 0.50 | 0.6016 |
| XLM-RoBERTa-XL + LoRA (3.5B multilingual encoder) | 0.25 | 0.5915 |
| TF-IDF (word 1-2 + char 2-5 grams) + logistic regression | 0.25 | 0.5277 |
| **Soft-vote blend** | | **0.6165** |

The notebook explains every step: preprocessing, folds, class weights, the three models, the blend, and the
evaluation (per class and per script).

## Files
```
herwill_final.ipynb   MAIN NOTEBOOK: the whole pipeline, from the CSVs to submission.csv
herwill_final.py      the same code as a Python script
adapters/             the trained LoRA adapters and classification heads, one per fold (fp16, about 660 MB)
  qwen3_4b/fold0-4/   for Qwen/Qwen3-4B-Base
  xlmr_xl/fold0-4/    for facebook/xlm-roberta-xl
training_scripts/     the exact scripts that trained the adapters (see "Training" below)
logs/                 training logs, the fp16 verification run and the model-selection log
requirements.txt      library versions used for the reported run
```
The competition data is not included, because it must not be redistributed.

## How to run
Internet access is needed to download the two pretrained base models from the Hugging Face Hub:
`Qwen/Qwen3-4B-Base` (8 GB, Apache-2.0) and `facebook/xlm-roberta-xl` (14 GB, MIT).

By default the notebook runs in **predict mode**:
* TF-IDF + logistic regression is trained inside the notebook, on CPU, in about 2 minutes.
* Qwen and XLM-R XL load the adapter of each fold from `adapters/`. They predict the test set and also each fold's
  held-out rows, so the notebook reproduces both the submission and the 5-fold CV score.
* It takes about 20 minutes on an RTX 4090. On a Kaggle T4 we estimate 1 to 2 hours.

### Option A: Kaggle notebook (GPU T4)
1. Upload this zip file (or the unzipped folder) as a new Kaggle Dataset.
2. Create a notebook from `herwill_final.ipynb` (File → Import Notebook).
3. Add Input: the competition data and the Dataset from step 1.
4. Settings: Accelerator **GPU T4**, Internet **ON**. Then Run All.

The notebook installs the pinned `transformers` and `peft`, and finds the data and the adapters anywhere under
`/kaggle/input`. If Kaggle kept the zip unextracted, the notebook extracts it. It writes
`/kaggle/working/submission.csv` and `/kaggle/working/cv_report.json`.

### Option B: local GPU (16 GB or more)
```bash
pip install -r requirements.txt
# put train.csv, test.csv and sample_submission.csv in data/ (next to this README)
python herwill_final.py      # or run herwill_final.ipynb; writes results/submission.csv and results/cv_report.json
```

Optional environment variables:

| Variable | Effect |
|---|---|
| `FORCE_FP16=1` | Use fp16 even on a bf16 GPU, which reproduces T4 numerics. |
| `RECOMPUTE_OOF=0` | Predict only the test set. This is about 2x faster, but gives no CV score. |
| `QUICK=1` | Smoke test on a few rows of folds 0 and 1, which takes about 2 minutes. |
| `TRAIN_LLM=1` | Retrain both LoRA models instead of loading `adapters/`. This needs a bf16 GPU (Ampere or newer) with 24 GB and takes about 7 hours on an RTX 4090. |

## Reproducibility checks
We ran this notebook in fp16, the precision a T4 uses, on an RTX 4090, and compared it with the training run
(bf16). The log is `logs/local_fp16_run.txt`, and the comparison is `logs/compare_local_fp16_vs_reference.txt`.

| Component | OOF F1, notebook fp16 | OOF F1, training run | Test labels identical |
|---|---|---|---|
| Qwen3-4B + LoRA | 0.6013 | 0.6016 | 99.76% |
| XLM-R XL + LoRA | 0.5915 | 0.5915 | 99.92% |
| TF-IDF + LR | 0.5277 | 0.5277 | 100% |
| **Blend** | **0.6164** | **0.6165** | **99.82% of the submitted labels** |

* **fp16 safety:** Qwen's largest activation is 6,559, about 10% of the fp16 limit (`logs/fp16_headroom_qwen.txt`). The
  notebook also stops with an error if any output is not finite.
* **Peak GPU memory:** 8.7 GB (Qwen) and 7.4 GB (XL), so a 16 GB T4 is enough.
* **fp16 adapters:** they are stored in fp16 to fit the 1 GB upload limit. In fp16 inference their outputs are
  bit-identical to the original fp32 adapters (400 held-out and 400 test rows per model; `logs/fp16_adapter_check.txt`).
* **Seeds and determinism:** seeds and folds are fixed. TF-IDF is bit-identical with the same number of CPU threads.
  GPU results can differ very slightly between hardware.

## Training
The adapters were trained with 5-fold CV on one RTX 4090, in bf16:
* **Qwen3-4B-Base + LoRA:** r=16, alpha=32 on all attention and MLP projections, a short task instruction before the
  text, lr 1e-4, 2 epochs, about 56 minutes per fold.
* **XLM-R XL + LoRA:** r=16, alpha=32 on the attention and FFN dense layers, lr 2e-4, 3 epochs, about 25 minutes per fold.
* **Both:** class-weighted cross-entropy with weight (N/n_c)^0.5 and effective batch 32. For each fold, the epoch with
  the best held-out macro-F1 is kept.

`training_scripts/qwen_lora_cv.py` and `training_scripts/xlmr_xl_cv.py` are the exact scripts that produced
`adapters/`. Their logs are `logs/qwen3_4b_lora_cv.txt` and `logs/xlmr_xl_cv.txt`. The scripts expect our original
project layout (`data/`, `qwen/`, `xlmr_xl/`). The notebook's `TRAIN_LLM=1` mode runs the same training loop, with the
same hyper-parameters and seeds, inside the notebook.

**Model selection:** replacing XLM-R XL with XLM-R large (560M, full fine-tuning) gives OOF 0.6131 (public LB 0.6097).
XL is better by +0.0034 macro-F1, with a paired-bootstrap 95% CI of [+0.0012, +0.0056]
(`logs/model_selection_qxt_vs_qlt.txt`). The blend weights were chosen among a few simple candidates on the OOF
predictions, not fitted by an optimiser.

## Rule compliance
* Only the competition data is used. There is no external or pseudo-labelled data, and no checkpoint that was
  fine-tuned on toxicity, hate or sentiment labels. The base models are plain pretrained models.
* Licenses allow commercial use: Qwen3-4B-Base (Apache-2.0) and XLM-RoBERTa-XL (MIT).
* The `id` column is never used as a feature. It is used only to write the submission rows.
