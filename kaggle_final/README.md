# Final reproducible notebook: Qwen3-4B + XLM-R XL + TF-IDF

The one end-to-end notebook for the code submission. It reproduces the final submission
(`final_qxt/submission/submission_final_qxt.csv`, 5-fold OOF macro-F1 0.6165) from the competition data.

* TF-IDF + logistic regression is trained inside the notebook, on CPU.
* Qwen3-4B-Base and XLM-RoBERTa-XL predict with their saved LoRA adapters, one per fold. The adapters are
  uploaded as a Kaggle Dataset. Their training code is in the notebook and runs with `TRAIN_LLM=1`; it needs a
  bf16 GPU with 24 GB, so it cannot run on a Kaggle T4.
* By default, every fold's held-out rows are predicted too, so the notebook also reproduces the 5-fold CV score.

## Files
```
herwill_final_kaggle.py        source with `# %%` cells (edit this one)
herwill_final_kaggle.ipynb     the notebook to upload, built with:
                               python tools/make_notebook.py kaggle_final/herwill_final_kaggle.py kaggle_final/herwill_final_kaggle.ipynb
dataset/herwill-lora-adapters/ the folder to upload as a private Kaggle Dataset (10 adapters + README, 1.3 GB)
scripts/compare_with_reference.py  compares a run with the training-run probabilities and the submitted CSV
scripts/fp16_headroom.py       measures Qwen's largest activations against the fp16 limit
logs/                          verification logs
results/local_fp16/            outputs of the local fp16 verification run
```

## Run it on Kaggle
1. **Dataset:** Datasets → New Dataset → upload the folder `dataset/herwill-lora-adapters` and keep it private.
   Any folder layout works, because the notebook searches for `adapter_config.json` files.
2. **Notebook:** Create → New Notebook → File → Import Notebook → `herwill_final_kaggle.ipynb`.
3. **Inputs:** Add Input → the competition data, and the adapters Dataset from step 1.
4. **Settings:** Accelerator **GPU T4** (x1 or x2) and Internet **ON**. Internet installs the pinned
   `transformers==4.57.6` and `peft==0.21.0` and downloads the base models from the Hugging Face Hub:
   Qwen3-4B-Base (8 GB) and XLM-R XL (14 GB). Kaggle turns Internet on only for phone-verified accounts.
5. **Run:** Save Version → Save & Run All. `submission.csv` and `cv_report.json` appear in the output.

Expected time on a T4 is roughly 1–2 hours. This is an estimate from the 21 minutes measured on an RTX 4090 in
fp16. To roughly halve the GPU time, set `RECOMPUTE_OOF = False` in the configuration cell.

## Local verification in fp16
The T4 has no bf16, so the notebook runs in fp16 there. The adapters were trained in bf16.
* **Activation headroom** (`logs/fp16_headroom_qwen.txt`): Qwen's largest activation is 6,559. The fp16 limit is
  65,504, so the model stays at about 10% of the range.
* **Full run** on the RTX 4090 with `FORCE_FP16=1` (`logs/local_fp16_run.txt`, `logs/compare_local_fp16_vs_reference.txt`):

| Component | OOF F1 fp16 run | OOF F1 reference (bf16) | Test argmax agreement |
|---|---|---|---|
| Qwen3-4B + LoRA | 0.6013 | 0.6016 | 99.76% |
| XLM-R XL + LoRA | 0.5915 | 0.5915 | 99.92% |
| TF-IDF + LR | 0.5277 | 0.5277 | 100% (bit-identical) |
| **Blend** | **0.6164** | **0.6165** | **99.82%** of submission labels |

* Time: Qwen 15.0 min, XL 3.7 min, TF-IDF 2.3 min. Peak GPU memory: Qwen 8.7 GB, XL 7.4 GB, which fits a 15 GB T4.
* TF-IDF is bit-identical when it uses the same number of CPU threads. With a different thread count (4 instead
  of 6), the L-BFGS path changes slightly: probabilities differed by up to 0.03 on the 16 rows checked.
* The CPU smoke test (`QUICK=1`, fp32) also matched the reference.
* Sorting the texts by length for batching does not change the outputs, only the speed.

## Run it locally
```bash
# from the project root
.venv/bin/python kaggle_final/herwill_final_kaggle.py                                                 # predict mode, bf16 on Ampere+
FORCE_FP16=1 OUT_DIR=kaggle_final/results/local_fp16 .venv/bin/python kaggle_final/herwill_final_kaggle.py  # T4 precision
QUICK=1 .venv/bin/python kaggle_final/herwill_final_kaggle.py                                         # smoke test
TRAIN_LLM=1 .venv/bin/python kaggle_final/herwill_final_kaggle.py                                     # retrain (about 7 h)
.venv/bin/python kaggle_final/scripts/compare_with_reference.py kaggle_final/results/local_fp16
```
