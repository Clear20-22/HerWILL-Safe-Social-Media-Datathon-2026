# XLM-RoBERTa-XL + LoRA (overnight experiment)

A 3.5B multilingual **encoder** (MIT license, CC-100 pretraining that includes Bangla). It is tested as a partner
for the Qwen3-4B decoder: the two architectures differ, so their errors may be less correlated.

## Model
* **Backbone:** `facebook/xlm-roberta-xl`. 36 layers, hidden size 2560, FFN 10240, 32 heads, pre-LayerNorm, 250k vocab. The backbone is frozen.
* **LoRA:** r=16, alpha=32, dropout 0.05 on the query/key/value, attention-output, intermediate and output dense layers of all 36 layers (216 modules).
* **Head:** the standard XLM-R classification head (dense, tanh, out_proj to 3 classes), fully trained in fp32.
* **Training:** class-weighted CE with weights (N/n_c)^0.5, AdamW with lr 2e-4 and weight decay 0.01, 6% warmup then linear decay, batch 16 x 2 accumulation, max 128 tokens, bf16, 3 epochs. `XLMRobertaXLForSequenceClassification` does not support gradient checkpointing (`logs/xlmr_xl_cv_failed_gradckpt_0043.txt`), so the run fits in 24 GB without it. The best epoch per fold is chosen by validation macro-F1.
* **Folds, preprocessing and seeds** are identical to `src/herwill_final.py` and `qwen/`.

## Pipeline (`scripts/run_after_qwen.sh`, tmux session `herwill_xl`)
1. Wait until the Qwen3-4B CV run writes `DONE` to `qwen/logs/qwen3_4b_lora_cv.txt`.
2. Train **fold 0**. **Gate:** continue only if its macro-F1 is at least 0.585, that is, clearly above XLM-R large at 0.578.
3. Train folds 1–4, then write `submission/submission_xlmr_xl.csv`.
4. If the Qwen OOF exists, blend Qwen and XL at 1:1 and 2:1, run a paired bootstrap on the full OOF, and write `submission_qwen_plus_xl_*.csv` plus a recommendation in `submission/report.json`. A blend is recommended only if the lower bound of its 95% CI is above 0.
5. On CUDA out-of-memory, retry automatically with batch 8 x 4 accumulation. Cached folds are kept. (It was not needed.)

Nothing outside `xlmr_xl/` is written. Submission #1 (`outputs/`) and Qwen (`qwen/`) are never modified.

## Folder layout
```
scripts/xlmr_xl_cv.py        CV + gate + submissions + blend report (QUICK=1 for a smoke test)
scripts/run_after_qwen.sh    overnight chain
logs/xlmr_xl_cv.txt          full log
results/cv/                  per-fold val/test probabilities, oof.npy, test.npy, config.json
adapters/fold{0-4}/          LoRA + classifier head
submission/                  submission_xlmr_xl.csv, submission_qwen_plus_xl_*.csv, report.json
```

## Results
5-fold OOF macro-F1 **0.5915**, against 0.5788 for XLM-R large with the same folds. XL is better on every fold:

| Fold | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| XLM-R XL + LoRA | 0.5949 | 0.5823 | 0.5908 | 0.5932 | 0.5965 |
| XLM-R large (full fine-tune) | 0.5781 | 0.5708 | 0.5788 | 0.5840 | 0.5824 |

The best epoch was 2 or 3; each fold took about 24.5 min on an RTX 4090. The fold-0 gate passed (0.5949 ≥ 0.585).

Blends on the full OOF (`final_qxt/logs/make_submission.txt`):
* Qwen + XL at 1:1 scores 0.6110, which is +0.0094 over Qwen alone (CI [+0.0062, +0.0126]).
* Qwen 0.50 + **XL** 0.25 + TF-IDF 0.25 scores **0.6165**. The same blend with XLM-R large (`final_qlt`) scores 0.6131,
  so XL adds +0.0034 (CI [+0.0012, +0.0056]). This is the final architecture (`final_qxt/`, `kaggle_final/`).
