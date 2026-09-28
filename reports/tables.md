| Model | OOF macro-F1 | F1 explicit | F1 subtle | F1 neutral | F1 Latin | F1 Bangla | F1 mixed |
|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg | 0.5277 | 0.652 | 0.504 | 0.427 | 0.544 | 0.458 | 0.462 |
| XLM-R base (full FT) | 0.5639 | 0.687 | 0.505 | 0.500 | 0.589 | 0.477 | 0.487 |
| mDeBERTa-v3 base (full FT) | 0.5719 | 0.692 | 0.514 | 0.509 | 0.601 | 0.482 | 0.468 |
| XLM-R large (full FT) | 0.5788 | 0.704 | 0.505 | 0.527 | 0.611 | 0.476 | 0.492 |
| XLM-R XL + LoRA | 0.5915 | 0.703 | 0.530 | 0.541 | 0.625 | 0.488 | 0.514 |
| Qwen3-4B-Base + LoRA | 0.6016 | 0.708 | 0.545 | 0.551 | 0.641 | 0.487 | 0.459 |
| **Final blend (0.5 Qwen + 0.25 XL + 0.25 TF-IDF)** | 0.6165 | 0.717 | 0.559 | 0.573 | 0.653 | 0.502 | 0.505 |

| Model | Train time, 5 folds (min) | Train GPU (GB) | Inference GPU fp16 (GB) | Parameters | Saved size | Inference posts/s |
|---|---|---|---|---|---|---|
| Qwen3-4B-Base + LoRA | 278 | ~10.0 (nvidia-smi) | 8.7 | 4.06B total / 33.0M trainable | 5 x 126 MB fp32 (63 MB fp16) | 120 |
| XLM-R XL + LoRA | 122 | ~18-20 (nvidia-smi) | 7.4 | 3.52B total / 39.7M trainable | 5 x 126 MB fp32 (63 MB fp16) | 485 |
| TF-IDF + LogReg | 2 | 0 (CPU) | 0.0 | ≤800k features x 3 | not saved (retrains in 2.3 min) | n/a |
| XLM-R large (full FT) | 50 | ~12 (nvidia-smi) | n/a | 560M | weights not saved | n/a |
| mDeBERTa-v3 base (full FT) | 29 | n/a | n/a | 278M | weights not saved | n/a |
| XLM-R base (full FT) | 22 | n/a | n/a | 278M | weights not saved | n/a |
