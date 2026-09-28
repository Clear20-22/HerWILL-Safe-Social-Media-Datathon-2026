# HerWILL 2026: LoRA adapters for the final notebook

LoRA adapters and classification heads used by the final HerWILL Safe Social Media Datathon 2026 notebook.
There is one adapter per cross-validation fold (5-fold `StratifiedKFold(shuffle=True, random_state=42)`).

| Folder | Base model (license) | Adapter |
|---|---|---|
| `qwen3_4b/fold0` … `fold4` | `Qwen/Qwen3-4B-Base` (Apache-2.0) | LoRA r=16, alpha=32 on q/k/v/o/gate/up/down, plus the `score` head |
| `xlmr_xl/fold0` … `fold4` | `facebook/xlm-roberta-xl` (MIT) | LoRA r=16, alpha=32 on the attention and FFN dense layers, plus the `classifier` head |

* Trained only on the competition's `train.csv`, with the training code in the notebook (`TRAIN_LLM=1`), in bf16 on one RTX 4090.
* The adapter of fold k never saw the held-out rows of fold k, so the notebook can recompute the 5-fold out-of-fold score.
* No competition text is stored here. The base weights are not included; they are downloaded from the Hugging Face Hub.
* Load with `transformers==4.57.6` and `peft==0.21.0` (the notebook pins both).
