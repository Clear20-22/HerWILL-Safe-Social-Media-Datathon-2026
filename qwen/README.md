# Qwen3-4B-Base + LoRA (candidate for Submission #2)

A decoder LLM fine-tuned as a 3-class classifier. It is kept separate from the Submission #1 encoder
ensemble (`src/`, `outputs/`), which is left unchanged.

## Model
* **Backbone:** `Qwen/Qwen3-4B-Base` (Apache-2.0). 36 layers, hidden size 2560, 32 query and 8 KV heads (GQA), SwiGLU MLP, RoPE.
* **Adaptation:** LoRA with r=16, alpha=32, dropout 0.05 on the q/k/v/o/gate/up/down projections. That is 33M trainable parameters (0.8% of 4.06B). The backbone is frozen.
* **Head:** a linear `score` layer (2560 to 3) on the last non-pad token, trained in fp32.
* **Input:** the instruction prefix `Classify the toxicity of this social media post (Bangla/English): explicit, subtle, or neutral.\nPost: ` followed by the cleaned text, max 152 tokens.
* **Training:** class-weighted CE with weights (N/n_c)^0.5, AdamW with lr 1e-4, 5% warmup then linear decay, batch 16 x 2 accumulation, bf16 with gradient checkpointing, 2 epochs. The best epoch per fold is chosen by validation macro-F1.
* **Folds, preprocessing and seeds** are identical to `src/herwill_final.py`, so OOF scores are directly comparable.

## Results

Fold 0 (n = 9,564), same split as every other model:

| Model | Macro F1 | Explicit | Subtle | Neutral | Latin | Bangla |
|---|---|---|---|---|---|---|
| XLM-R large | 0.578 | 0.708 | 0.493 | 0.533 | 0.609 | 0.462 |
| Submission #1 ensemble | 0.588 | 0.708 | 0.519 | 0.537 | 0.613 | 0.485 |
| **Qwen3-4B LoRA** | **0.609** | 0.709 | **0.562** | **0.557** | **0.647** | **0.493** |

* Qwen alone vs the Submission #1 ensemble: **+0.021**, paired-bootstrap 95% CI [+0.010, +0.031].
* Adding the encoders to Qwen does not help significantly. Equal weights score 0.601, lower than Qwen alone. Qwen weighted x4 plus the 4 encoders scores +0.005, CI [−0.002, +0.012].
* Epoch 1 reached 0.601 and epoch 2 reached 0.609.

5-fold CV: *in progress*. The score will be written to `submission/cv_score.json`.

## Folder layout
```
scripts/qwen_lora_fold0.py    fold-0 experiment (any Qwen3 base model name as an argument)
scripts/qwen_lora_cv.py       5-fold CV + test predictions + adapters + submission (resumable)
scripts/qwen_in_ensemble.py   fold-0 comparison against and with the encoder ensemble
logs/                         training logs
results/                      fold-0 probabilities; cv/ holds per-fold val/test probabilities, oof.npy, test.npy
adapters/fold{0-4}/           saved LoRA adapters + score head (about 130 MB each)
submission/                   submission_qwen3_4b.csv + cv_score.json (written automatically at the end)
```

## Reproduce
```bash
# from the project root; runs in about 4.5 h on one RTX 4090
tmux new -s herwill_qwen ".venv/bin/python -u qwen/scripts/qwen_lora_cv.py 0,1,2,3,4"
```
