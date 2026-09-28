"""fp16 headroom check for Qwen3-4B-Base + LoRA (fold 0 adapter). Kaggle T4 has no bf16, so inference runs in fp16,
which overflows above 65504. This runs real inputs one at a time (no padding) in fp32 on CPU and records the largest
absolute activation in the residual stream and in the MLP (down_proj input and output) of every layer.
Usage (from project root): CUDA_VISIBLE_DEVICES= python kaggle_final/scripts/fp16_headroom.py"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(4)
NAME = "Qwen/Qwen3-4B-Base"
PROMPT = "Classify the toxicity of this social media post (Bangla/English): explicit, subtle, or neutral.\nPost: "
FP16_MAX = 65504.0

_ws, _rep = re.compile(r"\s+"), re.compile(r"(.)\1{3,}")
clean = lambda t: _ws.sub(" ", _rep.sub(r"\1\1\1", unicodedata.normalize("NFKC", str(t)))).strip()
texts = pd.read_csv(ROOT / "data/train.csv").text.fillna("").map(clean).values
lens = np.array([len(t) for t in texts])
idx = list(np.random.default_rng(0).choice(len(texts), 12, replace=False)) + list(np.argsort(lens)[-12:])  # 12 random + 12 longest

tok = AutoTokenizer.from_pretrained(NAME)
base = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=3, dtype=torch.float32)
base.config.pad_token_id = tok.pad_token_id
model = PeftModel.from_pretrained(base, str(ROOT / "qwen/adapters/fold0")).eval()
layers = model.base_model.model.model.layers
stats = {k: np.zeros(len(layers)) for k in ("residual", "mlp_in", "mlp_out")}


def hook(key, i, use_input=False):
    def f(mod, inp, out):
        t = inp[0] if use_input else (out[0] if isinstance(out, tuple) else out)
        stats[key][i] = max(stats[key][i], t.detach().abs().max().item())
    return f


for i, layer in enumerate(layers):
    layer.register_forward_hook(hook("residual", i))
    layer.mlp.down_proj.register_forward_hook(hook("mlp_in", i, use_input=True))
    layer.mlp.down_proj.register_forward_hook(hook("mlp_out", i))
max_logit = 0.0
with torch.no_grad():
    for j in idx:
        enc = tok(PROMPT + texts[j], truncation=True, max_length=152, return_tensors="pt")
        max_logit = max(max_logit, model(**enc).logits.abs().max().item())

print(f"Qwen3-4B-Base + LoRA fold 0, {len(idx)} inputs (12 random, 12 longest), fp32 on CPU. fp16 max = {FP16_MAX:.0f}")
for k, v in stats.items():
    i = int(v.argmax())
    print(f"  {k:8s}: largest |activation| = {v.max():10.1f} (layer {i}), {v.max() / FP16_MAX:.1%} of fp16 max; "
          f"median over layers {np.median(v):.1f}")
print(f"  logits  : largest |logit| = {max_logit:.2f}")
print("  per-layer residual max:", np.round(stats["residual"]).astype(int).tolist())
