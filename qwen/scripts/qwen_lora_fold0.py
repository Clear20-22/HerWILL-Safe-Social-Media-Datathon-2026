"""Fold-0 test: Qwen3 base models (Apache-2.0, default Qwen3-4B-Base) + LoRA as a 3-class sequence classifier (last-token pooling).
Same preprocessing / fold / class weights as src/herwill_final.py.
Usage (from project root): python qwen/scripts/qwen_lora_fold0.py [hf_name] [epochs] [batch] [accum]"""
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "qwen/results"; OUT.mkdir(parents=True, exist_ok=True)
NAME = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B-Base"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
BS = int(sys.argv[3]) if len(sys.argv) > 3 else 16
ACCUM = int(sys.argv[4]) if len(sys.argv) > 4 else 2
LR, MAX_LEN = 1e-4, 128   # effective batch BS * ACCUM = 32
TAG = NAME.split("/")[-1].lower()
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
torch.manual_seed(42); np.random.seed(42)

tr = pd.read_csv(ROOT / "data/train.csv")
_ws, _rep = re.compile(r"\s+"), re.compile(r"(.)\1{3,}")
clean = lambda t: _ws.sub(" ", _rep.sub(r"\1\1\1", unicodedata.normalize("NFKC", str(t)))).strip()
X = tr.text.fillna("").map(clean).values; y = tr.y.values
a, b = next(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y))
cnt = np.bincount(y); W = torch.tensor((cnt.sum() / cnt) ** 0.5, dtype=torch.float); W = (W / W.mean()).cuda()

tok = AutoTokenizer.from_pretrained(NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"
model = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=3, torch_dtype=torch.bfloat16)
model.config.pad_token_id = tok.pad_token_id
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS", **LORA))
for n, p in model.named_parameters():          # classification head trained in fp32
    if "score" in n:
        p.data = p.data.float(); p.requires_grad = True
model.print_trainable_parameters()
model.cuda()

# A short instruction prefix helps a base LM frame the task; the label is read from the last token.
PROMPT = "Classify the toxicity of this social media post (Bangla/English): explicit, subtle, or neutral.\nPost: "


def collate(batch):
    t, l = zip(*batch)
    e = tok([PROMPT + s for s in t], truncation=True, max_length=MAX_LEN + 24, padding=True, return_tensors="pt")
    e["labels"] = torch.tensor(l); return e


g = torch.Generator(); g.manual_seed(42)
dl_tr = DataLoader(list(zip(X[a], y[a])), batch_size=BS, shuffle=True, collate_fn=collate, generator=g)
dl_va = DataLoader(list(zip(X[b], y[b])), batch_size=32, collate_fn=collate)
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
steps = len(dl_tr) // ACCUM * EPOCHS
sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
lossf = torch.nn.CrossEntropyLoss(weight=W)

hist, best, t0 = [], (-1, None, 0), time.time()
for ep in range(EPOCHS):
    model.train(); tl = []
    for i, batch in enumerate(dl_tr):
        batch = {k: v.cuda() for k, v in batch.items()}; lab = batch.pop("labels")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = lossf(model(**batch).logits.float(), lab) / ACCUM
        loss.backward(); tl.append(loss.item() * ACCUM)
        if (i + 1) % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        if (i + 1) % 400 == 0:
            print(f"  ep{ep + 1} step {i + 1}/{len(dl_tr)} loss={np.mean(tl[-400:]):.4f} {(time.time() - t0) / 60:.1f}min", flush=True)
    model.eval(); P = []
    with torch.no_grad():
        for batch in dl_va:
            batch = {k: v.cuda() for k, v in batch.items()}; batch.pop("labels")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                P.append(torch.softmax(model(**batch).logits.float(), -1).cpu().numpy())
    P = np.concatenate(P); f1 = f1_score(y[b], P.argmax(1), average="macro")
    rec = dict(epoch=ep + 1, train_loss=float(np.mean(tl)), val_f1=float(f1),
               per_class=f1_score(y[b], P.argmax(1), average=None).round(4).tolist(), minutes=round((time.time() - t0) / 60, 1))
    print(rec, flush=True); hist.append(rec)
    if f1 > best[0]:
        best = (f1, P, ep + 1)
np.save(OUT / f"{TAG}_lora_fold0.npy", best[1])
json.dump(dict(model=NAME, lora=LORA, lr=LR, hist=hist, best_f1=best[0], best_epoch=best[2]), open(OUT / f"{TAG}_lora.json", "w"), indent=1)
print(f"BEST: {best[0]:.4f} (epoch {best[2]})", flush=True)
