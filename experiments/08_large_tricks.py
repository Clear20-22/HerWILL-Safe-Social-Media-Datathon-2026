"""Fold-0 test: XLM-R large + label smoothing + LLRD + FGM, vs the plain recipe used in the final run.
Same preprocessing / fold / class weights as src/herwill_final.py. Usage: python 08_large_tricks.py <seed>"""
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/results/tricks"; OUT.mkdir(parents=True, exist_ok=True)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
NAME, EPOCHS, BS, MAX_LEN = "xlm-roberta-large", 3, 32, 128
TOP_LR, LLRD, HEAD_LR, SMOOTH, FGM_EPS = 2e-5, 0.92, 5e-5, 0.1, 1.0

torch.manual_seed(seed); np.random.seed(seed)
tr = pd.read_csv(ROOT / "data/train.csv")
_ws, _rep = re.compile(r"\s+"), re.compile(r"(.)\1{3,}")
clean = lambda t: _ws.sub(" ", _rep.sub(r"\1\1\1", unicodedata.normalize("NFKC", str(t)))).strip()
X = tr.text.fillna("").map(clean).values; y = tr.y.values
a, b = next(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y))
cnt = np.bincount(y); W = torch.tensor((cnt.sum() / cnt) ** 0.5, dtype=torch.float); W = (W / W.mean()).cuda()

tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=3).cuda()


def collate(batch):
    t, l = zip(*batch)
    e = tok(list(t), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    e["labels"] = torch.tensor(l); return e


g = torch.Generator(); g.manual_seed(seed)
dl_tr = DataLoader(list(zip(X[a], y[a])), batch_size=BS, shuffle=True, collate_fn=collate, generator=g)
dl_va = DataLoader(list(zip(X[b], y[b])), batch_size=128, collate_fn=collate)

# --- LLRD: head gets HEAD_LR, top encoder layer TOP_LR, each layer below x LLRD, embeddings lowest
n_layers = model.config.num_hidden_layers
groups = [{"params": [p for n, p in model.named_parameters() if n.startswith("classifier")], "lr": HEAD_LR}]
for i in range(n_layers):
    groups.append({"params": [p for n, p in model.named_parameters() if f"encoder.layer.{i}." in n],
                   "lr": TOP_LR * LLRD ** (n_layers - 1 - i)})
groups.append({"params": [p for n, p in model.named_parameters() if "embeddings" in n], "lr": TOP_LR * LLRD ** n_layers})
assigned = {id(p) for gr in groups for p in gr["params"]}
rest = [p for p in model.parameters() if id(p) not in assigned]
if rest:
    groups.append({"params": rest, "lr": TOP_LR})
opt = torch.optim.AdamW(groups, weight_decay=0.01)
steps = len(dl_tr) * EPOCHS
sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
lossf = torch.nn.CrossEntropyLoss(weight=W, label_smoothing=SMOOTH)
emb = model.get_input_embeddings().weight
print(f"seed={seed} lr: head={HEAD_LR} top={TOP_LR} emb={TOP_LR * LLRD ** n_layers:.2e} smooth={SMOOTH} fgm_eps={FGM_EPS}", flush=True)


def fwd(batch, lab):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return lossf(model(**batch).logits.float(), lab)


hist, best = [], (-1, None, 0)
t0 = time.time()
for ep in range(EPOCHS):
    model.train(); tl = []
    for batch in dl_tr:
        batch = {k: v.cuda() for k, v in batch.items()}; lab = batch.pop("labels")
        loss = fwd(batch, lab); loss.backward()
        # --- FGM: perturb word embeddings along the gradient, add adversarial loss, restore
        gnorm = emb.grad.norm()
        if torch.isfinite(gnorm) and gnorm > 0:
            delta = FGM_EPS * emb.grad / gnorm
            emb.data.add_(delta)
            fwd(batch, lab).backward()
            emb.data.sub_(delta)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True); tl.append(loss.item())
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
np.save(OUT / f"large_tricks_seed{seed}_fold0.npy", best[1])
json.dump(dict(seed=seed, hist=hist, best_f1=best[0], best_epoch=best[2]), open(OUT / f"large_tricks_seed{seed}.json", "w"), indent=1)
print(f"BEST seed={seed}: {best[0]:.4f} (epoch {best[2]})", flush=True)
