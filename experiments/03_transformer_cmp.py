"""Compare pretrained encoders on the same holdout (fold 0 of the 5-fold split used in 02_baselines)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/results/transformers"
OUT.mkdir(exist_ok=True)

model_name = sys.argv[1]
lr = float(sys.argv[2]) if len(sys.argv) > 2 else 2e-5
epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 3
bs, max_len = 32, 128
torch.manual_seed(42)

tr = pd.read_pickle(ROOT / "experiments/results/train_feats.pkl").reset_index(drop=True)
y = tr.y.values
a, b = next(StratifiedKFold(5, shuffle=True, random_state=42).split(tr.text, y))

tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).cuda()

lens = [len(x) for x in tok(tr.text.tolist()[:5000], truncation=False)["input_ids"]]
unk = tok.unk_token_id
bn = tr.script.values[:5000] == "bangla"
ids5k = tok(tr.text.tolist()[:5000])["input_ids"]
unk_rate = lambda m: np.mean([np.mean([t == unk for t in s]) for s, k in zip(ids5k, m) if k])
tok_stats = dict(p95_len=float(np.percentile(lens, 95)), p99_len=float(np.percentile(lens, 99)),
                 unk_bangla=float(unk_rate(bn)), unk_latin=float(unk_rate(~bn)))
print(model_name, tok_stats, flush=True)


def collate(batch):
    t, lab = zip(*batch)
    enc = tok(list(t), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    enc["labels"] = torch.tensor(lab)
    return enc


dl_tr = DataLoader(list(zip(tr.text.values[a], y[a])), batch_size=bs, shuffle=True, collate_fn=collate)
dl_va = DataLoader(list(zip(tr.text.values[b], y[b])), batch_size=128, collate_fn=collate)

# class weights ~ inverse sqrt frequency (mild; macro-F1 target)
cnt = np.bincount(y[a]); w = torch.tensor((cnt.sum() / cnt) ** 0.5, dtype=torch.float).cuda(); w /= w.mean()
lossf = torch.nn.CrossEntropyLoss(weight=w)
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
steps = len(dl_tr) * epochs
sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

hist, best = [], (-1, None)
t0 = time.time()
for ep in range(epochs):
    model.train(); tl = []
    for i, batch in enumerate(dl_tr):
        batch = {k: v.cuda() for k, v in batch.items()}
        lab = batch.pop("labels")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = lossf(model(**batch).logits.float(), lab)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad(); tl.append(loss.item())
    model.eval(); P = []
    with torch.no_grad():
        for batch in dl_va:
            batch = {k: v.cuda() for k, v in batch.items()}; batch.pop("labels")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                P.append(torch.softmax(model(**batch).logits.float(), -1).cpu().numpy())
    P = np.concatenate(P); pred = P.argmax(1)
    f1 = f1_score(y[b], pred, average="macro")
    per_script = {s: float(f1_score(y[b][m], pred[m], average="macro"))
                  for s in ["latin", "bangla", "mixed"] if (m := tr.script.values[b] == s).sum()}
    rec = dict(epoch=ep + 1, train_loss=float(np.mean(tl)), val_macro_f1=float(f1),
               per_class=f1_score(y[b], pred, average=None).round(4).tolist(), per_script=per_script,
               minutes=round((time.time() - t0) / 60, 1))
    print(rec, flush=True); hist.append(rec)
    if f1 > best[0]:
        best = (f1, P)

tag = model_name.replace("/", "__")
np.save(OUT / f"{tag}_fold0_probs.npy", best[1])
json.dump(dict(model=model_name, lr=lr, epochs=epochs, tok_stats=tok_stats, hist=hist, best_f1=best[0]),
          open(OUT / f"{tag}.json", "w"), indent=1)
