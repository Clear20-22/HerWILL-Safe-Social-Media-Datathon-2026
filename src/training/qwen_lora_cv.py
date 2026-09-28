"""Qwen3-4B-Base + LoRA, 5-fold CV with test predictions and saved adapters (for a Kaggle predict-only mode).
Same preprocessing / folds / class weights as src/herwill_final.py. Each fold is cached, so the run is resumable.
Usage (from project root): python qwen/scripts/qwen_lora_cv.py [folds, e.g. 0,1,2,3,4]"""
import gc
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
OUT = ROOT / "qwen/results/cv"; ADP = ROOT / "qwen/adapters"
OUT.mkdir(parents=True, exist_ok=True); ADP.mkdir(parents=True, exist_ok=True)
FOLDS = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 1, 2, 3, 4]
QUICK = __import__("os").environ.get("QUICK") == "1"  # smoke test: tiny subset, 1 epoch, separate output dir
if QUICK:
    OUT = Path(__import__("os").environ.get("QUICK_DIR", "/tmp")) / "qwen_quick"; ADP = OUT / "adapters"
    OUT.mkdir(parents=True, exist_ok=True); ADP.mkdir(parents=True, exist_ok=True)

CFG = dict(model="Qwen/Qwen3-4B-Base", epochs=2, batch=16, accum=2, lr=1e-4, warmup=0.05, max_len=128 + 24,
           lora=dict(r=16, lora_alpha=32, lora_dropout=0.05,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
           prompt="Classify the toxicity of this social media post (Bangla/English): explicit, subtle, or neutral.\nPost: ",
           seed=42)

tr = pd.read_csv(ROOT / "data/train.csv"); te = pd.read_csv(ROOT / "data/test.csv")
_ws, _rep = re.compile(r"\s+"), re.compile(r"(.)\1{3,}")
clean = lambda t: _ws.sub(" ", _rep.sub(r"\1\1\1", unicodedata.normalize("NFKC", str(t)))).strip()
X = tr.text.fillna("").map(clean).values; Xt = te.text.fillna("").map(clean).values; y = tr.y.values
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y))
if QUICK:
    CFG["epochs"] = 1; folds = [(a[:320], b[:96]) for a, b in folds]; Xt = Xt[:64]; te = te.head(64)
cnt = np.bincount(y); W = torch.tensor((cnt.sum() / cnt) ** 0.5, dtype=torch.float); W = (W / W.mean()).cuda()

tok = AutoTokenizer.from_pretrained(CFG["model"])
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"


def collate(batch):
    t, l = zip(*batch)
    e = tok([CFG["prompt"] + s for s in t], truncation=True, max_length=CFG["max_len"], padding=True, return_tensors="pt")
    e["labels"] = torch.tensor(l); return e


def predict(model, texts):
    dl = DataLoader(list(zip(texts, np.zeros(len(texts), int))), batch_size=32, collate_fn=collate)
    model.eval(); P = []
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.cuda() for k, v in batch.items()}; batch.pop("labels")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                P.append(torch.softmax(model(**batch).logits.float(), -1).cpu().numpy())
    return np.concatenate(P)


def build_model():
    m = AutoModelForSequenceClassification.from_pretrained(CFG["model"], num_labels=3, dtype=torch.bfloat16)
    m.config.pad_token_id = tok.pad_token_id
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.enable_input_require_grads()
    m = get_peft_model(m, LoraConfig(task_type="SEQ_CLS", **CFG["lora"]))
    for n, p in m.named_parameters():  # classification head trained in fp32
        if "score" in n:
            p.data = p.data.float(); p.requires_grad = True
    return m.cuda()


for f in FOLDS:
    cache = OUT / f"fold{f}.npz"
    if cache.exists():
        print(f"fold {f}: cached, skipping", flush=True); continue
    a, b = folds[f]
    torch.manual_seed(CFG["seed"] + f); np.random.seed(CFG["seed"] + f)
    model = build_model()
    g = torch.Generator(); g.manual_seed(CFG["seed"] + f)
    dl_tr = DataLoader(list(zip(X[a], y[a])), batch_size=CFG["batch"], shuffle=True, collate_fn=collate, generator=g)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=CFG["lr"], weight_decay=0.0)
    steps = len(dl_tr) // CFG["accum"] * CFG["epochs"]
    sch = get_linear_schedule_with_warmup(opt, int(CFG["warmup"] * steps), steps)
    lossf = torch.nn.CrossEntropyLoss(weight=W)
    hist, best, t0 = dict(step_loss=[], epoch=[]), dict(f1=-1), time.time()
    for ep in range(CFG["epochs"]):
        model.train(); tl = []
        for i, batch in enumerate(dl_tr):
            batch = {k: v.cuda() for k, v in batch.items()}; lab = batch.pop("labels")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lossf(model(**batch).logits.float(), lab) / CFG["accum"]
            loss.backward(); tl.append(loss.item() * CFG["accum"])
            if (i + 1) % CFG["accum"] == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            if (i + 1) % 50 == 0:
                hist["step_loss"].append(float(np.mean(tl[-50:])))
            if (i + 1) % 800 == 0:
                print(f"  fold {f} ep{ep + 1} step {i + 1}/{len(dl_tr)} loss={np.mean(tl[-800:]):.4f} {(time.time() - t0) / 60:.1f}min", flush=True)
        pv = predict(model, X[b])
        vloss = float(torch.nn.functional.cross_entropy(torch.log(torch.tensor(pv) + 1e-9), torch.tensor(y[b]), weight=W.cpu()))
        f1 = f1_score(y[b], pv.argmax(1), average="macro")
        hist["epoch"].append(dict(epoch=ep + 1, train_loss=float(np.mean(tl)), val_loss=vloss, val_f1=float(f1)))
        print(f"  fold {f} epoch {ep + 1}: train_loss={np.mean(tl):.4f} val_loss={vloss:.4f} val_macroF1={f1:.4f} "
              f"{(time.time() - t0) / 60:.1f}min", flush=True)
        if f1 > best["f1"]:
            best = dict(f1=f1, epoch=ep + 1, val=pv, test=predict(model, Xt))
            model.save_pretrained(ADP / f"fold{f}")  # LoRA weights + score head
    np.savez(cache, val=best["val"], test=best["test"], hist=json.dumps(hist))
    print(f"fold {f}: macro-F1={best['f1']:.4f} (best epoch {best['epoch']}, {(time.time() - t0) / 60:.1f} min)", flush=True)
    del model, opt; gc.collect(); torch.cuda.empty_cache()

json.dump(CFG, open(OUT / "config.json", "w"), indent=1)
done = [f for f in range(5) if (OUT / f"fold{f}.npz").exists()]
if len(done) == 5:
    oof = np.zeros((len(tr), 3)); tp = np.zeros((len(te), 3))
    for f, (a, b) in enumerate(folds):
        d = np.load(OUT / f"fold{f}.npz", allow_pickle=True); oof[b] = d["val"]; tp += d["test"] / 5
    np.save(OUT / "oof.npy", oof); np.save(OUT / "test.npy", tp)
    cv = f1_score(y, oof.argmax(1), average="macro")
    print(f"\nQwen3-4B 5-fold OOF macro-F1 = {cv:.4f}", flush=True)
    print("per-class F1:", f1_score(y, oof.argmax(1), average=None).round(4).tolist(), flush=True)

    # Submission for the single Qwen model, validated against sample_submission (format: id,y_pred)
    if True:
        SUB = OUT / "submission" if QUICK else ROOT / "qwen/submission"; SUB.mkdir(parents=True, exist_ok=True)
        sample = pd.read_csv(ROOT / "data/sample_submission.csv").head(len(te))
        sub = sample[["id"]].merge(pd.DataFrame({"id": te["id"].values, "y_pred": tp.argmax(1).astype(int)}), on="id", how="left")
        assert len(sub) == len(sample) and sub["id"].tolist() == sample["id"].tolist(), "id mismatch with sample"
        assert sub["y_pred"].notna().all() and set(sub["y_pred"].unique()) <= {0, 1, 2}, "bad predictions"
        sub["y_pred"] = sub["y_pred"].astype(int)
        sub.to_csv(SUB / "submission_qwen3_4b.csv", index=False)
        json.dump(dict(cv_macro_f1=float(cv), per_class=f1_score(y, oof.argmax(1), average=None).tolist(),
                       pred_dist=sub["y_pred"].value_counts(normalize=True).sort_index().to_dict()),
                  open(SUB / "cv_score.json", "w"), indent=1)
        print(f"saved {SUB / 'submission_qwen3_4b.csv'} rows={len(sub)} "
              f"dist={sub['y_pred'].value_counts(normalize=True).sort_index().round(3).to_dict()}", flush=True)
