"""XLM-RoBERTa-XL (3.5B, MIT) + LoRA, 5-fold CV with a fold-0 gate, test predictions, saved adapters,
its own submission, and a final OOF comparison against / blend with the Qwen3-4B CV run.
Same preprocessing / folds / class weights as src/herwill_final.py. Every fold is cached, so the run is resumable.
Usage (from project root): python xlmr_xl/scripts/xlmr_xl_cv.py
Env: QUICK=1 for a smoke test (tiny subset, writes to $QUICK_DIR/xl_quick), MODEL=<hf name> to override the model."""
import gc
import json
import os
import re
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
QUICK = os.environ.get("QUICK") == "1"
BASE = Path(os.environ.get("QUICK_DIR", "/tmp")) / "xl_quick" if QUICK else ROOT / "xlmr_xl"
OUT, ADP, SUB = BASE / "results/cv", BASE / "adapters", BASE / "submission"
for d in (OUT, ADP, SUB):
    d.mkdir(parents=True, exist_ok=True)

CFG = dict(model=os.environ.get("MODEL", "facebook/xlm-roberta-xl"), epochs=3, batch=int(os.environ.get("BATCH", 16)), accum=int(os.environ.get("ACCUM", 2)), lr=2e-4, warmup=0.06,
           max_len=128, seed=42,
           lora=dict(r=16, lora_alpha=32, lora_dropout=0.05,
                     target_modules=r".*encoder\.layer\.\d+\.(attention\.self\.(query|key|value)|attention\.output\.dense"
                                    r"|intermediate\.dense|output\.dense)"),
           # continue to folds 1-4 only if fold 0 beats XLM-R large (0.578) by a clear margin
           gate_fold0_f1=0.585)

tr = pd.read_csv(ROOT / "data/train.csv"); te = pd.read_csv(ROOT / "data/test.csv")
_ws, _rep = re.compile(r"\s+"), re.compile(r"(.)\1{3,}")
clean = lambda t: _ws.sub(" ", _rep.sub(r"\1\1\1", unicodedata.normalize("NFKC", str(t)))).strip()
X = tr.text.fillna("").map(clean).values; Xt = te.text.fillna("").map(clean).values; y = tr.y.values
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y))
if QUICK:
    CFG["epochs"] = 1; CFG["gate_fold0_f1"] = 0.0
    folds = [(a[:320], b[:96]) for a, b in folds]; Xt = Xt[:64]; te = te.head(64)
cnt = np.bincount(y); W = torch.tensor((cnt.sum() / cnt) ** 0.5, dtype=torch.float); W = (W / W.mean()).cuda()
tok = AutoTokenizer.from_pretrained(CFG["model"])


def log(msg):
    print(msg, flush=True)


def collate(batch):
    t, l = zip(*batch)
    e = tok(list(t), truncation=True, max_length=CFG["max_len"], padding=True, return_tensors="pt")
    e["labels"] = torch.tensor(l); return e


def predict(model, texts):
    dl = DataLoader(list(zip(texts, np.zeros(len(texts), int))), batch_size=64, collate_fn=collate)
    model.eval(); P = []
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.cuda() for k, v in batch.items()}; batch.pop("labels")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                P.append(torch.softmax(model(**batch).logits.float(), -1).cpu().numpy())
    return np.concatenate(P)


def build_model():
    m = AutoModelForSequenceClassification.from_pretrained(CFG["model"], num_labels=3, dtype=torch.bfloat16)
    if m.supports_gradient_checkpointing:  # XLMRobertaXL does not support it; batch 16 x 128 peaks at 16 GiB without it
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        m.enable_input_require_grads()
    m = get_peft_model(m, LoraConfig(task_type="SEQ_CLS", **CFG["lora"]))  # SEQ_CLS keeps `classifier` trainable + saved
    for n, p in m.named_parameters():  # classification head trained in fp32
        if "classifier" in n:
            p.data = p.data.float(); p.requires_grad = True
    m.print_trainable_parameters()
    return m.cuda()


def run_fold(f):
    cache = OUT / f"fold{f}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True); log(f"fold {f}: cached (macro-F1={float(d['f1']):.4f})"); return float(d["f1"])
    a, b = folds[f]
    torch.manual_seed(CFG["seed"] + f); np.random.seed(CFG["seed"] + f)
    model = build_model()
    g = torch.Generator(); g.manual_seed(CFG["seed"] + f)
    dl_tr = DataLoader(list(zip(X[a], y[a])), batch_size=CFG["batch"], shuffle=True, collate_fn=collate, generator=g)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=CFG["lr"], weight_decay=0.01)
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
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at fold {f} epoch {ep + 1} step {i + 1}")
            loss.backward(); tl.append(loss.item() * CFG["accum"])
            if (i + 1) % CFG["accum"] == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            if (i + 1) % 50 == 0:
                hist["step_loss"].append(float(np.mean(tl[-50:])))
            if (i + 1) % 400 == 0:
                log(f"  fold {f} ep{ep + 1} step {i + 1}/{len(dl_tr)} loss={np.mean(tl[-400:]):.4f} {(time.time() - t0) / 60:.1f}min")
        pv = predict(model, X[b])
        vloss = float(torch.nn.functional.cross_entropy(torch.log(torch.tensor(pv) + 1e-9), torch.tensor(y[b]), weight=W.cpu()))
        f1 = f1_score(y[b], pv.argmax(1), average="macro")
        hist["epoch"].append(dict(epoch=ep + 1, train_loss=float(np.mean(tl)), val_loss=vloss, val_f1=float(f1)))
        log(f"  fold {f} epoch {ep + 1}: train_loss={np.mean(tl):.4f} val_loss={vloss:.4f} val_macroF1={f1:.4f} "
            f"per_class={f1_score(y[b], pv.argmax(1), average=None).round(3).tolist()} {(time.time() - t0) / 60:.1f}min")
        if f1 > best["f1"]:
            best = dict(f1=f1, epoch=ep + 1, val=pv, test=predict(model, Xt))
            model.save_pretrained(ADP / f"fold{f}")  # LoRA weights + classifier head
    np.savez(cache, val=best["val"], test=best["test"], f1=best["f1"], epoch=best["epoch"], hist=json.dumps(hist))
    log(f"fold {f}: macro-F1={best['f1']:.4f} (best epoch {best['epoch']}, {(time.time() - t0) / 60:.1f} min)")
    del model, opt; gc.collect(); torch.cuda.empty_cache()
    return best["f1"]


def write_sub(probs, name, note):
    sample = pd.read_csv(ROOT / "data/sample_submission.csv").head(len(te))
    sub = sample[["id"]].merge(pd.DataFrame({"id": te["id"].values, "y_pred": probs.argmax(1).astype(int)}), on="id", how="left")
    assert len(sub) == len(sample) and sub["id"].tolist() == sample["id"].tolist(), "id mismatch with sample"
    assert sub["y_pred"].notna().all() and set(sub["y_pred"].unique()) <= {0, 1, 2}, "bad predictions"
    sub["y_pred"] = sub["y_pred"].astype(int)
    sub.to_csv(SUB / name, index=False)
    log(f"saved {SUB / name} ({note}) rows={len(sub)} dist={sub['y_pred'].value_counts(normalize=True).sort_index().round(3).to_dict()}")


json.dump(CFG, open(OUT / "config.json", "w"), indent=1)
log(f"=== XLM-R XL LoRA CV  model={CFG['model']}  gate={CFG['gate_fold0_f1']}")
f0 = run_fold(0)
report = dict(fold0_f1=f0, gate=CFG["gate_fold0_f1"])
if f0 < CFG["gate_fold0_f1"]:
    report["decision"] = f"STOPPED after fold 0: {f0:.4f} < gate {CFG['gate_fold0_f1']} (not better enough than XLM-R large 0.578)"
    log(report["decision"]); json.dump(report, open(SUB / "report.json", "w"), indent=1)
else:
    for f in range(1, 5):
        run_fold(f)
    oof = np.zeros((len(tr), 3)); tp = np.zeros((len(te), 3))
    for f, (a, b) in enumerate(folds):
        d = np.load(OUT / f"fold{f}.npz", allow_pickle=True); oof[b] = d["val"]; tp += d["test"] / 5
    np.save(OUT / "oof.npy", oof); np.save(OUT / "test.npy", tp)
    idx = np.concatenate([b for _, b in folds])  # rows that have OOF predictions (all rows in a full run)
    mf1 = lambda P: f1_score(y[idx], P[idx].argmax(1), average="macro")
    report["xl_oof_f1"] = mf1(oof)
    log(f"\nXLM-R XL 5-fold OOF macro-F1 = {report['xl_oof_f1']:.4f}")
    write_sub(tp, "submission_xlmr_xl.csv", "XLM-R XL alone")

    # --- compare with / blend with Qwen3-4B (only if its CV run finished); OOF-based, paired bootstrap
    qdir = Path(os.environ.get("QWEN_DIR", ROOT / "qwen/results/cv"))
    if (qdir / "oof.npy").exists():
        Qo, Qt = np.load(qdir / "oof.npy"), np.load(qdir / "test.npy")
        rng = np.random.default_rng(0); boot = [rng.integers(0, len(idx), len(idx)) for _ in range(1000)]
        yy = y[idx]

        def gain(P, P0):
            p, p0 = P[idx].argmax(1), P0[idx].argmax(1)
            d = np.array([f1_score(yy[i], p[i], average="macro") - f1_score(yy[i], p0[i], average="macro") for i in boot])
            return dict(mean=float(d.mean()), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], p_no_gain=float((d <= 0).mean()))

        report["qwen_oof_f1"] = mf1(Qo)
        blends = {"qwen+xl (1:1)": (1, 1), "qwen+xl (2:1)": (2, 1)}
        report["blends"] = {}
        for name, (wq, wx) in blends.items():
            P, Pt = (wq * Qo + wx * oof) / (wq + wx), (wq * Qt + wx * tp) / (wq + wx)
            g = gain(P, Qo); report["blends"][name] = dict(oof_f1=mf1(P), gain_vs_qwen=g)
            log(f"{name}: OOF {mf1(P):.4f}  vs Qwen alone {report['qwen_oof_f1']:.4f}  gain {g['mean']:+.4f} "
                f"CI[{g['ci'][0]:+.4f},{g['ci'][1]:+.4f}] P(no gain)={g['p_no_gain']:.3f}")
            write_sub(Pt, f"submission_{name.split(' ')[0].replace('+', '_plus_')}_{wq}to{wx}.csv", name)
        best = max(report["blends"].items(), key=lambda kv: kv[1]["oof_f1"])
        ok = best[1]["gain_vs_qwen"]["ci"][0] > 0
        report["recommendation"] = (f"use {best[0]} (significant OOF gain over Qwen alone)" if ok else
                                    "keep Qwen3-4B alone (no significant OOF gain from adding XLM-R XL)")
        log("RECOMMENDATION: " + report["recommendation"])
    json.dump(report, open(SUB / "report.json", "w"), indent=1, default=float)
log("XL PIPELINE FINISHED")
