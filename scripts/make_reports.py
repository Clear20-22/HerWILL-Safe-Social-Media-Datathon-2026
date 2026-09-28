"""Build the report artifacts from the saved runs (no training):
runs/<run>/summary.json, runs/<run>/curves.png, runs/final_blend_summary.json, reports/figures/*.png, reports/tables.md.
Needs data/raw/train.csv (labels) and data/processed/folds.csv. Usage: python scripts/make_reports.py"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
RUNS, FIG = ROOT / "runs", ROOT / "reports/figures"
FIG.mkdir(parents=True, exist_ok=True)
CFG = json.loads((ROOT / "configs/final_qxt.json").read_text())
y = pd.read_csv(ROOT / "data/raw/train.csv")["y"].values
meta = pd.read_csv(ROOT / "data/processed/folds.csv")
fold, script = meta["fold"].values, meta["script"].values

# Measured on the RTX 4090 (logs in runs/<run>/log.txt and logs/fp16_verification/). Train GPU memory is the
# nvidia-smi reading during training; inference memory is torch.cuda.max_memory_allocated in the fp16 notebook run.
# Inference posts/s: the fp16 notebook run predicts 47,817 OOF + 5 x 11,955 test = 107,592 posts per model (incl. loading).
EFF = {
    "qwen3_4b_lora":   dict(label="Qwen3-4B-Base + LoRA", train_min=5 * 55.6, train_gpu_gb="~10.0 (nvidia-smi)", infer_gpu_gb=8.7,
                            params="4.06B total / 33.0M trainable", saved="5 x 126 MB fp32 (63 MB fp16)", posts_s=107592 / (15.0 * 60)),
    "xlmr_xl_lora":    dict(label="XLM-R XL + LoRA", train_min=5 * 24.5, train_gpu_gb="~18-20 (nvidia-smi)", infer_gpu_gb=7.4,
                            params="3.52B total / 39.7M trainable", saved="5 x 126 MB fp32 (63 MB fp16)", posts_s=107592 / (3.7 * 60)),
    "tfidf_lr":        dict(label="TF-IDF + LogReg", train_min=2.3, train_gpu_gb="0 (CPU)", infer_gpu_gb=0.0,
                            params="≤800k features x 3", saved="not saved (retrains in 2.3 min)", posts_s=None),
    "xlmr_large":      dict(label="XLM-R large (full FT)", train_min=49.6, train_gpu_gb="~12 (nvidia-smi)", infer_gpu_gb=None,
                            params="560M", saved="weights not saved", posts_s=None),
    "mdeberta_v3_base": dict(label="mDeBERTa-v3 base (full FT)", train_min=28.6, train_gpu_gb="n/a", infer_gpu_gb=None,
                             params="278M", saved="weights not saved", posts_s=None),
    "xlmr_base":       dict(label="XLM-R base (full FT)", train_min=21.5, train_gpu_gb="n/a", infer_gpu_gb=None,
                            params="278M", saved="weights not saved", posts_s=None),
}


def scores(P):
    p = P.argmax(1)
    return dict(oof_macro_f1=f1_score(y, p, average="macro"),
                per_class_f1=dict(zip(["explicit", "subtle", "neutral"], f1_score(y, p, average=None).round(4).tolist())),
                per_script_f1={s: round(f1_score(y[script == s], p[script == s], average="macro"), 4)
                               for s in ["latin", "bangla", "mixed"]},
                per_fold_f1=[round(f1_score(y[fold == f], p[fold == f], average="macro"), 4) for f in range(5)])


def curves(run, label):
    hs = []
    for f in range(5):
        z = RUNS / run / f"fold{f}.npz"
        if z.exists():
            h = np.load(z, allow_pickle=True)["hist"]; hs.append(json.loads(str(h)))
    if not hs:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for f, h in enumerate(hs):
        ax[0].plot(np.arange(1, len(h["step_loss"]) + 1) * 50, h["step_loss"], lw=.8, label=f"fold {f}")
        e = [r["epoch"] for r in h["epoch"]]
        ax[1].plot(e, [r["val_loss"] for r in h["epoch"]], "o-", label=f"fold {f} val")
        ax[1].plot(e, [r["train_loss"] for r in h["epoch"]], "x--", alpha=.6, label=f"fold {f} train")
    ax[0].set(title=f"{label}: training loss (50-step mean)", xlabel="step (micro-batch)", ylabel="weighted CE")
    ax[1].set(title="per-epoch train vs validation loss", xlabel="epoch", ylabel="weighted CE")
    for x in ax:
        x.grid(alpha=.3); x.legend(fontsize=6)
    fig.tight_layout()
    for p in (RUNS / run / "curves.png", FIG / f"loss_{run}.png"):
        fig.savefig(p, dpi=110)
    plt.close(fig)
    return [dict(fold=f, best_epoch=int(np.argmax([r["val_f1"] for r in h["epoch"]]) + 1)) for f, h in enumerate(hs)]


rows = []
for run, e in EFF.items():
    S = scores(np.load(RUNS / run / "oof.npy"))
    S["best_epochs"] = curves(run, e["label"])
    S["efficiency"] = {k: v for k, v in e.items() if k != "label"}
    (RUNS / run / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    rows.append((run, e["label"], S))

W = CFG["weights"]
P = sum(w * np.load(RUNS / r / "oof.npy") for r, w in W.items())
B = scores(P); B["weights"] = W
(RUNS / "final_blend_summary.json").write_text(json.dumps(B, indent=1, default=float))
rows.append(("final_blend", "**Final blend (0.5 Qwen + 0.25 XL + 0.25 TF-IDF)**", B))

md = ["| Model | OOF macro-F1 | F1 explicit | F1 subtle | F1 neutral | F1 Latin | F1 Bangla | F1 mixed |", "|---|---|---|---|---|---|---|---|"]
for run, label, S in sorted(rows, key=lambda r: r[2]["oof_macro_f1"]):
    c, s = S["per_class_f1"], S["per_script_f1"]
    md.append(f"| {label} | {S['oof_macro_f1']:.4f} | {c['explicit']:.3f} | {c['subtle']:.3f} | {c['neutral']:.3f} | "
              f"{s['latin']:.3f} | {s['bangla']:.3f} | {s['mixed']:.3f} |")
md += ["", "| Model | Train time, 5 folds (min) | Train GPU (GB) | Inference GPU fp16 (GB) | Parameters | Saved size | Inference posts/s |",
       "|---|---|---|---|---|---|---|"]
for run, e in EFF.items():
    md.append(f"| {e['label']} | {e['train_min']:.0f} | {e['train_gpu_gb']} | {e['infer_gpu_gb'] if e['infer_gpu_gb'] is not None else 'n/a'} | "
              f"{e['params']} | {e['saved']} | {f'{e[chr(112)+"osts_s"]:.0f}' if e['posts_s'] else 'n/a'} |")
(ROOT / "reports/tables.md").write_text("\n".join(md) + "\n")

fig, ax = plt.subplots(figsize=(7.5, 3.6))
names = [r[1].replace("**", "") for r in sorted(rows, key=lambda r: r[2]["oof_macro_f1"])]
vals = [r[2]["oof_macro_f1"] for r in sorted(rows, key=lambda r: r[2]["oof_macro_f1"])]
ax.barh(names, vals, color=["#9aa"] * (len(vals) - 1) + ["#2a7"])
for i, v in enumerate(vals):
    ax.text(v + .002, i, f"{v:.4f}", va="center", fontsize=8)
ax.set(xlim=(.5, .64), xlabel="5-fold OOF macro-F1 (n = 47,817)", title="Model comparison"); ax.grid(axis="x", alpha=.3)
fig.tight_layout(); fig.savefig(FIG / "model_comparison.png", dpi=120); plt.close(fig)
print("\n".join(md)); print(f"\nfinal blend OOF macro-F1 = {B['oof_macro_f1']:.4f}")
