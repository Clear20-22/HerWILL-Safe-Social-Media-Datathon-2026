# %% [markdown]
# # HerWILL Safe Social Media Datathon 2026 — Final Solution
#
# **Task:** 3-class harmful-content classification of Bangla / English / Banglish social-media text
# (0 = Explicitly toxic, 1 = Subtly toxic, 2 = Neutral). **Metric:** Macro F1.
#
# **Method (one sentence):** a soft-voting ensemble of three multilingual pretrained transformers
# (XLM-RoBERTa-large, mDeBERTa-v3-base, XLM-RoBERTa-base) fine-tuned with 5-fold stratified CV,
# plus a word+char TF-IDF logistic-regression model for diversity.
#
# **Why this design** (all numbers from our fold-0 experiments, see `experiments/`):
#
# | Model (fold 0) | Macro F1 |
# |---|---|
# | TF-IDF word+char + LR | 0.528 |
# | XLM-R base | 0.562 |
# | mDeBERTa-v3 base | 0.577 |
# | XLM-R large | 0.579 |
# | **Ensemble of the 4 (equal weights)** | **0.597** (+0.018 vs best single, paired-bootstrap 95% CI [+0.010, +0.026]) |
#
# **Rule compliance**
# * Only the competition data is used. No external or pseudo-labelled data, and no checkpoints
#   fine-tuned on hate or sentiment labels. The models are plain self-supervised pretrained encoders.
# * All models use OSI-approved licenses that allow commercial use: XLM-R (MIT), mDeBERTa-v3 (MIT).
#   BanglaBERT (CC BY-NC-SA) and Twitter-XLM-R (no license) were evaluated but are excluded.
# * The `id` column is **never** used as a feature. It encodes row order of the source corpora and would leak labels.
# * Fully reproducible: fixed seeds, deterministic folds, and every hyper-parameter is in `CFG` below.
#
# **Runs on:** Kaggle GPU (T4 / P100, fp16) or any local CUDA GPU (bf16 when supported).
# On Kaggle, turn **Internet ON** (to download the pretrained weights) and attach the competition data.

# %%
import gc
import json
import os
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# %% [markdown]
# ## 1. Configuration
# All hyper-parameters live here. `QUICK=1` runs a tiny smoke test (subset, 1 epoch, 2 folds).
# `FOLDS` can be lowered (e.g. `FOLDS=3`) if the GPU budget is tight. The default 5 matches our reported CV.

# %%
QUICK = os.environ.get("QUICK", "0") == "1"

CFG = dict(
    seed=42,
    n_folds=int(os.environ.get("FOLDS", 2 if QUICK else 5)),
    max_len=128,             # 99th percentile of token length is ~90-113 across tokenizers
    warmup_ratio=0.06,
    weight_decay=0.01,
    grad_clip=1.0,
    class_weight_power=0.5,  # loss weight = (N / n_c) ** 0.5  -> milder than full inverse frequency
    # (hf_name, learning_rate, epochs, batch_size, grad_accum). Effective batch = 32 for every model.
    # The best epoch per fold is chosen on that fold's validation macro-F1.
    transformers=[
        ("xlm-roberta-large", 1e-5, 3, 16, 2),
        ("microsoft/mdeberta-v3-base", 3e-5, 3, 32, 1),
        ("xlm-roberta-base", 2e-5, 3, 32, 1),
    ],
    tfidf=dict(word_ngram=(1, 2), char_ngram=(2, 5), word_max=300_000, char_max=500_000, C=4.0),
    ensemble_weights=None,   # None = equal weights (weight tuning did not generalise in our split-half test)
)
if QUICK:
    CFG["transformers"] = [(m, lr, 1, bs, ga) for m, lr, _, bs, ga in CFG["transformers"]]


HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT = HERE.parent if HERE.name in ("src", "notebooks") else HERE


def find_data_dir():
    """Kaggle: /kaggle/input/<competition>/ ; local: <project>/data/."""
    cands = []
    if Path("/kaggle/input").exists():
        cands += [p.parent for p in Path("/kaggle/input").rglob("train.csv")]
    cands += [PROJECT / "data", Path.cwd() / "data"]
    for c in cands:
        if (c / "train.csv").exists() and (c / "test.csv").exists():
            return c
    sys.exit("train.csv / test.csv not found. Attach the competition data or place it in ./data")


DATA = find_data_dir()
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else PROJECT / "outputs"
OUT = OUT / "quick" if QUICK else OUT
(OUT / "cache").mkdir(parents=True, exist_ok=True)
(OUT / "plots").mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# FORCE_FP16=1 simulates Kaggle T4/P100 (no bf16) on a newer GPU
USE_BF16 = DEVICE == "cuda" and torch.cuda.is_bf16_supported() and os.environ.get("FORCE_FP16") != "1"
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
print(f"data={DATA}  out={OUT}  device={DEVICE}  "
      f"gpu={torch.cuda.get_device_name(0) if DEVICE == 'cuda' else '-'}  amp={AMP_DTYPE}")
print("torch", torch.__version__, "| transformers", __import__("transformers").__version__)


def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    os.environ["PYTHONHASHSEED"] = str(s)


seed_everything(CFG["seed"])

# %% [markdown]
# ## 2. Data loading and preprocessing
# Preprocessing is intentionally **light**, because pretrained tokenizers are trained on raw web text:
# * Unicode NFKC normalisation, which unifies look-alike Bangla/Latin code points and full-width characters.
# * Collapse repeated whitespace.
# * Cap character runs longer than 3 (e.g. "soooooo" to "sooo") to reduce tokenizer fragmentation.
# * Emojis, punctuation and casing are **kept**, because they carry sarcasm and toxicity signal.
#
# The EDA (`experiments/logs/01_eda.txt`) shows the train/test length and language distributions match
# (KS p=0.15), so a random stratified K-fold is a faithful proxy for the leaderboard.

# %%
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
sample = pd.read_csv(DATA / "sample_submission.csv") if (DATA / "sample_submission.csv").exists() else None
if QUICK:
    train = train.sample(3000, random_state=0).reset_index(drop=True)
    test = test.head(500).copy()

_ws = re.compile(r"\s+")
_rep = re.compile(r"(.)\1{3,}")


def clean(t):
    t = unicodedata.normalize("NFKC", str(t))
    t = _rep.sub(r"\1\1\1", t)
    return _ws.sub(" ", t).strip()


train["clean"] = train["text"].fillna("").map(clean)
test["clean"] = test["text"].fillna("").map(clean)
y = train["y"].astype(int).values
print(f"train={len(train)} test={len(test)} label dist={np.bincount(y) / len(y)}")

folds = list(StratifiedKFold(CFG["n_folds"], shuffle=True, random_state=CFG["seed"]).split(train, y))
cnt = np.bincount(y, minlength=3)
CLASS_W = torch.tensor((cnt.sum() / cnt) ** CFG["class_weight_power"], dtype=torch.float)
CLASS_W = CLASS_W / CLASS_W.mean()
print("class loss weights:", CLASS_W.numpy().round(3))


def macro_f1(yt, p):
    return f1_score(yt, p, average="macro")

# %% [markdown]
# ## 3. Model A: TF-IDF (word 1-2 gram + char_wb 2-5 gram) with Logistic Regression
# Character n-grams are robust to the heavy spelling variation in Banglish and censored slurs (e.g. "জা*রজ*").

# %%
def run_tfidf():
    cache = OUT / "cache/tfidf.npz"
    if cache.exists():
        d = np.load(cache); return d["oof"], d["test"]
    c = CFG["tfidf"]
    oof = np.zeros((len(train), 3)); tp = np.zeros((len(test), 3))
    X, Xt = train["clean"].values, test["clean"].values
    for f, (a, b) in enumerate(folds):
        wv = TfidfVectorizer(analyzer="word", ngram_range=c["word_ngram"], min_df=2, max_features=c["word_max"],
                             sublinear_tf=True, token_pattern=r"(?u)[ঀ-৿]+|\b\w+\b")
        cv = TfidfVectorizer(analyzer="char_wb", ngram_range=c["char_ngram"], min_df=3, max_features=c["char_max"],
                             sublinear_tf=True)
        Xa = hstack([wv.fit_transform(X[a]), cv.fit_transform(X[a])]).tocsr()
        m = LogisticRegression(C=c["C"], max_iter=3000, class_weight="balanced")
        m.fit(Xa, y[a])
        oof[b] = m.predict_proba(hstack([wv.transform(X[b]), cv.transform(X[b])]).tocsr())
        tp += m.predict_proba(hstack([wv.transform(Xt), cv.transform(Xt)]).tocsr()) / len(folds)
        print(f"  tfidf fold {f}: macro-F1={macro_f1(y[b], oof[b].argmax(1)):.4f}", flush=True)
    np.savez(cache, oof=oof, test=tp)
    return oof, tp

# %% [markdown]
# ## 4. Models B–D: fine-tuned transformers
# * Architecture: pretrained encoder plus the standard sequence-classification head (dropout, dense, tanh, linear to 3).
# * Loss: class-weighted cross-entropy (weights above), which targets macro-F1 under the 50/38/12 class split.
# * Optimiser: AdamW, linear warm-up (6%) then linear decay, gradient clipping 1.0, and mixed precision
#   (bf16 on Ampere+, fp16 with loss scaling on T4/P100).
# * Dynamic padding, max 128 tokens. For every fold we keep the validation and test probabilities of the best epoch.
# * Each (model, fold) result is cached to `cache/`, so a restarted Kaggle session resumes where it stopped.

# %%
class Collate:
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, batch):
        texts, labels = zip(*batch)
        enc = self.tok(list(texts), truncation=True, max_length=CFG["max_len"], padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc


def predict(model, dl):
    model.eval(); out = []
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}; batch.pop("labels")
            with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=DEVICE == "cuda"):
                logits = model(**batch).logits
            out.append(torch.softmax(logits.float(), -1).cpu().numpy())
    return np.concatenate(out)


def make_scaler():
    enabled = DEVICE == "cuda" and not USE_BF16
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # torch < 2.3
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_fold(name, lr, epochs, bs, accum, f, a, b, tok):
    seed_everything(CFG["seed"] + f)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=3).to(DEVICE)
    col = Collate(tok)
    g = torch.Generator(); g.manual_seed(CFG["seed"] + f)
    dl_tr = DataLoader(list(zip(train["clean"].values[a], y[a])), batch_size=bs, shuffle=True, collate_fn=col, generator=g)
    dl_va = DataLoader(list(zip(train["clean"].values[b], y[b])), batch_size=128, collate_fn=col)
    dl_te = DataLoader(list(zip(test["clean"].values, np.zeros(len(test), int))), batch_size=128, collate_fn=col)

    lossf = torch.nn.CrossEntropyLoss(weight=CLASS_W.to(DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=CFG["weight_decay"])
    total = (len(dl_tr) // accum) * epochs
    sch = get_linear_schedule_with_warmup(opt, int(CFG["warmup_ratio"] * total), total)
    scaler = make_scaler()

    hist = dict(step_loss=[], epoch=[])
    best = dict(f1=-1)
    for ep in range(epochs):
        model.train(); run = []
        for i, batch in enumerate(dl_tr):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            lab = batch.pop("labels")
            with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=DEVICE == "cuda"):
                logits = model(**batch).logits
            loss = lossf(logits.float(), lab) / accum
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sch.step()
            run.append(loss.item() * accum)
            if (i + 1) % 50 == 0:
                hist["step_loss"].append(float(np.mean(run[-50:])))
        pv = predict(model, dl_va)
        # same class-weighted CE as training, so the two curves are comparable
        vloss = float(torch.nn.functional.cross_entropy(torch.log(torch.tensor(pv) + 1e-9), torch.tensor(y[b]), weight=CLASS_W))
        f1 = macro_f1(y[b], pv.argmax(1))
        hist["epoch"].append(dict(epoch=ep + 1, train_loss=float(np.mean(run)), val_loss=vloss, val_f1=float(f1)))
        print(f"    epoch {ep + 1}: train_loss={np.mean(run):.4f} val_loss={vloss:.4f} val_macroF1={f1:.4f}", flush=True)
        if f1 > best["f1"]:
            best = dict(f1=f1, epoch=ep + 1, val=pv, test=predict(model, dl_te))
    del model, opt; gc.collect(); torch.cuda.empty_cache()
    return best, hist


def run_transformer(name, lr, epochs, bs, accum):
    tag = name.replace("/", "__")
    tok = AutoTokenizer.from_pretrained(name)
    oof = np.zeros((len(train), 3)); tp = np.zeros((len(test), 3)); hists = []
    for f, (a, b) in enumerate(folds):
        cache = OUT / f"cache/{tag}_fold{f}.npz"
        if cache.exists():
            d = np.load(cache, allow_pickle=True)
            oof[b], fold_test, h = d["val"], d["test"], json.loads(str(d["hist"]))
        else:
            t0 = time.time()
            print(f"  {name} fold {f} ...", flush=True)
            best, h = train_fold(name, lr, epochs, bs, accum, f, a, b, tok)
            oof[b], fold_test = best["val"], best["test"]
            h["best_epoch"], h["minutes"] = best["epoch"], round((time.time() - t0) / 60, 1)
            np.savez(cache, val=best["val"], test=best["test"], hist=json.dumps(h))
        tp += fold_test / len(folds); hists.append(h)
        print(f"  {name} fold {f}: macro-F1={macro_f1(y[b], oof[b].argmax(1)):.4f} "
              f"(best epoch {h['best_epoch']}, {h.get('minutes', '?')} min)", flush=True)
    plot_losses(name, hists)
    return oof, tp, hists


def plot_losses(name, hists):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for f, h in enumerate(hists):
        ax[0].plot(np.arange(1, len(h["step_loss"]) + 1) * 50, h["step_loss"], lw=1, label=f"fold {f}")
        e = [r["epoch"] for r in h["epoch"]]
        ax[1].plot(e, [r["val_loss"] for r in h["epoch"]], "o-", label=f"fold {f} val")
        ax[1].plot(e, [r["train_loss"] for r in h["epoch"]], "x--", alpha=.6, label=f"fold {f} train")
    ax[0].set(title=f"{name}: training loss (50-step avg)", xlabel="step", ylabel="CE loss")
    ax[1].set(title="per-epoch train vs validation loss", xlabel="epoch", ylabel="CE loss")
    for x in ax:
        x.grid(alpha=.3); x.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / f"plots/loss_{name.replace('/', '__')}.png", dpi=120); plt.close(fig)

# %% [markdown]
# ## 5. Train everything (5-fold CV)

# %%
T0 = time.time()
OOF, TEST, HIST = {}, {}, {}
print("\n=== TF-IDF + LR ===")
OOF["tfidf-lr"], TEST["tfidf-lr"] = run_tfidf()
for name, lr, ep, bs, ga in CFG["transformers"]:
    print(f"\n=== {name} (lr={lr}, epochs={ep}, batch={bs}x{ga}) ===")
    OOF[name], TEST[name], HIST[name] = run_transformer(name, lr, ep, bs, ga)
print(f"\ntotal training time: {(time.time() - T0) / 60:.1f} min")

# %% [markdown]
# ## 6. Evaluation: out-of-fold CV scores, ensemble, and per-language breakdown

# %%
names = list(OOF)
w = CFG["ensemble_weights"] or {n: 1.0 for n in names}
ens_oof = sum(w[n] * OOF[n] for n in names) / sum(w.values())
ens_test = sum(w[n] * TEST[n] for n in names) / sum(w.values())

_bn, _lat = re.compile(r"[ঀ-৿]"), re.compile(r"[A-Za-z]")


def script_of(t):
    b, l = len(_bn.findall(t)), len(_lat.findall(t))
    return "other" if b + l == 0 else "bangla" if b / (b + l) > .9 else "latin" if b / (b + l) < .1 else "mixed"


scr = train["text"].map(script_of).values
report = {"cv_folds": CFG["n_folds"], "models": {}}
rows = []
for n, p in list(OOF.items()) + [("ENSEMBLE", ens_oof)]:
    pr = p.argmax(1)
    r = dict(model=n, macro_f1=macro_f1(y, pr), **{f"f1_class{k}": v for k, v in enumerate(f1_score(y, pr, average=None))},
             **{f"f1_{s}": macro_f1(y[scr == s], pr[scr == s]) for s in ["latin", "bangla", "mixed"] if (scr == s).any()})
    rows.append(r); report["models"][n] = r
table = pd.DataFrame(rows).set_index("model").round(4)
print(table.to_string())
print("\nEnsemble classification report (OOF):\n", classification_report(y, ens_oof.argmax(1), digits=4))
print("Ensemble confusion matrix (rows = true 0/1/2):\n", confusion_matrix(y, ens_oof.argmax(1)))
report["ensemble_confusion"] = confusion_matrix(y, ens_oof.argmax(1)).tolist()
report["config"] = {k: v for k, v in CFG.items()}
report["histories"] = HIST
json.dump(report, open(OUT / "cv_report.json", "w"), indent=1, default=str)
table.to_csv(OUT / "cv_scores.csv")
np.save(OUT / "oof_ensemble.npy", ens_oof); np.save(OUT / "test_ensemble.npy", ens_test)

fig, ax = plt.subplots(figsize=(7, 3.5))
table["macro_f1"].sort_values().plot.barh(ax=ax, color=["#888"] * (len(table) - 1) + ["#2a6"])
ax.set(xlabel=f"OOF macro-F1 ({CFG['n_folds']}-fold)", title="Model comparison"); ax.grid(axis="x", alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "plots/model_comparison.png", dpi=120); plt.close(fig)

# %% [markdown]
# ## 7. Submission file
# The format is exactly `id,y_pred`, with rows in `sample_submission.csv` order and integer labels in {0, 1, 2}.
# The file is validated before writing.

# %%
sub = pd.DataFrame({"id": test["id"].values, "y_pred": ens_test.argmax(1).astype(int)})
if sample is not None and not QUICK:
    sub = sample[["id"]].merge(sub, on="id", how="left")
    assert len(sub) == len(sample), "row count differs from sample_submission"
    assert sub["id"].tolist() == sample["id"].tolist(), "id order differs from sample_submission"
assert sub["y_pred"].notna().all(), "missing predictions"
sub["y_pred"] = sub["y_pred"].astype(int)
assert set(sub["y_pred"].unique()) <= {0, 1, 2}, "labels outside {0,1,2}"
assert list(sub.columns) == ["id", "y_pred"]
sub.to_csv(OUT / "submission.csv", index=False)
print(f"\nsaved {OUT / 'submission.csv'}  rows={len(sub)}  pred dist={sub.y_pred.value_counts(normalize=True).sort_index().round(3).to_dict()}")
print(sub.head())
