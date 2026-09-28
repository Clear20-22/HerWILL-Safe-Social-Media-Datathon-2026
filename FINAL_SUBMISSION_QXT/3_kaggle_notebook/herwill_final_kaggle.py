# %% [markdown]
# # HerWILL Safe Social Media Datathon 2026 — final solution
#
# **Task:** 3-class harmful-content classification of Bangla / English / Banglish social-media text
# (0 = explicitly toxic, 1 = subtly toxic, 2 = neutral). **Metric:** macro F1.
#
# **Method:** a soft vote of three models, all trained on the same 5-fold stratified split:
#
# | Component | Weight | 5-fold OOF macro-F1 |
# |---|---|---|
# | Qwen3-4B-Base + LoRA (decoder LLM used as a classifier) | 0.50 | 0.6016 |
# | XLM-RoBERTa-XL + LoRA (3.5B multilingual encoder) | 0.25 | 0.5915 |
# | TF-IDF (word 1-2 + char 2-5 grams) + logistic regression | 0.25 | 0.5277 |
# | **Soft-vote blend** | | **0.6165** |
#
# The OOF scores come from the training run (5 folds, n = 47,817). This notebook recomputes them (section 6).
# The previous version of the blend used XLM-RoBERTa-large (560M, full fine-tuning) instead of XL: OOF 0.6131,
# public LB 0.6097. XL is better by +0.0034 macro-F1 (paired bootstrap over rows, 95% CI [+0.0012, +0.0056]).
#
# **Rule compliance**
# * Only the competition data is used. There is no external or pseudo-labelled data, and no checkpoint that was
#   fine-tuned on toxicity, hate or sentiment labels. The backbones are plain pretrained models.
# * Licenses allow commercial use: Qwen3-4B-Base (Apache-2.0) and XLM-RoBERTa-XL (MIT).
# * The `id` column is **never** used as a feature. It is used only to write the submission rows.
# * Reproducible: fixed seeds, deterministic folds, and every hyper-parameter is in `CFG`.
#
# **How to run on Kaggle:** accelerator GPU T4, Internet ON, and attach the competition data plus the
# Dataset with the LoRA adapters. Then choose Run All. The notebook writes `submission.csv` to `/kaggle/working/`.
# **Locally:** put the competition CSVs in `data/`. The adapters are read from `adapters/` next to this notebook.
# See README.md for details.
#
# **Verified in fp16**, the precision a T4 uses (the adapters were trained in bf16). On an RTX 4090 forced to fp16,
# this notebook reproduced the blend OOF (0.6164 vs 0.6165), and 99.8% of its test labels are identical to the
# submitted file. It took 21 min there (Qwen 15, XL 4, TF-IDF 2), with a peak GPU memory of 8.7 GB.
# A T4 is slower, so expect roughly 1–2 hours (an estimate). `RECOMPUTE_OOF=0` skips the held-out rows and
# roughly halves the GPU time.

# %%
import contextlib
import gc
import json
import os
import random
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

IN_KAGGLE = Path("/kaggle/input").exists()
# The library versions used for the reported run. They matter for loading the saved LoRA adapters.
PINNED = ["transformers==4.57.6", "peft==0.21.0"]
if IN_KAGGLE:  # needs Internet ON, which is also needed to download the pretrained weights
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PINNED], check=True)

import numpy as np
import pandas as pd
import torch
import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
transformers.logging.set_verbosity_error()  # hides "score/classifier newly initialized": the adapter replaces that head

# %% [markdown]
# ## 1. Configuration
# Run modes are set with environment variables, or by editing the defaults below.
# * `TRAIN_LLM=0` (default): the two LoRA models predict with the adapters from the attached Kaggle Dataset.
#   The adapters were trained by exactly the training code in section 4. TF-IDF is always trained here, on CPU.
# * `TRAIN_LLM=1`: retrain both LoRA models from scratch. This needs a GPU with bf16 (Ampere or newer) and 24 GB.
#   It took 4.6 h (Qwen) + 2 h (XL) on an RTX 4090, so it is not feasible on a Kaggle T4.
# * `RECOMPUTE_OOF=1` (default): also predict every fold's held-out rows, so the 5-fold CV score is reproduced here.
# * `QUICK=1`: smoke test on a few rows of folds 0 and 1.

# %%
QUICK = os.environ.get("QUICK") == "1"
TRAIN_LLM = os.environ.get("TRAIN_LLM") == "1"
RECOMPUTE_OOF = os.environ.get("RECOMPUTE_OOF", "1") == "1"

CFG = dict(
    seed=42,
    n_folds=5,
    weights=dict(qwen=0.50, xl=0.25, tfidf=0.25),  # soft-vote weights, see section 6
    class_weight_power=0.5,                         # loss weight of class c = (N / n_c) ** 0.5
    tfidf=dict(word_ngram=(1, 2), char_ngram=(2, 5), word_max=300_000, char_max=500_000, C=4.0),
    llm=dict(
        qwen=dict(model="Qwen/Qwen3-4B-Base", head="score", max_len=152, epochs=2, batch=16, accum=2,
                  lr=1e-4, warmup=0.05, weight_decay=0.0,
                  prompt="Classify the toxicity of this social media post (Bangla/English): explicit, subtle, or neutral.\nPost: ",
                  lora=dict(r=16, lora_alpha=32, lora_dropout=0.05,
                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])),
        xl=dict(model="facebook/xlm-roberta-xl", head="classifier", max_len=128, epochs=3, batch=16, accum=2,
                lr=2e-4, warmup=0.06, weight_decay=0.01, prompt="",
                lora=dict(r=16, lora_alpha=32, lora_dropout=0.05,
                          target_modules=r".*encoder\.layer\.\d+\.(attention\.self\.(query|key|value)"
                                         r"|attention\.output\.dense|intermediate\.dense|output\.dense)")),
    ),
    predict_batch=64,
)

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT = HERE.parent if HERE.name in ("kaggle_final", "notebooks", "src") else HERE


def find_data_dir():
    """Kaggle: /kaggle/input/<competition>/ ; local: <project>/data/."""
    cands = [p.parent for p in Path("/kaggle/input").rglob("train.csv")] if IN_KAGGLE else []
    cands += [PROJECT / "data", Path.cwd() / "data"]
    for c in cands:
        if (c / "train.csv").exists() and (c / "test.csv").exists():
            return c
    sys.exit("train.csv / test.csv not found. Attach the competition data or put it in ./data")


DATA = find_data_dir()
OUT = Path("/kaggle/working") if IN_KAGGLE else Path(os.environ.get("OUT_DIR", HERE / "results"))
OUT = OUT / "quick" if QUICK else OUT
for d in ("cache", "probs"):
    (OUT / d).mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# bf16 needs compute capability >= 8 (Ampere or newer). torch.cuda.is_bf16_supported() also returns True on a T4,
# through slow emulation, so the capability is checked directly. Kaggle T4 / P100 therefore run in fp16.
CAP = torch.cuda.get_device_capability() if DEVICE == "cuda" else (0, 0)
AMP = torch.bfloat16 if CAP[0] >= 8 and os.environ.get("FORCE_FP16") != "1" else torch.float16
LOAD_DTYPE = AMP if DEVICE == "cuda" else torch.float32
print(f"data={DATA}  out={OUT}  device={DEVICE} "
      f"({torch.cuda.get_device_name(0) if DEVICE == 'cuda' else '-'}, capability {CAP})  dtype={LOAD_DTYPE}")
print(f"torch {torch.__version__} | transformers {transformers.__version__} | peft {__import__('peft').__version__}")
print(f"mode: {'TRAIN' if TRAIN_LLM else 'PREDICT from adapters'}  recompute_oof={RECOMPUTE_OOF}  quick={QUICK}")
if TRAIN_LLM:
    assert AMP == torch.bfloat16, "TRAIN_LLM=1 needs a bf16 GPU (Ampere or newer); the adapters were trained in bf16"


def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    os.environ["PYTHONHASHSEED"] = str(s)


def amp():
    return torch.autocast("cuda", dtype=AMP) if DEVICE == "cuda" else contextlib.nullcontext()


seed_everything(CFG["seed"])

# %% [markdown]
# ## 2. Data, preprocessing and folds
# Preprocessing is intentionally light, because the pretrained tokenizers were trained on raw web text:
# * Unicode NFKC normalisation, which unifies look-alike Bangla/Latin code points and full-width characters.
# * Character runs longer than 3 are capped at 3 (e.g. "soooooo" becomes "sooo").
# * Repeated whitespace is collapsed.
# * Emojis, punctuation and casing are kept, because they carry sarcasm and toxicity signal.
#
# All models share one 5-fold `StratifiedKFold(shuffle=True, random_state=42)` split, so their out-of-fold (OOF)
# predictions can be blended and compared row by row. Train and test have the same length and script distributions,
# so the OOF score is a faithful proxy for the leaderboard.

# %%
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
sample = pd.read_csv(DATA / "sample_submission.csv")

_ws, _rep = re.compile(r"\s+"), re.compile(r"(.)\1{3,}")


def clean(t):
    t = unicodedata.normalize("NFKC", str(t))
    t = _rep.sub(r"\1\1\1", t)
    return _ws.sub(" ", t).strip()


X = train["text"].fillna("").map(clean).values
Xt = test["text"].fillna("").map(clean).values
y = train["y"].astype(int).values
folds = list(StratifiedKFold(CFG["n_folds"], shuffle=True, random_state=CFG["seed"]).split(train, y))
FOLDS_RUN = [0, 1] if QUICK else list(range(CFG["n_folds"]))
if QUICK:  # keep the real splits (the adapters were trained on them), but only a few held-out and test rows
    n = int(os.environ.get("QUICK_N", 32))
    folds = [(a[:320] if TRAIN_LLM else a, b[:n]) for a, b in folds]
    test, Xt = test.head(2 * n).copy(), Xt[:2 * n]
HELD_OUT = np.zeros(len(X), bool)
for f in FOLDS_RUN:
    HELD_OUT[folds[f][1]] = True

cnt = np.bincount(y, minlength=3)
CLASS_W = torch.tensor((cnt.sum() / cnt) ** CFG["class_weight_power"], dtype=torch.float)
CLASS_W = CLASS_W / CLASS_W.mean()
print(f"train={len(X)} test={len(Xt)} label dist={np.round(cnt / cnt.sum(), 3)} class loss weights={CLASS_W.numpy().round(3)}")


def macro_f1(yt, p):
    return f1_score(yt, p, average="macro")

# %% [markdown]
# ## 3. Component A: TF-IDF with logistic regression
# Word 1-2 grams plus `char_wb` 2-5 grams. Character n-grams are robust to the heavy spelling variation in Banglish
# and to censored slurs. It is the weakest model alone, but its errors differ from the transformers' errors.

# %%
def run_tfidf():
    c = CFG["tfidf"]
    oof, tp = np.zeros((len(X), 3)), np.zeros((len(Xt), 3))
    for f in FOLDS_RUN:
        a, b = folds[f]
        wv = TfidfVectorizer(analyzer="word", ngram_range=c["word_ngram"], min_df=2, max_features=c["word_max"],
                             sublinear_tf=True, token_pattern=r"(?u)[ঀ-৿]+|\b\w+\b")
        cv = TfidfVectorizer(analyzer="char_wb", ngram_range=c["char_ngram"], min_df=3, max_features=c["char_max"],
                             sublinear_tf=True)
        Xa = hstack([wv.fit_transform(X[a]), cv.fit_transform(X[a])]).tocsr()
        m = LogisticRegression(C=c["C"], max_iter=3000, class_weight="balanced").fit(Xa, y[a])
        oof[b] = m.predict_proba(hstack([wv.transform(X[b]), cv.transform(X[b])]).tocsr())
        tp += m.predict_proba(hstack([wv.transform(Xt), cv.transform(Xt)]).tocsr()) / len(FOLDS_RUN)
        print(f"  tfidf fold {f}: macro-F1={macro_f1(y[b], oof[b].argmax(1)):.4f}", flush=True)
    return oof, tp

# %% [markdown]
# ## 4. Components B and C: LoRA fine-tuned Qwen3-4B-Base and XLM-RoBERTa-XL
# Both use `AutoModelForSequenceClassification`, i.e. a linear head on the last non-pad token (Qwen) or on `<s>`
# (XLM-R), trained together with LoRA adapters while the backbone stays frozen.
# * **Qwen3-4B-Base** (decoder): the text is prefixed with a short task instruction, max 152 tokens. LoRA r=16,
#   alpha=32 on all attention and MLP projections (33M trainable parameters). lr 1e-4, 2 epochs.
# * **XLM-RoBERTa-XL** (encoder): raw text, max 128 tokens. LoRA r=16, alpha=32 on the attention and FFN dense
#   layers of all 36 layers. lr 2e-4, 3 epochs.
# * Both: class-weighted cross-entropy (weights above), AdamW, linear warm-up then linear decay, gradient clipping
#   1.0, effective batch 32, bf16. The classification head is trained in fp32. For every fold, the epoch with the
#   best held-out macro-F1 is kept, and its adapter is saved.
#
# Prediction sorts the texts by length, so batches need little padding. The model output does not depend on the order.

# %%
def make_tokenizer(spec):
    tok = AutoTokenizer.from_pretrained(spec["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


class Collate:
    def __init__(self, tok, spec):
        self.tok, self.prompt, self.max_len = tok, spec["prompt"], spec["max_len"]

    def __call__(self, batch):
        texts, labels = zip(*batch)
        enc = self.tok([self.prompt + t for t in texts], truncation=True, max_length=self.max_len,
                       padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(labels)
        return enc


def predict(model, collate, texts):
    order = np.argsort([len(t) for t in texts], kind="stable")
    dl = DataLoader([(texts[i], 0) for i in order], batch_size=CFG["predict_batch"], collate_fn=collate)
    model.eval(); probs = []
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}; batch.pop("labels")
            with amp():
                logits = model(**batch).logits.float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"non-finite logits in {LOAD_DTYPE} inference")
            probs.append(torch.softmax(logits, -1).cpu().numpy())
    out = np.empty((len(texts), 3))
    out[order] = np.concatenate(probs)
    return out


def build_lora_model(spec, tok):
    m = AutoModelForSequenceClassification.from_pretrained(spec["model"], num_labels=3, dtype=torch.bfloat16)
    m.config.pad_token_id = tok.pad_token_id
    if m.supports_gradient_checkpointing:  # Qwen3 supports it, XLM-R XL does not
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        m.enable_input_require_grads()
    m = get_peft_model(m, LoraConfig(task_type="SEQ_CLS", **spec["lora"]))  # SEQ_CLS also trains and saves the head
    for n, p in m.named_parameters():  # classification head in fp32
        if spec["head"] in n:
            p.data = p.data.float(); p.requires_grad = True
    return m.to(DEVICE)


def train_lora_fold(name, spec, tok, collate, f, adapter_dir):
    """Training code that produced the adapters. Returns (held-out probs, test probs) of the best epoch."""
    a, b = folds[f]
    torch.manual_seed(CFG["seed"] + f); np.random.seed(CFG["seed"] + f)
    model = build_lora_model(spec, tok)
    g = torch.Generator(); g.manual_seed(CFG["seed"] + f)
    dl = DataLoader(list(zip(X[a], y[a])), batch_size=spec["batch"], shuffle=True, collate_fn=collate, generator=g)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=spec["lr"], weight_decay=spec["weight_decay"])
    epochs = 1 if QUICK else spec["epochs"]
    steps = len(dl) // spec["accum"] * epochs
    sch = get_linear_schedule_with_warmup(opt, int(spec["warmup"] * steps), steps)
    lossf = torch.nn.CrossEntropyLoss(weight=CLASS_W.to(DEVICE))
    best, t0 = dict(f1=-1), time.time()
    for ep in range(epochs):
        model.train(); tl = []
        for i, batch in enumerate(dl):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}; lab = batch.pop("labels")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lossf(model(**batch).logits.float(), lab) / spec["accum"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at {name} fold {f} epoch {ep + 1} step {i + 1}")
            loss.backward(); tl.append(loss.item() * spec["accum"])
            if (i + 1) % spec["accum"] == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            if (i + 1) % 400 == 0:
                print(f"    {name} fold {f} ep{ep + 1} step {i + 1}/{len(dl)} loss={np.mean(tl[-400:]):.4f} "
                      f"{(time.time() - t0) / 60:.1f}min", flush=True)
        pv = predict(model, collate, X[b])
        f1 = macro_f1(y[b], pv.argmax(1))
        print(f"    {name} fold {f} epoch {ep + 1}: train_loss={np.mean(tl):.4f} val_macroF1={f1:.4f} "
              f"{(time.time() - t0) / 60:.1f}min", flush=True)
        if f1 > best["f1"]:
            best = dict(f1=f1, val=pv, test=predict(model, collate, Xt))
            model.save_pretrained(adapter_dir)  # LoRA weights + classification head
    del model, opt; gc.collect(); torch.cuda.empty_cache()
    return best["val"], best["test"]


def walk(root):  # os.walk that follows symlinks: Kaggle may mount inputs as links, which Path.rglob skips
    return os.walk(root, followlinks=True)


def scan_adapters(roots, base_model):
    found = {}
    for root in roots:
        for d, _, files in sorted(walk(root)):
            m = re.search(r"fold[_-]?(\d+)$", Path(d).name)
            if m and "adapter_config.json" in files and \
                    json.loads((Path(d) / "adapter_config.json").read_text()).get("base_model_name_or_path") == base_model:
                found.setdefault(int(m.group(1)), Path(d))
    return found


def find_adapters(base_model):
    """{fold: folder} of the LoRA adapters trained on `base_model`. Searches the attached Kaggle Datasets in any
    folder layout, and extracts .zip archives that contain adapters. Locally it searches `adapters/` next to this
    notebook (code-submission layout) and the training folders. ADAPTER_DIR=<folder> overrides the search root."""
    if os.environ.get("ADAPTER_DIR"):
        roots = [Path(os.environ["ADAPTER_DIR"])]
    elif IN_KAGGLE:
        roots = [Path("/kaggle/input")]
    else:
        roots = [p for p in (PROJECT / "adapters", PROJECT / "qwen/adapters", PROJECT / "xlmr_xl/adapters") if p.exists()]
    found = scan_adapters(roots, base_model)
    if any(f not in found for f in FOLDS_RUN):  # e.g. the Dataset was uploaded as a .zip that Kaggle kept zipped
        import zipfile
        for root in roots:
            for d, _, files in walk(root):
                for z in (Path(d) / n for n in files if n.endswith(".zip")):
                    with zipfile.ZipFile(z) as zf:
                        if any(n.endswith("adapter_config.json") for n in zf.namelist()):
                            dest = OUT / "adapters_unzipped" / z.stem
                            if not dest.exists():
                                print(f"extracting {z} -> {dest}", flush=True); zf.extractall(dest)
                            for f, p in scan_adapters([dest], base_model).items():
                                found.setdefault(f, p)
    missing = [f for f in FOLDS_RUN if f not in found]
    if missing:
        seen = [d for r in roots for d, _, _ in walk(r) if len(Path(d).relative_to(r).parts) <= 3][:40]
        sys.exit(f"LoRA adapters of {base_model} for folds {missing} not found under {roots}. "
                 f"Folders seen: {seen}. Attach the adapters Dataset (with its fold0..fold4 folders), or set TRAIN_LLM=1.")
    return found


def predict_from_adapter(spec, tok, collate, f, adapter_dir):
    base = AutoModelForSequenceClassification.from_pretrained(spec["model"], num_labels=3, dtype=LOAD_DTYPE)
    base.config.pad_token_id = tok.pad_token_id
    base.config.use_cache = False  # no KV cache for classification: saves ~1.4 GB per Qwen batch, same outputs
    model = PeftModel.from_pretrained(base, str(adapter_dir)).to(DEVICE)
    val = predict(model, collate, X[folds[f][1]]) if RECOMPUTE_OOF else np.zeros((0, 3))
    tst = predict(model, collate, Xt)
    del model, base; gc.collect(); torch.cuda.empty_cache()
    return val, tst


def run_llm(name):
    spec = CFG["llm"][name]
    tok = make_tokenizer(spec); collate = Collate(tok, spec)
    adapters = None if TRAIN_LLM else find_adapters(spec["model"])
    if adapters:
        print(f"  {name}: adapters from {adapters[FOLDS_RUN[0]].parent}", flush=True)
    oof, tp = np.zeros((len(X), 3)), np.zeros((len(Xt), 3))
    for f in FOLDS_RUN:
        cache = OUT / f"cache/{name}_{'train' if TRAIN_LLM else 'predict'}_fold{f}.npz"  # a restarted session resumes here
        if cache.exists() and (np.load(cache)["val"].size or not RECOMPUTE_OOF):
            d = np.load(cache); val, tst = d["val"], d["test"]
        else:
            t0 = time.time()
            if TRAIN_LLM:
                val, tst = train_lora_fold(name, spec, tok, collate, f, OUT / f"adapters/{name}/fold{f}")
            else:
                val, tst = predict_from_adapter(spec, tok, collate, f, adapters[f])
            np.savez(cache, val=val, test=tst)
            print(f"  {name} fold {f}: {time.time() - t0:.0f}s", flush=True)
        if val.size:
            oof[folds[f][1]] = val
            print(f"  {name} fold {f}: macro-F1={macro_f1(y[folds[f][1]], val.argmax(1)):.4f}", flush=True)
        tp += tst / len(FOLDS_RUN)
    return oof, tp

# %% [markdown]
# ## 5. Run all components

# %%
OOF, TEST, MINUTES, PEAK_GB = {}, {}, {}, {}
for name in CFG["weights"]:
    t0 = time.time()
    print(f"\n=== {name} ===", flush=True)
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    OOF[name], TEST[name] = run_tfidf() if name == "tfidf" else run_llm(name)
    MINUTES[name] = round((time.time() - t0) / 60, 1)
    PEAK_GB[name] = round(torch.cuda.max_memory_allocated() / 2**30, 1) if DEVICE == "cuda" else 0
    np.save(OUT / f"probs/{name}_test.npy", TEST[name]); np.save(OUT / f"probs/{name}_oof.npy", OOF[name])
print("\nminutes per component:", MINUTES, "| peak GPU memory (GB):", PEAK_GB)

# %% [markdown]
# ## 6. Blend and out-of-fold evaluation
# The blend is a weighted average of the class probabilities, followed by argmax. Qwen, the strongest model,
# gets half the weight, and the other two share the rest equally. These weights were chosen among a few simple
# candidates on the 5-fold OOF predictions of the XLM-R-large version, not fitted by an optimiser, and were reused
# unchanged when XL replaced XLM-R large. Tuning per-class thresholds did not generalise in a split-half test.

# %%
W = CFG["weights"]
blend_oof = sum(W[n] * OOF[n] for n in W) / sum(W.values())
blend_test = sum(W[n] * TEST[n] for n in W) / sum(W.values())
np.save(OUT / "probs/blend_test.npy", blend_test)

_bn, _lat = re.compile(r"[ঀ-৿]"), re.compile(r"[A-Za-z]")


def script_of(t):
    b, l = len(_bn.findall(t)), len(_lat.findall(t))
    return "other" if b + l == 0 else "bangla" if b / (b + l) > .9 else "latin" if b / (b + l) < .1 else "mixed"


report = dict(mode="train" if TRAIN_LLM else "predict", dtype=str(LOAD_DTYPE), weights=W, minutes=MINUTES,
              peak_gpu_gb=PEAK_GB, models={})
if TRAIN_LLM or RECOMPUTE_OOF:
    m, scr = HELD_OUT, train["text"].map(script_of).values
    rows = []
    for n, p in list(OOF.items()) + [("BLEND", blend_oof)]:
        pr = p[m].argmax(1)
        r = dict(model=n, weight=W.get(n, ""), macro_f1=macro_f1(y[m], pr),
                 **{f"f1_class{k}": v for k, v in enumerate(f1_score(y[m], pr, average=None, labels=[0, 1, 2]))},
                 **{f"f1_{s}": macro_f1(y[m][scr[m] == s], pr[scr[m] == s]) for s in ("latin", "bangla")})
        rows.append(r); report["models"][n] = r
    print(pd.DataFrame(rows).set_index("model").round(4).to_string())
    print("\nBlend classification report (OOF):\n", classification_report(y[m], blend_oof[m].argmax(1), digits=4))
    cm = confusion_matrix(y[m], blend_oof[m].argmax(1))
    print("Blend confusion matrix (rows = true 0/1/2):\n", cm)
    report["blend_confusion"] = cm.tolist()
json.dump(report, open(OUT / "cv_report.json", "w"), indent=1, default=str)

# %% [markdown]
# ## 7. Submission file
# The format is `id,y_pred`, with rows in `sample_submission.csv` order and integer labels in {0, 1, 2}.
# The file is validated before it is written.

# %%
sub = pd.DataFrame({"id": test["id"].values, "y_pred": blend_test.argmax(1).astype(int)})
sub = sample[["id"]].head(len(test)).merge(sub, on="id", how="left")
assert len(sub) == len(test) and sub["id"].tolist() == sample["id"].head(len(test)).tolist(), "id order differs from sample"
assert sub["y_pred"].notna().all(), "missing predictions"
sub["y_pred"] = sub["y_pred"].astype(int)
assert set(sub["y_pred"].unique()) <= {0, 1, 2} and list(sub.columns) == ["id", "y_pred"]
sub.to_csv(OUT / "submission.csv", index=False)
print(f"saved {OUT / 'submission.csv'}  rows={len(sub)}  "
      f"pred dist={sub['y_pred'].value_counts(normalize=True).sort_index().round(3).to_dict()}")
print(sub.head())
