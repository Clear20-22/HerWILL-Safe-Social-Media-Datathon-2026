"""Statistical EDA for HerWILL Safe Social Media Datathon 2026."""
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
tr = pd.read_csv(ROOT / "data/train.csv")
te = pd.read_csv(ROOT / "data/test.csv")
ss = pd.read_csv(ROOT / "data/sample_submission.csv")

BN = re.compile(r"[ঀ-৿]")
LAT = re.compile(r"[A-Za-z]")
URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
HASHTAG = re.compile(r"#\w+")
EMOJI = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")


def script(t):
    b, l = len(BN.findall(t)), len(LAT.findall(t))
    if b == 0 and l == 0:
        return "other"
    r = b / (b + l)
    if r > 0.9:
        return "bangla"
    if r < 0.1:
        return "latin"
    return "mixed"


def feats(df):
    t = df["text"].fillna("").astype(str)
    df = df.assign(
        n_char=t.str.len(),
        n_word=t.str.split().str.len(),
        script=t.map(script),
        has_url=t.str.contains(URL),
        has_mention=t.str.contains(MENTION),
        has_hashtag=t.str.contains(HASHTAG),
        has_emoji=t.map(lambda s: bool(EMOJI.search(s))),
        upper_ratio=t.map(lambda s: sum(c.isupper() for c in s) / max(1, sum(c.isalpha() for c in s))),
        n_excl=t.str.count("!"),
        n_q=t.str.count(r"\?"),
        has_newline=t.str.contains("\n"),
    )
    return df


def sec(title):
    print("\n" + "=" * 80 + f"\n{title}\n" + "=" * 80)


sec("1. BASIC SHAPE / INTEGRITY")
print("train", tr.shape, "test", te.shape, "sample", ss.shape)
print("train nulls:\n", tr.isna().sum().to_dict(), "\ntest nulls:", te.isna().sum().to_dict())
print("id overlap train/test:", len(set(tr.id) & set(te.id)))
print("sample ids == test ids:", set(ss.id) == set(te.id), "order same:", (ss.id.values == te.id.values).all())
print("id range train", tr.id.min(), tr.id.max(), "test", te.id.min(), te.id.max())
print("sample y_pred values:", ss.y_pred.value_counts().to_dict())

sec("2. LABEL DISTRIBUTION")
vc = tr.y.value_counts().sort_index()
print(pd.DataFrame({"count": vc, "pct": (vc / len(tr) * 100).round(2)}))
print("imbalance ratio max/min:", round(vc.max() / vc.min(), 2))
# Does id correlate with label? (leak check)
print("id vs y Spearman:", stats.spearmanr(tr.id, tr.y))
tr["id_bin"] = pd.qcut(tr.id, 10, labels=False)
print("label pct by id decile:\n", pd.crosstab(tr.id_bin, tr.y, normalize="index").round(3))

tr, te = feats(tr), feats(te)

sec("3. SCRIPT / LANGUAGE MIX")
print("train script dist:\n", tr.script.value_counts(normalize=True).round(4))
print("test  script dist:\n", te.script.value_counts(normalize=True).round(4))
print("label dist by script (row %):\n", pd.crosstab(tr.script, tr.y, normalize="index").round(3))
print("counts:\n", pd.crosstab(tr.script, tr.y, margins=True))
chi2, p, dof, _ = stats.chi2_contingency(pd.crosstab(tr.script, tr.y))
n = len(tr); k = min(pd.crosstab(tr.script, tr.y).shape) - 1
print(f"chi2={chi2:.1f} p={p:.2e} Cramer's V={np.sqrt(chi2 / (n * k)):.3f}")

sec("4. LENGTH STATS")
for col in ["n_char", "n_word"]:
    print(f"\n{col} by label:\n", tr.groupby("y")[col].describe(percentiles=[.5, .9, .95, .99]).round(1))
    print(f"{col} by script:\n", tr.groupby("script")[col].describe(percentiles=[.5, .95, .99]).round(1))
    h = stats.kruskal(*[g[col] for _, g in tr.groupby("y")])
    print(f"Kruskal-Wallis {col} ~ y: H={h.statistic:.1f} p={h.pvalue:.2e}")
    print(f"test {col}:", te[col].describe(percentiles=[.5, .95, .99]).round(1).to_dict())
    ks = stats.ks_2samp(tr[col], te[col])
    print(f"KS train vs test {col}: D={ks.statistic:.4f} p={ks.pvalue:.3f}")
print("\nlength-bucket x label (row %):")
tr["len_bin"] = pd.cut(tr.n_word, [0, 3, 7, 15, 30, 60, 120, 10000])
print(pd.crosstab(tr.len_bin, tr.y, normalize="index").round(3).assign(n=tr.len_bin.value_counts().sort_index()))

sec("5. SURFACE FEATURES x LABEL")
for c in ["has_url", "has_mention", "has_hashtag", "has_emoji", "has_newline"]:
    ct = pd.crosstab(tr[c], tr.y)
    chi2, p, *_ = stats.chi2_contingency(ct)
    print(f"{c}: train rate={tr[c].mean():.4f} test rate={te[c].mean():.4f} | label% if True:",
          (ct.loc[True] / ct.loc[True].sum()).round(3).to_dict() if True in ct.index else None, f"p={p:.1e}")
for c in ["upper_ratio", "n_excl", "n_q"]:
    print(c, tr.groupby("y")[c].mean().round(4).to_dict())

sec("6. DUPLICATES / LABEL NOISE / LEAKAGE")
norm = lambda s: re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s)).lower()).strip()
tr["norm"] = tr.text.map(norm); te["norm"] = te.text.map(norm)
print("exact dup texts in train:", tr.text.duplicated().sum(), "| normalized dups:", tr.norm.duplicated().sum())
g = tr.groupby("norm").y.agg(["nunique", "count"])
conf = g[g["nunique"] > 1]
print("normalized texts with conflicting labels:", len(conf), "rows involved:", conf["count"].sum())
print(tr[tr.norm.isin(conf.index)].sort_values("norm")[["text", "y"]].head(20).to_string())
overlap = te.norm.isin(set(tr.norm))
print(f"test rows whose normalized text appears in train: {overlap.sum()} ({overlap.mean()*100:.2f}%)")
print("test internal dups:", te.norm.duplicated().sum())

sec("7. SAMPLES PER CLASS x SCRIPT")
for s in ["latin", "bangla", "mixed"]:
    for y in [0, 1, 2]:
        sub = tr[(tr.script == s) & (tr.y == y)]
        print(f"\n--- script={s} y={y} (n={len(sub)})")
        for t in sub.sample(min(5, len(sub)), random_state=0).text:
            print("  *", t[:200].replace("\n", " "))

sec("8. TOP DISCRIMINATIVE TOKENS (log-odds w/ informative Dirichlet prior)")
tok = lambda s: re.findall(r"[ঀ-৿]+|[a-z']+", s.lower())
cnt = {y: Counter() for y in [0, 1, 2]}
for t, y in zip(tr.text, tr.y):
    cnt[y].update(set(tok(t)))
allc = sum(cnt.values(), Counter())
a0 = sum(allc.values())
for y in [0, 1, 2]:
    rest = allc - cnt[y]
    ny, nr = sum(cnt[y].values()), sum(rest.values())
    sc = {}
    for w, aw in allc.items():
        if aw < 15:
            continue
        a = aw / a0 * 1000
        yi, ri = cnt[y][w], rest[w]
        d = np.log((yi + a) / (ny + 1000 - yi - a)) - np.log((ri + a) / (nr + 1000 - ri - a))
        v = 1 / (yi + a) + 1 / (ri + a)
        sc[w] = d / np.sqrt(v)
    top = sorted(sc.items(), key=lambda x: -x[1])[:30]
    print(f"\nclass {y}:", ", ".join(f"{w}({s:.0f})" for w, s in top))

sec("9. VOCAB / OOV")
vtr = Counter(w for t in tr.text for w in tok(t))
vte = Counter(w for t in te.text for w in tok(t))
print("train vocab:", len(vtr), "test vocab:", len(vte))
oov = sum(c for w, c in vte.items() if w not in vtr) / sum(vte.values())
print(f"test token OOV rate vs train: {oov*100:.2f}%")
print("hapax share of train vocab:", round(sum(1 for c in vtr.values() if c == 1) / len(vtr), 3))

tr.drop(columns=["norm"]).to_pickle(ROOT / "experiments/results/train_feats.pkl")
te.drop(columns=["norm"]).to_pickle(ROOT / "experiments/results/test_feats.pkl")
