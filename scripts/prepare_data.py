"""Step 1 of the full pipeline: write data/processed/ from data/raw/.
* folds.csv     id, fold (StratifiedKFold(5, shuffle=True, random_state=42) on y, the split every model used), script
* test_ids.csv  id in test.csv order (the row order of every saved test-probability array)
The id column is only a row key here. It is never a model feature.
Usage: python scripts/prepare_data.py [--raw data/raw] [--out data/processed]"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

BN, LAT = re.compile(r"[ঀ-৿]"), re.compile(r"[A-Za-z]")


def script_of(t):
    """Writing system of a post: bangla (>90% Bengali letters), latin (<10%), mixed, or other."""
    b, l = len(BN.findall(str(t))), len(LAT.findall(str(t)))
    if b + l == 0:
        return "other"
    r = b / (b + l)
    return "bangla" if r > 0.9 else "latin" if r < 0.1 else "mixed"


def prepare(raw: Path, out: Path):
    tr, te = pd.read_csv(raw / "train.csv"), pd.read_csv(raw / "test.csv")
    folds = np.zeros(len(tr), int)
    for f, (_, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, tr["y"].values)):
        folds[b] = f
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": tr["id"], "fold": folds, "script": tr["text"].map(script_of)}).to_csv(out / "folds.csv", index=False)
    te[["id"]].to_csv(out / "test_ids.csv", index=False)
    print(f"wrote {out / 'folds.csv'} ({len(tr)} rows) and {out / 'test_ids.csv'} ({len(te)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw"); ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()
    prepare(Path(a.raw), Path(a.out))
