"""HerWILL Safe Social Media Datathon 2026 - team ScatteredBrains - entry point.

  python main.py --quick     Rebuild the final Kaggle CSV from the saved 5-fold probabilities in runs/ (CPU, seconds).
  python main.py --predict   Recompute every probability with the saved LoRA adapters in adapters/ and retrain TF-IDF,
                             then blend (GPU; about 21 min on an RTX 4090 in fp16).
  python main.py --full      Everything from raw data: prepare folds, train TF-IDF, train Qwen3-4B LoRA and
                             XLM-R XL LoRA (5 folds each), then blend (GPU with bf16 and 24 GB; about 7 h on an RTX 4090).

The final model is a soft vote of three text-only models: 0.50 * Qwen3-4B-Base+LoRA + 0.25 * XLM-R-XL+LoRA
+ 0.25 * TF-IDF+LogisticRegression (weights in configs/final_qxt.json). The id column is only a row key.
See README.md for setup, data placement and measured run times."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "configs/final_qxt.json").read_text())


def fail(msg, section):
    sys.exit(f"ERROR: {msg}\n       See README.md, section '{section}'.")


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def load_sample():
    p = ROOT / CFG["raw_dir"] / "sample_submission.csv"
    if not p.exists():
        fail(f"{p.relative_to(ROOT)} not found. Put train.csv, test.csv and sample_submission.csv in data/raw/.", "Data")
    return pd.read_csv(p)


def blend_test_probs(runs_dir):
    """0.50 Qwen + 0.25 XL + 0.25 TF-IDF on the 5-fold-averaged test probabilities (rows in test.csv order)."""
    P = 0
    for run, w in CFG["weights"].items():
        f = runs_dir / run / "test.npy"
        if not f.exists():
            fail(f"{f.relative_to(ROOT) if f.is_relative_to(ROOT) else f} not found (saved test probabilities of run '{run}').",
                 "Quick reproduction")
        P = P + w * np.load(f)
    return P


def write_submission(P, sample, test_ids, out_csv, extra):
    """Validate and write id,y_pred in sample_submission order, plus a JSON manifest."""
    if len(P) != len(test_ids):
        fail(f"{len(P)} probability rows but {len(test_ids)} test ids.", "Quick reproduction")
    sub = sample[["id"]].merge(pd.DataFrame({"id": test_ids, "y_pred": P.argmax(1).astype(int)}), on="id", how="left")
    assert list(sub.columns) == ["id", "y_pred"], "columns must be exactly id,y_pred"
    assert len(sub) == CFG["n_test"] == len(sample), f"expected {CFG['n_test']} rows"
    assert sub["id"].tolist() == sample["id"].tolist() and sub["id"].is_unique, "ids must equal sample_submission.csv"
    assert sub["y_pred"].notna().all(), "missing predictions"
    sub["y_pred"] = sub["y_pred"].astype(int)
    assert set(sub["y_pred"].unique()) <= {0, 1, 2}, "y_pred must be in {0,1,2}"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv, index=False)
    digest = md5(out_csv)
    manifest = dict(file=str(out_csv.relative_to(ROOT)), md5=digest, rows=len(sub),
                    matches_submitted_file=digest == CFG["expected_md5"], submitted_md5=CFG["expected_md5"],
                    weights=CFG["weights"], features=CFG["features"],
                    label_share=sub["y_pred"].value_counts(normalize=True).sort_index().round(4).to_dict(), **extra)
    (ROOT / CFG["manifest"]).write_text(json.dumps(manifest, indent=1, default=str))
    return sub, manifest


def quick():
    runs = ROOT / CFG["runs_dir"]
    ids_file = ROOT / CFG["processed_dir"] / "test_ids.csv"
    if not ids_file.exists():
        fail(f"{ids_file.relative_to(ROOT)} not found. Run `python scripts/prepare_data.py` after placing the raw CSVs.", "Data")
    sample = load_sample()
    test_ids = pd.read_csv(ids_file)["id"].values
    P = blend_test_probs(runs)
    oof = {r: json.loads((runs / r / "summary.json").read_text())["oof_macro_f1"]
           for r in CFG["weights"] if (runs / r / "summary.json").exists()}
    extra = dict(mode="quick", source_runs={r: str((runs / r / "test.npy").relative_to(ROOT)) for r in CFG["weights"]},
                 run_oof_macro_f1=oof, input_md5={r: md5(runs / r / "test.npy") for r in CFG["weights"]})
    blend_summary = runs / "final_blend_summary.json"
    if blend_summary.exists():
        extra["blend_oof_macro_f1"] = json.loads(blend_summary.read_text())["oof_macro_f1"]
    sub, man = write_submission(P, sample, test_ids, ROOT / CFG["output_csv"], extra)
    print(f"wrote {CFG['output_csv']}  rows={man['rows']}  md5={man['md5']}  label share={man['label_share']}")
    print(f"byte-identical to the file submitted to Kaggle: {man['matches_submitted_file']}")
    print(f"manifest: {CFG['manifest']}")
    return 0 if man["matches_submitted_file"] else 1


def run_pipeline(train_llm):
    """Runs src/herwill_final_kaggle.py (the same code as the Kaggle notebook) locally."""
    raw = ROOT / CFG["raw_dir"]
    for f in ("train.csv", "test.csv", "sample_submission.csv"):
        if not (raw / f).exists():
            fail(f"data/raw/{f} not found.", "Data")
    if not train_llm and not (ROOT / "adapters").exists():
        fail("adapters/ not found (LoRA adapters of Qwen3-4B and XLM-R XL, folds 0-4).", "LoRA adapters")
    print("[1/4] prepare folds and test ids", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/prepare_data.py"), "--raw", str(raw),
                    "--out", str(ROOT / CFG["processed_dir"])], check=True)
    out = ROOT / ("runs_full" if train_llm else "runs_predict")
    env = dict(os.environ, DATA_DIR=str(raw), OUT_DIR=str(out), TRAIN_LLM="1" if train_llm else "0")
    if not train_llm:
        env["ADAPTER_DIR"] = str(ROOT / "adapters")
    print(f"[2/4] TF-IDF (train, CPU)  [3/4] Qwen3-4B + XLM-R XL ({'train 5 folds each' if train_llm else 'predict with adapters'})"
          f"  [4/4] blend  ->  {out.relative_to(ROOT)}/", flush=True)
    subprocess.run([sys.executable, str(ROOT / "src/herwill_final_kaggle.py")], env=env, check=True)
    csv = out / "submission.csv"
    print(f"done: {csv.relative_to(ROOT)}  (agreement with the submitted CSV: "
          f"{(pd.read_csv(csv).y_pred.values == pd.read_csv(ROOT / CFG['output_csv']).y_pred.values).mean():.4f})"
          if (ROOT / CFG["output_csv"]).exists() else f"done: {csv.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true", help="rebuild the final CSV from saved probabilities (CPU)")
    g.add_argument("--predict", action="store_true", help="recompute probabilities from the LoRA adapters (GPU)")
    g.add_argument("--full", action="store_true", help="train every model from raw data, then blend (GPU, ~7 h)")
    a = ap.parse_args()
    sys.exit(quick() if a.quick else run_pipeline(train_llm=a.full))
