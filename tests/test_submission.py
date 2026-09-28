"""Fast CPU tests. Run: python -m pytest -q tests/
The data/raw tests are skipped when the competition CSVs are not in data/raw/ (see README, section 'Data')."""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/final_qxt.json").read_text())
RAW = ROOT / CFG["raw_dir"]
has_raw = pytest.mark.skipif(not (RAW / "sample_submission.csv").exists(), reason="competition CSVs not in data/raw/")


def test_weights_sum_to_one():
    assert abs(sum(CFG["weights"].values()) - 1) < 1e-12


def test_saved_probabilities_are_valid():
    for run in CFG["weights"]:
        for name, n in (("test.npy", CFG["n_test"]), ("oof.npy", 47817)):
            p = np.load(ROOT / "runs" / run / name)
            assert p.shape == (n, 3) and np.isfinite(p).all()
            assert np.allclose(p.sum(1), 1, atol=1e-4), f"{run}/{name} rows are not probabilities"


def test_folds_cover_every_train_row_once():
    f = pd.read_csv(ROOT / "data/processed/folds.csv")
    assert f["id"].is_unique and sorted(f["fold"].unique()) == [0, 1, 2, 3, 4]
    assert set(f["script"]) <= {"latin", "bangla", "mixed", "other"}
    assert pd.read_csv(ROOT / "data/processed/test_ids.csv")["id"].is_unique


def test_final_csv_format():
    s = pd.read_csv(ROOT / CFG["output_csv"])
    assert list(s.columns) == ["id", "y_pred"] and len(s) == CFG["n_test"] and s["id"].is_unique
    assert s["y_pred"].dtype.kind == "i" and set(s["y_pred"].unique()) <= {0, 1, 2}


def test_blend_reproduces_final_labels():
    P = sum(w * np.load(ROOT / "runs" / r / "test.npy") for r, w in CFG["weights"].items())
    ids = pd.read_csv(ROOT / "data/processed/test_ids.csv")["id"].values
    s = pd.read_csv(ROOT / CFG["output_csv"]).set_index("id").loc[ids]
    assert (s["y_pred"].values == P.argmax(1)).all()


@has_raw
def test_quick_is_byte_identical_to_submitted_file():
    r = subprocess.run([sys.executable, str(ROOT / "main.py"), "--quick"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert hashlib.md5((ROOT / CFG["output_csv"]).read_bytes()).hexdigest() == CFG["expected_md5"]


@has_raw
def test_ids_match_sample_submission():
    s = pd.read_csv(ROOT / CFG["output_csv"]); ss = pd.read_csv(RAW / "sample_submission.csv")
    assert s["id"].tolist() == ss["id"].tolist()


def test_no_id_feature_in_model_code():
    """The id column may only be read as a row key (merge / output / fold file). Flag arithmetic or feature use."""
    bad = re.compile(r"(id\s*//|id_bin|\[\s*[\"']id[\"']\s*\]\s*[-+*/%]|np\.\w+\(\s*\w*\.id\b|features?.*[\"']id[\"'])")
    files = [ROOT / "main.py", *sorted((ROOT / "src").rglob("*.py")), *sorted((ROOT / "scripts").glob("*.py"))]
    hits = [f"{p.relative_to(ROOT)}:{i}: {l.strip()}" for p in files for i, l in enumerate(p.read_text().splitlines(), 1)
            if bad.search(l) and not l.lstrip().startswith("#")]
    assert not hits, "possible id feature use:\n" + "\n".join(hits)
