"""Build dist/ScatteredBrains_submission.zip for the code-submission form (one file, < 1 GB).
Includes code, configs, tests, requirements, README, reports, data/processed, per-run artifacts, the final CSV and
manifest, and the fp16 LoRA adapters (only if the zip stays under 900 MB). Excludes data/raw, caches, .venv and
experiment/diagnosis code (the id-leak diagnosis is not part of the package). Usage: python scripts/build_zip.py"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist/ScatteredBrains_submission.zip"
PREFIX = "ScatteredBrains_submission"
ADAPTERS = ROOT / "code_submission/herwill_final_submission/adapters"   # fp16 copies of qwen/adapters, xlmr_xl/adapters
LOGS = ROOT / "FINAL_SUBMISSION_QXT/logs"

files = [ROOT / p for p in ("main.py", "requirements.txt", "README.md", "notebooks/herwill_final_kaggle.ipynb")]
for d in ("src", "configs", "scripts", "tests", "reports", "data/processed", "runs", "submissions"):
    files += [p for p in sorted((ROOT / d).rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
entries = [(p, p.relative_to(ROOT)) for p in files]
entries += [(p, Path("logs") / p.relative_to(LOGS)) for p in sorted(LOGS.rglob("*"))
            if p.is_file() and p.parent.name in ("fp16_verification", "model_selection")]
adapter_entries = [(p, Path("adapters") / p.relative_to(ADAPTERS)) for p in sorted(ADAPTERS.rglob("*")) if p.is_file()]
assert not any("data/raw" in str(a) or a.suffix == ".pyc" for _, a in entries)

base = sum(p.stat().st_size for p, _ in entries)
adapters_size = sum(p.stat().st_size for p, _ in adapter_entries)
include_adapters = base + adapters_size < 900 * 2**20
OUT.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p, arc in entries + (adapter_entries if include_adapters else []):
        z.write(p, f"{PREFIX}/{arc.as_posix()}")
size = OUT.stat().st_size
print(f"wrote {OUT.relative_to(ROOT)}: {size / 2**20:.1f} MB, {len(entries) + (len(adapter_entries) if include_adapters else 0)} files, "
      f"adapters {'included' if include_adapters else 'NOT included (size)'}")
assert size < 2**30, "zip is over the 1 GB form limit"
