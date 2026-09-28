"""Build the one-file code submission (form limit: 1 file, 1 GB): code_submission/herwill_final_submission.zip.
It holds the final notebook and script, README, requirements, the LoRA adapters converted to fp16 (half the size;
a Kaggle T4 computes in fp16 anyway), the exact training scripts, and the logs.
Usage (from project root): python code_submission/make_package.py"""
import re
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
CS = ROOT / "code_submission"
PKG = CS / "herwill_final_submission"
if PKG.exists():
    shutil.rmtree(PKG)
for sub in ("adapters", "training_scripts", "logs"):
    (PKG / sub).mkdir(parents=True)

for name, src in (("qwen3_4b", ROOT / "qwen/adapters"), ("xlmr_xl", ROOT / "xlmr_xl/adapters")):
    for f in range(5):
        s, d = src / f"fold{f}", PKG / f"adapters/{name}/fold{f}"
        d.mkdir(parents=True)
        shutil.copy2(s / "adapter_config.json", d)
        with safe_open(s / "adapter_model.safetensors", "pt") as fh:
            meta = fh.metadata()
        tensors = {k: v.half() if v.is_floating_point() else v for k, v in load_file(s / "adapter_model.safetensors").items()}
        save_file(tensors, d / "adapter_model.safetensors", metadata=meta)
shutil.copy2(ROOT / "kaggle_final/dataset/herwill-lora-adapters/README.md", PKG / "adapters/README.md")

shutil.copy2(ROOT / "kaggle_final/herwill_final_kaggle.ipynb", PKG / "herwill_final.ipynb")
shutil.copy2(ROOT / "kaggle_final/herwill_final_kaggle.py", PKG / "herwill_final.py")
for p in ("README.md", "requirements.txt"):
    shutil.copy2(CS / "package_files" / p, PKG / p)
for s in ("qwen/scripts/qwen_lora_cv.py", "xlmr_xl/scripts/xlmr_xl_cv.py"):
    shutil.copy2(ROOT / s, PKG / "training_scripts")
LOGS = {"qwen/logs/qwen3_4b_lora_cv.txt": "qwen3_4b_lora_cv.txt",
        "xlmr_xl/logs/xlmr_xl_cv.txt": "xlmr_xl_cv.txt",
        "kaggle_final/logs/local_fp16_run.txt": "local_fp16_run.txt",
        "kaggle_final/logs/compare_local_fp16_vs_reference.txt": "compare_local_fp16_vs_reference.txt",
        "kaggle_final/logs/fp16_headroom_qwen.txt": "fp16_headroom_qwen.txt",
        "kaggle_final/logs/fp16_adapter_check.txt": "fp16_adapter_check.txt",
        "final_qxt/logs/make_submission.txt": "model_selection_qxt_vs_qlt.txt"}
for s, d in LOGS.items():  # drop the carriage-return progress bars of model loading
    text = (ROOT / s).read_text().replace("\r", "\n")
    text = re.sub(r"(?m)^Loading checkpoint shards:.*\n", "", text)
    (PKG / "logs" / d).write_text(re.sub(r"\n{3,}", "\n\n", text))

zip_path = Path(shutil.make_archive(str(PKG), "zip", root_dir=CS, base_dir=PKG.name))
size = zip_path.stat().st_size
for p in sorted(PKG.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(CS)}  {p.stat().st_size / 2**20:.1f} MB")
print(f"wrote {zip_path.relative_to(ROOT)}: {size / 2**20:.0f} MB")
assert size < 2**30, "over the 1 GB form limit"
