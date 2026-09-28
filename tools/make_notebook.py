"""Convert a script with `# %%` cell markers into a Jupyter notebook for Kaggle / code submission.
Usage: python tools/make_notebook.py [SRC.py DST.ipynb]
Default: src/herwill_final.py -> notebooks/herwill_final.ipynb (Submission #1)."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src_path, out = ((Path(sys.argv[1]), Path(sys.argv[2])) if len(sys.argv) == 3 else
                 (ROOT / "src/herwill_final.py", ROOT / "notebooks/herwill_final.ipynb"))
src = src_path.read_text()
cells = []
for chunk in re.split(r"^# %%", src, flags=re.M)[1:]:
    is_md = chunk.startswith(" [markdown]")
    body = chunk.split("\n", 1)[1] if "\n" in chunk else ""
    body = body.strip("\n")
    if is_md:
        body = "\n".join(re.sub(r"^# ?", "", l) for l in body.splitlines())
        cells.append(dict(cell_type="markdown", metadata={}, source=body.splitlines(keepends=True)))
    else:
        cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
                          source=body.splitlines(keepends=True)))
nb = dict(nbformat=4, nbformat_minor=5, cells=cells,
          metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python"),
                        language_info=dict(name="python")))
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"wrote {out} ({len(cells)} cells)")
