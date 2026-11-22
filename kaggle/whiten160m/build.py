"""Build the self-contained Kaggle script: embeds train_sae.py and eval_sae.py."""
import base64
from pathlib import Path

here = Path(__file__).parent
scripts = here.parents[1] / "whitening" / "scripts"
files = {n: base64.b64encode((scripts / n).read_bytes()).decode() for n in ["train_sae.py", "eval_sae.py"]}
import sys

tokens, seeds, out = sys.argv[1], sys.argv[2], sys.argv[3]
template = (here / "driver_template.py").read_text()
code = template.replace("__FILES__", repr(files)).replace("__TOKENS__", tokens).replace("__SEEDS__", seeds)
(here / out).write_text(code)
print("built", out)
