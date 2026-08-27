import sys
from pathlib import Path

# scripts/ isn't a package (no __init__.py, matches the CLI convention of
# running each file directly with `python scripts/x.py`), so tests import
# modules like `ad_tracker` by putting scripts/ on sys.path here rather
# than via package-relative imports.
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
