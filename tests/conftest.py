"""pytest configuration.

The scripts in gui_scripts/ open a window and block in app.exec(), so they are not collected; run one by hand from
the repo root, e.g. `python tests/gui_scripts/raster_display.py` (needs the full desktop dependencies).
"""
import os
import sys
from pathlib import Path

collect_ignore_glob = ["gui_scripts/*"]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
