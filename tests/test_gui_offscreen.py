"""The desktop GUI still builds (offscreen) and its raster display uses utils/display_stretch.

Runs in a fresh subprocess (other Qt bindings in a shared environment, e.g. PySide6, can load a different QtCore
into the pytest process first). Skipped when the desktop dependencies are missing. PDAL is blocked (optional: only
LIDAR ground classification needs it), so this also checks that the GUI starts without it.
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NEEDED = ("PyQt6", "rasterio", "geopandas", "serial", "requests", "torch", "laspy", "matplotlib", "sqlite3")

CODE = """
import sys
import numpy as np
sys.modules['pdal'] = None  # import pdal -> ImportError
from PyQt6.QtWidgets import QApplication
app = QApplication([])
from geo_segmenter.gui.main_window import MainWindow
from geo_segmenter.main import main  # the console-script entry point exists
from geo_segmenter.data.providers.lidar_provider import LidarProvider
from geo_segmenter.data.layer import RasterLayer
from geo_segmenter.utils.display_stretch import prepare_display
w = MainWindow()
assert w.windowTitle() == 'Geospatial Segmentation Tool' and w.model_combo.findText('Random Forest') >= 0
layer = RasterLayer.__new__(RasterLayer)
layer.band_indices, layer.stretch_params = [3, 2, 1], {'type': 'Linear 5%', 'min': 0, 'max': 1}
data = np.random.default_rng(0).random((3, 9, 8))
assert np.array_equal(layer.prepare_display_data(data), prepare_display(data, [3, 2, 1], layer.stretch_params))
w.close()
print('GUI_OK')
"""


@pytest.mark.skipif(not all(importlib.util.find_spec(m) for m in NEEDED), reason="needs the desktop dependencies")
def test_main_window_builds_and_display_uses_shared_stretch():
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    r = subprocess.run([sys.executable, "-c", CODE], cwd=ROOT, capture_output=True, text=True, timeout=180, env=env)
    out = r.stdout + r.stderr
    if r.returncode and "sqlite3" in out and "Symbol not found" in out:
        pytest.skip("this Python's sqlite3 extension cannot load (set DYLD_LIBRARY_PATH to a matching libsqlite)")
    assert "GUI_OK" in r.stdout, out[-3000:]


DIALOGS = """
import numpy as np
from PyQt6.QtWidgets import QApplication
app = QApplication([])
from geo_segmenter.gui.dialogs.feature_dialog import FeatureDialog
from geo_segmenter.gui.dialogs.results_dialog import ResultsDialog
img = np.random.default_rng(0).random((32, 32, 4))
FeatureDialog(img)
ResultsDialog(img, np.zeros((32, 32), int), np.full((32, 32, 2), 0.5), ['a', 'b'])
print('DIALOGS_OK')
"""


@pytest.mark.skipif(not all(importlib.util.find_spec(m) for m in NEEDED), reason="needs the desktop dependencies")
def test_feature_and_results_dialogs_build():
    """Regression: both dialogs used QWidget without importing it (NameError on opening)."""
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    r = subprocess.run([sys.executable, "-c", DIALOGS], cwd=ROOT, capture_output=True, text=True, timeout=180, env=env)
    out = r.stdout + r.stderr
    if r.returncode and "sqlite3" in out and "Symbol not found" in out:
        pytest.skip("this Python's sqlite3 extension cannot load (set DYLD_LIBRARY_PATH to a matching libsqlite)")
    assert "DIALOGS_OK" in r.stdout, out[-3000:]
