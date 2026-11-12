"""The desktop GUI still builds (offscreen) and its raster display uses utils/display_stretch.

Runs in a fresh subprocess (other Qt bindings in a shared environment, e.g. PySide6, can load a different QtCore
into the pytest process first). Skipped when the desktop dependencies are missing; PDAL (conda only) is replaced by a
stub module, as only lidar_utils uses it.
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
import sys, types
import numpy as np
sys.modules['pdal'] = types.ModuleType('pdal')
from PyQt6.QtWidgets import QApplication
app = QApplication([])
from geo_segmenter.gui.main_window import MainWindow
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
