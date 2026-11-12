"""The feature extractors and RandomForestModel must import without the heavy optional stack.

Regression test: geo_segmenter/__init__.py imported utils (rasterio, pyproj, shapely, laspy, pdal), models (PyTorch
via cnn.py) and gui eagerly, so `from geo_segmenter.models.features import SpectralFeatureExtractor` failed unless
GDAL, PDAL and PyTorch were all installed (PDAL cannot be installed with pip).
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BLOCKED = ["torch", "pdal", "laspy", "rasterio", "pyproj", "shapely", "geopandas", "PyQt6"]


def run(code: str) -> subprocess.CompletedProcess:
    # a meta-path finder that refuses the blocked packages (sys.modules[name] = None confuses scipy's torch check)
    block = ("import sys\nclass Block:\n    def find_spec(self, name, path=None, target=None):\n"
             f"        if name.split('.')[0] in {BLOCKED!r}: raise ImportError('blocked: ' + name)\n"
             "sys.meta_path.insert(0, Block())\n")
    return subprocess.run([sys.executable, "-c", block + code], cwd=ROOT, capture_output=True, text=True)


def test_features_and_random_forest_import_without_heavy_deps():
    r = run("from geo_segmenter.models.features import SpectralFeatureExtractor, TextureFeatureExtractor\n"
            "from geo_segmenter.models.random_forest import RandomForestModel\n"
            "from geo_segmenter.models import RandomForestModel as RF2\n"
            "assert RandomForestModel is RF2\nprint('ok')")
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]


def test_cnn_still_reachable_from_models_package():
    pytest.importorskip("torch")
    r = subprocess.run([sys.executable, "-c", "from geo_segmenter.models import CNNModel; print(CNNModel.__name__)"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and "CNNModel" in r.stdout, r.stderr[-2000:]


def test_utils_reexports_are_lazy():
    r = run("import geo_segmenter.utils as u\nfrom geo_segmenter.utils import setup_logger\nprint('ok')")
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]
