"""The Python-vs-JS fixtures are current, the JS GeoTIFF writer is readable by tifffile, and the pure helpers behave.

The node tests (geo_segmenter/web/static/*.test.mjs, run by tests/test_web.py) compare the JS ports with these
fixtures; regenerate them with `python scripts/make_web_fixtures.py` after changing the Python side.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
FIX = ROOT / "tests" / "fixtures"
STATIC = ROOT / "geo_segmenter" / "web" / "static"


def _decode(case_old, case_new, key, dtype=np.float64):
    import base64
    a = np.frombuffer(base64.b64decode(case_old[key]), dtype)
    b = np.frombuffer(base64.b64decode(case_new[key]), dtype)
    return a, b


@pytest.fixture(scope="module")
def fresh():
    import make_web_fixtures as m  # not tiff_case: writing LZW with tifffile needs imagecodecs
    m.quiet_logs()
    return {"features_case.json": m.features_case(), "stretch_case.json": m.stretch_case(),
            "forest_case.json": m.forest_case()}


def test_feature_fixture_current(fresh):
    old = json.loads((FIX / "features_case.json").read_text())
    for co, cn in zip(old["cases"], fresh["features_case.json"]["cases"]):
        assert co["names"] == cn["names"]
        a, b = _decode(co, cn, "features")
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)


def test_stretch_fixture_current(fresh):
    old = json.loads((FIX / "stretch_case.json").read_text())
    for co, cn in zip(old["cases"], fresh["stretch_case.json"]["cases"]):
        a, b = _decode(co, cn, "rgb", np.uint8)
        assert np.mean(a != b) < 1e-3  # uint8 truncation may flip a value on another libm


def test_forest_fixture_current(fresh):
    old = json.loads((FIX / "forest_case.json").read_text())
    new = fresh["forest_case.json"]
    a, b = _decode(old, new, "pred", np.uint8)
    assert np.mean(a == b) > 0.99
    assert old["forest"]["n_features"] == new["forest"]["n_features"]


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_geotiff_writer_read_by_tifffile(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    out = tmp_path / "classes.tif"
    r = subprocess.run(["node", str(STATIC / "geotiff.test.mjs")], env=dict(os.environ, GEOTIFF_OUT=str(out)),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    with tifffile.TiffFile(out) as tf:
        page = tf.pages[0]
        assert page.asarray().tolist() == (np.arange(42) % 5).reshape(6, 7).tolist()
        assert page.tags["ModelPixelScaleTag"].value == (10.0, 10.0, 0.0)
        assert tf.geotiff_metadata["ProjectedCSTypeGeoKey"] == 32633
    multi = tifffile.imread(str(out) + ".multi.tif")
    assert multi.shape == (2, 2, 2) and multi[..., 1].ravel().tolist() == [5, 6, 7, 8]


def test_synthetic_scene_is_deterministic():
    from geo_segmenter.utils.synthetic_scene import make_scene
    a, b = make_scene(size=48, seed=3), make_scene(size=48, seed=3)
    assert a["image"].dtype == np.uint16 and a["image"].shape == (48, 48, 4)
    np.testing.assert_array_equal(a["image"], b["image"])
    for name, m in a["strokes"].items():
        k = [c["name"] for c in a["classes"]].index(name)
        assert np.all(a["truth"][m] == k)  # strokes lie inside their class


def test_display_stretch():
    from geo_segmenter.utils.display_stretch import prepare_display
    data = np.stack([np.arange(100, dtype=float).reshape(10, 10)] * 3)
    rgb = prepare_display(data)
    assert rgb.shape == (10, 10, 3) and rgb.dtype == np.uint8 and rgb[0, 0, 0] == 0 and rgb[-1, -1, 0] == 255
    p2 = prepare_display(data, stretch_params={"type": "Linear 2%", "min": 0, "max": 99})
    assert p2[0, 1, 0] == 0 and p2[-1, -2, 0] == 255
    const = prepare_display(np.ones((3, 4, 4)), stretch_params={"type": "Standard Deviation", "min": 0, "max": 1})
    assert np.all(const == 0)
    grey = prepare_display(data[:1])
    assert np.array_equal(grey[..., 0], grey[..., 2])


def test_pipeline_matches_desktop_band_map():
    from geo_segmenter.models.pipeline import default_band_map
    assert default_band_map(3) == {"red": 0, "green": 1, "blue": 2}
    assert default_band_map(4) == {"red": 0, "green": 1, "blue": 2, "nir": 3}
