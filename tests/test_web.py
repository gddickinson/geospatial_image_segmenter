"""Web interface tests: every exported endpoint answers, the static build is complete, and the JS engine tests pass.

Run: python -m pytest -q tests/test_web.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from geo_segmenter.web.app import STATIC, STATIC_ENDPOINTS, app  # noqa: E402

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "static/main.js" in r.text


@pytest.mark.parametrize("path", STATIC_ENDPOINTS)
def test_endpoint(path):
    r = client.get("/api" + path)
    assert r.status_code == 200, r.text
    assert r.json() is not None


def test_static_build(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    assert (tmp_path / "site" / "index.html").read_text().count("APP_STATIC") == 1
    for path in STATIC_ENDPOINTS:
        assert (tmp_path / "site" / "api" / (path.strip("/") + ".json")).exists()


def test_static_assets_referenced_exist():
    html = (STATIC / "index.html").read_text()
    for name in ("style.css", "main.js"):
        assert f"static/{name}" in html and (STATIC / name).exists()


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_engine():
    tests = sorted(STATIC.glob("*.test.mjs"))
    for t in tests:
        r = subprocess.run(["node", str(t)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


def test_info_has_config_defaults():
    from geo_segmenter import config
    j = client.get("/api/info").json()
    assert j["defaults"]["n_estimators"] == config.RF_N_ESTIMATORS and j["defaults"]["max_depth"] == config.RF_MAX_DEPTH
    assert j["features"]["spectral_indices"] == config.SPECTRAL_INDICES
    assert j["demo"]["n_features"] == 15 and 0.5 < j["demo"]["accuracy"] <= 1


def test_demo_payload_is_consistent():
    import base64
    j = client.get("/api/demo").json()
    s = j["summary"]
    n = s["width"] * s["height"]
    assert len(j["bands"]) == 4 and all(len(base64.b64decode(b)) == 2 * n for b in j["bands"])
    assert len(base64.b64decode(j["python"]["pred"])) == n
    assert len(j["python"]["forest"]["trees"]) == s["n_estimators"]
    assert list(j["python"]["metrics"]["feature_importances"]) == j["python"]["feature_names"]


def test_classify_runs_python_pipeline_and_maps_class_indices():
    import base64
    import numpy as np
    from geo_segmenter.utils.synthetic_scene import make_scene
    from geo_segmenter.web.classify_service import pack_mask
    sc = make_scene(size=64, seed=2)
    img, truth = sc["image"], sc["truth"]
    names = [c["name"] for c in sc["classes"]]
    present = [k for k in range(5) if (sc["strokes"][names[k]]).sum() > 10]
    classes = [{"name": "unused", "mask": pack_mask(np.zeros(truth.shape, bool))}]  # index 0 not painted
    classes += [{"name": names[k], "mask": pack_mask(sc["strokes"][names[k]])} for k in present]
    body = {"width": 64, "height": 64, "seed": 0, "n_estimators": 20,
            "bands": [base64.b64encode(img[..., b].astype("<f4").tobytes()).decode() for b in range(4)],
            "classes": classes}
    r = client.post("/api/classify", json=body)
    assert r.status_code == 200, r.text
    j = r.json()
    pred = np.frombuffer(base64.b64decode(j["pred"]), np.uint8)
    assert set(np.unique(pred)) <= set(range(1, len(classes))) and j["class_index"] == list(range(1, len(classes)))
    lut = np.array([0] + [1 + present.index(k) if k in present else 255 for k in range(5)])
    assert np.mean(pred.reshape(64, 64) == lut[1:][truth]) > 0.6


def test_classify_rejects_one_class():
    import base64
    import numpy as np
    from geo_segmenter.web.classify_service import pack_mask
    m = np.zeros((4, 4), bool); m[0] = True
    body = {"width": 4, "height": 4, "bands": [base64.b64encode(np.ones(16, "<f4").tobytes()).decode()] * 3,
            "classes": [{"name": "a", "mask": pack_mask(m)}, {"name": "b", "mask": pack_mask(np.zeros((4, 4), bool))}]}
    r = client.post("/api/classify", json=body)
    assert r.status_code == 400 and "two classes" in r.text
