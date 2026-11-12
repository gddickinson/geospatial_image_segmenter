"""FastAPI web app for the Geospatial Image Segmenter: serves the front end in static/ and the JSON API under /api.

Run with `python run_web.py` (http://127.0.0.1:8235, local only). `scripts/build_static.py` exports every path in
STATIC_ENDPOINTS to JSON for the GitHub Pages site, so keep that list in step with the GET routes below. The POST route
(train and classify with the repo's RandomForestModel) exists only in the local app; on the static site the browser
runs the JS port instead (static_api.js).
"""
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from geo_segmenter import __version__, config
from geo_segmenter.web.classify_service import b64, from_b64, metrics_json, quiet_logs, run_pipeline, unpack_mask
from geo_segmenter.web.demo import demo_payload, demo_summary

STATIC = Path(__file__).resolve().parent / "static"
MAX_PIXELS = 4_000_000

quiet_logs()
app = FastAPI(title="Geospatial Image Segmenter")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    """Project summary, the desktop defaults from config.py and the demo summary."""
    return {
        "title": "Geospatial Image Segmenter", "version": __version__,
        "defaults": {"n_estimators": config.RF_N_ESTIMATORS, "max_depth": config.RF_MAX_DEPTH,
                     "brush_size": config.DEFAULT_BRUSH_SIZE, "overlay_opacity": config.OVERLAY_OPACITY,
                     "validation_split": 0.2, "max_features": "sqrt", "class_weight": "balanced"},
        "features": {"spectral_indices": config.SPECTRAL_INDICES, "texture_windows": config.TEXTURE_WINDOWS},
        "stretches": ["None", "Linear", "Linear 2%", "Linear 5%", "Standard Deviation"],
        "demo": demo_summary(),
    }


@app.get("/api/demo")
def demo():
    """Synthetic demo scene, known classes, example strokes and the Python forest trained on them."""
    return demo_payload()


class ClassIn(BaseModel):
    name: str
    mask: str  # base64 np.packbits of the painted pixels, row-major


class ClassifyIn(BaseModel):
    width: int
    height: int
    bands: List[str]  # base64 float32 per band, row-major
    band_map: Optional[Dict[str, int]] = None  # roles red/green/blue/nir -> 0-based band; None = desktop default
    classes: List[ClassIn]
    n_estimators: int = config.RF_N_ESTIMATORS
    max_depth: int = config.RF_MAX_DEPTH
    seed: Optional[int] = None


@app.post("/api/classify")
def classify(body: ClassifyIn):
    """Train RandomForestModel on the painted classes (main_window.train_model) and classify the whole image."""
    h, w = body.height, body.width
    if h * w > MAX_PIXELS or h < 1 or w < 1:
        raise HTTPException(400, f"Image must have between 1 and {MAX_PIXELS:,} pixels")
    if not 1 <= body.n_estimators <= 500 or not 1 <= body.max_depth <= 40:
        raise HTTPException(400, "Trees must be 1-500 and depth 1-40")
    try:
        image = np.stack([from_b64(b, np.float32, (h, w)) for b in body.bands], axis=-1).astype(np.float64)
        masks = [unpack_mask(c.mask, (h, w)) for c in body.classes]
    except Exception as e:
        raise HTTPException(400, f"Bad image or mask data: {e}")
    if body.band_map and any(not 0 <= v < image.shape[-1] for v in body.band_map.values()):
        raise HTTPException(400, "Band map refers to a band the image does not have")
    names = [c.name for c in body.classes]
    if len(set(names)) != len(names):
        raise HTTPException(400, "Class names must be unique")
    painted = [i for i, m in enumerate(masks) if m.any()]
    try:
        r = run_pipeline(image, {names[i]: masks[i] for i in painted}, body.band_map, body.n_estimators,
                         body.max_depth, body.seed)
    except ValueError as e:
        raise HTTPException(400, str(e))
    # the model numbers the painted classes 0..k-1; map back to the page's class indices
    lut = np.array(painted, dtype=np.uint8)
    return {
        "engine": "python", "pred": b64(lut[r["pred"]], np.uint8), "metrics": metrics_json(r["metrics"]),
        "classes": [names[i] for i in painted], "class_index": painted,
        "feature_names": r["model"].feature_set.enabled_feature_names(),
        "train_s": r["train_s"], "predict_s": r["predict_s"], "n_train": r["n_train"],
    }


# GET paths (without the /api prefix) written to the static site as api/<path>.json.
STATIC_ENDPOINTS = ["/info", "/demo"]
