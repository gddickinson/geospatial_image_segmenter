"""Python side of the web interface: run the repo's real pipeline (models/pipeline.py + RandomForestModel) on an image
and painted label masks, and export a trained forest for the JS port (forest.js).

All arrays cross the wire as base64 of raw little-endian bytes; masks as base64 of np.packbits (row-major).
"""
import base64
import logging
import time
from typing import Dict, List, Optional

import numpy as np

from .. import config
from ..models.pipeline import add_default_extractors
from ..models.random_forest import RandomForestModel


def quiet_logs():
    """The package logs every step at DEBUG to stdout and a file; keep the web server's output readable."""
    logging.getLogger("geo_segmenter").setLevel(logging.WARNING)
    for name in list(logging.root.manager.loggerDict):
        if name.startswith("geo_segmenter"):
            logging.getLogger(name).setLevel(logging.WARNING)


def b64(a: np.ndarray, dtype=None) -> str:
    a = np.ascontiguousarray(a, dtype=dtype or a.dtype)
    return base64.b64encode(a.astype(a.dtype.newbyteorder("<"), copy=False).tobytes()).decode()


def from_b64(s: str, dtype, shape) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.dtype(dtype).newbyteorder("<")).reshape(shape).copy()


def pack_mask(m: np.ndarray) -> str:
    return base64.b64encode(np.packbits(m.astype(bool).ravel()).tobytes()).decode()


def unpack_mask(s: str, shape) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(base64.b64decode(s), dtype=np.uint8))
    return bits[: shape[0] * shape[1]].reshape(shape).astype(bool)


def build_model(n_bands: int, band_map: Optional[Dict[str, int]] = None, n_estimators: int = config.RF_N_ESTIMATORS,
                max_depth: int = config.RF_MAX_DEPTH) -> RandomForestModel:
    model = RandomForestModel(n_estimators=n_estimators, max_depth=max_depth)
    add_default_extractors(model, n_bands, band_map)
    return model


def run_pipeline(image: np.ndarray, labels: Dict[str, np.ndarray], band_map: Optional[Dict[str, int]] = None,
                 n_estimators: int = config.RF_N_ESTIMATORS, max_depth: int = config.RF_MAX_DEPTH,
                 seed: Optional[int] = None, validation_split: float = 0.2) -> dict:
    """Train on the painted masks (as main_window.train_model) and classify the whole image (run_segmentation).

    image: (height, width, bands); labels: {class name: bool mask}, empty masks are dropped as the GUI does.
    """
    labels = {k: v for k, v in labels.items() if v is not None and np.any(v)}
    if len(labels) < 2:
        raise ValueError("Paint training pixels for at least two classes")
    model = build_model(image.shape[-1], band_map, n_estimators, max_depth)
    kw = {} if seed is None else {"random_state": int(seed)}
    t0 = time.perf_counter()
    metrics = model.train(image, labels, validation_split=validation_split, **kw)
    t1 = time.perf_counter()
    pred = model.predict(image)
    t2 = time.perf_counter()
    return dict(model=model, metrics=metrics, pred=pred.astype(np.uint8), train_s=t1 - t0, predict_s=t2 - t1,
                class_names=list(labels), n_train=int(sum(int(np.sum(v)) for v in labels.values())))


def metrics_json(metrics: dict) -> dict:
    """JSON-safe copy of RandomForestModel.train's metrics."""
    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v
    return clean(metrics)


def export_forest(model: RandomForestModel) -> dict:
    """Trees of the fitted sklearn forest as plain arrays (value = class fractions per node, left = -1 at leaves)."""
    clf = model.classifier
    trees: List[dict] = []
    for est in clf.estimators_:
        t = est.tree_
        v = t.value[:, 0, :]
        v = v / v.sum(axis=1, keepdims=True)
        trees.append(dict(feature=t.feature.tolist(), threshold=t.threshold.tolist(),
                          left=t.children_left.tolist(), right=t.children_right.tolist(),
                          value=v.ravel().tolist()))
    return dict(classes=[int(c) for c in clf.classes_], n_features=int(clf.n_features_in_), trees=trees)
