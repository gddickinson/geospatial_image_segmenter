"""The web demo: the seeded synthetic scene (utils/synthetic_scene.py) classified by the repo's own pipeline.

demo_payload() is what /api/demo returns (and the static site's api/demo.json): the scene as uint16 bands, the known
class map, example strokes, and a RandomForestModel (config defaults, random_state=0) trained on those strokes, with
its metrics, prediction and exported trees so the browser can re-run it.
"""
from functools import lru_cache

import numpy as np

from .. import config
from ..utils.synthetic_scene import make_scene
from .classify_service import b64, export_forest, metrics_json, pack_mask, run_pipeline

DEMO_SEED = 0  # random_state of the demo forest (the desktop app sets none)


@lru_cache(maxsize=1)
def demo_data() -> dict:
    scene = make_scene()
    names = [c["name"] for c in scene["classes"]]
    r = run_pipeline(scene["image"], {n: scene["strokes"][n] for n in names}, seed=DEMO_SEED)
    return dict(scene=scene, result=r)


def demo_summary() -> dict:
    d = demo_data()
    s, r = d["scene"], d["result"]
    h, w, b = s["image"].shape
    acc = float(np.mean(r["pred"] == s["truth"]))
    return dict(width=w, height=h, bands=b, dtype="uint16", pixel_size_m=s["pixel_size_m"],
                classes=len(s["classes"]), stroke_pixels=r["n_train"], accuracy=acc,
                n_estimators=config.RF_N_ESTIMATORS, max_depth=config.RF_MAX_DEPTH, seed=DEMO_SEED,
                n_features=int(r["model"].classifier.n_features_in_))


def demo_payload() -> dict:
    d = demo_data()
    s, r = d["scene"], d["result"]
    img = s["image"]
    return dict(
        summary=demo_summary(),
        band_names=["Red", "Green", "Blue", "NIR"],
        bands=[b64(img[..., k], np.uint16) for k in range(img.shape[-1])],
        truth=b64(s["truth"], np.uint8),
        classes=s["classes"],
        strokes=[pack_mask(s["strokes"][c["name"]]) for c in s["classes"]],
        python=dict(pred=b64(r["pred"], np.uint8), metrics=metrics_json(r["metrics"]),
                    feature_names=r["model"].feature_set.enabled_feature_names(),
                    train_s=r["train_s"], predict_s=r["predict_s"], forest=export_forest(r["model"])),
    )
