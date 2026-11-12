"""Write the Python-vs-JS fixtures in tests/fixtures/ (read by geo_segmenter/web/static/*.test.mjs).

  features_case.json  spectral + texture features (models/pipeline.py) on a crop of the demo scene, a 3-band random
                      uint8 image and a 1-band float image
  stretch_case.json   utils/display_stretch.prepare_display for every stretch type on a demo crop
  forest_case.json    a seeded 10-tree RandomForestModel trained on the demo strokes: exported trees, the pipeline's
                      features on a crop and RandomForestModel.predict on it
  tiff_case.json      GeoTIFFs written by tifffile (multi-band chunky/planar, uint8/uint16/int16/float32,
                      none/zlib/LZW+predictor/PackBits, tiled, big-endian) with their geokeys

Run: python scripts/make_web_fixtures.py   (tests/test_web_engine.py checks that the fixtures are current)
"""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "tests" / "fixtures"

from geo_segmenter.utils.display_stretch import prepare_display  # noqa: E402
from geo_segmenter.utils.synthetic_scene import make_scene  # noqa: E402
from geo_segmenter.web.classify_service import b64, build_model, export_forest, quiet_logs, run_pipeline  # noqa: E402

CROP = (slice(120, 150), slice(140, 176))  # 30 x 36 px of the demo: town, river, fields


def pipeline_features(img, band_map=None):
    model = build_model(img.shape[-1], band_map)
    feats = model.feature_set.extract_all_features(img)
    return np.vstack([f for f in feats.values()]), model.feature_set.enabled_feature_names()


def features_case():
    scene = make_scene()
    rng = np.random.default_rng(3)
    cases = []
    for name, img, bm in (("demo_crop", scene["image"][CROP], None),
                          ("demo_crop_no_nir", scene["image"][CROP], {"red": 0, "green": 1, "blue": 2}),
                          ("rgb_uint8", rng.integers(0, 256, size=(9, 11, 3)).astype(np.uint8), None),
                          ("one_band_float", rng.random((8, 10, 1)), None)):
        f, names = pipeline_features(img, bm)
        cases.append(dict(name=name, shape=list(img.shape), dtype=str(img.dtype), band_map=bm,
                          bands=[b64(img[..., k], np.float64) for k in range(img.shape[-1])],
                          names=names, features=b64(f, np.float64)))
    return dict(cases=cases)


def stretch_case():
    img = make_scene()["image"][CROP].transpose(2, 0, 1)  # (bands, H, W) as RasterLayer gets it
    lo, hi = float(img.min()), float(img.max())
    out = []
    for params, bands in ((None, [1, 2, 3]), ({"type": "Linear 2%", "min": lo, "max": hi}, [4, 1, 2]),
                          ({"type": "Linear 5%", "min": lo, "max": hi}, [1, 2, 3]),
                          ({"type": "Linear", "min": 200.0, "max": 3000.0}, [1, 2, 3]),
                          ({"type": "Standard Deviation", "min": lo, "max": hi}, [4, 3, 2])):
        rgb = prepare_display(img, bands, params)
        out.append(dict(params=params, band_indices=bands, rgb=b64(rgb, np.uint8)))
    return dict(shape=list(img.shape), bands=[b64(b, np.float64) for b in img], cases=out)


def forest_case():
    scene = make_scene()
    names = [c["name"] for c in scene["classes"]]
    r = run_pipeline(scene["image"], {n: scene["strokes"][n] for n in names}, n_estimators=10, seed=0)
    model = r["model"]
    crop = scene["image"][CROP]
    feats, _ = pipeline_features(crop)
    return dict(shape=list(crop.shape), forest=export_forest(model), features=b64(feats, np.float64),
                pred=b64(model.predict(crop), np.uint8), full_pred=b64(r["pred"], np.uint8),
                proba=b64(model.predict_proba(crop), np.float64))


def tiff_case():
    import tifffile
    rng = np.random.default_rng(5)
    geo = {33550: (12, 3, (10.0, 10.0, 0.0)), 33922: (12, 6, (0.0, 0.0, 0.0, 500000.0, 4100000.0, 0.0)),
           34735: (3, 16, (1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, 32633)),
           42113: (2, None, "0")}
    tags = [(k, t, n, v, True) for k, (t, n, v) in geo.items()]
    files = []
    variants = [
        ("rgbn_uint16_chunky", (4, 6, 7), np.uint16, dict(photometric="minisblack", planarconfig="contig")),
        ("rgbn_uint16_planar_lzw_pred", (4, 6, 7), np.uint16,
         dict(photometric="minisblack", planarconfig="separate", compression="lzw", predictor=True)),
        ("rgb_uint8_deflate", (3, 5, 9), np.uint8, dict(photometric="rgb", compression="zlib")),
        ("gray_int16_packbits_be", (1, 6, 5), np.int16, dict(photometric="minisblack", compression="packbits",
                                                             byteorder=">")),
        ("float32_tiled", (2, 40, 37), np.float32, dict(photometric="minisblack", planarconfig="contig",
                                                        tile=(16, 16))),
    ]
    for name, (b, h, w), dt, kw in variants:
        if np.dtype(dt).kind == "f":
            a = rng.random((b, h, w)).astype(dt)
        else:
            info = np.iinfo(dt)
            a = rng.integers(max(info.min, -3000), min(info.max, 60000), size=(b, h, w)).astype(dt)
        data = a if kw.get("planarconfig") == "separate" or b == 1 else np.moveaxis(a, 0, -1)
        if b == 1:
            data = a[0]
        buf = io.BytesIO()
        tifffile.imwrite(buf, data, extratags=tags, **kw)
        files.append(dict(name=name, tiff=base64.b64encode(buf.getvalue()).decode(), dtype=str(np.dtype(dt)),
                          shape=[b, h, w], bands=b64(a.astype(np.float64), np.float64)))
    return dict(files=files, pixel_scale=[10.0, 10.0, 0.0], tiepoint=[0, 0, 0, 500000.0, 4100000.0, 0], epsg=32633)


def all_cases() -> dict:
    quiet_logs()
    return {"features_case.json": features_case(), "stretch_case.json": stretch_case(),
            "forest_case.json": forest_case(), "tiff_case.json": tiff_case()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in all_cases().items():
        (OUT / name).write_text(json.dumps(data, separators=(",", ":")))
        print(f"wrote tests/fixtures/{name} ({(OUT / name).stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
