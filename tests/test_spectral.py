"""Spectral indices must not depend on the integer type of the raster.

Regression test: rasterio returns uint8/uint16 arrays and the desktop app passes them straight to
SpectralFeatureExtractor, where `nir - red` wrapped around (e.g. 100 - 200 = 65436 for uint16), so NDVI, NDWI,
SAVI and EVI were wrong wherever the first band was smaller than the second (water, bare soil, urban).
"""
import numpy as np
import pytest

from geo_segmenter.models.features.spectral import SpectralFeatureExtractor

BAND_MAP = {"red": 0, "green": 1, "blue": 2, "nir": 3}  # as main_window.train_model sets it for 4 bands


def extractor():
    s = SpectralFeatureExtractor()
    s.set_band_map(dict(BAND_MAP))
    return s


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.float32])
def test_indices_match_float_computation(dtype):
    rng = np.random.default_rng(0)
    img = rng.integers(1, 250, size=(6, 7, 4)).astype(dtype)
    got = extractor().extract_features(img)
    ref = extractor().extract_features(img.astype(np.float64))
    np.testing.assert_allclose(got, ref, rtol=1e-6)


def test_ndvi_negative_for_water_like_pixel_uint16():
    img = np.zeros((1, 1, 4), dtype=np.uint16)
    img[0, 0] = [200, 300, 400, 100]  # red > nir
    f = extractor().extract_features(img)
    names = extractor().get_feature_names()
    ndvi = f[names.index("NDVI"), 0, 0]
    assert ndvi == pytest.approx((100 - 200) / (100 + 200), rel=1e-6)
    assert -1 <= ndvi <= 1


def test_feature_names_match_feature_count():
    img = np.ones((3, 3, 4), dtype=np.uint16)
    s = extractor()
    assert s.extract_features(img).shape[0] == len(s.get_feature_names())
