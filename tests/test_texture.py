"""Texture features: the 'standard deviation' feature is a local standard deviation and the names match the stack.

Regression tests:
- extract_features labelled the first feature of each window "Standard deviation" but computed
  ndimage.uniform_filter, i.e. the local *mean* of the grey image (a brightness, not a texture measure).
- get_feature_names listed 120+ GLCM/entropy names from an older version while extract_features returns 9 maps, so
  RandomForestModel.train paired the importances with the wrong names (the Training dialog showed e.g.
  "texture_GLCM_contrast_3px_1d_0deg" for what was the local mean).
- A constant image divided by zero when normalising to 0-255 (NaN cast to uint8, RuntimeWarning).
"""
import warnings

import numpy as np
from scipy import ndimage

from geo_segmenter.models.features.texture import TextureFeatureExtractor


def gray_u8(img):
    g = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]) if img.ndim == 3 else img
    return ((g - g.min()) / (g.max() - g.min()) * 255).astype(np.uint8)


def test_std_feature_is_local_standard_deviation():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 4000, size=(20, 23, 4)).astype(np.uint16)
    t = TextureFeatureExtractor()
    feats = t.extract_features(img)
    g = gray_u8(img.astype(float)).astype(float)
    names = t.get_feature_names()
    for w in t.parameters["window_sizes"]:
        ref = ndimage.generic_filter(g, np.std, size=w, mode="reflect")
        np.testing.assert_allclose(feats[names.index(f"StdDev_{w}px")], ref, rtol=1e-9, atol=1e-9)


def test_names_match_features():
    t = TextureFeatureExtractor()
    feats = t.extract_features(np.random.default_rng(2).random((12, 13)))
    names = t.get_feature_names()
    assert feats.shape[0] == len(names) == 3 * len(t.parameters["window_sizes"])
    assert names[:3] == ["StdDev_3px", "Range_3px", "GradientMagnitude_3px"]


def test_constant_image_gives_zeros_without_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        feats = TextureFeatureExtractor().extract_features(np.full((8, 9, 3), 7, dtype=np.uint16))
    assert np.all(feats == 0)
