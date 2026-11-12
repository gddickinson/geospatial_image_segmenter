"""RandomForestModel: predict_proba returns probabilities; train/predict round trip on a separable image.

Regression test: predict_proba was a copy of predict and returned the label map instead of class probabilities.
"""
import numpy as np

from geo_segmenter.models.features import SpectralFeatureExtractor, TextureFeatureExtractor
from geo_segmenter.models.random_forest import RandomForestModel


def make():
    m = RandomForestModel(n_estimators=10, max_depth=5)
    s = SpectralFeatureExtractor(); s.set_band_map({"red": 0, "green": 1, "blue": 2, "nir": 3})
    m.add_feature_extractor("spectral", s)
    m.add_feature_extractor("texture", TextureFeatureExtractor())
    rng = np.random.default_rng(0)
    img = rng.integers(500, 700, size=(16, 16, 4)).astype(np.uint16)
    img[:, 8:, 3] += 2000  # right half: high NIR (vegetation-like)
    left = np.zeros((16, 16), bool); left[:, :6] = True
    right = np.zeros((16, 16), bool); right[:, 10:] = True
    m.train(img, {"bare": left, "veg": right}, validation_split=0, random_state=0)
    return m, img


def test_predict_proba_returns_probability_maps():
    m, img = make()
    p = m.predict_proba(img)
    assert p.shape == (2, 16, 16) and p.dtype.kind == "f"
    np.testing.assert_allclose(p.sum(axis=0), 1.0)
    np.testing.assert_array_equal(p.argmax(axis=0), m.predict(img))


def test_predict_separates_halves():
    m, img = make()
    pred = m.predict(img)
    assert (pred[:, :6] == 0).all() and (pred[:, 10:] == 1).all()
