"""Texture option local_mean (config.TEXTURE_LOCAL_MEAN, default off) and names following set_parameters."""
import numpy as np
from scipy import ndimage

from geo_segmenter import config
from geo_segmenter.models.features.texture import TextureFeatureExtractor
from geo_segmenter.models.pipeline import add_default_extractors
from geo_segmenter.models.random_forest import RandomForestModel


def _img():
    return np.random.default_rng(4).integers(0, 4000, size=(18, 21, 4)).astype(np.uint16)


def test_default_is_off_and_keeps_nine_maps():
    assert config.TEXTURE_LOCAL_MEAN is False
    t = TextureFeatureExtractor()
    assert t.extract_features(_img()).shape[0] == 9 == len(t.get_feature_names())


def test_local_mean_adds_one_map_per_window_after_the_default_ones():
    img = _img()
    base = TextureFeatureExtractor().extract_features(img)
    t = TextureFeatureExtractor()
    t.set_parameters({'local_mean': True})
    feats, names = t.extract_features(img), t.get_feature_names()
    assert feats.shape[0] == len(names) == 12
    assert names[:4] == ["StdDev_3px", "Range_3px", "GradientMagnitude_3px", "LocalMean_3px"]
    keep = [i for i, n in enumerate(names) if not n.startswith("LocalMean")]
    np.testing.assert_array_equal(feats[keep], base)
    rgb = img[..., :3].astype(float)
    g = rgb[..., 0] * 0.2989 + rgb[..., 1] * 0.5870 + rgb[..., 2] * 0.1140
    g = ((g - g.min()) / (g.max() - g.min()) * 255).astype(np.uint8).astype(float)
    np.testing.assert_allclose(feats[names.index("LocalMean_5px")], ndimage.uniform_filter(g, 5), atol=1e-9)


def test_set_parameters_updates_names():
    t = TextureFeatureExtractor()
    t.set_parameters({'window_sizes': [3]})
    assert t.get_feature_names() == ["StdDev_3px", "Range_3px", "GradientMagnitude_3px"]


def test_pipeline_option():
    m = RandomForestModel(n_estimators=5)
    add_default_extractors(m, 4, texture_local_mean=True)
    assert sum(n.startswith("texture_LocalMean") or "LocalMean" in n for n in m.feature_set.enabled_feature_names()) == 3
