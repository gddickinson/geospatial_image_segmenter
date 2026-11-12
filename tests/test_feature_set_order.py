"""Feature columns must come in a fixed order that matches the feature names.

Regression test: FeatureSet.extract_all_features iterated `enabled_extractors`, a set of strings, whose order
depends on Python's per-process hash seed, while the names used for the importances came from the extractors
dict (insertion order). In about half of all runs the desktop app's feature importances were attributed to the
wrong extractor (spectral <-> texture).
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CODE = """
import numpy as np
from geo_segmenter.models.features import FeatureSet, SpectralFeatureExtractor, TextureFeatureExtractor
fs = FeatureSet()
s = SpectralFeatureExtractor(); s.set_band_map({'red': 0, 'green': 1, 'blue': 2, 'nir': 3})
fs.add_extractor('spectral', s); fs.add_extractor('texture', TextureFeatureExtractor())
print(','.join(fs.extract_all_features(np.ones((4, 4, 4))).keys()))
"""


def test_extractor_order_independent_of_hash_seed():
    orders = set()
    for seed in range(8):
        env = dict(os.environ, PYTHONHASHSEED=str(seed))
        r = subprocess.run([sys.executable, "-c", CODE], cwd=ROOT, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-1500:]
        orders.add(r.stdout.strip().splitlines()[-1])
    assert orders == {"spectral,texture"}


def test_importance_names_follow_columns():
    """Disabled extractors must not contribute names (they contribute no columns)."""
    from geo_segmenter.models.features import SpectralFeatureExtractor, TextureFeatureExtractor
    from geo_segmenter.models.random_forest import RandomForestModel
    m = RandomForestModel(n_estimators=5, max_depth=3)
    s = SpectralFeatureExtractor(); s.set_band_map({"red": 0, "green": 1, "blue": 2, "nir": 3})
    m.add_feature_extractor("texture", TextureFeatureExtractor())
    m.add_feature_extractor("spectral", s)
    m.disable_feature_extractor("texture")
    rng = np.random.default_rng(0)
    img = rng.random((10, 10, 4))
    a = np.zeros((10, 10), bool); a[:5] = True
    metrics = m.train(img, {"a": a, "b": ~a}, validation_split=0, random_state=0)
    assert list(metrics["feature_importances"]) == [f"spectral_{n}" for n in s.get_feature_names()]
