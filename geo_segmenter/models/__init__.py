# geo_segmenter/models/__init__.py:
"""Model implementations.

    base.GeospatialModel          abstract model (feature set, training-data preparation)
    random_forest.RandomForestModel  scikit-learn random forest
    cnn.CNNModel                  PyTorch CNN (imported on first use: needs torch)
    pipeline                      band map and feature extractors as the desktop app sets them up
    features/                     spectral, texture and terrain feature extractors
"""
from .base import GeospatialModel
from .random_forest import RandomForestModel


def __getattr__(name):
    if name == "CNNModel":  # lazy: PyTorch is only needed for the CNN
        from .cnn import CNNModel
        return CNNModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
