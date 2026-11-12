"""Geospatial segmentation and analysis package.

Subpackages (imported on first use, so the pure parts work without GDAL, PDAL, PyTorch or Qt):
    config   settings (paths, feature and model defaults)
    utils    logger, geo/image/LIDAR helpers, display stretch, synthetic demo scene
    data     layers and data providers (rasterio, geopandas, laspy)
    models   feature extractors, RandomForestModel, CNNModel (PyTorch)
    gui      the PyQt6 desktop application
    web      FastAPI web interface and static site
    main     main(): desktop app entry point (console script geo-segmenter)
"""
import importlib

__version__ = '0.1.0'

from . import config  # noqa: E402,F401

_SUBPACKAGES = {"utils", "data", "models", "gui", "web"}


def __getattr__(name):
    if name in _SUBPACKAGES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
