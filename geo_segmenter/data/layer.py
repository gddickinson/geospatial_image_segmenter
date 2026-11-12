"""Map layer classes (re-exported here so `from geo_segmenter.data.layer import ...` keeps working).

    layer_base.py      Layer: base class (name, visibility, opacity, CRS, render cache, signals)
    layer_raster.py    RasterLayer: GeoTIFF rasters, band combination + stretch (utils/display_stretch)
    layer_vector.py    VectorLayer (geopandas), LidarLayer (point clouds)
    layer_overlays.py  SegmentationLayer (results), TrainingLabelsLayer (painted labels)
"""
from .layer_base import Layer
from .layer_raster import RasterLayer
from .layer_vector import VectorLayer, LidarLayer
from .layer_overlays import SegmentationLayer, TrainingLabelsLayer

__all__ = ["Layer", "RasterLayer", "VectorLayer", "LidarLayer", "SegmentationLayer", "TrainingLabelsLayer"]
