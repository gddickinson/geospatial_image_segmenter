# geo_segmenter/utils/__init__.py:
"""Utility functions and helpers.

    logger           setup_logger, log_exception
    geo_utils        coordinate transforms, raster info (pyproj, rasterio, shapely)
    image_utils      normalisation, indices, overviews, cloud mask (rasterio, scipy, skimage)
    lidar_utils      LAS reading, DEM, ground classification (laspy, pdal)
    display_stretch  band combination and contrast stretch used to display rasters (numpy only)
    synthetic_scene  seeded synthetic 4-band scene for the web demo (numpy, scipy)

The re-exported helpers below are imported on first use, so importing one light module does not need GDAL/PDAL.
"""
import importlib

from .logger import setup_logger, log_exception  # noqa: F401

_LAZY = {
    "transform_coordinates": "geo_utils", "calculate_tile_bounds": "geo_utils", "get_raster_info": "geo_utils",
    "pixel_to_coords": "geo_utils", "coords_to_pixel": "geo_utils",
    "normalize_image": "image_utils", "calculate_spectral_index": "image_utils",
    "calculate_texture_features": "image_utils", "create_overview_pyramid": "image_utils", "mask_clouds": "image_utils",
    "read_las_file": "lidar_utils", "create_dem": "lidar_utils", "classify_ground_points": "lidar_utils",
    "create_intensity_image": "lidar_utils",
}


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
