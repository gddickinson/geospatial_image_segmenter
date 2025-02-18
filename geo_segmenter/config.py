"""Configuration settings for the geospatial segmentation application."""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / "cache"  # For tile caching

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Logging configuration
LOG_LEVEL = "DEBUG"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = PROJECT_ROOT / "geo_segmentation.log"

# Map settings
DEFAULT_ZOOM = 13
DEFAULT_CENTER = (0, 0)  # (lat, lon)
MAP_TILE_SIZE = 256
AVAILABLE_BASE_MAPS = {
    "OpenStreetMap": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    "Satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "Terrain": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}"
}

# Feature extraction parameters
SPECTRAL_INDICES = [
    "NDVI",    # Normalized Difference Vegetation Index
    "NDWI",    # Normalized Difference Water Index
    "SAVI",    # Soil Adjusted Vegetation Index
    "EVI"      # Enhanced Vegetation Index
]

TEXTURE_WINDOWS = [3, 5, 7]  # Window sizes for texture analysis
TERRAIN_ANALYSIS_SCALES = [10, 25, 50]  # Scales in meters

# Model parameters
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10

# CNN parameters
CNN_BATCH_SIZE = 4
CNN_LEARNING_RATE = 0.001
CNN_EPOCHS = 50

# LIDAR parameters
LIDAR_MAX_POINTS = 1000000  # Maximum points to load at once
LIDAR_GROUND_CLASSIFICATION = 2  # Standard LAS classification for ground points

# GUI parameters
DEFAULT_BRUSH_SIZE = 5
OVERLAY_OPACITY = 0.5
MAX_RECENT_FILES = 10
CACHE_SIZE_LIMIT = 500  # MB

# Coordinate Reference Systems
DEFAULT_CRS = "EPSG:4326"  # WGS 84
SUPPORTED_CRS = [
    "EPSG:4326",  # WGS 84
    "EPSG:3857",  # Web Mercator
    "EPSG:32633"  # UTM Zone 33N
]

# Data handling
SUPPORTED_RASTER_FORMATS = [".tif", ".tiff", ".img", ".jp2"]
SUPPORTED_VECTOR_FORMATS = [".shp", ".geojson", ".kml", ".gpkg"]
SUPPORTED_LIDAR_FORMATS = [".las", ".laz", ".xyz"]

MAX_RASTER_DIMENSION = 10000  # Maximum pixel dimension for loading full raster
RASTER_BLOCK_SIZE = 1024  # Size of blocks for processing large rasters