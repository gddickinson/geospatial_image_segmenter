"""Geospatial utility functions."""
import numpy as np
from typing import Tuple, List, Dict, Union
import pyproj
from pyproj import CRS, Transformer
from shapely.geometry import Point, Polygon, box
import rasterio
from rasterio.warp import transform_bounds
from .. import config
from .logger import setup_logger

logger = setup_logger(__name__)

def create_transformer(src_crs: Union[str, CRS], dst_crs: Union[str, CRS]) -> Transformer:
    """Create a coordinate transformer.
    
    Args:
        src_crs: Source coordinate reference system
        dst_crs: Destination coordinate reference system
        
    Returns:
        Transformer: PyProj transformer object
    """
    try:
        return Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    except Exception as e:
        logger.error(f"Error creating transformer: {str(e)}")
        raise

def transform_coordinates(
    coords: Union[Tuple[float, float], List[Tuple[float, float]]],
    src_crs: Union[str, CRS],
    dst_crs: Union[str, CRS]
) -> Union[Tuple[float, float], List[Tuple[float, float]]]:
    """Transform coordinates between coordinate reference systems.
    
    Args:
        coords: Single coordinate pair or list of coordinate pairs
        src_crs: Source coordinate reference system
        dst_crs: Destination coordinate reference system
        
    Returns:
        Transformed coordinates in the same format as input
    """
    try:
        transformer = create_transformer(src_crs, dst_crs)
        
        if isinstance(coords, tuple):
            x, y = transformer.transform(coords[0], coords[1])
            return (x, y)
        else:
            return [transformer.transform(x, y) for x, y in coords]
            
    except Exception as e:
        logger.error(f"Error transforming coordinates: {str(e)}")
        raise

def get_raster_info(raster_path: str) -> Dict:
    """Get basic information about a raster dataset.
    
    Args:
        raster_path: Path to raster file
        
    Returns:
        dict: Dictionary containing raster information
    """
    try:
        with rasterio.open(raster_path) as src:
            info = {
                'crs': src.crs.to_string(),
                'bounds': src.bounds,
                'shape': src.shape,
                'resolution': src.res,
                'count': src.count,
                'dtypes': src.dtypes,
                'nodata': src.nodata,
                'transform': src.transform
            }
        return info
        
    except Exception as e:
        logger.error(f"Error reading raster info: {str(e)}")
        raise

def pixel_to_coords(
    row: int,
    col: int,
    transform: rasterio.transform.Affine
) -> Tuple[float, float]:
    """Convert pixel indices to coordinates.
    
    Args:
        row: Pixel row index
        col: Pixel column index
        transform: Raster transform matrix
        
    Returns:
        tuple: (x, y) coordinates
    """
    try:
        x, y = transform * (col + 0.5, row + 0.5)
        return (x, y)
        
    except Exception as e:
        logger.error(f"Error converting pixel to coordinates: {str(e)}")
        raise

def coords_to_pixel(
    x: float,
    y: float,
    transform: rasterio.transform.Affine
) -> Tuple[int, int]:
    """Convert coordinates to pixel indices.
    
    Args:
        x: X coordinate
        y: Y coordinate
        transform: Raster transform matrix
        
    Returns:
        tuple: (row, col) pixel indices
    """
    try:
        rev_transform = ~transform
        col, row = rev_transform * (x, y)
        return (int(row), int(col))
        
    except Exception as e:
        logger.error(f"Error converting coordinates to pixel: {str(e)}")
        raise

def calculate_tile_bounds(
    z: int,
    x: int,
    y: int,
    tile_size: int = config.MAP_TILE_SIZE
) -> Tuple[float, float, float, float]:
    """Calculate geographic bounds for a map tile.
    
    Args:
        z: Zoom level
        x: Tile X coordinate
        y: Tile Y coordinate
        tile_size: Tile size in pixels
        
    Returns:
        tuple: (min_lon, min_lat, max_lon, max_lat)
    """
    try:
        n = 2.0 ** z
        lon_deg = x / n * 360.0 - 180.0
        lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * y / n)))
        lat_deg = np.degrees(lat_rad)
        
        # Calculate bounds of next tile
        next_lon_deg = (x + 1) / n * 360.0 - 180.0
        next_lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * (y + 1) / n)))
        next_lat_deg = np.degrees(next_lat_rad)
        
        return (lon_deg, next_lat_deg, next_lon_deg, lat_deg)
        
    except Exception as e:
        logger.error(f"Error calculating tile bounds: {str(e)}")
        raise

def get_tile_indices(
    lat: float,
    lon: float,
    zoom: int
) -> Tuple[int, int]:
    """Get tile indices for a given coordinate and zoom level.
    
    Args:
        lat: Latitude
        lon: Longitude
        zoom: Zoom level
        
    Returns:
        tuple: (tile_x, tile_y)
    """
    try:
        lat_rad = np.radians(lat)
        n = 2.0 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - np.log(np.tan(lat_rad) + (1 / np.cos(lat_rad))) / np.pi) / 2.0 * n)
        return (x, y)
        
    except Exception as e:
        logger.error(f"Error calculating tile indices: {str(e)}")
        raise

def calculate_scale(
    center_lat: float,
    zoom: int,
    tile_size: int = config.MAP_TILE_SIZE
) -> float:
    """Calculate the scale (meters per pixel) at a given latitude and zoom level.
    
    Args:
        center_lat: Latitude at center of view
        zoom: Zoom level
        tile_size: Tile size in pixels
        
    Returns:
        float: Scale in meters per pixel
    """
    try:
        # Length of a degree of latitude in meters (approximately)
        lat_degree_meters = 111320.0
        
        # Length of a degree of longitude in meters at this latitude
        lon_degree_meters = lat_degree_meters * np.cos(np.radians(center_lat))
        
        # Number of tiles at this zoom level
        n = 2.0 ** zoom
        
        # Degrees per tile
        degrees_per_tile = 360.0 / n
        
        # Meters per tile
        meters_per_tile = degrees_per_tile * lon_degree_meters
        
        # Meters per pixel
        return meters_per_tile / tile_size
        
    except Exception as e:
        logger.error(f"Error calculating scale: {str(e)}")
        raise

def calculate_viewport_bounds(
    center: Tuple[float, float],
    width: int,
    height: int,
    scale: float
) -> Tuple[float, float, float, float]:
    """Calculate the geographic bounds of the viewport.
    
    Args:
        center: (latitude, longitude) of viewport center
        width: Viewport width in pixels
        height: Viewport height in pixels
        scale: Scale in meters per pixel
        
    Returns:
        tuple: (min_lon, min_lat, max_lon, max_lat)
    """
    try:
        center_lat, center_lon = center
        
        # Convert pixel dimensions to meters
        width_meters = width * scale
        height_meters = height * scale
        
        # Convert to degrees (approximate)
        lat_degree_meters = 111320.0
        lon_degree_meters = lat_degree_meters * np.cos(np.radians(center_lat))
        
        width_degrees = width_meters / lon_degree_meters
        height_degrees = height_meters / lat_degree_meters
        
        return (
            center_lon - width_degrees/2,
            center_lat - height_degrees/2,
            center_lon + width_degrees/2,
            center_lat + height_degrees/2
        )
        
    except Exception as e:
        logger.error(f"Error calculating viewport bounds: {str(e)}")
        raise

def create_bbox_polygon(bounds: Tuple[float, float, float, float]) -> Polygon:
    """Create a Shapely polygon from bounds.
    
    Args:
        bounds: (min_lon, min_lat, max_lon, max_lat)
        
    Returns:
        Polygon: Shapely polygon representing the bounding box
    """
    try:
        return box(bounds[0], bounds[1], bounds[2], bounds[3])
        
    except Exception as e:
        logger.error(f"Error creating bbox polygon: {str(e)}")
        raise