"""Utility functions for processing LIDAR point cloud data."""
import numpy as np
from typing import Tuple, List, Dict, Optional
import laspy
import pdal
import json
from pathlib import Path
from .. import config
from .logger import setup_logger

logger = setup_logger(__name__)

def read_las_file(
    path: str,
    max_points: Optional[int] = None
) -> Tuple[np.ndarray, Dict]:
    """Read LAS/LAZ file and return point cloud data.
    
    Args:
        path: Path to LAS/LAZ file
        max_points: Maximum number of points to read
        
    Returns:
        tuple: (point_cloud_array, metadata)
    """
    try:
        las = laspy.read(path)
        
        # Get point records as numpy array
        points = np.vstack((las.x, las.y, las.z)).transpose()
        
        # Get classification if available
        if hasattr(las, 'classification'):
            classification = las.classification
        else:
            classification = np.zeros(len(points))
            
        # Get intensity if available
        if hasattr(las, 'intensity'):
            intensity = las.intensity
        else:
            intensity = np.zeros(len(points))
            
        # Combine all attributes
        point_cloud = np.column_stack((points, classification, intensity))
        
        # Subsample if needed
        if max_points and len(point_cloud) > max_points:
            indices = np.random.choice(len(point_cloud), max_points, replace=False)
            point_cloud = point_cloud[indices]
        
        # Collect metadata
        metadata = {
            'point_count': len(point_cloud),
            'point_format': las.point_format.id,
            'crs': las.header.parse_crs(),
            'min_bounds': las.header.mins,
            'max_bounds': las.header.maxs,
            'version': las.header.version
        }
        
        logger.debug(f"Loaded {len(point_cloud)} points from {path}")
        return point_cloud, metadata
        
    except Exception as e:
        logger.error(f"Error reading LAS file: {str(e)}")
        raise

def create_dem(
    point_cloud: np.ndarray,
    resolution: float,
    bounds: Optional[Tuple[float, float, float, float]] = None
) -> Tuple[np.ndarray, Dict]:
    """Create Digital Elevation Model from point cloud.
    
    Args:
        point_cloud: Point cloud array (x, y, z, classification, intensity)
        resolution: Grid resolution in same units as point cloud
        bounds: Optional (xmin, ymin, xmax, ymax) to limit extent
        
    Returns:
        tuple: (dem_array, metadata)
    """
    try:
        # Extract ground points (typically class 2 in LAS classification)
        ground_points = point_cloud[point_cloud[:, 3] == config.LIDAR_GROUND_CLASSIFICATION]
        
        if len(ground_points) == 0:
            raise ValueError("No ground points found in point cloud")
        
        # Determine grid extent
        if bounds is None:
            xmin, ymin = ground_points[:, 0].min(), ground_points[:, 1].min()
            xmax, ymax = ground_points[:, 0].max(), ground_points[:, 1].max()
        else:
            xmin, ymin, xmax, ymax = bounds
        
        # Calculate grid dimensions
        nx = int((xmax - xmin) / resolution) + 1
        ny = int((ymax - ymin) / resolution) + 1
        
        # Create empty grid
        grid = np.full((ny, nx), np.nan)
        
        # Convert points to grid indices
        xi = ((ground_points[:, 0] - xmin) / resolution).astype(int)
        yi = ((ground_points[:, 1] - ymin) / resolution).astype(int)
        
        # Mask out points outside the grid
        mask = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
        xi, yi = xi[mask], yi[mask]
        z = ground_points[mask, 2]
        
        # Calculate mean elevation for each cell
        for i in range(len(xi)):
            if np.isnan(grid[yi[i], xi[i]]):
                grid[yi[i], xi[i]] = z[i]
            else:
                grid[yi[i], xi[i]] = (grid[yi[i], xi[i]] + z[i]) / 2
        
        # Fill gaps using nearest neighbor interpolation
        mask = np.isnan(grid)
        if np.any(mask):
            from scipy.interpolate import griddata
            y, x = np.mgrid[0:ny, 0:nx]
            grid[mask] = griddata(
                (y[~mask], x[~mask]),
                grid[~mask],
                (y[mask], x[mask]),
                method='nearest'
            )
        
        metadata = {
            'resolution': resolution,
            'bounds': (xmin, ymin, xmax, ymax),
            'shape': grid.shape
        }
        
        logger.debug(f"Created DEM with shape {grid.shape}")
        return grid, metadata
        
    except Exception as e:
        logger.error(f"Error creating DEM: {str(e)}")
        raise

def calculate_slope_aspect(
    dem: np.ndarray,
    resolution: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate slope and aspect from DEM.
    
    Args:
        dem: Digital Elevation Model array
        resolution: Grid resolution
        
    Returns:
        tuple: (slope_array, aspect_array)
    """
    try:
        from scipy import ndimage
        
        # Calculate gradients
        dy, dx = np.gradient(dem, resolution)
        
        # Calculate slope
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        
        # Calculate aspect
        aspect = np.degrees(np.arctan2(-dx, dy))
        aspect = np.mod(aspect + 360, 360)
        
        logger.debug("Calculated slope and aspect")
        return slope, aspect
        
    except Exception as e:
        logger.error(f"Error calculating slope and aspect: {str(e)}")
        raise

def classify_ground_points(
    point_cloud: np.ndarray,
    max_angle: float = 3.0,
    max_distance: float = 0.5,
    cell_size: float = 3.0
) -> np.ndarray:
    """Classify ground points using Progressive Morphological Filter.
    
    Args:
        point_cloud: Point cloud array
        max_angle: Maximum angle (degrees) for ground classification
        max_distance: Maximum distance to ground surface
        cell_size: Initial cell size for ground surface estimation
        
    Returns:
        numpy.ndarray: Updated point cloud with ground classification
    """
    try:
        # Create PDAL pipeline for ground classification
        pipeline_json = {
            "pipeline": [
                {
                    "type": "filters.pmf",
                    "max_window_size": 33,
                    "slope": max_angle,
                    "max_distance": max_distance,
                    "initial_distance": max_distance / 2,
                    "cell_size": cell_size
                }
            ]
        }
        
        # Convert numpy array to PDAL array
        dtype = np.dtype([
            ('X', np.float64),
            ('Y', np.float64),
            ('Z', np.float64),
            ('Classification', np.uint8),
            ('Intensity', np.uint16)
        ])
        
        pdal_data = np.empty(len(point_cloud), dtype=dtype)
        pdal_data['X'] = point_cloud[:, 0]
        pdal_data['Y'] = point_cloud[:, 1]
        pdal_data['Z'] = point_cloud[:, 2]
        pdal_data['Classification'] = point_cloud[:, 3]
        pdal_data['Intensity'] = point_cloud[:, 4]
        
        # Run pipeline
        pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        pipeline.execute(pdal_data)
        
        # Get results
        arrays = pipeline.arrays
        if len(arrays) == 0:
            raise RuntimeError("PDAL pipeline produced no output")
            
        # Update classification in original point cloud
        point_cloud[:, 3] = arrays[0]['Classification']
        
        logger.debug("Completed ground point classification")
        return point_cloud
        
    except Exception as e:
        logger.error(f"Error classifying ground points: {str(e)}")
        raise

def create_intensity_image(
    point_cloud: np.ndarray,
    resolution: float,
    bounds: Optional[Tuple[float, float, float, float]] = None
) -> Tuple[np.ndarray, Dict]:
    """Create intensity image from point cloud.
    
    Args:
        point_cloud: Point cloud array
        resolution: Grid resolution
        bounds: Optional (xmin, ymin, xmax, ymax) to limit extent
        
    Returns:
        tuple: (intensity_array, metadata)
    """
    try:
        # Determine grid extent
        if bounds is None:
            xmin, ymin = point_cloud[:, 0].min(), point_cloud[:, 1].min()
            xmax, ymax = point_cloud[:, 0].max(), point_cloud[:, 1].max()
        else:
            xmin, ymin, xmax, ymax = bounds
        
        # Calculate grid dimensions
        nx = int((xmax - xmin) / resolution) + 1
        ny = int((ymax - ymin) / resolution) + 1
        
        # Create empty grid
        grid = np.zeros((ny, nx))
        counts = np.zeros((ny, nx))
        
        # Convert points to grid indices
        xi = ((point_cloud[:, 0] - xmin) / resolution).astype(int)
        yi = ((point_cloud[:, 1] - ymin) / resolution).astype(int)
        
        # Mask out points outside the grid
        mask = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
        xi, yi = xi[mask], yi[mask]
        intensity = point_cloud[mask, 4]
        
        # Accumulate intensity values
        np.add.at(grid, (yi, xi), intensity)
        np.add.at(counts, (yi, xi), 1)
        
        # Average intensity values
        mask = counts > 0
        grid[mask] /= counts[mask]
        
        metadata = {
            'resolution': resolution,
            'bounds': (xmin, ymin, xmax, ymax),
            'shape': grid.shape
        }
        
        logger.debug(f"Created intensity image with shape {grid.shape}")
        return grid, metadata
        
    except Exception as e:
        logger.error(f"Error creating intensity image: {str(e)}")
        raise