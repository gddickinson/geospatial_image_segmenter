"""Provider for handling LIDAR point cloud data."""
import numpy as np
from typing import Optional, Dict, Tuple, List
import laspy
import pdal
import json
from pathlib import Path

from ...utils.logger import setup_logger
from ... import config

logger = setup_logger(__name__)

class LidarProvider:
    """Provider for reading and processing LIDAR data."""
    
    def __init__(self):
        """Initialize LIDAR provider."""
        self.open_files = {}  # Track open LAS files
    
    def read_lidar_file(
        self,
        path: str,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        max_points: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Read LIDAR data from file.
        
        Args:
            path: Path to LAS/LAZ file
            bounds: Optional (min_x, min_y, max_x, max_y) to filter points
            max_points: Maximum number of points to read
            
        Returns:
            tuple: (point cloud array, metadata)
        """
        try:
            # Open file
            if path not in self.open_files:
                self.open_files[path] = laspy.read(path)
            las = self.open_files[path]
            
            # Extract points
            points = np.vstack((las.x, las.y, las.z)).transpose()
            
            # Apply spatial filter if bounds specified
            if bounds:
                mask = ((points[:, 0] >= bounds[0]) &
                       (points[:, 0] <= bounds[2]) &
                       (points[:, 1] >= bounds[1]) &
                       (points[:, 1] <= bounds[3]))
                points = points[mask]
            
            # Get classification if available
            if hasattr(las, 'classification'):
                classification = las.classification
                if bounds:
                    classification = classification[mask]
            else:
                classification = np.zeros(len(points))
            
            # Get intensity if available
            if hasattr(las, 'intensity'):
                intensity = las.intensity
                if bounds:
                    intensity = intensity[mask]
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
                'point_format': las.header.point_format.id,
                'crs': las.header.parse_crs(),
                'min_bounds': las.header.mins,
                'max_bounds': las.header.maxs,
                'version': las.header.version,
                'point_count_by_class': self._count_by_classification(point_cloud[:, 3])
            }
            
            logger.debug(f"Read {len(point_cloud)} points from {path}")
            return point_cloud, metadata
            
        except Exception as e:
            logger.error(f"Error reading LIDAR file: {str(e)}")
            logger.exception(e)
            raise
    
    def classify_ground_points(
        self,
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
            logger.exception(e)
            raise
    
    def create_dem(
        self,
        point_cloud: np.ndarray,
        resolution: float,
        bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Create Digital Elevation Model from ground points.
        
        Args:
            point_cloud: Point cloud array
            resolution: Grid resolution
            bounds: Optional (min_x, min_y, max_x, max_y) to limit extent
            
        Returns:
            tuple: (dem_array, metadata)
        """
        try:
            # Extract ground points (class 2)
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
            logger.exception(e)
            raise
    
    def close_file(self, path: str) -> None:
        """Close an open LIDAR file.
        
        Args:
            path: Path to file
        """
        try:
            if path in self.open_files:
                self.open_files[path].close()
                del self.open_files[path]
                logger.debug(f"Closed file: {path}")
                
        except Exception as e:
            logger.error(f"Error closing file: {str(e)}")
            logger.exception(e)
    
    def close_all(self) -> None:
        """Close all open files."""
        try:
            for path in list(self.open_files.keys()):
                self.close_file(path)
            logger.debug("Closed all files")
            
        except Exception as e:
            logger.error("Error closing all files")
            logger.exception(e)
    
    def _count_by_classification(self, classifications: np.ndarray) -> Dict[int, int]:
        """Count points by classification value.
        
        Args:
            classifications: Array of classification values
            
        Returns:
            dict: Mapping of classification values to counts
        """
        unique, counts = np.unique(classifications, return_counts=True)
        return dict(zip(unique.astype(int), counts.astype(int)))
    
    def __del__(self):
        """Cleanup when object is deleted."""
        self.close_all()