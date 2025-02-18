"""Provider for handling raster data sources."""
import numpy as np
from typing import Optional, Dict, Tuple, List
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio import warp
from pathlib import Path

from ...utils.logger import setup_logger
from ... import config

logger = setup_logger(__name__)

class RasterProvider:
    """Provider for reading and processing raster data."""
    
    def __init__(self):
        """Initialize raster provider."""
        self.open_datasets = {}  # Track open rasterio datasets
        
    def get_raster_info(self, path: str) -> Dict:
        """Get information about a raster dataset.
        
        Args:
            path: Path to raster file
            
        Returns:
            dict: Raster information
        """
        try:
            with rasterio.open(path) as src:
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
                
                # Add band statistics
                stats = []
                for i in range(1, src.count + 1):
                    band_stats = {
                        'min': float(src.statistics(i).min),
                        'max': float(src.statistics(i).max),
                        'mean': float(src.statistics(i).mean),
                        'std': float(src.statistics(i).std)
                    }
                    stats.append(band_stats)
                info['band_stats'] = stats
                
                return info
                
        except Exception as e:
            logger.error(f"Error getting raster info: {str(e)}")
            logger.exception(e)
            raise
    
    def read_raster_window(
        self,
        path: str,
        bounds: Tuple[float, float, float, float],
        width: int,
        height: int,
        bands: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, rasterio.transform.Affine]:
        """Read a portion of a raster dataset.
        
        Args:
            path: Path to raster file
            bounds: (min_x, min_y, max_x, max_y) in raster's CRS
            width: Output width in pixels
            height: Output height in pixels
            bands: List of band indices to read (1-based)
            
        Returns:
            tuple: (data array, transform)
        """
        try:
            # Get or open dataset
            if path not in self.open_datasets:
                self.open_datasets[path] = rasterio.open(path)
            src = self.open_datasets[path]
            
            # Default to all bands if none specified
            if bands is None:
                bands = list(range(1, src.count + 1))
            
            # Calculate window
            window = src.window(*bounds)
            
            # Read data
            data = src.read(
                bands,
                window=window,
                out_shape=(len(bands), height, width),
                resampling=Resampling.bilinear
            )
            
            # Get transform
            transform = src.window_transform(window)
            
            return data, transform
            
        except Exception as e:
            logger.error(f"Error reading raster window: {str(e)}")
            logger.exception(e)
            raise
    
    def reproject_raster(
        self,
        path: str,
        dst_crs: str,
        resolution: Optional[Tuple[float, float]] = None
    ) -> Tuple[np.ndarray, rasterio.transform.Affine]:
        """Reproject a raster dataset.
        
        Args:
            path: Path to raster file
            dst_crs: Target CRS
            resolution: Optional (x_res, y_res) for output
            
        Returns:
            tuple: (reprojected data, transform)
        """
        try:
            with rasterio.open(path) as src:
                # Calculate transform for reprojection
                transform, width, height = warp.calculate_default_transform(
                    src.crs,
                    dst_crs,
                    src.width,
                    src.height,
                    *src.bounds,
                    resolution=resolution
                )
                
                # Initialize output array
                out_shape = (src.count, height, width)
                out_data = np.zeros(out_shape, dtype=src.dtypes[0])
                
                # Reproject each band
                for i in range(1, src.count + 1):
                    warp.reproject(
                        source=rasterio.band(src, i),
                        destination=out_data[i-1],
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear
                    )
                
                return out_data, transform
                
        except Exception as e:
            logger.error(f"Error reprojecting raster: {str(e)}")
            logger.exception(e)
            raise
    
    def calculate_statistics(
        self,
        path: str,
        band: int = 1,
        sample_size: Optional[int] = None
    ) -> Dict:
        """Calculate statistics for a raster band.
        
        Args:
            path: Path to raster file
            band: Band number (1-based)
            sample_size: Optional number of pixels to sample
            
        Returns:
            dict: Statistics dictionary
        """
        try:
            with rasterio.open(path) as src:
                if band < 1 or band > src.count:
                    raise ValueError(f"Invalid band number: {band}")
                
                # Read data (potentially using sample)
                if sample_size:
                    # Calculate row/col indices for random sample
                    rows = np.random.randint(0, src.height, sample_size)
                    cols = np.random.randint(0, src.width, sample_size)
                    data = src.read(band, indexes=(rows, cols))
                else:
                    data = src.read(band)
                
                # Calculate statistics
                stats = {
                    'min': float(np.min(data)),
                    'max': float(np.max(data)),
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'median': float(np.median(data)),
                    'percentiles': {
                        '1': float(np.percentile(data, 1)),
                        '5': float(np.percentile(data, 5)),
                        '95': float(np.percentile(data, 95)),
                        '99': float(np.percentile(data, 99))
                    }
                }
                
                return stats
                
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            logger.exception(e)
            raise
    
    def create_overview_pyramid(
        self,
        path: str,
        factors: List[int] = [2, 4, 8, 16]
    ) -> None:
        """Create overview pyramid for a raster dataset.
        
        Args:
            path: Path to raster file
            factors: List of overview reduction factors
        """
        try:
            with rasterio.open(path, 'r+') as src:
                src.build_overviews(factors, Resampling.average)
                src.update_tags(ns='rio_overview', resampling='average')
                logger.info(f"Created overview pyramid for {path}")
                
        except Exception as e:
            logger.error(f"Error creating overview pyramid: {str(e)}")
            logger.exception(e)
            raise
    
    def close_dataset(self, path: str) -> None:
        """Close an open raster dataset.
        
        Args:
            path: Path to raster file
        """
        try:
            if path in self.open_datasets:
                self.open_datasets[path].close()
                del self.open_datasets[path]
                logger.debug(f"Closed dataset: {path}")
                
        except Exception as e:
            logger.error(f"Error closing dataset: {str(e)}")
            logger.exception(e)
    
    def close_all(self) -> None:
        """Close all open datasets."""
        try:
            for path in list(self.open_datasets.keys()):
                self.close_dataset(path)
            logger.debug("Closed all datasets")
            
        except Exception as e:
            logger.error("Error closing all datasets")
            logger.exception(e)
    
    def __del__(self):
        """Cleanup when object is deleted."""
        self.close_all()