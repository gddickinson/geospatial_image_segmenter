"""Utility functions for image processing and raster analysis."""
import numpy as np
from typing import Tuple, List, Optional, Union
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from scipy import ndimage
from skimage import exposure, filters
from .. import config
from .logger import setup_logger

logger = setup_logger(__name__)

def load_raster_window(
    path: str,
    window: Optional[Window] = None,
    bands: Optional[List[int]] = None,
    out_shape: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, rasterio.transform.Affine]:
    """Load a portion of a raster file.
    
    Args:
        path: Path to raster file
        window: Rasterio Window object defining the region to load
        bands: List of band indices to load (1-based)
        out_shape: Desired output shape for resampling
        
    Returns:
        tuple: (image array, transform)
    """
    try:
        with rasterio.open(path) as src:
            if bands is None:
                bands = list(range(1, src.count + 1))
                
            # Read specified bands
            data = src.read(
                bands,
                window=window,
                out_shape=out_shape,
                resampling=Resampling.bilinear
            )
            
            # Get transform
            transform = src.window_transform(window) if window else src.transform
            
            # Transpose to (height, width, bands)
            data = np.transpose(data, (1, 2, 0))
            
            logger.debug(f"Loaded raster data with shape {data.shape}")
            return data, transform
            
    except Exception as e:
        logger.error(f"Error loading raster: {str(e)}")
        raise

def normalize_image(
    image: np.ndarray,
    method: str = 'minmax',
    percentiles: Tuple[float, float] = (2, 98)
) -> np.ndarray:
    """Normalize image data.
    
    Args:
        image: Input image array
        method: Normalization method ('minmax' or 'percentile')
        percentiles: Percentile values for contrast stretching
        
    Returns:
        numpy.ndarray: Normalized image
    """
    try:
        if not isinstance(image, np.ndarray):
            raise TypeError("Input must be a numpy array")
            
        if method == 'minmax':
            return (image - image.min()) / (image.max() - image.min())
            
        elif method == 'percentile':
            p_low, p_high = np.percentile(image, percentiles)
            return np.clip((image - p_low) / (p_high - p_low), 0, 1)
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
    except Exception as e:
        logger.error(f"Error normalizing image: {str(e)}")
        raise

def calculate_spectral_index(
    image: np.ndarray,
    index_name: str,
    band_map: dict
) -> np.ndarray:
    """Calculate spectral indices from multi-band imagery.
    
    Args:
        image: Multi-band image array
        index_name: Name of spectral index to calculate
        band_map: Dictionary mapping band names to array indices
        
    Returns:
        numpy.ndarray: Calculated index
    """
    try:
        eps = 1e-8  # Small value to prevent division by zero
        
        if index_name == "NDVI":
            nir = image[:, :, band_map['nir']]
            red = image[:, :, band_map['red']]
            return (nir - red) / (nir + red + eps)
            
        elif index_name == "NDWI":
            nir = image[:, :, band_map['nir']]
            green = image[:, :, band_map['green']]
            return (green - nir) / (green + nir + eps)
            
        elif index_name == "SAVI":
            # Soil Adjusted Vegetation Index
            L = 0.5  # soil brightness correction factor
            nir = image[:, :, band_map['nir']]
            red = image[:, :, band_map['red']]
            return ((nir - red) * (1 + L)) / (nir + red + L + eps)
            
        elif index_name == "EVI":
            # Enhanced Vegetation Index
            G = 2.5  # gain factor
            C1, C2 = 6.0, 7.5  # aerosol resistance terms
            L = 1.0  # canopy background adjustment
            
            nir = image[:, :, band_map['nir']]
            red = image[:, :, band_map['red']]
            blue = image[:, :, band_map['blue']]
            
            return G * (nir - red) / (nir + C1 * red - C2 * blue + L + eps)
            
        else:
            raise ValueError(f"Unknown spectral index: {index_name}")
            
    except Exception as e:
        logger.error(f"Error calculating spectral index: {str(e)}")
        raise

def calculate_texture_features(
    image: np.ndarray,
    window_size: int = 5
) -> List[np.ndarray]:
    """Calculate texture features using sliding windows.
    
    Args:
        image: Input image array
        window_size: Size of sliding window
        
    Returns:
        list: List of texture feature arrays
    """
    try:
        features = []
        
        # Mean
        mean = ndimage.uniform_filter(image, size=window_size)
        features.append(mean)
        
        # Variance
        variance = ndimage.uniform_filter(image**2, size=window_size) - mean**2
        features.append(variance)
        
        # Range
        footprint = np.ones((window_size, window_size))
        local_min = ndimage.minimum_filter(image, footprint=footprint)
        local_max = ndimage.maximum_filter(image, footprint=footprint)
        range_img = local_max - local_min
        features.append(range_img)
        
        # Entropy (using uniform filter as approximation)
        eps = 1e-8
        p = image + eps
        p = p / p.sum()
        entropy = -p * np.log2(p)
        entropy = ndimage.uniform_filter(entropy, size=window_size)
        features.append(entropy)
        
        return features
        
    except Exception as e:
        logger.error(f"Error calculating texture features: {str(e)}")
        raise

def create_overview_pyramid(
    image: np.ndarray,
    max_levels: int = 4
) -> List[np.ndarray]:
    """Create image pyramid for multi-scale analysis.
    
    Args:
        image: Input image array
        max_levels: Maximum number of pyramid levels
        
    Returns:
        list: List of increasingly downsampled images
    """
    try:
        pyramid = [image]
        current = image
        
        for _ in range(max_levels - 1):
            reduced = ndimage.zoom(current, 0.5, order=1)
            pyramid.append(reduced)
            current = reduced
            
        return pyramid
        
    except Exception as e:
        logger.error(f"Error creating image pyramid: {str(e)}")
        raise

def mask_clouds(
    image: np.ndarray,
    method: str = 'threshold',
    threshold: float = 0.9
) -> np.ndarray:
    """Create cloud mask for satellite imagery.
    
    Args:
        image: Input image array
        method: Cloud detection method
        threshold: Brightness threshold for cloud detection
        
    Returns:
        numpy.ndarray: Boolean cloud mask
    """
    try:
        if method == 'threshold':
            # Simple brightness thresholding
            if image.ndim == 3:
                brightness = np.mean(image, axis=2)
            else:
                brightness = image
            return brightness > threshold
            
        else:
            raise ValueError(f"Unknown cloud masking method: {method}")
            
    except Exception as e:
        logger.error(f"Error creating cloud mask: {str(e)}")
        raise