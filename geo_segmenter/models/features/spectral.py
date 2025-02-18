"""Spectral feature extraction for multi-band imagery."""
import numpy as np
from typing import List, Dict, Optional
from .base import FeatureExtractor
from ...utils.logger import setup_logger
from ... import config

logger = setup_logger(__name__)

class SpectralFeatureExtractor(FeatureExtractor):
    """Extract spectral features from multi-band imagery."""
    
    def __init__(self):
        """Initialize spectral feature extractor."""
        super().__init__()
        
        # Default parameters
        self.parameters = {
            'indices': config.SPECTRAL_INDICES,
            'band_map': {
                'blue': 0,
                'green': 1,
                'red': 2,
                'nir': 3  # If available
            }
        }
        
        self._update_feature_names()
    
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """Extract spectral features from image data.
        
        Args:
            data: Multi-band image array (height, width, bands)
            
        Returns:
            numpy.ndarray: Feature array
        """
        try:
            features = []
            eps = 1e-8  # Small value to prevent division by zero
            
            for index in self.parameters['indices']:
                if index == 'NDVI' and self._has_bands(['red', 'nir']):
                    # Normalized Difference Vegetation Index
                    nir = data[:, :, self.parameters['band_map']['nir']]
                    red = data[:, :, self.parameters['band_map']['red']]
                    ndvi = (nir - red) / (nir + red + eps)
                    features.append(ndvi)
                    
                elif index == 'NDWI' and self._has_bands(['green', 'nir']):
                    # Normalized Difference Water Index
                    green = data[:, :, self.parameters['band_map']['green']]
                    nir = data[:, :, self.parameters['band_map']['nir']]
                    ndwi = (green - nir) / (green + nir + eps)
                    features.append(ndwi)
                    
                elif index == 'SAVI' and self._has_bands(['red', 'nir']):
                    # Soil Adjusted Vegetation Index
                    L = 0.5  # soil brightness correction factor
                    nir = data[:, :, self.parameters['band_map']['nir']]
                    red = data[:, :, self.parameters['band_map']['red']]
                    savi = ((nir - red) * (1 + L)) / (nir + red + L + eps)
                    features.append(savi)
                    
                elif index == 'EVI' and self._has_bands(['blue', 'red', 'nir']):
                    # Enhanced Vegetation Index
                    G = 2.5  # gain factor
                    C1, C2 = 6.0, 7.5  # aerosol resistance terms
                    L = 1.0  # canopy background adjustment
                    
                    nir = data[:, :, self.parameters['band_map']['nir']]
                    red = data[:, :, self.parameters['band_map']['red']]
                    blue = data[:, :, self.parameters['band_map']['blue']]
                    
                    evi = G * (nir - red) / (nir + C1 * red - C2 * blue + L + eps)
                    features.append(evi)
                    
                elif index == 'ARVI' and self._has_bands(['blue', 'red', 'nir']):
                    # Atmospherically Resistant Vegetation Index
                    nir = data[:, :, self.parameters['band_map']['nir']]
                    red = data[:, :, self.parameters['band_map']['red']]
                    blue = data[:, :, self.parameters['band_map']['blue']]
                    
                    rb = red - (red - blue)  # Correction for atmospheric effects
                    arvi = (nir - rb) / (nir + rb + eps)
                    features.append(arvi)
                    
                elif index == 'SIPI' and self._has_bands(['blue', 'red', 'nir']):
                    # Structure Insensitive Pigment Index
                    nir = data[:, :, self.parameters['band_map']['nir']]
                    red = data[:, :, self.parameters['band_map']['red']]
                    blue = data[:, :, self.parameters['band_map']['blue']]
                    
                    sipi = (nir - blue) / (nir - red + eps)
                    features.append(sipi)
            
            # Add band ratios
            if self._has_bands(['red', 'green']):
                rg_ratio = (data[:, :, self.parameters['band_map']['red']] /
                          (data[:, :, self.parameters['band_map']['green']] + eps))
                features.append(rg_ratio)
            
            if self._has_bands(['nir', 'red']):
                nir_ratio = (data[:, :, self.parameters['band_map']['nir']] /
                           (data[:, :, self.parameters['band_map']['red']] + eps))
                features.append(nir_ratio)
            
            # Stack features
            if features:
                return np.stack(features, axis=0)
            else:
                raise ValueError("No features could be calculated with available bands")
            
        except Exception as e:
            logger.error("Error extracting spectral features")
            logger.exception(e)
            raise
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names.
        
        Returns:
            list: Feature names
        """
        return self.feature_names
    
    def set_band_map(self, band_map: Dict[str, int]) -> None:
        """Set band mapping.
        
        Args:
            band_map: Dictionary mapping band names to indices
        """
        self.parameters['band_map'] = band_map
        logger.debug(f"Updated band mapping: {band_map}")
        self._update_feature_names()
    
    def _has_bands(self, required_bands: List[str]) -> bool:
        """Check if required bands are available.
        
        Args:
            required_bands: List of required band names
            
        Returns:
            bool: True if all required bands are available
        """
        return all(band in self.parameters['band_map'] 
                  for band in required_bands)
    
    def _update_feature_names(self) -> None:
        """Update list of feature names based on current parameters."""
        self.feature_names = []
        
        # Add spectral indices
        for index in self.parameters['indices']:
            if ((index == 'NDVI' and self._has_bands(['red', 'nir'])) or
                (index == 'NDWI' and self._has_bands(['green', 'nir'])) or
                (index == 'SAVI' and self._has_bands(['red', 'nir'])) or
                (index == 'EVI' and self._has_bands(['blue', 'red', 'nir'])) or
                (index == 'ARVI' and self._has_bands(['blue', 'red', 'nir'])) or
                (index == 'SIPI' and self._has_bands(['blue', 'red', 'nir']))):
                self.feature_names.append(index)
        
        # Add band ratios
        if self._has_bands(['red', 'green']):
            self.feature_names.append('RG_ratio')
        if self._has_bands(['nir', 'red']):
            self.feature_names.append('NIR_ratio')