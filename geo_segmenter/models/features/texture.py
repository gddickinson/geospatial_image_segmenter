"""Texture feature extraction for image data."""
import numpy as np
from typing import List, Dict, Optional
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from scipy import ndimage
from .base import FeatureExtractor
from ...utils.logger import setup_logger
from ... import config

logger = setup_logger(__name__)

class TextureFeatureExtractor(FeatureExtractor):
    """Extract texture features using various methods."""
    
    def __init__(self):
        """Initialize texture feature extractor."""
        super().__init__()
        
        # Default parameters
        self.parameters = {
            'window_sizes': config.TEXTURE_WINDOWS,
            'glcm_distances': [1, 2, 4],
            'glcm_angles': [0, np.pi/4, np.pi/2, 3*np.pi/4],
            'glcm_properties': [
                'contrast',
                'dissimilarity',
                'homogeneity',
                'energy',
                'correlation'
            ]
        }
        
        self._update_feature_names()
    
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """Extract texture features from image data.
        
        Args:
            data: Input image array (height, width) or (height, width, bands)
            
        Returns:
            numpy.ndarray: Feature array
        """
        try:
            # Convert to grayscale if needed
            if data.ndim == 3:
                if data.shape[2] >= 3:
                    # Use standard RGB to grayscale conversion
                    gray = np.dot(data[..., :3], [0.2989, 0.5870, 0.1140])
                else:
                    gray = data[..., 0]
            else:
                gray = data
            
            # Normalize to [0, 255] range for GLCM
            gray = ((gray - gray.min()) / (gray.max() - gray.min()) * 255).astype(np.uint8)
            
            features = []
            
            # Calculate GLCM features
            for window_size in self.parameters['window_sizes']:
                # Use sliding window to calculate local GLCM features
                pad = window_size // 2
                padded = np.pad(gray, pad, mode='reflect')
                
                for d in self.parameters['glcm_distances']:
                    glcm_features = []
                    
                    for i in range(pad, padded.shape[0] - pad):
                        for j in range(pad, padded.shape[1] - pad):
                            # Extract window
                            window = padded[i-pad:i+pad+1, j-pad:j+pad+1]
                            
                            # Calculate GLCM
                            glcm = graycomatrix(window,
                                              distances=[d],
                                              angles=self.parameters['glcm_angles'],
                                              levels=256,
                                              symmetric=True,
                                              normed=True)
                            
                            # Calculate GLCM properties
                            props = []
                            for prop in self.parameters['glcm_properties']:
                                p = graycoprops(glcm, prop)[0]
                                props.extend(p)
                            
                            glcm_features.append(props)
                    
                    # Convert to array and reshape
                    glcm_features = np.array(glcm_features)
                    glcm_features = glcm_features.reshape(gray.shape[0], gray.shape[1], -1)
                    
                    # Add each feature channel
                    for i in range(glcm_features.shape[2]):
                        features.append(glcm_features[..., i])
            
            # Calculate local entropy
            for window_size in self.parameters['window_sizes']:
                entropy = ndimage.generic_filter(
                    gray,
                    shannon_entropy,
                    size=window_size
                )
                features.append(entropy)
            
            # Calculate gradient features
            for window_size in self.parameters['window_sizes']:
                # Gaussian gradient magnitude
                gradient = ndimage.gaussian_gradient_magnitude(
                    gray,
                    sigma=window_size/3
                )
                features.append(gradient)
                
                # Local gradient standard deviation
                grad_std = ndimage.generic_filter(
                    gradient,
                    np.std,
                    size=window_size
                )
                features.append(grad_std)
            
            # Calculate local statistics
            for window_size in self.parameters['window_sizes']:
                # Standard deviation
                std = ndimage.generic_filter(
                    gray,
                    np.std,
                    size=window_size
                )
                features.append(std)
                
                # Range (max - min)
                def local_range(x):
                    return np.max(x) - np.min(x)
                
                range_feat = ndimage.generic_filter(
                    gray,
                    local_range,
                    size=window_size
                )
                features.append(range_feat)
            
            # Stack all features
            return np.stack(features, axis=0)
            
        except Exception as e:
            logger.error("Error extracting texture features")
            logger.exception(e)
            raise
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names.
        
        Returns:
            list: Feature names
        """
        return self.feature_names
    
    def _update_feature_names(self) -> None:
        """Update list of feature names based on current parameters."""
        self.feature_names = []
        
        # GLCM features
        for w in self.parameters['window_sizes']:
            for d in self.parameters['glcm_distances']:
                for angle in self.parameters['glcm_angles']:
                    angle_deg = int(np.degrees(angle))
                    for prop in self.parameters['glcm_properties']:
                        name = f"GLCM_{prop}_{w}px_{d}d_{angle_deg}deg"
                        self.feature_names.append(name)
        
        # Entropy features
        for w in self.parameters['window_sizes']:
            self.feature_names.append(f"Entropy_{w}px")
        
        # Gradient features
        for w in self.parameters['window_sizes']:
            self.feature_names.append(f"GradientMagnitude_{w}px")
            self.feature_names.append(f"GradientStd_{w}px")
        
        # Statistical features
        for w in self.parameters['window_sizes']:
            self.feature_names.append(f"StdDev_{w}px")
            self.feature_names.append(f"Range_{w}px")