"""Terrain feature extraction from elevation data."""
import numpy as np
from typing import List, Dict, Optional
from scipy import ndimage
from skimage import feature
from .base import FeatureExtractor
from ...utils.logger import setup_logger
from ... import config

logger = setup_logger(__name__)

class TerrainFeatureExtractor(FeatureExtractor):
    """Extract terrain features from elevation data."""
    
    def __init__(self):
        """Initialize terrain feature extractor."""
        super().__init__()
        
        # Default parameters
        self.parameters = {
            'scales': config.TERRAIN_ANALYSIS_SCALES,  # Analysis scales in meters
            'resolution': 1.0,  # Pixel resolution in meters
            'gaussian_sigma': 1.0  # Smoothing parameter
        }
        
        self._update_feature_names()
    
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """Extract terrain features from elevation data.
        
        Args:
            data: Elevation array (height, width)
            
        Returns:
            numpy.ndarray: Feature array
        """
        try:
            features = []
            
            # Convert scales from meters to pixels
            pixel_scales = [int(s / self.parameters['resolution'])
                          for s in self.parameters['scales']]
            
            # Basic terrain parameters
            for scale in pixel_scales:
                sigma = scale / 3.0  # Approximate Gaussian sigma
                
                # Smooth elevation data
                smoothed = ndimage.gaussian_filter(data, sigma)
                
                # Calculate slope and aspect
                dy, dx = np.gradient(smoothed)
                slope = np.degrees(np.arctan(np.sqrt(dx*dx + dy*dy)))
                aspect = np.degrees(np.arctan2(-dx, dy))
                aspect = np.mod(aspect + 360, 360)
                
                features.extend([slope, aspect])
                
                # Calculate curvature
                dyy, dyx = np.gradient(dy)
                dxy, dxx = np.gradient(dx)
                
                # Plan curvature (across slope)
                plan_curv = ((dxx * dy * dy + dyy * dx * dx - 2 * dxy * dx * dy) /
                            (dx * dx + dy * dy))
                
                # Profile curvature (in direction of slope)
                profile_curv = -2 * ((dxx * dx * dx + dyy * dy * dy + dxy * dx * dy) /
                                   (dx * dx + dy * dy))
                
                features.extend([plan_curv, profile_curv])
                
                # Calculate Topographic Position Index (TPI)
                kernel_size = int(2 * scale + 1)
                if kernel_size > 1:
                    neighborhood_mean = ndimage.uniform_filter(
                        smoothed,
                        size=kernel_size
                    )
                    tpi = smoothed - neighborhood_mean
                    features.append(tpi)
                
                # Calculate Terrain Ruggedness Index (TRI)
                kernel = np.ones((kernel_size, kernel_size))
                n = kernel.sum() - 1  # Exclude center pixel
                if n > 0:
                    neighborhood_sum = ndimage.correlate(smoothed, kernel)
                    center_val = smoothed * kernel.sum()
                    mean_diff = np.abs(neighborhood_sum - center_val) / n
                    features.append(mean_diff)
                
                # Calculate surface roughness
                if kernel_size > 1:
                    roughness = ndimage.generic_filter(
                        smoothed,
                        lambda x: np.max(x) - np.min(x),
                        size=kernel_size
                    )
                    features.append(roughness)
            
            # Multi-scale ridge detection
            for scale in pixel_scales:
                sigma = scale / 3.0
                
                # Calculate Hessian matrix
                Hxx, Hxy, Hyy = feature.hessian_matrix(
                    data,
                    sigma=sigma,
                    mode='reflect'
                )
                
                # Calculate eigenvalues
                eigen_vals = feature.hessian_matrix_eigvals(Hxx, Hxy, Hyy)
                
                # Ridge measure (positive eigenvalue indicates ridge)
                ridge = np.maximum(eigen_vals[0], 0)
                valley = np.maximum(-eigen_vals[1], 0)
                
                features.extend([ridge, valley])
            
            # Calculate flow accumulation
            # (Simple D8 algorithm)
            flow_acc = np.zeros_like(data)
            
            # Calculate flow direction (8 directions)
            dy, dx = np.gradient(data)
            flow_dir = np.degrees(np.arctan2(-dy, -dx))
            flow_dir = np.mod(flow_dir + 360, 360)
            
            # Accumulate flow
            for i in range(1, data.shape[0] - 1):
                for j in range(1, data.shape[1] - 1):
                    # Get flow direction
                    direction = flow_dir[i, j]
                    
                    # Determine target cell
                    if 337.5 <= direction or direction < 22.5:  # East
                        ti, tj = i, j+1
                    elif 22.5 <= direction < 67.5:  # Northeast
                        ti, tj = i-1, j+1
                    elif 67.5 <= direction < 112.5:  # North
                        ti, tj = i-1, j
                    elif 112.5 <= direction < 157.5:  # Northwest
                        ti, tj = i-1, j-1
                    elif 157.5 <= direction < 202.5:  # West
                        ti, tj = i, j-1
                    elif 202.5 <= direction < 247.5:  # Southwest
                        ti, tj = i+1, j-1
                    elif 247.5 <= direction < 292.5:  # South
                        ti, tj = i+1, j
                    else:  # Southeast
                        ti, tj = i+1, j+1
                    
                    # Accumulate flow
                    flow_acc[ti, tj] += flow_acc[i, j] + 1
            
            features.append(flow_acc)
            
            # Calculate topographic wetness index
            slope_rad = np.radians(slope)
            # Avoid division by zero
            slope_rad = np.maximum(slope_rad, 0.001)
            twi = np.log((flow_acc + 1) / np.tan(slope_rad))
            features.append(twi)
            
            # Stack all features
            return np.stack(features, axis=0)
            
        except Exception as e:
            logger.error("Error extracting terrain features")
            logger.exception(e)
            raise
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names.
        
        Returns:
            list: Feature names
        """
        return self.feature_names
    
    def set_resolution(self, resolution: float) -> None:
        """Set the spatial resolution of the data.
        
        Args:
            resolution: Resolution in meters
        """
        self.parameters['resolution'] = resolution
        logger.debug(f"Set resolution to {resolution}m")
        self._update_feature_names()
    
    def _update_feature_names(self) -> None:
        """Update list of feature names based on current parameters."""
        self.feature_names = []
        
        # Basic terrain parameters at each scale
        for scale in self.parameters['scales']:
            self.feature_names.extend([
                f"Slope_{scale}m",
                f"Aspect_{scale}m",
                f"PlanCurvature_{scale}m",
                f"ProfileCurvature_{scale}m",
                f"TPI_{scale}m",
                f"TRI_{scale}m",
                f"Roughness_{scale}m"
            ])
        
        # Ridge detection features
        for scale in self.parameters['scales']:
            self.feature_names.extend([
                f"Ridge_{scale}m",
                f"Valley_{scale}m"
            ])
        
        # Hydrological features
        self.feature_names.extend([
            "FlowAccumulation",
            "TopographicWetnessIndex"
        ])