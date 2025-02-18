"""Base model interface for geospatial segmentation."""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple, Optional, List
import json
from pathlib import Path

from .features.base import FeatureSet
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class GeospatialModel(ABC):
    """Abstract base class for geospatial segmentation models."""
    
    def __init__(self):
        """Initialize base model components."""
        self.feature_set = FeatureSet()
        self.is_trained = False
        self.class_names = []
        self.model_info = {}
    
    def prepare_training_data(
        self,
        image: np.ndarray,
        labels: Dict[str, np.ndarray],
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for training.
        
        Args:
            image: Input image data
            labels: Dictionary mapping class names to boolean masks
            mask: Optional mask of valid data areas
            
        Returns:
            tuple: (features, labels) prepared for training
        """
        try:
            logger.debug("Preparing training data")
            
            # Extract features
            feature_dict = self.feature_set.extract_all_features(image)
            
            # Combine all features
            features = []
            for feat_array in feature_dict.values():
                if feat_array.ndim == 2:
                    features.append(feat_array[np.newaxis, ...])
                else:
                    features.append(feat_array)
            features = np.vstack(features)
            
            # Prepare training points
            X = []
            y = []
            
            for label_idx, (label_name, label_mask) in enumerate(labels.items()):
                if np.any(label_mask):
                    # Apply valid data mask if provided
                    if mask is not None:
                        label_mask = label_mask & mask
                    
                    # Get features for labeled pixels
                    f = features[:, label_mask].T
                    X.append(f)
                    y.extend([label_idx] * f.shape[0])
                    
                    logger.debug(f"Added {f.shape[0]} training points for class '{label_name}'")
            
            if not X:
                raise ValueError("No labeled pixels found in any class")
            
            X_array = np.vstack(X)
            y_array = np.array(y)
            
            # Store class names
            self.class_names = list(labels.keys())
            
            logger.debug(f"Prepared training data with shape: X={X_array.shape}, y={y_array.shape}")
            return X_array, y_array
            
        except Exception as e:
            logger.error("Error preparing training data")
            logger.exception(e)
            raise
    
    @abstractmethod
    def train(
        self,
        image: np.ndarray,
        labels: Dict[str, np.ndarray],
        mask: Optional[np.ndarray] = None,
        **kwargs
    ) -> None:
        """Train the model.
        
        Args:
            image: Training image
            labels: Dictionary mapping class names to boolean masks
            mask: Optional mask of valid data areas
            **kwargs: Additional model-specific parameters
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Predict segmentation for new image.
        
        Args:
            image: Image to segment
            mask: Optional mask of valid data areas
            
        Returns:
            numpy.ndarray: Predicted segmentation mask
        """
        pass
    
    def predict_proba(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            image: Image to segment
            mask: Optional mask of valid data areas
            
        Returns:
            numpy.ndarray: Class probability maps
        """
        raise NotImplementedError("Probability prediction not implemented for this model")
    
    def save_model(self, path: str) -> None:
        """Save model to disk.
        
        Args:
            path: Save path
        """
        raise NotImplementedError("Save functionality not implemented for this model")
    
    def load_model(self, path: str) -> None:
        """Load model from disk.
        
        Args:
            path: Load path
        """
        raise NotImplementedError("Load functionality not implemented for this model")
    
    def add_feature_extractor(self, name: str, extractor) -> None:
        """Add a feature extractor to the model.
        
        Args:
            name: Extractor name
            extractor: FeatureExtractor instance
        """
        self.feature_set.add_extractor(name, extractor)
    
    def enable_feature_extractor(self, name: str) -> None:
        """Enable a feature extractor.
        
        Args:
            name: Extractor name
        """
        self.feature_set.enable_extractor(name)
    
    def disable_feature_extractor(self, name: str) -> None:
        """Disable a feature extractor.
        
        Args:
            name: Extractor name
        """
        self.feature_set.disable_extractor(name)
    
    def get_feature_info(self) -> Dict[str, List[str]]:
        """Get information about available features.
        
        Returns:
            dict: Dictionary mapping extractor names to feature lists
        """
        return self.feature_set.get_feature_info()
    
    def get_model_info(self) -> Dict:
        """Get model information and parameters.
        
        Returns:
            dict: Model information dictionary
        """
        return self.model_info.copy()
    
    def validate_image(self, image: np.ndarray) -> None:
        """Validate input image.
        
        Args:
            image: Input image to validate
            
        Raises:
            ValueError: If image is invalid
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
        if image.ndim not in [2, 3]:
            raise ValueError("Image must be 2D or 3D array")
        if not np.issubdtype(image.dtype, np.number):
            raise ValueError("Image must contain numeric values")