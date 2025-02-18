"""Base class for feature extraction."""
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Optional, Any
from ...utils.logger import setup_logger

logger = setup_logger(__name__)

class FeatureExtractor(ABC):
    """Abstract base class for feature extractors."""
    
    def __init__(self):
        """Initialize feature extractor."""
        self.parameters = {}
        self.feature_names = []
    
    @abstractmethod
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """Extract features from input data.
        
        Args:
            data: Input data array
            
        Returns:
            numpy.ndarray: Extracted features
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get list of feature names.
        
        Returns:
            list: List of feature names
        """
        pass
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set feature extraction parameters.
        
        Args:
            params: Parameter dictionary
        """
        self.parameters.update(params)
        logger.debug(f"Updated parameters: {params}")
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters.
        
        Returns:
            dict: Parameter dictionary
        """
        return self.parameters.copy()

class FeatureSet:
    """Container for multiple feature extractors."""
    
    def __init__(self):
        """Initialize feature set."""
        self.extractors = {}
        self.enabled_extractors = set()
    
    def add_extractor(self, name: str, extractor: FeatureExtractor) -> None:
        """Add a feature extractor.
        
        Args:
            name: Extractor name
            extractor: FeatureExtractor instance
        """
        self.extractors[name] = extractor
        self.enabled_extractors.add(name)
        logger.debug(f"Added feature extractor: {name}")
    
    def enable_extractor(self, name: str) -> None:
        """Enable a feature extractor.
        
        Args:
            name: Extractor name
        """
        if name in self.extractors:
            self.enabled_extractors.add(name)
            logger.debug(f"Enabled feature extractor: {name}")
    
    def disable_extractor(self, name: str) -> None:
        """Disable a feature extractor.
        
        Args:
            name: Extractor name
        """
        self.enabled_extractors.discard(name)
        logger.debug(f"Disabled feature extractor: {name}")
    
    def extract_all_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract features using all enabled extractors.
        
        Args:
            data: Input data array
            
        Returns:
            dict: Dictionary mapping extractor names to feature arrays
        """
        try:
            features = {}
            for name in self.enabled_extractors:
                extractor = self.extractors[name]
                features[name] = extractor.extract_features(data)
                logger.debug(f"Extracted features using {name}")
            return features
            
        except Exception as e:
            logger.error("Error extracting features")
            logger.exception(e)
            raise
    
    def get_feature_info(self) -> Dict[str, List[str]]:
        """Get information about available features.
        
        Returns:
            dict: Dictionary mapping extractor names to feature lists
        """
        return {
            name: extractor.get_feature_names()
            for name, extractor in self.extractors.items()
        }
    
    def set_parameters(self, name: str, params: Dict[str, Any]) -> None:
        """Set parameters for a specific extractor.
        
        Args:
            name: Extractor name
            params: Parameter dictionary
        """
        if name in self.extractors:
            self.extractors[name].set_parameters(params)
            logger.debug(f"Set parameters for {name}: {params}")
        else:
            raise KeyError(f"Unknown extractor: {name}")
    
    def get_parameters(self, name: str) -> Dict[str, Any]:
        """Get parameters for a specific extractor.
        
        Args:
            name: Extractor name
            
        Returns:
            dict: Parameter dictionary
        """
        if name in self.extractors:
            return self.extractors[name].get_parameters()
        else:
            raise KeyError(f"Unknown extractor: {name}")
