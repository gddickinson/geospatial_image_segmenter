"""Dialog for Random Forest model configuration."""
from PyQt6.QtWidgets import (QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
                           QMessageBox)
from PyQt6.QtCore import Qt

from .model_dialog_base import ModelDialogBase
from ...utils.logger import setup_logger
from ... import config

logger = setup_logger(__name__)

class RandomForestDialog(ModelDialogBase):
    """Dialog for configuring Random Forest model."""
    
    def __init__(self, parent=None):
        """Initialize dialog.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent, "Random Forest Configuration")
        
        # Add model-specific parameters
        self.setup_rf_parameters()
        
        # Set default values
        self.reset_parameters()
    
    def setup_rf_parameters(self):
        """Set up Random Forest specific parameters."""
        try:
            # Basic parameters
            n_estimators = QSpinBox()
            n_estimators.setRange(1, 1000)
            n_estimators.setValue(config.RF_N_ESTIMATORS)
            n_estimators.setToolTip("Number of trees in the forest")
            self.add_parameter("n_estimators", n_estimators,
                             label="Number of Trees")
            
            max_depth = QSpinBox()
            max_depth.setRange(1, 100)
            max_depth.setValue(config.RF_MAX_DEPTH)
            max_depth.setToolTip("Maximum depth of trees")
            self.add_parameter("max_depth", max_depth,
                             label="Maximum Depth")
            
            min_samples_split = QSpinBox()
            min_samples_split.setRange(2, 20)
            min_samples_split.setValue(2)
            min_samples_split.setToolTip("Minimum samples required to split a node")
            self.add_parameter("min_samples_split", min_samples_split)
            
            min_samples_leaf = QSpinBox()
            min_samples_leaf.setRange(1, 20)
            min_samples_leaf.setValue(1)
            min_samples_leaf.setToolTip("Minimum samples required in a leaf node")
            self.add_parameter("min_samples_leaf", min_samples_leaf)
            
            # Feature selection
            max_features = QComboBox()
            max_features.addItems(['sqrt', 'log2', 'auto'])
            max_features.setCurrentText('sqrt')
            max_features.setToolTip("Number of features to consider for best split")
            self.add_parameter("max_features", max_features)
            
            # Advanced parameters
            class_weight = QComboBox()
            class_weight.addItems(['balanced', 'balanced_subsample', 'None'])
            class_weight.setCurrentText('balanced')
            class_weight.setToolTip("Class weighting scheme")
            self.add_parameter("class_weight", class_weight,
                             group="Advanced")
            
            bootstrap = QCheckBox()
            bootstrap.setChecked(True)
            bootstrap.setToolTip("Whether to use bootstrapping")
            self.add_parameter("bootstrap", bootstrap,
                             group="Advanced")
            
            oob_score = QCheckBox()
            oob_score.setChecked(False)
            oob_score.setToolTip("Whether to use out-of-bag samples")
            self.add_parameter("oob_score", oob_score,
                             group="Advanced",
                             label="Use OOB Score")
            
            # Training parameters
            validation_split = QDoubleSpinBox()
            validation_split.setRange(0.0, 0.5)
            validation_split.setValue(0.2)
            validation_split.setSingleStep(0.1)
            validation_split.setToolTip("Fraction of data to use for validation")
            self.add_parameter("validation_split", validation_split,
                             group="Advanced")
            
            n_jobs = QSpinBox()
            n_jobs.setRange(-1, 16)
            n_jobs.setValue(-1)
            n_jobs.setToolTip("Number of parallel jobs (-1 for all cores)")
            self.add_parameter("n_jobs", n_jobs,
                             group="Advanced")
            
        except Exception as e:
            logger.error("Error setting up Random Forest parameters")
            logger.exception(e)
            raise
    
    def reset_parameters(self):
        """Reset parameters to defaults."""
        try:
            default_params = {
                'n_estimators': config.RF_N_ESTIMATORS,
                'max_depth': config.RF_MAX_DEPTH,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'class_weight': 'balanced',
                'bootstrap': True,
                'oob_score': False,
                'validation_split': 0.2,
                'n_jobs': -1,
                'enabled_features': ['Spectral', 'Texture', 'Terrain']
            }
            
            self.set_parameters(default_params)
            logger.debug("Reset Random Forest parameters to defaults")
            
        except Exception as e:
            logger.error("Error resetting parameters")
            logger.exception(e)
            QMessageBox.critical(self, "Error",
                               "Failed to reset parameters to defaults")
    
    def validate_parameters(self) -> bool:
        """Validate parameter values.
        
        Returns:
            bool: True if parameters are valid
        """
        try:
            params = self.get_parameters()
            
            # Check required parameters
            if params['n_estimators'] < 1:
                raise ValueError("Number of trees must be at least 1")
            
            if params['max_depth'] < 1:
                raise ValueError("Maximum depth must be at least 1")
            
            if not params['enabled_features']:
                raise ValueError("At least one feature type must be enabled")
            
            return True
            
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Parameters", str(e))
            return False
            
        except Exception as e:
            logger.error("Error validating parameters")
            logger.exception(e)
            QMessageBox.critical(self, "Error",
                               "Failed to validate parameters")
            return False
    
    def accept(self):
        """Handle dialog acceptance."""
        if self.validate_parameters():
            super().accept()