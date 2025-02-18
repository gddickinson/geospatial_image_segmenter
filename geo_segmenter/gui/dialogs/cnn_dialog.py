"""Dialog for CNN model configuration."""
from PyQt6.QtWidgets import (QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
                           QMessageBox, QGroupBox, QVBoxLayout, QLabel)
from PyQt6.QtCore import Qt

from .model_dialog_base import ModelDialogBase
from ...utils.logger import setup_logger
from ... import config

logger = setup_logger(__name__)

class CNNDialog(ModelDialogBase):
    """Dialog for configuring CNN model."""
    
    def __init__(self, parent=None):
        """Initialize dialog.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent, "CNN Configuration")
        
        # Add model-specific parameters
        self.setup_cnn_parameters()
        
        # Set default values
        self.reset_parameters()
    
    def setup_cnn_parameters(self):
        """Set up CNN specific parameters."""
        try:
            # Architecture parameters
            patch_size = QSpinBox()
            patch_size.setRange(64, 512)
            patch_size.setValue(256)
            patch_size.setSingleStep(32)
            patch_size.setToolTip("Size of image patches for training")
            self.add_parameter("patch_size", patch_size,
                             label="Patch Size")
            
            batch_size = QSpinBox()
            batch_size.setRange(1, 32)
            batch_size.setValue(config.CNN_BATCH_SIZE)
            batch_size.setToolTip("Training batch size")
            self.add_parameter("batch_size", batch_size,
                             label="Batch Size")
            
            # Training parameters
            learning_rate = QDoubleSpinBox()
            learning_rate.setRange(0.0001, 0.1)
            learning_rate.setValue(config.CNN_LEARNING_RATE)
            learning_rate.setDecimals(4)
            learning_rate.setSingleStep(0.0001)
            learning_rate.setToolTip("Learning rate for training")
            self.add_parameter("learning_rate", learning_rate)
            
            n_epochs = QSpinBox()
            n_epochs.setRange(1, 200)
            n_epochs.setValue(config.CNN_EPOCHS)
            n_epochs.setToolTip("Number of training epochs")
            self.add_parameter("n_epochs", n_epochs,
                             label="Epochs")
            
            # Optimizer selection
            optimizer = QComboBox()
            optimizer.addItems(['Adam', 'SGD', 'RMSprop'])
            optimizer.setCurrentText('Adam')
            optimizer.setToolTip("Optimization algorithm")
            self.add_parameter("optimizer", optimizer)
            
            # Advanced parameters
            validation_split = QDoubleSpinBox()
            validation_split.setRange(0.0, 0.5)
            validation_split.setValue(0.2)
            validation_split.setSingleStep(0.1)
            validation_split.setToolTip("Fraction of data to use for validation")
            self.add_parameter("validation_split", validation_split,
                             group="Advanced")
            
            early_stopping = QCheckBox()
            early_stopping.setChecked(True)
            early_stopping.setToolTip("Enable early stopping")
            self.add_parameter("early_stopping", early_stopping,
                             group="Advanced")
            
            patience = QSpinBox()
            patience.setRange(1, 20)
            patience.setValue(5)
            patience.setToolTip("Number of epochs to wait before early stopping")
            self.add_parameter("patience", patience,
                             group="Advanced")
            
            use_augmentation = QCheckBox()
            use_augmentation.setChecked(True)
            use_augmentation.setToolTip("Enable data augmentation")
            self.add_parameter("use_augmentation", use_augmentation,
                             group="Advanced",
                             label="Data Augmentation")
            
            # Hardware settings
            use_gpu = QCheckBox()
            use_gpu.setChecked(True)
            use_gpu.setToolTip("Use GPU for training if available")
            self.add_parameter("use_gpu", use_gpu,
                             group="Advanced")
            
            mixed_precision = QCheckBox()
            mixed_precision.setChecked(True)
            mixed_precision.setToolTip("Use mixed precision training")
            self.add_parameter("mixed_precision", mixed_precision,
                             group="Advanced")
            
        except Exception as e:
            logger.error("Error setting up CNN parameters")
            logger.exception(e)
            raise
    
    def setup_features_tab(self):
        """Override features tab setup for CNN."""
        tab = super().setup_features_tab()
        
        # Add note about feature channel concatenation
        note = QLabel(
            "Note: All enabled features will be concatenated as input channels "
            "to the CNN. More features will increase memory usage."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        tab.layout().insertWidget(1, note)
    
    def reset_parameters(self):
        """Reset parameters to defaults."""
        try:
            default_params = {
                'patch_size': 256,
                'batch_size': config.CNN_BATCH_SIZE,
                'learning_rate': config.CNN_LEARNING_RATE,
                'n_epochs': config.CNN_EPOCHS,
                'optimizer': 'Adam',
                'validation_split': 0.2,
                'early_stopping': True,
                'patience': 5,
                'use_augmentation': True,
                'use_gpu': True,
                'mixed_precision': True,
                'enabled_features': ['Spectral', 'Texture', 'Terrain']
            }
            
            self.set_parameters(default_params)
            logger.debug("Reset CNN parameters to defaults")
            
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
            if params['batch_size'] < 1:
                raise ValueError("Batch size must be at least 1")
            
            if params['n_epochs'] < 1:
                raise ValueError("Number of epochs must be at least 1")
            
            if params['learning_rate'] <= 0:
                raise ValueError("Learning rate must be positive")
            
            if not params['enabled_features']:
                raise ValueError("At least one feature type must be enabled")
            
            # Memory estimation
            total_features = sum(1 for _ in params['enabled_features'])
            memory_estimate = (
                params['batch_size'] *  # Batch size
                params['patch_size'] * params['patch_size'] *  # Spatial dimensions
                total_features *  # Number of features
                4  # Bytes per float32
            ) / (1024 * 1024)  # Convert to MB
            
            if memory_estimate > 4096:  # Warning threshold: 4GB
                result = QMessageBox.warning(
                    self,
                    "High Memory Usage",
                    f"The current configuration may require approximately "
                    f"{memory_estimate:.1f}MB of GPU memory per batch. "
                    f"Continue anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if result == QMessageBox.StandardButton.No:
                    return False
            
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