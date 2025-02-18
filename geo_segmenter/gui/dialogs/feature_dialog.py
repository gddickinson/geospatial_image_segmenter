"""Dialog for feature configuration and visualization."""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
                           QPushButton, QGroupBox, QLabel, QSpinBox,
                           QDoubleSpinBox, QComboBox, QFormLayout)
from PyQt6.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from ...models.features import SpectralFeatureExtractor, TextureFeatureExtractor, TerrainFeatureExtractor
from ...utils.logger import setup_logger

logger = setup_logger(__name__)

class FeatureDialog(QDialog):
    """Dialog for configuring and visualizing features."""
    
    def __init__(self, image, parent=None):
        """Initialize dialog.
        
        Args:
            image: Input image data
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Feature Configuration")
        self.setModal(True)
        self.resize(1000, 600)
        
        self.image = image
        self.feature_extractors = {
            'Spectral': SpectralFeatureExtractor(),
            'Texture': TextureFeatureExtractor(),
            'Terrain': TerrainFeatureExtractor()
        }
        
        self.setup_ui()
        self.update_preview()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Create tabs
        tabs = QTabWidget()
        
        # Feature configuration tabs
        for name, extractor in self.feature_extractors.items():
            tab = self.create_feature_tab(name, extractor)
            tabs.addTab(tab, name)
        
        layout.addWidget(tabs)
        
        # Preview section
        preview_group = QGroupBox("Feature Preview")
        preview_layout = QVBoxLayout()
        
        # Matplotlib figure for preview
        self.figure, self.axes = plt.subplots(2, 3, figsize=(12, 8))
        self.canvas = FigureCanvasQTAgg(self.figure)
        preview_layout.addWidget(self.canvas)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        update_btn = QPushButton("Update Preview")
        update_btn.clicked.connect(self.update_preview)
        button_layout.addWidget(update_btn)
        
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        
        layout.addLayout(button_layout)
    
    def create_feature_tab(self, name, extractor):
        """Create tab for feature configuration.
        
        Args:
            name: Feature type name
            extractor: Feature extractor instance
            
        Returns:
            QWidget: Tab widget
        """
        tab = QWidget()
        layout = QFormLayout(tab)
        
        # Add controls for each parameter
        params = extractor.get_parameters()
        controls = {}
        
        for param_name, value in params.items():
            if isinstance(value, float):
                control = QDoubleSpinBox()
                control.setRange(0.0, 100.0)
                control.setValue(value)
                control.setDecimals(3)
            elif isinstance(value, int):
                control = QSpinBox()
                control.setRange(0, 1000)
                control.setValue(value)
            elif isinstance(value, str):
                control = QComboBox()
                control.addItems([value])
                control.setCurrentText(value)
            else:
                continue
            
            controls[param_name] = control
            layout.addRow(param_name.replace('_', ' ').title(), control)
        
        extractor.controls = controls
        return tab
    
    def update_preview(self):
        """Update feature preview."""
        try:
            # Clear axes
            for ax in self.axes.flat:
                ax.clear()
            
            # Calculate and display features
            for idx, (name, extractor) in enumerate(self.feature_extractors.items()):
                # Update parameters from controls
                params = {}
                for param_name, control in extractor.controls.items():
                    if isinstance(control, (QSpinBox, QDoubleSpinBox)):
                        params[param_name] = control.value()
                    elif isinstance(control, QComboBox):
                        params[param_name] = control.currentText()
                
                extractor.set_parameters(params)
                
                # Extract features
                features = extractor.extract_features(self.image)
                
                # Display first feature
                if features.ndim == 3 and features.shape[0] > 0:
                    feature = features[0]
                    self.axes[idx//3, idx%3].imshow(feature, cmap='viridis')
                    self.axes[idx//3, idx%3].set_title(f"{name}\n{extractor.get_feature_names()[0]}")
                    self.axes[idx//3, idx%3].axis('off')
            
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            logger.error("Error updating feature preview")
            logger.exception(e)
    
    def get_feature_extractors(self):
        """Get configured feature extractors.
        
        Returns:
            dict: Dictionary of configured feature extractors
        """
        return self.feature_extractors