"""Dialog for adjusting raster stretch parameters."""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QComboBox,
                           QLabel, QPushButton, QDoubleSpinBox)
from PyQt6.QtCore import Qt
import numpy as np
import rasterio

from ...utils.logger import setup_logger

logger = setup_logger(__name__)

class StretchDialog(QDialog):
    """Dialog for controlling raster stretching."""

    def __init__(self, layer, parent=None):
        """Initialize dialog.

        Args:
            layer: RasterLayer instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Stretch Parameters")
        self.layer = layer
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Stretch type selection
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Stretch Type:"))

        self.type_combo = QComboBox()
        self.type_combo.addItems([
            "None",
            "Linear",
            "Linear 2%",
            "Linear 5%",
            "Standard Deviation"
        ])
        self.type_combo.currentTextChanged.connect(self.update_stretch)
        type_layout.addWidget(self.type_combo)
        layout.addLayout(type_layout)

        # Min/Max controls
        min_layout = QHBoxLayout()
        min_layout.addWidget(QLabel("Min:"))
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setRange(-1e6, 1e6)
        self.min_spin.valueChanged.connect(self.update_stretch)
        min_layout.addWidget(self.min_spin)
        layout.addLayout(min_layout)

        max_layout = QHBoxLayout()
        max_layout.addWidget(QLabel("Max:"))
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(-1e6, 1e6)
        self.max_spin.valueChanged.connect(self.update_stretch)
        max_layout.addWidget(self.max_spin)
        layout.addLayout(max_layout)

        # Buttons
        button_layout = QHBoxLayout()

        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self.reset_stretch)
        button_layout.addWidget(reset_btn)

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.accept)
        button_layout.addWidget(apply_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

        # Initialize values
        self.reset_stretch()

    def update_stretch(self):
        """Update layer stretch parameters."""
        try:
            stretch_type = self.type_combo.currentText()

            if stretch_type == "None":
                self.layer.stretch_params = None
            else:
                self.layer.stretch_params = {
                    'type': stretch_type,
                    'min': self.min_spin.value(),
                    'max': self.max_spin.value()
                }

            # Update display
            if hasattr(self.parent(), 'map_canvas'):
                self.parent().map_canvas.update()

        except Exception as e:
            logger.error("Error updating stretch")
            logger.exception(e)

    def reset_stretch(self):
        """Reset stretch parameters to defaults."""
        try:
            # Get data min/max
            with rasterio.open(self.layer.path) as src:
                data = src.read()
                data_min = float(np.nanmin(data))
                data_max = float(np.nanmax(data))

            self.min_spin.setValue(data_min)
            self.max_spin.setValue(data_max)
            self.type_combo.setCurrentText("Linear 2%")
            self.update_stretch()

        except Exception as e:
            logger.error("Error resetting stretch")
            logger.exception(e)
