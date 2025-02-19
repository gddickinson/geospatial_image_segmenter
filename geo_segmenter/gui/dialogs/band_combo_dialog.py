"""Dialog for selecting band combinations."""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QComboBox,
                           QLabel, QPushButton, QFormLayout)
import rasterio
import numpy as np

from ...utils.logger import setup_logger

logger = setup_logger(__name__)

class BandComboDialog(QDialog):
    """Dialog for selecting band combinations."""

    def __init__(self, layer, parent=None):
        """Initialize dialog.

        Args:
            layer: RasterLayer instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Band Combination")
        self.layer = layer

        # Get number of bands
        with rasterio.open(layer.path) as src:
            self.n_bands = src.count

        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Band selection
        form = QFormLayout()

        # RGB band selection
        self.band_combos = []
        for band_name in ["Red", "Green", "Blue"]:
            combo = QComboBox()
            combo.addItems([f"Band {i+1}" for i in range(self.n_bands)])
            form.addRow(f"{band_name} Band:", combo)
            self.band_combos.append(combo)

        # Set current values
        for i, combo in enumerate(self.band_combos):
            combo.setCurrentIndex(self.layer.band_indices[i] - 1)

        layout.addLayout(form)

        # Preset combinations
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Presets:"))

        preset_combo = QComboBox()
        preset_combo.addItems([
            "Natural Color (RGB)",
            "Color Infrared (NIR-R-G)",
            "False Color (SWIR-NIR-R)",
            "Agriculture (NIR-R-G)",
            "Atmospheric (SWIR-NIR-R)"
        ])
        preset_combo.currentTextChanged.connect(self.apply_preset)
        preset_layout.addWidget(preset_combo)

        layout.addLayout(preset_layout)

        # Buttons
        button_layout = QHBoxLayout()

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_bands)
        button_layout.addWidget(apply_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

    def apply_bands(self):
        """Apply selected band combination."""
        try:
            # Get selected bands (convert to 1-based indices)
            bands = [combo.currentIndex() + 1 for combo in self.band_combos]

            # Update layer
            self.layer.band_indices = bands

            # Refresh display
            if hasattr(self.parent(), 'map_canvas'):
                self.parent().map_canvas.update()

            self.accept()

        except Exception as e:
            logger.error("Error applying band combination")
            logger.exception(e)

    def apply_preset(self, preset: str):
        """Apply a preset band combination.

        Args:
            preset: Preset name
        """
        try:
            if preset == "Natural Color (RGB)":
                bands = [1, 2, 3]  # Assuming RGB are first three bands
            elif preset == "Color Infrared (NIR-R-G)":
                bands = [4, 1, 2]  # Assuming NIR is band 4
            elif preset == "False Color (SWIR-NIR-R)":
                bands = [5, 4, 1]  # Assuming SWIR is band 5
            elif preset == "Agriculture (NIR-R-G)":
                bands = [4, 1, 2]  # Same as Color Infrared
            elif preset == "Atmospheric (SWIR-NIR-R)":
                bands = [5, 4, 1]  # Same as False Color

            # Update combos if we have enough bands
            if max(bands) <= self.n_bands:
                for combo, band in zip(self.band_combos, bands):
                    combo.setCurrentIndex(band - 1)

        except Exception as e:
            logger.error("Error applying preset")
            logger.exception(e)
