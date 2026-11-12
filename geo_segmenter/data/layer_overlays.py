"""SegmentationLayer (classification result) and TrainingLabelsLayer (painted training labels)."""
from typing import Dict
import numpy as np
from PyQt6.QtGui import QPainter, QImage, QColor
from PyQt6.QtCore import QRectF, Qt
from PIL import Image
from ..utils.logger import setup_logger
from .layer_base import Layer

logger = setup_logger(__name__)


class SegmentationLayer(Layer):
    """Layer for displaying segmentation/classification results."""

    def __init__(self, name: str, data: np.ndarray, class_colors: Dict[str, QColor], extent: tuple):
        """Initialize segmentation layer.

        Args:
            name: Layer name
            data: Segmentation mask (2D array of class indices)
            class_colors: Dictionary mapping class names to colors
            extent: Layer extent (min_x, min_y, max_x, max_y)
        """
        super().__init__(name)
        self.data = data
        self.class_colors = class_colors
        self._extent = extent

        # Create QImage for rendering
        self._create_image()

    def _create_image(self):
        """Create QImage from segmentation data."""
        height, width = self.data.shape
        self.image = QImage(width, height, QImage.Format.Format_ARGB32)
        self.image.fill(0)  # Initialize transparent

        # Create mapping of indices to colors
        class_indices = {i: color for i, (_, color) in enumerate(self.class_colors.items())}

        # Create array of pixels
        pixels = np.zeros((height, width, 4), dtype=np.uint8)
        for idx, color in class_indices.items():
            mask = self.data == idx
            pixels[mask] = [color.red(), color.green(), color.blue(), 128]  # Semi-transparent

        # Convert to QImage
        for y in range(height):
            for x in range(width):
                self.image.setPixelColor(x, y, QColor(*pixels[y, x]))

    def extent(self) -> tuple:
        """Get layer extent."""
        return self._extent

    def render(self, painter: QPainter, map_canvas) -> None:
        """Render segmentation results.

        Args:
            painter: QPainter object
            map_canvas: MapCanvas widget
        """
        if not self.visible:
            return

        try:
            # Get viewport dimensions
            width = map_canvas.width()
            height = map_canvas.height()

            # Calculate display size and position
            viewport_bounds = map_canvas.viewport_bounds
            if viewport_bounds:
                # Calculate scale factors
                world_width = viewport_bounds[2] - viewport_bounds[0]
                world_height = viewport_bounds[3] - viewport_bounds[1]

                if world_width > 0 and world_height > 0:
                    # Calculate image position in viewport
                    x_scale = width / world_width
                    y_scale = height / world_height

                    img_x = (self._extent[0] - viewport_bounds[0]) * x_scale
                    img_y = (viewport_bounds[3] - self._extent[3]) * y_scale
                    img_width = (self._extent[2] - self._extent[0]) * x_scale
                    img_height = (self._extent[3] - self._extent[1]) * y_scale

                    # Set opacity
                    painter.setOpacity(self.opacity)

                    # Draw image
                    target_rect = QRectF(img_x, img_y, img_width, img_height)
                    painter.drawImage(target_rect, self.image)

        except Exception as e:
            logger.error(f"Error rendering segmentation layer: {str(e)}")
            logger.exception(e)


class TrainingLabelsLayer(Layer):
    """Layer for displaying training labels and masks."""

    def __init__(self, name: str, shape: tuple):
        """Initialize training labels layer.

        Args:
            name: Layer name
            shape: Image shape (height, width)
        """
        super().__init__(name)
        self.shape = shape
        self.label_masks = {}  # Dictionary mapping label names to masks
        self.label_colors = {}  # Dictionary mapping label names to colors
        self.extent = None
        self.transform = None  # Store the transform to ensure consistent coordinates
        logger.debug(f"Initialized TrainingLabelsLayer with shape {shape}")

    def set_transform(self, transform):
        """Set the geotransform for coordinate consistency.

        Args:
            transform: Rasterio transform object
        """
        self.transform = transform
        logger.debug(f"Set transform for TrainingLabelsLayer: {transform}")

    # Add this method to the TrainingLabelsLayer class to help diagnose issues

    def update_labels(self, labels: Dict):
        """Update label data."""
        try:
            # Store masks and colors
            self.label_masks = {name: label.mask.copy() if label.mask is not None else None
                              for name, label in labels.items()}
            self.label_colors = {name: label.color for name, label in labels.items()}

            # Enhanced logging
            logger.debug(f"Updated {len(self.label_masks)} label masks in {self.name}:")
            for name, mask in self.label_masks.items():
                if mask is not None:
                    pixel_count = np.sum(mask)
                    logger.debug(f"  Label '{name}': {pixel_count} pixels")
                else:
                    logger.debug(f"  Label '{name}': No mask")

            # Invalidate any caches
            if hasattr(self, '_cache'):
                self._cache_valid = False
                self._cache.clear()

        except Exception as e:
            logger.error(f"Error updating training labels: {e}")
            logger.exception(e)

    def render(self, painter: QPainter, map_canvas) -> None:
        """Render training labels."""
        # CRITICAL: First check visibility directly - must be first line
        if not self._visible:
            logger.debug(f"SKIPPING RENDER for training labels layer {self.name} - not visible (_visible=False)")
            return

        try:
            logger.debug(f"Rendering training labels layer: {self.name}")

            # Get viewport dimensions
            width = map_canvas.width()
            height = map_canvas.height()

            # Calculate display size and position
            viewport_bounds = map_canvas.viewport_bounds
            if not viewport_bounds or not self.extent:
                logger.debug("Missing viewport bounds or extent, cannot render training labels")
                return

            # Calculate scale factors exactly the same as in RasterLayer
            world_width = viewport_bounds[2] - viewport_bounds[0]
            world_height = viewport_bounds[3] - viewport_bounds[1]

            if world_width <= 0 or world_height <= 0:
                logger.debug("Invalid world dimensions, cannot render training labels")
                return

            # Calculate image position in viewport exactly like RasterLayer
            x_scale = width / world_width
            y_scale = height / world_height

            img_x = (self.extent[0] - viewport_bounds[0]) * x_scale
            img_y = (viewport_bounds[3] - self.extent[3]) * y_scale
            img_width = (self.extent[2] - self.extent[0]) * x_scale
            img_height = (self.extent[3] - self.extent[1]) * y_scale

            logger.debug(f"Training labels position: ({img_x}, {img_y}), dimensions: {img_width}x{img_height}")

            # Set opacity
            old_opacity = painter.opacity()
            painter.setOpacity(self.opacity)

            # Draw directly on the painter without using an image
            for name, mask in self.label_masks.items():
                if mask is not None and np.any(mask):
                    color = self.label_colors[name]
                    overlay_color = QColor(color.red(), color.green(), color.blue(), 160)

                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(overlay_color)

                    # Calculate pixel size
                    pixel_width = img_width / self.shape[1]  # width
                    pixel_height = img_height / self.shape[0]  # height

                    # Draw each pixel directly - precision is critical here
                    y_coords, x_coords = np.where(mask)

                    # Log some details about the mask
                    logger.debug(f"Label {name}: Drawing {len(x_coords)} pixels")

                    # Draw each pixel
                    for x, y in zip(x_coords, y_coords):
                        pixel_x = img_x + x * pixel_width
                        pixel_y = img_y + y * pixel_height
                        painter.drawRect(QRectF(pixel_x, pixel_y, pixel_width, pixel_height))

            # Restore original opacity
            painter.setOpacity(old_opacity)

            logger.debug(f"Completed rendering {len(self.label_masks)} label masks")

        except Exception as e:
            logger.error(f"Error rendering training labels layer: {str(e)}")
            logger.exception(e)
