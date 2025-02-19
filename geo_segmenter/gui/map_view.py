"""Map view widget for rendering map content."""
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPaintEvent, QResizeEvent, QColor, QImage

import numpy as np

from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class MapViewWidget(QWidget):
    """Widget for rendering map content."""

    def __init__(self, parent=None):
        """Initialize map view widget."""
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAutoFillBackground(True)

        # Default background
        palette = self.palette()
        palette.setColor(self.backgroundRole(), Qt.GlobalColor.white)
        self.setPalette(palette)

    def paintEvent(self, event: QPaintEvent):
        """Handle paint events."""
        try:
            logger.debug("MapViewWidget paint event started")

            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

            # Fill background
            painter.fillRect(event.rect(), Qt.GlobalColor.white)

            # Get layers from parent canvas
            canvas = self.parent()
            if canvas and hasattr(canvas, '_layers'):
                layers = canvas._layers
                logger.debug(f"Drawing {len(layers)} layers")

                if layers:  # Only proceed if there are layers
                    for layer in layers:
                        try:
                            if layer.visible:  # Only render if layer is visible
                                logger.debug(f"Starting render of layer: {layer.name} (visible={layer.visible})")
                                layer.render(painter, canvas)
                                logger.debug(f"Completed render of layer: {layer.name}")
                            else:
                                logger.debug(f"Skipping invisible layer: {layer.name}")
                        except Exception as e:
                            logger.error(f"Error rendering layer {layer.name}: {str(e)}")
                            logger.exception(e)
                else:
                    logger.debug("No layers to draw")

            # Draw label overlay if in training mode
            if hasattr(canvas, 'training_mode') and canvas.training_mode:
                main_window = canvas.parent()
                if main_window and hasattr(main_window, 'label_dialog'):
                    self.draw_label_overlay(painter, main_window.label_dialog)

            painter.end()
            logger.debug("MapViewWidget paint event completed")

        except Exception as e:
            logger.error("Error in MapViewWidget paint event")
            logger.exception(e)

    def resizeEvent(self, event: QResizeEvent):
        """Handle resize events."""
        try:
            super().resizeEvent(event)
            logger.debug(f"MapViewWidget resized to {event.size().width()}x{event.size().height()}")

            # Notify parent canvas of resize
            if hasattr(self.parent(), "zoom_to_layer") and hasattr(self.parent(), "_layers"):
                if self.parent()._layers:
                    # Re-zoom to current layer to maintain view
                    self.parent().zoom_to_layer(self.parent()._layers[0])

        except Exception as e:
            logger.error("Error in MapViewWidget resize event")
            logger.exception(e)

    def get_viewport_rect(self):
        """Get the viewport rectangle in widget coordinates."""
        return self.rect()

    def world_to_screen(self, x: float, y: float) -> tuple:
        """Convert world coordinates to screen coordinates.

        Args:
            x: World X coordinate
            y: World Y coordinate

        Returns:
            tuple: (screen_x, screen_y)
        """
        try:
            canvas = self.parent()
            if not canvas or not canvas.viewport_bounds:
                return (0, 0)

            bounds = canvas.viewport_bounds
            width = self.width()
            height = self.height()

            # Calculate fractions within viewport
            x_frac = (x - bounds[0]) / (bounds[2] - bounds[0])
            y_frac = (bounds[3] - y) / (bounds[3] - bounds[1])

            # Convert to screen coordinates
            screen_x = x_frac * width
            screen_y = y_frac * height

            return (screen_x, screen_y)

        except Exception as e:
            logger.error("Error converting world to screen coordinates")
            logger.exception(e)
            return (0, 0)

    def screen_to_world(self, screen_x: float, screen_y: float) -> tuple:
        """Convert screen coordinates to world coordinates.

        Args:
            screen_x: Screen X coordinate
            screen_y: Screen Y coordinate

        Returns:
            tuple: (world_x, world_y)
        """
        try:
            canvas = self.parent()
            if not canvas or not canvas.viewport_bounds:
                return (0, 0)

            bounds = canvas.viewport_bounds
            width = self.width()
            height = self.height()

            # Calculate fractions within viewport
            x_frac = screen_x / width
            y_frac = screen_y / height

            # Convert to world coordinates
            world_x = bounds[0] + x_frac * (bounds[2] - bounds[0])
            world_y = bounds[1] + (1 - y_frac) * (bounds[3] - bounds[1])

            return (world_x, world_y)

        except Exception as e:
            logger.error("Error converting screen to world coordinates")
            logger.exception(e)
            return (0, 0)

    def draw_label_overlay(self, painter: QPainter, label_dialog):
        """Draw the training label overlay.

        Args:
            painter: QPainter instance
            label_dialog: LabelingDialog instance with label information
        """
        try:
            if not label_dialog or not hasattr(self.parent(), 'training_mode') or not self.parent().training_mode:
                return

            # Get all label masks
            label_masks = label_dialog.get_label_masks()
            if not label_masks:
                return

            for label_name, mask in label_masks.items():
                if mask is None:
                    continue

                # Get label color
                label = label_dialog.labels[label_name]
                color = label.color

                # Create semi-transparent color for overlay
                overlay_color = QColor(color.red(), color.green(), color.blue(), 128)

                # Create image from mask
                height, width = mask.shape
                mask_image = QImage(width, height, QImage.Format.Format_ARGB32)
                mask_image.fill(Qt.GlobalColor.transparent)

                # Draw mask
                mask_painter = QPainter(mask_image)
                mask_painter.setPen(Qt.PenStyle.NoPen)
                mask_painter.setBrush(overlay_color)

                # Convert boolean mask to coordinates
                y_coords, x_coords = np.where(mask)
                for x, y in zip(x_coords, y_coords):
                    mask_painter.drawRect(x, y, 1, 1)

                mask_painter.end()

                # Draw the mask image onto the map
                painter.drawImage(0, 0, mask_image)

        except Exception as e:
            logger.error("Error drawing label overlay")
            logger.exception(e)
