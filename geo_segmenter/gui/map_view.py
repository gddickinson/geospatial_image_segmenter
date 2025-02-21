"""Map view widget for rendering map content."""
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QPaintEvent, QResizeEvent, QColor, QImage, QPixmap

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

        # Track connected layers to avoid duplicate connections
        self._connected_layers = set()

        # Default background
        palette = self.palette()
        palette.setColor(self.backgroundRole(), Qt.GlobalColor.white)
        self.setPalette(palette)

        # Schedule connecting to layer signals after initialization
        QTimer.singleShot(100, self.connect_to_layers)

    # Replace the paintEvent method in MapViewWidget


    def paintEvent(self, event: QPaintEvent):
        """Handle paint events."""
        try:
            logger.debug("\n==== MAP VIEW WIDGET PAINT EVENT START ====")

            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

            # Fill background
            painter.fillRect(event.rect(), Qt.GlobalColor.white)

            # Get layers from parent canvas
            canvas = self.parent()
            if canvas and hasattr(canvas, '_layers'):
                layers = canvas._layers

                # CRITICAL: Filter visible layers manually
                visible_layers = []
                for layer in layers:
                    # Check visibility state directly
                    if layer._visible:
                        visible_layers.append(layer)
                        logger.debug(f"Layer {layer.name} IS visible (will render)")
                    else:
                        logger.debug(f"Layer {layer.name} is NOT visible (will skip)")

                logger.debug(f"Found {len(visible_layers)} visible layers out of {len(layers)} total")

                # Render only visible layers
                for layer in visible_layers:
                    try:
                        logger.debug(f"Starting render of visible layer: {layer.name}")
                        layer.render(painter, canvas)
                        logger.debug(f"Completed rendering layer: {layer.name}")
                    except Exception as e:
                        logger.error(f"Error rendering layer {layer.name}: {str(e)}")
                        logger.exception(e)

            else:
                logger.debug("No layers to draw")

            painter.end()
            logger.debug("==== MAP VIEW WIDGET PAINT EVENT COMPLETED ====\n")

        except Exception as e:
            logger.error("Error in MapViewWidget paint event")
            logger.exception(e)


    # Add this method to explicitly render visible layers
    def render_visible_layers(self, force_render=False):
        """Explicitly render visible layers."""
        logger.debug("render_visible_layers called with force_render=%s", force_render)

        # Create a pixmap to render onto
        pixmap = QPixmap(self.size())
        pixmap.fill(Qt.GlobalColor.white)

        # Create painter
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Get canvas
        canvas = self.parent()
        if canvas and hasattr(canvas, '_layers'):
            # Render each visible layer
            for layer in canvas._layers:
                # Direct property access to avoid any issues with getters
                is_visible = layer._visible if force_render else layer.visible

                if is_visible:
                    logger.debug(f"Manually rendering {layer.name}, visible={is_visible}")
                    layer.render(painter, canvas)
                else:
                    logger.debug(f"Manually skipping {layer.name}, visible={is_visible}")

        painter.end()

        # Draw the pixmap to screen
        screen_painter = QPainter(self)
        screen_painter.drawPixmap(0, 0, pixmap)
        screen_painter.end()

        logger.debug("Manual rendering completed")

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
        """Draw the training label overlay."""
        try:
            if not label_dialog:
                logger.debug("No label dialog provided")
                return

            # Get all label masks
            label_masks = label_dialog.get_label_masks()
            logger.debug(f"Retrieved {len(label_masks)} label masks")

            if not label_masks:
                logger.debug("No label masks to draw")
                return

            for label_name, mask in label_masks.items():
                if mask is None:
                    logger.debug(f"Mask for {label_name} is None")
                    continue

                # Get label color
                label = label_dialog.labels[label_name]
                color = label.color

                # Log mask information
                non_zero = np.sum(mask) if mask is not None else 0
                logger.debug(f"Drawing mask for {label_name} with {non_zero} active pixels")

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
                logger.debug(f"Completed drawing overlay for {label_name}")

        except Exception as e:
            logger.error(f"Error drawing label overlay: {str(e)}")
            logger.exception(e)

    # Add this method to the MapViewWidget class to ensure a complete repaint
    def force_repaint(self):
        """Force a complete repaint of the map view."""
        try:
            logger.debug("Forcing complete repaint of MapViewWidget")
            # Invalidate the entire widget area
            self.update()

            # Log information about layers for debugging
            canvas = self.parent()
            if canvas and hasattr(canvas, '_layers'):
                layers = canvas._layers
                logger.debug(f"Current layers for repaint: {len(layers)}")
                for layer in layers:
                    logger.debug(f"  Layer: {layer.name}, Visible: {layer.visible}")

        except Exception as e:
            logger.error("Error forcing repaint")
            logger.exception(e)

    # Add this to MapViewWidget class to check if signals are properly connected
    def connect_layer_signals(self, canvas):
        """Connect to layer visibility signals."""
        if not hasattr(canvas, '_layers'):
            return

        logger.debug("Connecting to layer visibility signals")
        for layer in canvas._layers:
            # Create a custom slot for each layer
            def create_visibility_handler(layer_name):
                def handle_visibility(visible):
                    logger.debug(f"SIGNAL: Received visibility_changed signal from {layer_name}: {visible}")
                    self.update()  # Force update when visibility changes
                return handle_visibility

            # Connect the signal
            handler = create_visibility_handler(layer.name)
            layer.visibility_changed.connect(handler)
            logger.debug(f"Connected visibility signal for layer: {layer.name}")

    def connect_to_layers(self):
        """Connect to all layer signals."""
        try:
            logger.debug("Connecting to layer signals")
            canvas = self.parent()
            if not canvas or not hasattr(canvas, '_layers'):
                logger.debug("No canvas or layers found")
                return

            for layer in canvas._layers:
                self.connect_to_layer(layer)

        except Exception as e:
            logger.error("Error connecting to layers")
            logger.exception(e)

    def connect_to_layer(self, layer):
        """Connect to a single layer's signals."""
        # Skip if already connected
        layer_id = id(layer)
        if layer_id in self._connected_layers:
            logger.debug(f"Already connected to layer {layer.name}")
            return

        try:
            logger.debug(f"Connecting to signals for layer {layer.name}")

            # Connect to visibility changed signal
            layer.visibility_changed.connect(self.on_layer_visibility_changed)

            # Connect to general changed signal
            layer.changed.connect(self.on_layer_changed)

            # Mark as connected
            self._connected_layers.add(layer_id)
            logger.debug(f"Connected to layer {layer.name} signals")

        except Exception as e:
            logger.error(f"Error connecting to layer {layer.name}")
            logger.exception(e)

    def on_layer_visibility_changed(self, is_visible):
        """Handle layer visibility change."""
        try:
            logger.debug(f"Layer visibility changed to {is_visible}")
            # Force immediate repaint
            self.repaint()

            # Schedule another repaint to ensure UI is updated
            QTimer.singleShot(50, self.repaint)

        except Exception as e:
            logger.error("Error handling layer visibility change")
            logger.exception(e)

    def on_layer_changed(self):
        """Handle layer change."""
        try:
            logger.debug("Layer changed")
            # Force immediate repaint
            self.repaint()

        except Exception as e:
            logger.error("Error handling layer change")
            logger.exception(e)

    def manual_render(self):
        """Manually render only visible layers."""
        logger.debug("\n==== MANUAL RENDER START ====")

        # Create a new pixmap
        pixmap = QPixmap(self.size())
        pixmap.fill(Qt.GlobalColor.white)

        # Create painter on pixmap
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Get canvas and layers
        canvas = self.parent()
        if not canvas or not hasattr(canvas, '_layers'):
            logger.debug("No canvas or layers")
            painter.end()
            return

        # Get layers directly
        layers = canvas._layers

        # Track which layers we actually rendered
        rendered_layers = []
        skipped_layers = []

        # Render each layer
        for layer in layers:
            try:
                # Direct check of visibility
                if layer._visible:
                    logger.debug(f"Manually rendering layer {layer.name}")
                    layer.render(painter, canvas)
                    rendered_layers.append(layer.name)
                else:
                    logger.debug(f"Manually skipping layer {layer.name} (not visible)")
                    skipped_layers.append(layer.name)
            except Exception as e:
                logger.error(f"Error rendering layer {layer.name}: {e}")

        # End pixmap painter
        painter.end()

        # Draw the pixmap to screen
        screen_painter = QPainter(self)
        screen_painter.drawPixmap(0, 0, pixmap)
        screen_painter.end()

        logger.debug(f"Rendered layers: {rendered_layers}")
        logger.debug(f"Skipped layers: {skipped_layers}")
        logger.debug("==== MANUAL RENDER COMPLETE ====\n")
