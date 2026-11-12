"""Map canvas widget for displaying and interacting with geospatial data.

Coordinate transforms are in map_canvas_coords.py and measurement / label overlays in map_canvas_overlays.py
(mixins of MapCanvas).
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout
from PyQt6.QtCore import Qt

from ..utils.logger import setup_logger
from .. import config
from ..gui.map_view import MapViewWidget
from .map_canvas_coords import MapCanvasCoordsMixin
from .map_canvas_overlays import MapCanvasOverlaysMixin

logger = setup_logger(__name__)

class MapCanvas(MapCanvasCoordsMixin, MapCanvasOverlaysMixin, QWidget):
    """Widget for displaying and interacting with map data."""

    def __init__(self, parent=None):
        """Initialize the map canvas."""
        super().__init__(parent)

        self.training_mode = False

        # Initialize state
        self._layers = []  # Use list to maintain layer order
        self.center = config.DEFAULT_CENTER
        self.zoom = config.DEFAULT_ZOOM
        self.scale = 1.0
        self.viewport_bounds = None

        # Mouse tracking
        self.setMouseTracking(True)
        self.last_mouse_pos = None
        self.is_panning = False

        # Initialize UI
        self.setup_ui()

        logger.debug("MapCanvas initialized")

    def setup_ui(self):
        """Set up the user interface."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create map view widget
        self.map_view = MapViewWidget(self)
        layout.addWidget(self.map_view, 1)  # Add with stretch

        # Status bar
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        status_layout.setContentsMargins(2, 2, 2, 2)

        self.coord_label = QLabel("Coordinates: ")
        status_layout.addWidget(self.coord_label)

        status_layout.addStretch()

        self.scale_label = QLabel("Scale: 1:1")
        status_layout.addWidget(self.scale_label)

        layout.addWidget(status_widget)

    def set_tool(self, tool: str):
        """Set the current map tool."""
        try:
            # Update training mode flag
            self.training_mode = (tool == "training")
            logger.debug(f"Set training mode to {self.training_mode}")

            self.current_tool = tool
            self.measuring_points = []
            self.update()
            logger.debug(f"Set map tool to: {tool}")

        except Exception as e:
            logger.error("Error setting map tool")
            logger.exception(e)

    def add_layer(self, layer) -> None:
        """Add a layer to the map."""
        try:
            # Add to internal list if not already present
            if layer not in self._layers:
                self._layers.append(layer)
                logger.debug(f"Added layer to map canvas: {layer.name}")
                logger.debug(f"Current layer count: {len(self._layers)}")

                # For debugging - log all layers
                for i, l in enumerate(self._layers):
                    logger.debug(f"Layer[{i}]: {l.name}, visible={l.visible}")

                # Initial zoom to first layer
                if len(self._layers) == 1:
                    self.zoom_to_layer(layer)

                # Force a repaint
                if hasattr(self, 'map_view'):
                    self.map_view.update()

        except Exception as e:
            logger.error("Error adding layer to map canvas")
            logger.exception(e)

    def remove_layer(self, layer) -> None:
        """Remove a layer from the map."""
        try:
            # Make sure we're using the correct attribute name
            if hasattr(self, '_layers') and layer in self._layers:
                logger.debug(f"Removing layer from map canvas: {layer.name}")
                self._layers.remove(layer)

                # For debugging - log remaining layers
                logger.debug("Remaining layers:")
                for i, l in enumerate(self._layers):
                    logger.debug(f"Layer[{i}]: {l.name}, visible={l.visible}")

                # Force a repaint
                self.update()
                logger.debug(f"Removed layer: {layer.name}")
            else:
                logger.debug(f"Layer {layer.name} not found in layers list or wrong attribute name")

        except Exception as e:
            logger.error("Error removing layer")
            logger.exception(e)
            raise

    def zoom_to_layer(self, layer) -> None:
        """Zoom to show a specific layer."""
        try:
            extent = layer.extent()
            if extent:
                logger.debug(f"Zooming to layer extent: {extent}")
                # Calculate viewport bounds with padding
                x_min, y_min, x_max, y_max = extent
                x_pad = (x_max - x_min) * 0.05
                y_pad = (y_max - y_min) * 0.05

                self.viewport_bounds = (
                    x_min - x_pad,
                    y_min - y_pad,
                    x_max + x_pad,
                    y_max + y_pad
                )

                # Update display
                if hasattr(self, 'map_view'):
                    self.map_view.update()
                logger.debug(f"Set viewport bounds: {self.viewport_bounds}")

        except Exception as e:
            logger.error("Error zooming to layer")
            logger.exception(e)

    def zoom_to_extent(self, extent: tuple):
        """Zoom to a specific extent.
        Args:
            extent: (min_x, min_y, max_x, max_y)
        """
        try:
            self.viewport_bounds = extent

            # Calculate scale
            world_width = extent[2] - extent[0]
            world_height = extent[3] - extent[1]
            view_width = self.width()
            view_height = self.height()

            if view_width > 0 and view_height > 0:
                x_scale = world_width / view_width
                y_scale = world_height / view_height
                self.scale = max(x_scale, y_scale)

                # Update display
                self.update()
                self.update_scale_label()
                logger.debug(f"Zoomed to extent: {extent}")

                # Update map view if it exists and synced
                if hasattr(self.parent(), 'map_widget'):
                    center_lat = (extent[1] + extent[3]) / 2
                    center_lon = (extent[0] + extent[2]) / 2
                    self.parent().map_widget.set_center(center_lat, center_lon)

                    # Calculate appropriate zoom level
                    zoom = self.parent().calculate_zoom_level(world_width, world_height)
                    self.parent().map_widget.zoom_level = zoom
                    self.parent().map_widget.update()

        except Exception as e:
            logger.error("Error zooming to extent")
            logger.exception(e)

    def mousePressEvent(self, event):
        """Handle mouse press events."""
        if hasattr(self, 'training_mode') and self.training_mode and event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            # Pass event to paint tool when in training mode and shift is held
            super().mousePressEvent(event)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        """Handle mouse move events."""
        try:
            if hasattr(self, 'training_mode') and self.training_mode and event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                # Pass event to paint tool when in training mode and shift is held
                super().mouseMoveEvent(event)
                return

            # Convert coordinates for display
            if self.viewport_bounds:
                # Get mouse position relative to map view
                pos = self.map_view.mapFrom(self, event.pos())

                # Convert to world coordinates
                width = self.map_view.width()
                height = self.map_view.height()

                if width > 0 and height > 0:
                    x_frac = pos.x() / width
                    y_frac = pos.y() / height
                    world_x = self.viewport_bounds[0] + x_frac * (self.viewport_bounds[2] - self.viewport_bounds[0])
                    world_y = self.viewport_bounds[1] + (1 - y_frac) * (self.viewport_bounds[3] - self.viewport_bounds[1])
                    self.coord_label.setText(f"X: {world_x:.1f}, Y: {world_y:.1f}")

            # Handle panning
            if self.is_panning and self.last_mouse_pos:
                dx = event.pos().x() - self.last_mouse_pos.x()
                dy = event.pos().y() - self.last_mouse_pos.y()

                if self.viewport_bounds:
                    # Convert screen distance to world distance
                    width = self.map_view.width()
                    height = self.map_view.height()
                    world_width = self.viewport_bounds[2] - self.viewport_bounds[0]
                    world_height = self.viewport_bounds[3] - self.viewport_bounds[1]

                    world_dx = -(dx * world_width / width)
                    world_dy = dy * world_height / height

                    # Update viewport bounds
                    self.viewport_bounds = (
                        self.viewport_bounds[0] + world_dx,
                        self.viewport_bounds[1] + world_dy,
                        self.viewport_bounds[2] + world_dx,
                        self.viewport_bounds[3] + world_dy
                    )

                    # Update map view if exists and synced
                    if hasattr(self.parent(), 'map_widget'):
                        center_lat = (self.viewport_bounds[1] + self.viewport_bounds[3]) / 2
                        center_lon = (self.viewport_bounds[0] + self.viewport_bounds[2]) / 2
                        self.parent().map_widget.set_center(center_lat, center_lon)

                    self.map_view.update()

            self.last_mouse_pos = event.pos()

        except Exception as e:
            logger.error("Error handling mouse move")
            logger.exception(e)

    def update_scale_label(self):
        """Update the scale display label."""
        try:
            if self.scale != 0:
                scale_text = f"Scale: 1:{int(1/self.scale):,}"
                self.scale_label.setText(scale_text)

        except Exception as e:
            logger.error("Error updating scale label")
            logger.exception(e)

    def cleanup(self):
        """Clean up resources."""
        try:
            # Clear layers
            if hasattr(self, '_layers'):
                self._layers.clear()
            logger.debug("MapCanvas cleanup completed")

        except Exception as e:
            logger.error("Error during cleanup")
            logger.exception(e)

    def paintEvent(self, event) -> None:
        """Handle paint events."""
        try:
            logger.debug(f"Starting paint event for {len(self._layers)} layers")

            # Paint on the map view
            if hasattr(self, 'map_view'):
                self.map_view.update()

            logger.debug("Paint event completed")

        except Exception as e:
            logger.error("Error in paint event")
            logger.exception(e)

    def resizeEvent(self, event) -> None:
        """Handle resize events."""
        super().resizeEvent(event)
        logger.debug(f"Canvas resized to {self.width()}x{self.height()}")
        if self.viewport_bounds:
            self.update()  # Force redraw with new size

    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming."""
        try:
            if self.viewport_bounds:
                # Get mouse position in world coordinates
                pos = self.map_view.mapFrom(self, event.position())
                world_x, world_y = self.map_view.screen_to_world(pos.x(), pos.y())

                # Calculate zoom factor
                zoom_in = event.angleDelta().y() > 0
                factor = 1.2 if zoom_in else 1/1.2

                # Calculate new bounds
                width = self.viewport_bounds[2] - self.viewport_bounds[0]
                height = self.viewport_bounds[3] - self.viewport_bounds[1]

                new_width = width / factor
                new_height = height / factor

                # Center new bounds on mouse position
                mouse_frac_x = (world_x - self.viewport_bounds[0]) / width
                mouse_frac_y = (world_y - self.viewport_bounds[1]) / height

                self.viewport_bounds = (
                    world_x - mouse_frac_x * new_width,
                    world_y - mouse_frac_y * new_height,
                    world_x + (1 - mouse_frac_x) * new_width,
                    world_y + (1 - mouse_frac_y) * new_height
                )

                # Update scale
                self.scale *= factor
                self.update_scale_label()

                self.map_view.update()

        except Exception as e:
            logger.error("Error handling wheel event")
            logger.exception(e)


    # Add this method to your MapCanvas class
    def force_complete_repaint(self):
        """Force a complete repaint of all layers."""
        logger.debug("FORCE COMPLETE REPAINT: manually redrawing all layers")

        # Get all layers
        visible_layers = [layer for layer in self._layers if layer._visible]
        invisible_layers = [layer for layer in self._layers if not layer._visible]

        logger.debug(f"Found {len(visible_layers)} visible and {len(invisible_layers)} invisible layers")
        for layer in visible_layers:
            logger.debug(f"- Visible: {layer.name}")
        for layer in invisible_layers:
            logger.debug(f"- Invisible: {layer.name}")

        # Force the view to redraw
        if hasattr(self, 'map_view'):
            self.map_view.repaint()  # Immediate repaint

        # Force ourselves to redraw too
        self.repaint()
