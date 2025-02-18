"""Map canvas widget for displaying and interacting with geospatial data."""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt6.QtGui import QPainter, QPen, QColor, QMouseEvent

import numpy as np
from typing import Optional, List, Tuple
from ..utils.logger import setup_logger
from ..utils.geo_utils import (transform_coordinates, calculate_scale,
                             calculate_viewport_bounds)
from .. import config
from ..gui.map_view import MapViewWidget

logger = setup_logger(__name__)

class MapCanvas(QWidget):
    """Widget for displaying and interacting with map data."""

    def __init__(self, parent=None):
        """Initialize the map canvas."""
        super().__init__(parent)

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
        """Set the current map tool.

        Args:
            tool: Tool identifier
        """
        self.current_tool = tool
        self.measuring_points = []  # Clear any measurement in progress
        self.update()
        logger.debug(f"Set map tool to: {tool}")


    def add_layer(self, layer) -> None:
        """Add a layer to the map."""
        try:
            # Add to internal list if not already present
            if layer not in self._layers:
                self._layers.append(layer)
                logger.debug(f"Added layer to map canvas: {layer.name}")
                logger.debug(f"Current layer count: {len(self._layers)}")

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
        """Remove a layer from the map.

        Args:
            layer: Layer object to remove
        """
        try:
            if layer in self.layers:
                self.layers.remove(layer)
                # Force a repaint
                self.update()
                logger.debug(f"Removed layer: {layer.name}")

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

    def zoom_to_extent(self, extent: tuple) -> None:
        """Zoom the map to show a specific extent.

        Args:
            extent: (min_lon, min_lat, max_lon, max_lat)
        """
        try:
            min_lon, min_lat, max_lon, max_lat = extent

            # Calculate center
            center_lon = (min_lon + max_lon) / 2
            center_lat = (min_lat + max_lat) / 2
            self.center = (center_lat, center_lon)

            # Calculate appropriate zoom level
            width = abs(max_lon - min_lon)
            height = abs(max_lat - min_lat)
            max_dimension = max(width, height)

            # Approximate zoom level
            zoom = int(np.log2(360 / max_dimension))
            self.zoom = max(0, min(zoom, 18))

            self.scale = calculate_scale(self.center[0], self.zoom)
            self.update_scale_label()
            self.update()

            logger.debug(f"Zoomed to extent: {extent}")

        except Exception as e:
            logger.error("Error zooming to extent")
            logger.exception(e)
            raise

    def update_scale_label(self):
        """Update the scale display label."""
        try:
            if self.scale != 0:
                scale_text = f"Scale: 1:{int(1/self.scale):,}"
                self.scale_label.setText(scale_text)

        except Exception as e:
            logger.error("Error updating scale label")
            logger.exception(e)



    def screen_to_geo(self, pos: QPointF) -> Tuple[float, float]:
        """Convert screen coordinates to geographic coordinates.

        Args:
            pos: Screen position

        Returns:
            tuple: (latitude, longitude)
        """
        try:
            # Get viewport bounds
            bounds = calculate_viewport_bounds(
                self.center,
                self.width(),
                self.height(),
                self.scale
            )

            # Calculate fractions
            x_frac = pos.x() / self.width()
            y_frac = pos.y() / self.height()

            # Interpolate coordinates
            lon = bounds[0] + x_frac * (bounds[2] - bounds[0])
            lat = bounds[3] - y_frac * (bounds[3] - bounds[1])

            return (lat, lon)

        except Exception as e:
            logger.error("Error converting screen to geo coordinates")
            logger.exception(e)
            raise

    def geo_to_screen(self, coords: Tuple[float, float]) -> QPointF:
        """Convert geographic coordinates to screen coordinates.

        Args:
            coords: (latitude, longitude)

        Returns:
            QPointF: Screen position
        """
        try:
            lat, lon = coords
            bounds = calculate_viewport_bounds(
                self.center,
                self.width(),
                self.height(),
                self.scale
            )

            # Calculate fractions
            x_frac = (lon - bounds[0]) / (bounds[2] - bounds[0])
            y_frac = (bounds[3] - lat) / (bounds[3] - bounds[1])

            # Convert to screen coordinates
            x = x_frac * self.width()
            y = y_frac * self.height()

            return QPointF(x, y)

        except Exception as e:
            logger.error("Error converting geo to screen coordinates")
            logger.exception(e)
            raise


    def draw_measurements(self, painter: QPainter) -> None:
        """Draw measurement overlay.

        Args:
            painter: QPainter object
        """
        try:
            if not self.measuring_points:
                return

            # Set up pen
            pen = QPen(QColor(255, 0, 0))
            pen.setWidth(2)
            painter.setPen(pen)

            # Draw points and lines
            for i in range(len(self.measuring_points)):
                # Draw point
                point_screen = self.geo_to_screen(self.measuring_points[i])
                painter.drawEllipse(point_screen, 3, 3)

                # Draw line to previous point
                if i > 0:
                    prev_point = self.geo_to_screen(self.measuring_points[i-1])
                    painter.drawLine(prev_point, point_screen)

            # Calculate and display measurement
            if len(self.measuring_points) > 1:
                if self.current_tool == "measure_distance":
                    self.display_distance_measurement()
                elif self.current_tool == "measure_area":
                    self.display_area_measurement()

        except Exception as e:
            logger.error("Error drawing measurements")
            logger.exception(e)

    def display_distance_measurement(self) -> None:
        """Display distance measurement result."""
        try:
            from geopy.distance import geodesic

            total_distance = 0
            for i in range(1, len(self.measuring_points)):
                point1 = self.measuring_points[i-1]
                point2 = self.measuring_points[i]
                distance = geodesic(point1, point2).meters
                total_distance += distance

            if total_distance < 1000:
                text = f"Distance: {total_distance:.1f} m"
            else:
                text = f"Distance: {total_distance/1000:.2f} km"

            self.coord_label.setText(text)

        except Exception as e:
            logger.error("Error displaying distance measurement")
            logger.exception(e)

    def display_area_measurement(self) -> None:
        """Display area measurement result."""
        try:
            from shapely.geometry import Polygon

            if len(self.measuring_points) < 3:
                return

            # Create polygon
            poly = Polygon([(p[1], p[0]) for p in self.measuring_points])

            # Calculate area (approximate)
            area = abs(poly.area * 111320 * 111320)  # Convert to square meters

            if area < 10000:
                text = f"Area: {area:.1f} m²"
            else:
                text = f"Area: {area/10000:.2f} ha"

            self.coord_label.setText(text)

        except Exception as e:
            logger.error("Error displaying area measurement")
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

    def get_viewport_bounds(self) -> Tuple[float, float, float, float]:
        """Get current viewport bounds."""
        if self.viewport_bounds:
            return self.viewport_bounds

        # Default to layer extent or world bounds
        if self.layers:
            return self.layers[0].extent()
        return (-180, -90, 180, 90)

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

    def get_transform(self) -> Tuple[float, float, float, float]:
        """Get the transformation between viewport and world coordinates."""
        try:
            if not self.viewport_bounds:
                return (1, 0, 0, 1)  # Identity transform

            # Get viewport and world dimensions
            vp_width = self.width()
            vp_height = self.height()

            world_width = self.viewport_bounds[2] - self.viewport_bounds[0]
            world_height = self.viewport_bounds[3] - self.viewport_bounds[1]

            # Calculate scale factors
            scale_x = vp_width / world_width if world_width != 0 else 1
            scale_y = vp_height / world_height if world_height != 0 else 1

            return (scale_x, scale_y, self.viewport_bounds[0], self.viewport_bounds[1])

        except Exception as e:
            logger.error("Error calculating transform")
            logger.exception(e)
            return (1, 0, 0, 1)

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

                    self.map_view.update()

            self.last_mouse_pos = event.pos()

        except Exception as e:
            logger.error("Error handling mouse move")
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
