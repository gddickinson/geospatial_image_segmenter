"""MapCanvas coordinate transforms: screen <-> geographic coordinates, viewport bounds, transform."""
from PyQt6.QtCore import QPointF
from typing import Tuple
from ..utils.logger import setup_logger
from ..utils.geo_utils import calculate_viewport_bounds

logger = setup_logger(__name__)


class MapCanvasCoordsMixin:
    """Methods of MapCanvas (mixed in; see map_canvas.py)."""

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

    def get_viewport_bounds(self) -> Tuple[float, float, float, float]:
        """Get current viewport bounds."""
        if self.viewport_bounds:
            return self.viewport_bounds

        # Default to layer extent or world bounds
        if self.layers:
            return self.layers[0].extent()
        return (-180, -90, 180, 90)

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
