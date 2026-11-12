"""VectorLayer (geopandas features) and LidarLayer (point clouds)."""
from typing import Dict, Tuple
import numpy as np
from PyQt6.QtGui import QPainter
from ..utils.logger import setup_logger
from .layer_base import Layer

logger = setup_logger(__name__)


class VectorLayer(Layer):
    """Layer for vector data."""

    def __init__(self, name: str, path: str, data):
        """Initialize vector layer.

        Args:
            name: Layer name
            path: Path to vector file
            data: GeoDataFrame with vector data
        """
        super().__init__(name)
        self.path = path
        self.data = data

        # Style settings
        self.fill_color = (200, 200, 200, 128)  # RGBA
        self.stroke_color = (0, 0, 0, 255)  # RGBA
        self.stroke_width = 1
        self.symbol_size = 5

        # Label settings
        self.label_field = None
        self.label_font_size = 10
        self.label_color = (0, 0, 0, 255)  # RGBA

        # Initialize from data
        self.crs = str(data.crs)
        bounds = data.total_bounds
        self._extent = (bounds[0], bounds[1], bounds[2], bounds[3])

    def extent(self) -> Tuple[float, float, float, float]:
        """Get layer extent."""
        return self._extent

    def render(self, painter: QPainter, map_canvas) -> None:
        """Render vector data.

        Args:
            painter: QPainter object
            map_canvas: MapCanvas widget
        """
        if not self.visible:
            return

        try:
            from shapely.geometry import Point, LineString, Polygon

            # Set up painter
            painter.setOpacity(self.opacity)

            # Iterate through features
            for _, feature in self.data.iterrows():
                geom = feature.geometry

                if isinstance(geom, Point):
                    self._render_point(painter, map_canvas, geom)
                elif isinstance(geom, LineString):
                    self._render_line(painter, map_canvas, geom)
                elif isinstance(geom, Polygon):
                    self._render_polygon(painter, map_canvas, geom)

                # Render label if specified
                if self.label_field and self.label_field in feature:
                    self._render_label(painter, map_canvas, geom,
                                     str(feature[self.label_field]))

        except Exception as e:
            logger.error(f"Error rendering vector layer: {str(e)}")
            logger.exception(e)

    def _render_point(self, painter: QPainter, map_canvas, point) -> None:
        """Render a point geometry."""
        screen_pos = map_canvas.geo_to_screen((point.y, point.x))

        # Draw symbol
        painter.setPen(self.stroke_color)
        painter.setBrush(self.fill_color)
        size = self.symbol_size
        painter.drawEllipse(screen_pos, size, size)

    def _render_line(self, painter: QPainter, map_canvas, line) -> None:
        """Render a line geometry."""
        # Convert coordinates to screen space
        points = []
        for x, y in line.coords:
            screen_pos = map_canvas.geo_to_screen((y, x))
            points.append(screen_pos)

        # Draw line
        painter.setPen(self.stroke_color)
        for i in range(len(points) - 1):
            painter.drawLine(points[i], points[i + 1])

    def _render_polygon(self, painter: QPainter, map_canvas, polygon) -> None:
        """Render a polygon geometry."""
        # Convert coordinates to screen space
        points = []
        for x, y in polygon.exterior.coords:
            screen_pos = map_canvas.geo_to_screen((y, x))
            points.append(screen_pos)

        # Draw polygon
        painter.setPen(self.stroke_color)
        painter.setBrush(self.fill_color)
        painter.drawPolygon(points)

    def _render_label(self, painter: QPainter, map_canvas, geom, text: str) -> None:
        """Render a feature label."""
        # Get label position (center of geometry)
        center = geom.centroid
        screen_pos = map_canvas.geo_to_screen((center.y, center.x))

        # Set up label style
        font = painter.font()
        font.setPointSize(self.label_font_size)
        painter.setFont(font)
        painter.setPen(self.label_color)

        # Draw text
        painter.drawText(screen_pos, text)


class LidarLayer(Layer):
    """Layer for LIDAR point cloud data."""

    def __init__(self, name: str, path: str, point_cloud: np.ndarray, metadata: Dict):
        """Initialize LIDAR layer.

        Args:
            name: Layer name
            path: Path to LIDAR file
            point_cloud: Point cloud array
            metadata: LIDAR metadata
        """
        super().__init__(name)
        self.path = path
        self.point_cloud = point_cloud
        self.metadata = metadata

        # Rendering settings
        self.point_size = 2
        self.color_by = 'elevation'  # 'elevation', 'intensity', 'classification'
        self.min_value = None
        self.max_value = None

        # Initialize extent from metadata
        self._extent = (
            metadata['min_bounds'][0],
            metadata['min_bounds'][1],
            metadata['max_bounds'][0],
            metadata['max_bounds'][1]
        )

    def extent(self) -> Tuple[float, float, float, float]:
        """Get layer extent."""
        return self._extent

    def render(self, painter: QPainter, map_canvas) -> None:
        """Render LIDAR data.

        Args:
            painter: QPainter object
            map_canvas: MapCanvas widget
        """
        if not self.visible:
            return

        try:
            # Get viewport bounds
            bounds = map_canvas.get_viewport_bounds()

            # Filter points within view
            mask = ((self.point_cloud[:, 0] >= bounds[0]) &
                   (self.point_cloud[:, 0] <= bounds[2]) &
                   (self.point_cloud[:, 1] >= bounds[1]) &
                   (self.point_cloud[:, 1] <= bounds[3]))
            visible_points = self.point_cloud[mask]

            if len(visible_points) == 0:
                return

            # Get colors based on selected attribute
            if self.color_by == 'elevation':
                values = visible_points[:, 2]
            elif self.color_by == 'intensity':
                values = visible_points[:, 4]
            elif self.color_by == 'classification':
                values = visible_points[:, 3]

            # Normalize values
            if self.min_value is not None and self.max_value is not None:
                normalized = np.clip(values, self.min_value, self.max_value)
                normalized = (normalized - self.min_value) / (self.max_value - self.min_value)
            else:
                normalized = (values - values.min()) / (values.max() - values.min())

            # Convert to colors (using a simple blue-red color ramp)
            colors = np.zeros((len(normalized), 4), dtype=np.uint8)
            colors[:, 0] = (normalized * 255).astype(np.uint8)  # Red
            colors[:, 2] = ((1 - normalized) * 255).astype(np.uint8)  # Blue
            colors[:, 3] = int(self.opacity * 255)  # Alpha

            # Draw points
            painter.setOpacity(self.opacity)
            for i, point in enumerate(visible_points):
                screen_pos = map_canvas.geo_to_screen((point[1], point[0]))
                painter.setPen(tuple(colors[i]))
                painter.drawEllipse(screen_pos, self.point_size, self.point_size)

        except Exception as e:
            logger.error(f"Error rendering LIDAR layer: {str(e)}")
            logger.exception(e)
