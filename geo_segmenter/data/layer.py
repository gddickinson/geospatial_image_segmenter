"""Base classes for map layers."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List
import numpy as np
from PyQt6.QtGui import QPainter, QImage, QColor
from PyQt6.QtCore import QObject, pyqtSignal, QRectF
from PIL import Image
import rasterio

from ..utils.logger import setup_logger
from ..utils.geo_utils import transform_coordinates
from .. import config


logger = setup_logger(__name__)


class Layer(QObject):
    """Base class for all map layers."""

    # Signals
    changed = pyqtSignal()  # Emitted when layer changes
    visibility_changed = pyqtSignal(bool)  # Emitted when visibility changes

    def __init__(self, name: str):
        """Initialize layer.

        Args:
            name: Layer name
        """
        super().__init__()
        self._name = name
        self._visible = True
        self._opacity = 1.0
        self._crs = config.DEFAULT_CRS

        # Caching for rendering optimization
        self._cache = {}
        self._cache_valid = False

    @property
    def name(self) -> str:
        """Get layer name."""
        return self._name

    @name.setter
    def name(self, value: str):
        """Set layer name."""
        self._name = value
        self.changed.emit()

    @property
    def visible(self) -> bool:
        """Get layer visibility."""
        return self._visible

    @visible.setter
    def visible(self, value: bool):
        """Set layer visibility."""
        if self._visible != value:
            self._visible = value
            self.visibility_changed.emit(value)
            self.changed.emit()
            logger.debug(f"Layer {self.name} visibility changed to {value}")

    @property
    def opacity(self) -> float:
        """Get layer opacity."""
        return self._opacity

    @opacity.setter
    def opacity(self, value: float):
        """Set layer opacity."""
        self._opacity = max(0.0, min(1.0, value))
        self._cache_valid = False
        self.changed.emit()

    @property
    def crs(self) -> str:
        """Get layer CRS."""
        return self._crs

    @crs.setter
    def crs(self, value: str):
        """Set layer CRS."""
        self._crs = value
        self._cache_valid = False
        self.changed.emit()

    @abstractmethod
    def extent(self) -> Optional[Tuple[float, float, float, float]]:
        """Get layer extent (min_lon, min_lat, max_lon, max_lat)."""
        pass

    @abstractmethod
    def render(self, painter: QPainter, map_canvas) -> None:
        """Render layer to map canvas.

        Args:
            painter: QPainter object
            map_canvas: MapCanvas widget
        """
        pass

    def invalidate_cache(self):
        """Invalidate render cache."""
        self._cache_valid = False
        self._cache.clear()

class RasterLayer(Layer):
    """Layer for raster data."""

    def __init__(self, name: str, path: str, info: Dict):
        super().__init__(name)
        self.path = path
        self.info = info
        self.band_indices = [1, 2, 3]  # Default to first three bands for RGB
        self.stretch_params = None

        # Initialize from info
        self.crs = info['crs']
        self._extent = (
            info['bounds'].left,
            info['bounds'].bottom,
            info['bounds'].right,
            info['bounds'].top
        )
        logger.debug(f"Initialized RasterLayer {name} with extent: {self._extent}")

    def prepare_display_data(self, data: np.ndarray) -> np.ndarray:
        """Prepare raster data for display.

        Args:
            data: Input raster data array (bands, height, width)

        Returns:
            numpy.ndarray: Processed data in RGB format
        """
        try:
            # Use specified bands or first three bands
            if data.shape[0] >= 3:
                rgb = np.stack([
                    data[i-1] for i in self.band_indices[:3]
                ])
            else:
                # Replicate single band to RGB
                rgb = np.repeat(data[:1], 3, axis=0)

            # Apply stretch
            if self.stretch_params:
                stretch_type = self.stretch_params['type']

                if stretch_type.startswith("Linear"):
                    # Get percentiles if needed
                    if "2%" in stretch_type:
                        p_min, p_max = 2, 98
                    elif "5%" in stretch_type:
                        p_min, p_max = 5, 95
                    else:
                        p_min, p_max = 0, 100

                    normalized = np.zeros_like(rgb, dtype=np.uint8)
                    for i in range(3):
                        band = rgb[i].astype(float)
                        if p_min == 0 and p_max == 100:
                            min_val = self.stretch_params['min']
                            max_val = self.stretch_params['max']
                        else:
                            min_val, max_val = np.nanpercentile(
                                band[~np.isnan(band)],
                                (p_min, p_max)
                            )
                        if max_val > min_val:
                            normalized[i] = np.clip(
                                (band - min_val) * 255 / (max_val - min_val),
                                0, 255
                            )

                elif stretch_type == "Standard Deviation":
                    normalized = np.zeros_like(rgb, dtype=np.uint8)
                    for i in range(3):
                        band = rgb[i].astype(float)
                        valid = ~np.isnan(band)
                        mean = np.mean(band[valid])
                        std = np.std(band[valid])
                        min_val = mean - 2 * std
                        max_val = mean + 2 * std
                        normalized[i] = np.clip(
                            (band - min_val) * 255 / (max_val - min_val),
                            0, 255
                        )

            else:
                # Default normalization
                normalized = np.zeros_like(rgb, dtype=np.uint8)
                for i in range(3):
                    band = rgb[i].astype(float)
                    min_val = np.nanmin(band)
                    max_val = np.nanmax(band)
                    if max_val > min_val:
                        normalized[i] = np.clip(
                            (band - min_val) * 255 / (max_val - min_val),
                            0, 255
                        )

            # Transpose to height, width, channels and ensure contiguous memory
            return np.ascontiguousarray(normalized.transpose(1, 2, 0))

        except Exception as e:
            logger.error("Error preparing display data")
            logger.exception(e)
            raise

    def extent(self) -> Tuple[float, float, float, float]:
        """Get layer extent."""
        return self._extent

    def render(self, painter: QPainter, map_canvas) -> None:
        """Render raster data."""
        if not self.visible:
            return

        try:
            logger.debug(f"Starting render for layer {self.name}")

            # Get viewport dimensions
            width = map_canvas.width()
            height = map_canvas.height()
            logger.debug(f"Canvas dimensions: {width}x{height}")

            # Get raster data
            with rasterio.open(self.path) as src:
                # Read data
                data = src.read()
                logger.debug(f"Read raster data: shape={data.shape}, dtype={data.dtype}")

                # Convert to RGB display format
                display_data = self.prepare_display_data(data)
                logger.debug(f"Prepared display data: shape={display_data.shape}")

                # Create QImage
                img_height, img_width, channels = display_data.shape
                bytes_per_line = channels * img_width
                qimage = QImage(
                    display_data.tobytes(),
                    img_width,
                    img_height,
                    bytes_per_line,
                    QImage.Format.Format_RGB888
                )

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

                        # Calculate image dimensions in viewport
                        img_x = (self._extent[0] - viewport_bounds[0]) * x_scale
                        img_y = (viewport_bounds[3] - self._extent[3]) * y_scale
                        img_width = (self._extent[2] - self._extent[0]) * x_scale
                        img_height = (self._extent[3] - self._extent[1]) * y_scale

                        logger.debug(f"Image position: ({img_x}, {img_y})")
                        logger.debug(f"Image dimensions: {img_width}x{img_height}")

                        # Set opacity
                        painter.setOpacity(self.opacity)

                        # Draw image with transformation
                        target_rect = QRectF(img_x, img_y, img_width, img_height)
                        painter.drawImage(target_rect, qimage)
                        logger.debug("Image drawn successfully")

        except Exception as e:
            logger.error(f"Error rendering layer {self.name}: {str(e)}")
            logger.exception(e)

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

"""Layer for displaying segmentation results."""

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
