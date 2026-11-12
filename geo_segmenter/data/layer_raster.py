"""RasterLayer: GeoTIFF raster layer (band combination and stretch via utils/display_stretch)."""
from typing import Dict, Tuple
import numpy as np
from PyQt6.QtGui import QPainter, QImage
from PyQt6.QtCore import QRectF
from PIL import Image
import rasterio
from ..utils.logger import setup_logger
from .layer_base import Layer

logger = setup_logger(__name__)


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
            from ..utils.display_stretch import prepare_display
            return prepare_display(data, self.band_indices, self.stretch_params)

        except Exception as e:
            logger.error("Error preparing display data")
            logger.exception(e)
            raise

    def extent(self) -> Tuple[float, float, float, float]:
        """Get layer extent."""
        return self._extent

    # Replace the render method in your RasterLayer class with this version

    def render(self, painter: QPainter, map_canvas) -> None:
        """Render raster data."""
        # VERY FIRST THING: Check visibility and return if not visible
        if not self._visible:
            logger.debug(f"SKIPPING RENDER for layer {self.name} - not visible (_visible=False)")
            return

        try:
            logger.debug(f"Starting render for layer {self.name}")
            logger.debug(f"Canvas dimensions: {map_canvas.width()}x{map_canvas.height()}")
            logger.debug(f"Viewport bounds: {map_canvas.viewport_bounds}")
            logger.debug(f"Layer extent: {self._extent}")

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

                        logger.debug(f"Transform factors: x_scale={x_scale}, y_scale={y_scale}")
                        logger.debug(f"Image position: ({img_x}, {img_y})")
                        logger.debug(f"Image dimensions: {img_width}x{img_height}")

                        # Save current opacity
                        old_opacity = painter.opacity()

                        # Set opacity
                        painter.setOpacity(self.opacity)

                        # Draw image with transformation
                        target_rect = QRectF(img_x, img_y, img_width, img_height)
                        painter.drawImage(target_rect, qimage)

                        # Restore opacity
                        painter.setOpacity(old_opacity)

                        logger.debug("Image drawn successfully")

        except Exception as e:
            logger.error(f"Error rendering layer {self.name}: {str(e)}")
            logger.exception(e)
