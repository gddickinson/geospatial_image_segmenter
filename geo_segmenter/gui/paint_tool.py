"""Paint tool for selecting and labeling regions in geospatial imagery."""
from PyQt6.QtWidgets import QGraphicsObject, QWidget
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, QPoint
from PyQt6.QtGui import QPainter, QPainterPath, QColor
import numpy as np
from typing import Optional, Tuple
from ..utils.logger import setup_logger
from ..utils.geo_utils import coords_to_pixel, pixel_to_coords

logger = setup_logger(__name__)

class PaintTool(QGraphicsObject):
    """Paint tool for image labeling with coordinate system awareness."""

    def __init__(self, parent=None):
        """Initialize paint tool.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.brush_size = 5
        self.image_shape = None
        self.transform = None  # Geotransform from raster
        self.parent_window = None  # Set by MainWindow
        self.current_path = QPainterPath()
        self.is_painting = False
        self.setAcceptHoverEvents(True)
        logger.debug("Initialized PaintTool")

    def boundingRect(self) -> QRectF:
        """Define bounding rectangle for the paint tool.

        Returns:
            QRectF: Bounding rectangle
        """
        if self.image_shape is None:
            return QRectF(0, 0, 0, 0)
        return QRectF(0, 0, self.image_shape[1], self.image_shape[0])

    def paint(self, painter: QPainter, *args):
        """Paint current path.

        Args:
            painter: QPainter instance
        """
        if self.current_path and self.parent_window and self.parent_window.active_label:
            painter.setPen(Qt.PenStyle.NoPen)
            color = self.parent_window.active_label.color
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 128))
            painter.drawPath(self.current_path)

    def mousePressEvent(self, event):
        """Handle mouse press events."""
        if event.button() == Qt.MouseButton.LeftButton:
            try:
                self.is_painting = True
                self.start_painting(event.pos())
                event.accept()
            except Exception as e:
                logger.error("Error in mouse press event")
                logger.exception(e)

    def mouseMoveEvent(self, event):
        """Handle mouse move events."""
        if event.buttons() & Qt.MouseButton.LeftButton and self.is_painting:
            try:
                self.continue_painting(event.pos())
                event.accept()
            except Exception as e:
                logger.error("Error in mouse move event")
                logger.exception(e)

    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        if event.button() == Qt.MouseButton.LeftButton and self.is_painting:
            try:
                self.finish_painting()
                event.accept()
            except Exception as e:
                logger.error("Error in mouse release event")
                logger.exception(e)

    def start_painting(self, pos: QPointF):
        """Start a new painting operation.

        Args:
            pos: Mouse position
        """
        try:
            if not self.parent_window or not self.parent_window.active_label:
                logger.debug("No active label for painting")
                return

            # Convert screen coordinates to image coordinates
            image_pos = self.get_image_coordinates(pos)
            if image_pos is None:
                return

            # Start new path
            self.current_path = QPainterPath()
            self.current_path.addEllipse(
                image_pos.x() - self.brush_size/2,
                image_pos.y() - self.brush_size/2,
                self.brush_size,
                self.brush_size
            )

            # Update mask
            self.update_mask(image_pos)
            self.update()

        except Exception as e:
            logger.error("Error starting painting")
            logger.exception(e)

    def continue_painting(self, pos: QPointF):
        """Continue painting operation.

        Args:
            pos: Mouse position
        """
        try:
            if not self.is_painting:
                return

            image_pos = self.get_image_coordinates(pos)
            if image_pos is None:
                return

            # Add to path
            self.current_path.addEllipse(
                image_pos.x() - self.brush_size/2,
                image_pos.y() - self.brush_size/2,
                self.brush_size,
                self.brush_size
            )

            # Update mask
            self.update_mask(image_pos)
            self.update()

        except Exception as e:
            logger.error("Error continuing painting")
            logger.exception(e)

    def finish_painting(self):
        """Finish painting operation and update the label mask."""
        try:
            self.is_painting = False
            self.current_path = QPainterPath()
            self.update()

            # Update display in main window
            if self.parent_window:
                self.parent_window.update_overlay()

        except Exception as e:
            logger.error("Error finishing painting")
            logger.exception(e)

    def get_image_coordinates(self, pos: QPointF) -> Optional[QPointF]:
        """Convert screen coordinates to image coordinates.

        Args:
            pos: Screen position

        Returns:
            QPointF: Image coordinates or None if invalid
        """
        try:
            if self.image_shape is None or self.transform is None:
                return None

            # Convert to scene coordinates
            scene_pos = self.mapToScene(pos)

            # Convert to geo coordinates
            geo_x, geo_y = scene_pos.x(), scene_pos.y()

            # Convert to pixel coordinates
            row, col = coords_to_pixel(geo_x, geo_y, self.transform)

            # Check bounds
            if (0 <= row < self.image_shape[0] and
                0 <= col < self.image_shape[1]):
                return QPointF(col, row)
            return None

        except Exception as e:
            logger.error("Error converting coordinates")
            logger.exception(e)
            return None

    def update_mask(self, pos: QPointF):
        """Update the label mask at the given position.

        Args:
            pos: Position in image coordinates
        """
        try:
            if not self.parent_window or not self.parent_window.active_label:
                return

            x, y = int(pos.x()), int(pos.y())

            # Get current label's mask
            label = self.parent_window.active_label
            if label.layer_id not in label.masks:
                label.masks[label.layer_id] = np.zeros(self.image_shape, dtype=bool)

            mask = label.masks[label.layer_id]

            # Create circular brush
            y_idx, x_idx = np.ogrid[-self.brush_size:self.brush_size+1,
                                  -self.brush_size:self.brush_size+1]
            dist = np.sqrt(x_idx*x_idx + y_idx*y_idx)
            brush = dist <= self.brush_size/2

            # Calculate brush bounds
            y_start = max(0, y - self.brush_size)
            y_end = min(self.image_shape[0], y + self.brush_size + 1)
            x_start = max(0, x - self.brush_size)
            x_end = min(self.image_shape[1], x + self.brush_size + 1)

            # Calculate brush array bounds
            brush_y_start = max(0, -(y - self.brush_size))
            brush_y_end = brush.shape[0] - max(0, y_end - self.image_shape[0])
            brush_x_start = max(0, -(x - self.brush_size))
            brush_x_end = brush.shape[1] - max(0, x_end - self.image_shape[1])

            # Apply brush
            mask[y_start:y_end, x_start:x_end] |= \
                brush[brush_y_start:brush_y_end,
                      brush_x_start:brush_x_end]

        except Exception as e:
            logger.error("Error updating mask")
            logger.exception(e)

    def set_image_shape(self, shape: tuple):
        """Set the shape of the image being painted.

        Args:
            shape: Image shape tuple
        """
        try:
            self.image_shape = shape
            self.prepareGeometryChange()
            logger.debug(f"Set image shape to {shape}")
        except Exception as e:
            logger.error("Error setting image shape")
            logger.exception(e)

    def set_transform(self, transform):
        """Set the geotransform for coordinate conversion.

        Args:
            transform: Rasterio transform object
        """
        try:
            self.transform = transform
            logger.debug("Set geotransform")
        except Exception as e:
            logger.error("Error setting transform")
            logger.exception(e)





"""Paint tool for selecting training pixels."""

class TrainingPaintTool(QWidget):
    """Paint tool for selecting training pixels."""

    # Signal emitted when selection changes
    selection_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.brush_size = 5
        self.image_shape = None
        self.transform = None
        self.current_path = QPainterPath()
        self.is_painting = False
        self.label_dialog = None  # Will be set by main window
        self.extent = None  # Store the extent to ensure consistent coordinates
        self._brush_positions = []  # Store all brush positions for visualization during painting

        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        logger.debug("Initialized TrainingPaintTool")

    def set_extent(self, extent):
        """Set the geographic extent for coordinate consistency.

        Args:
            extent: Layer extent (min_x, min_y, max_x, max_y)
        """
        self.extent = extent
        logger.debug(f"Set extent for TrainingPaintTool: {extent}")

    def paintEvent(self, event):
        """Paint current path."""
        if self.label_dialog and self.label_dialog.active_label:
            painter = QPainter(self)
            painter.setPen(Qt.PenStyle.NoPen)
            color = self.label_dialog.active_label.color
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 128))

            # Draw all brush positions while painting
            if self._brush_positions:
                # Get map canvas reference
                map_canvas = self.parent().parent()

                # Use the same coordinate transformation as in get_image_coordinates
                if map_canvas and self.extent and self.image_shape:
                    viewport_bounds = map_canvas.viewport_bounds
                    if viewport_bounds:
                        width = map_canvas.width()
                        height = map_canvas.height()
                        world_width = viewport_bounds[2] - viewport_bounds[0]
                        world_height = viewport_bounds[3] - viewport_bounds[1]

                        # Calculate the correct size for the brush on screen
                        brush_size_screen = (self.brush_size / self.image_shape[1]) * width

                        # Draw each brush position
                        for pos in self._brush_positions:
                            pixel_x, pixel_y = pos.x(), pos.y()

                            # Convert from pixel coordinates back to screen coordinates
                            world_x = self.extent[0] + (pixel_x / self.image_shape[1]) * (self.extent[2] - self.extent[0])
                            world_y = self.extent[3] - (pixel_y / self.image_shape[0]) * (self.extent[3] - self.extent[1])

                            screen_x = ((world_x - viewport_bounds[0]) / world_width) * width
                            screen_y = ((viewport_bounds[3] - world_y) / world_height) * height

                            # Draw the brush at the correct screen position
                            painter.drawEllipse(
                                QPointF(screen_x, screen_y),
                                brush_size_screen/2,
                                brush_size_screen/2
                            )
                        return

            # Fall back to drawing the path if we can't calculate the correct positions
            if self.current_path:
                painter.drawPath(self.current_path)

    def mousePressEvent(self, event):
        """Handle mouse press events."""
        if event.button() == Qt.MouseButton.LeftButton and event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            if not self.label_dialog or not self.label_dialog.active_label:
                return

            self.is_painting = True
            self.start_painting(event.pos())
            event.accept()

    def mouseMoveEvent(self, event):
        """Handle mouse move events."""
        if event.buttons() & Qt.MouseButton.LeftButton and event.modifiers() & Qt.KeyboardModifier.ShiftModifier and self.is_painting:
            self.continue_painting(event.pos())
            event.accept()

    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        if event.button() == Qt.MouseButton.LeftButton and self.is_painting:
            self.finish_painting()
            event.accept()

    def start_painting(self, pos: QPoint):
        """Start a new painting operation."""
        image_pos = self.get_image_coordinates(pos)
        if image_pos is None:
            return

        # Clear previous brush positions and add the current one
        self._brush_positions = [image_pos]

        # The path is now just a placeholder, we don't actually use it for rendering
        self.current_path = QPainterPath()

        # Update mask
        self.update_mask(image_pos)
        self.update()

    def continue_painting(self, pos: QPoint):
        """Continue painting operation."""
        if not self.is_painting:
            return

        image_pos = self.get_image_coordinates(pos)
        if image_pos is None:
            return

        # Add the current position to the brush positions list
        self._brush_positions.append(image_pos)

        # Limit the number of stored brush positions to prevent memory issues
        if len(self._brush_positions) > 1000:
            self._brush_positions = self._brush_positions[-1000:]

        # Update mask
        self.update_mask(image_pos)
        self.update()

    def finish_painting(self):
        """Finish painting operation."""
        if self.is_painting:
            logger.debug("Finishing painting operation")
            self.is_painting = False
            self.current_path = QPainterPath()
            self._brush_positions = []  # Clear brush positions

            # Emit signal to notify parent about selection change
            self.update()
            self.selection_changed.emit()

            # Force a repaint of the entire view after a short delay
            # This ensures the painted pixels show up immediately
            from PyQt6.QtCore import QTimer
            parent = self.parent()
            while parent:
                if hasattr(parent, 'update'):
                    logger.debug(f"Scheduling update for parent: {parent.__class__.__name__}")
                    QTimer.singleShot(10, parent.update)
                parent = parent.parent()

            logger.debug("Painting finished, signal emitted")

    def get_image_coordinates(self, pos: QPoint) -> Optional[QPoint]:
        """Convert screen coordinates to image coordinates."""
        if self.image_shape is None or not self.extent:
            logger.debug("Missing image shape or extent, cannot convert coordinates")
            return None

        # Get map canvas reference and check viewport bounds
        map_canvas = self.parent().parent()  # Get reference to map canvas
        viewport_bounds = map_canvas.viewport_bounds
        if not viewport_bounds:
            logger.debug("No viewport bounds available")
            return None

        # Convert screen coordinates to world coordinates precisely the same way as in rendering
        width = map_canvas.width()
        height = map_canvas.height()
        world_width = viewport_bounds[2] - viewport_bounds[0]
        world_height = viewport_bounds[3] - viewport_bounds[1]

        # Convert screen coordinates to world coordinates
        world_x = viewport_bounds[0] + (pos.x() / width) * world_width
        world_y = viewport_bounds[3] - (pos.y() / height) * world_height

        # Convert world coordinates to pixel coordinates using the same transformation as in rendering
        # This is the critical part for alignment
        pixel_x = int((world_x - self.extent[0]) / (self.extent[2] - self.extent[0]) * self.image_shape[1])
        pixel_y = int((self.extent[3] - world_y) / (self.extent[3] - self.extent[1]) * self.image_shape[0])

        # Store the exact world coordinates for debugging
        self._last_world_coords = (world_x, world_y)

        # Check bounds
        if (0 <= pixel_x < self.image_shape[1] and 0 <= pixel_y < self.image_shape[0]):
            logger.debug(f"Converted screen ({pos.x()}, {pos.y()}) to pixel ({pixel_x}, {pixel_y})")
            return QPoint(pixel_x, pixel_y)

        logger.debug(f"Coordinates out of bounds: ({pixel_x}, {pixel_y})")
        return None

    def update_mask(self, pos: QPoint):
        """Update the label mask at the given position."""
        if not self.label_dialog or not self.label_dialog.active_label:
            return

        # Get pixel coordinates directly
        pixel_x, pixel_y = pos.x(), pos.y()

        label = self.label_dialog.active_label

        if label.mask is None:
            if self.image_shape is not None:
                self.label_dialog.initialize_masks(self.image_shape)
            else:
                return

        # Create circular brush with exact dimensions
        y_idx, x_idx = np.ogrid[-self.brush_size:self.brush_size+1,
                              -self.brush_size:self.brush_size+1]
        dist = np.sqrt(x_idx*x_idx + y_idx*y_idx)
        brush = dist <= self.brush_size/2

        # Calculate exact brush bounds
        y_start = int(max(0, pixel_y - self.brush_size))
        y_end = int(min(self.image_shape[0], pixel_y + self.brush_size + 1))
        x_start = int(max(0, pixel_x - self.brush_size))
        x_end = int(min(self.image_shape[1], pixel_x + self.brush_size + 1))

        # Calculate exact brush array bounds
        brush_y_start = int(max(0, -(pixel_y - self.brush_size)))
        brush_y_end = int(brush.shape[0] - max(0, y_end - self.image_shape[0]))
        brush_x_start = int(max(0, -(pixel_x - self.brush_size)))
        brush_x_end = int(brush.shape[1] - max(0, x_end - self.image_shape[1]))

        logger.debug(f"Updating mask at {pixel_x}, {pixel_y}")
        logger.debug(f"Brush bounds: x={x_start}-{x_end}, y={y_start}-{y_end}")

        # Apply brush with exact dimensions
        try:
            brush_section = brush[brush_y_start:brush_y_end, brush_x_start:brush_x_end]
            mask_section = label.mask[y_start:y_end, x_start:x_end]

            if brush_section.shape == mask_section.shape:
                label.mask[y_start:y_end, x_start:x_end] |= brush_section
            else:
                logger.error(f"Shape mismatch - Brush: {brush_section.shape}, Mask: {mask_section.shape}")
        except Exception as e:
            logger.error(f"Error applying brush: {e}")

    def set_image_shape(self, shape: tuple):
        """Set the shape of the image being painted."""
        self.image_shape = shape
        self.resize(shape[1], shape[0])
        logger.debug(f"Set image shape to {shape}")

    def set_transform(self, transform):
        """Set the geotransform for coordinate conversion."""
        self.transform = transform
        logger.debug(f"Set transform to {transform}")

    def set_brush_size(self, size: int):
        """Set the brush size.

        Args:
            size: Brush size in pixels
        """
        self.brush_size = size
        logger.debug(f"Set brush size to {size}")
