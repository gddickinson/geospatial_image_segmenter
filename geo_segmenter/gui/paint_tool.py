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

        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

    def paintEvent(self, event):
        """Paint current path."""
        if self.current_path and self.label_dialog and self.label_dialog.active_label:
            painter = QPainter(self)
            painter.setPen(Qt.PenStyle.NoPen)
            color = self.label_dialog.active_label.color
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 128))
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

    def continue_painting(self, pos: QPoint):
        """Continue painting operation."""
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

    def finish_painting(self):
        """Finish painting operation."""
        self.is_painting = False
        self.current_path = QPainterPath()
        self.update()
        self.selection_changed.emit()

    def get_image_coordinates(self, pos: QPoint) -> Optional[QPoint]:
        """Convert screen coordinates to image coordinates."""
        if self.image_shape is None or self.transform is None:
            return None

        # Get pixel coordinates
        x, y = pos.x(), pos.y()

        # Check bounds
        if (0 <= x < self.image_shape[1] and
            0 <= y < self.image_shape[0]):
            return QPoint(x, y)
        return None

    def update_mask(self, pos: QPoint):
        """Update the label mask at the given position."""
        if not self.label_dialog or not self.label_dialog.active_label:
            return

        x, y = int(pos.x()), int(pos.y())
        label = self.label_dialog.active_label

        if label.mask is None:
            if self.image_shape is not None:
                self.label_dialog.initialize_masks(self.image_shape)
            else:
                return

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
        label.mask[y_start:y_end, x_start:x_end] |= \
            brush[brush_y_start:brush_y_end,
                  brush_x_start:brush_x_end]

    def set_image_shape(self, shape: tuple):
        """Set the shape of the image being painted."""
        self.image_shape = shape
        self.resize(shape[1], shape[0])

    def set_transform(self, transform):
        """Set the geotransform for coordinate conversion."""
        self.transform = transform

    def set_brush_size(self, size: int):
        """Set the brush size.

        Args:
            size: Brush size in pixels
        """
        self.brush_size = size
