"""Manager for various selection tools and modes."""
from PyQt6.QtWidgets import QGraphicsObject
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt6.QtGui import QPainter, QPainterPath, QColor, QPolygonF
import numpy as np
from typing import List, Optional, Tuple
from ..utils.logger import setup_logger
from ..utils.geo_utils import coords_to_pixel, pixel_to_coords

logger = setup_logger(__name__)

class SelectionMode:
    """Enumeration of selection modes."""
    PAINT = "paint"
    POLYGON = "polygon"
    RECTANGLE = "rectangle"
    MAGIC_WAND = "magic_wand"

class SelectionManager(QGraphicsObject):
    """Manages different selection tools and modes."""
    
    # Signal emitted when selection is completed
    selection_completed = pyqtSignal(np.ndarray)  # Emits the selection mask
    
    def __init__(self, parent=None):
        """Initialize selection manager.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.mode = SelectionMode.PAINT
        self.image_shape = None
        self.transform = None
        self.brush_size = 5
        self.tolerance = 0.1  # For magic wand
        
        # State variables
        self.is_selecting = False
        self.current_path = QPainterPath()
        self.points = []  # For polygon selection
        self.start_pos = None  # For rectangle selection
        self.current_pos = None
        
        self.setAcceptHoverEvents(True)
        logger.debug("Initialized SelectionManager")
    
    def boundingRect(self) -> QRectF:
        """Define bounding rectangle for the selection area."""
        if self.image_shape is None:
            return QRectF(0, 0, 0, 0)
        return QRectF(0, 0, self.image_shape[1], self.image_shape[0])
    
    def paint(self, painter: QPainter, *args):
        """Paint current selection visualization."""
        if not self.is_selecting:
            return
            
        painter.setPen(Qt.PenStyle.SolidLine)
        painter.setBrush(QColor(255, 255, 255, 64))
        
        if self.mode == SelectionMode.PAINT:
            painter.drawPath(self.current_path)
            
        elif self.mode == SelectionMode.POLYGON:
            if len(self.points) > 1:
                # Draw completed lines
                poly = QPolygonF(self.points)
                painter.drawPolyline(poly)
                
                # Draw line to current position
                if self.current_pos:
                    painter.drawLine(self.points[-1], self.current_pos)
                    
        elif self.mode == SelectionMode.RECTANGLE:
            if self.start_pos and self.current_pos:
                rect = QRectF(self.start_pos, self.current_pos).normalized()
                painter.drawRect(rect)
    
    def mousePressEvent(self, event):
        """Handle mouse press events."""
        if event.button() == Qt.MouseButton.LeftButton:
            try:
                pos = self.get_image_coordinates(event.pos())
                if pos is None:
                    return
                
                self.is_selecting = True
                
                if self.mode == SelectionMode.PAINT:
                    self.start_paint_selection(pos)
                elif self.mode == SelectionMode.POLYGON:
                    self.add_polygon_point(pos)
                elif self.mode == SelectionMode.RECTANGLE:
                    self.start_rectangle_selection(pos)
                elif self.mode == SelectionMode.MAGIC_WAND:
                    self.do_magic_wand_selection(pos)
                
                event.accept()
                
            except Exception as e:
                logger.error("Error in mouse press event")
                logger.exception(e)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events."""
        try:
            pos = self.get_image_coordinates(event.pos())
            if pos is None:
                return
                
            self.current_pos = pos
            
            if event.buttons() & Qt.MouseButton.LeftButton and self.is_selecting:
                if self.mode == SelectionMode.PAINT:
                    self.continue_paint_selection(pos)
                elif self.mode == SelectionMode.RECTANGLE:
                    self.update_rectangle_selection(pos)
            
            self.update()
            event.accept()
            
        except Exception as e:
            logger.error("Error in mouse move event")
            logger.exception(e)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        if event.button() == Qt.MouseButton.LeftButton:
            try:
                if self.mode == SelectionMode.PAINT:
                    self.finish_paint_selection()
                elif self.mode == SelectionMode.RECTANGLE:
                    self.finish_rectangle_selection()
                
                event.accept()
                
            except Exception as e:
                logger.error("Error in mouse release event")
                logger.exception(e)
    
    def mouseDoubleClickEvent(self, event):
        """Handle mouse double click events."""
        if event.button() == Qt.MouseButton.LeftButton:
            try:
                if self.mode == SelectionMode.POLYGON:
                    self.finish_polygon_selection()
                event.accept()
                
            except Exception as e:
                logger.error("Error in mouse double click event")
                logger.exception(e)
    
    def start_paint_selection(self, pos: QPointF):
        """Start paint selection.
        
        Args:
            pos: Starting position
        """
        self.current_path = QPainterPath()
        self.current_path.addEllipse(
            pos.x() - self.brush_size/2,
            pos.y() - self.brush_size/2,
            self.brush_size,
            self.brush_size
        )
        self.update()
    
    def continue_paint_selection(self, pos: QPointF):
        """Continue paint selection.
        
        Args:
            pos: Current position
        """
        self.current_path.addEllipse(
            pos.x() - self.brush_size/2,
            pos.y() - self.brush_size/2,
            self.brush_size,
            self.brush_size
        )
        self.update()
    
    def finish_paint_selection(self):
        """Finish paint selection and emit result."""
        mask = self.path_to_mask(self.current_path)
        self.selection_completed.emit(mask)
        self.reset_selection()
    
    def add_polygon_point(self, pos: QPointF):
        """Add point to polygon selection.
        
        Args:
            pos: New point position
        """
        self.points.append(pos)
        self.update()
    
    def finish_polygon_selection(self):
        """Finish polygon selection and emit result."""
        if len(self.points) >= 3:
            path = QPainterPath()
            path.addPolygon(QPolygonF(self.points))
            mask = self.path_to_mask(path)
            self.selection_completed.emit(mask)
        self.reset_selection()
    
    def start_rectangle_selection(self, pos: QPointF):
        """Start rectangle selection.
        
        Args:
            pos: Starting corner position
        """
        self.start_pos = pos
        self.current_pos = pos
        self.update()
    
    def update_rectangle_selection(self, pos: QPointF):
        """Update rectangle selection.
        
        Args:
            pos: Current corner position
        """
        self.current_pos = pos
        self.update()
    
    def finish_rectangle_selection(self):
        """Finish rectangle selection and emit result."""
        if self.start_pos and self.current_pos:
            path = QPainterPath()
            rect = QRectF(self.start_pos, self.current_pos).normalized()
            path.addRect(rect)
            mask = self.path_to_mask(path)
            self.selection_completed.emit(mask)
        self.reset_selection()
    
    def do_magic_wand_selection(self, pos: QPointF):
        """Perform magic wand selection.
        
        Args:
            pos: Seed point position
        """
        if self.parent() and hasattr(self.parent(), 'get_current_layer'):
            try:
                layer = self.parent().get_current_layer()
                if layer is None:
                    return
                
                # Convert position to pixel coordinates
                x, y = int(pos.x()), int(pos.y())
                
                # Get image data
                image_data = layer.get_data()
                if image_data is None:
                    return
                
                # Perform flood fill
                mask = self.flood_fill(image_data, (y, x), self.tolerance)
                self.selection_completed.emit(mask)
                
            except Exception as e:
                logger.error("Error in magic wand selection")
                logger.exception(e)
    
    def flood_fill(self, image: np.ndarray, seed_point: Tuple[int, int], tolerance: float) -> np.ndarray:
        """Perform flood fill for magic wand tool.
        
        Args:
            image: Image array
            seed_point: Starting point (row, col)
            tolerance: Color difference tolerance
            
        Returns:
            numpy.ndarray: Boolean mask of selected region
        """
        try:
            if image.ndim == 3:
                # For multi-band images, use mean value
                image = np.mean(image, axis=2)
            
            mask = np.zeros(image.shape, dtype=bool)
            seed_value = image[seed_point]
            
            # Stack for flood fill
            stack = [seed_point]
            mask[seed_point] = True
            
            while stack:
                y, x = stack.pop()
                
                # Check 4-connected neighbors
                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ny, nx = y + dy, x + dx
                    
                    if (0 <= ny < image.shape[0] and 
                        0 <= nx < image.shape[1] and 
                        not mask[ny, nx]):
                        
                        # Check if neighbor is within tolerance
                        if abs(image[ny, nx] - seed_value) <= tolerance:
                            mask[ny, nx] = True
                            stack.append((ny, nx))
            
            return mask
            
        except Exception as e:
            logger.error("Error in flood fill")
            logger.exception(e)
            return np.zeros(image.shape, dtype=bool)
    
    def path_to_mask(self, path: QPainterPath) -> np.ndarray:
        """Convert QPainterPath to boolean mask.
        
        Args:
            path: QPainterPath defining selection
            
        Returns:
            numpy.ndarray: Boolean mask of selected region
        """
        try:
            mask = np.zeros(self.image_shape, dtype=bool)
            
            # Create QImage to rasterize the path
            from PyQt6.QtGui import QImage, QPainter
            img = QImage(self.image_shape[1], self.image_shape[0],
                        QImage.Format.Format_ARGB32)
            img.fill(0)
            
            # Draw path
            painter = QPainter(img)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(Qt.GlobalColor.white)
            painter.drawPath(path)
            painter.end()
            
            # Convert to numpy array
            buffer = img.bits().asarray(img.height() * img.width() * 4)
            arr = np.frombuffer(buffer, dtype=np.uint8).reshape(
                (img.height(), img.width(), 4))
            
            # Use alpha channel as mask
            mask = arr[:, :, 3] > 0
            return mask
            
        except Exception as e:
            logger.error("Error converting path to mask")
            logger.exception(e)
            return np.zeros(self.image_shape, dtype=bool)
    
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
    
    def reset_selection(self):
        """Reset selection state."""
        self.is_selecting = False
        self.current_path = QPainterPath()
        self.points = []
        self.start_pos = None
        self.current_pos = None
        self.update()
    
    def set_mode(self, mode: str):
        """Set the current selection mode.
        
        Args:
            mode: Selection mode from SelectionMode
        """
        self.mode = mode
        self.reset_selection()
        logger.debug(f"Set selection mode to {mode}")
    
    def set_brush_size(self, size: int):
        """Set the brush size for paint selection.
        
        Args:
            size: Brush size in pixels
        """
        self.brush_size = size
        logger.debug(f"Set brush size to {size}")
    
    def set_tolerance(self, tolerance: float):
        """Set the tolerance for magic wand selection.
        
        Args:
            tolerance: Color difference tolerance (0-1)
        """
        self.tolerance = tolerance
        logger.debug(f"Set magic wand tolerance to {tolerance}")
    
    def set_image_shape(self, shape: tuple):
        """Set the shape of the image being processed.
        
        Args:
            shape: Image shape tuple
        """
        self.image_shape = shape
        self.prepareGeometryChange()
        logger.debug(f"Set image shape to {shape}")
    
    def set_transform(self, transform):
        """Set the geotransform for coordinate conversion.
        
        Args:
            transform: Rasterio transform object
        """
        self.transform = transform
        logger.debug("Set geotransform")