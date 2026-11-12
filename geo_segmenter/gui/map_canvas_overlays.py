"""MapCanvas overlays: distance/area measurements and the training-label overlay."""
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPen, QColor
import numpy as np
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class MapCanvasOverlaysMixin:
    """Methods of MapCanvas (mixed in; see map_canvas.py)."""

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

    def draw_label_overlay(self, painter: QPainter, label_dialog):
        """Draw the training label overlay."""
        try:
            if not label_dialog:
                return

            label_masks = label_dialog.get_label_masks()
            if not label_masks:
                return

            for label_name, mask in label_masks.items():
                if mask is None:
                    continue

                label = label_dialog.labels[label_name]
                color = label.color

                # Create semi-transparent color for overlay
                overlay_color = QColor(color.red(), color.green(), color.blue(), 128)
                painter.setBrush(overlay_color)
                painter.setPen(Qt.PenStyle.NoPen)

                # Draw each masked pixel
                y_coords, x_coords = np.where(mask)
                for x, y in zip(x_coords, y_coords):
                    painter.drawRect(x, y, 1, 1)

            logger.debug(f"Drew overlays for {len(label_masks)} labels")

        except Exception as e:
            logger.error("Error drawing label overlay")
            logger.exception(e)
