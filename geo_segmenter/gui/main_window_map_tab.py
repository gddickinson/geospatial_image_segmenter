"""MainWindow base-map tab: GPS / IP location, centring and zoom, auto-follow, capture of the map view as a layer."""
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox, QComboBox, QCheckBox
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPainter
import os
from datetime import datetime
import rasterio
import rasterio.transform
import platform
import serial
import math
from ..utils.logger import setup_logger
from ..utils.geo_utils import get_raster_info
from .layer_manager import RasterLayer
from ..gui.map_widget import MapWidget
from ..data.providers.gps_providers import SerialGPSProvider, IPLocationProvider

logger = setup_logger(__name__)


class MainWindowMapTabMixin:
    """Methods of MainWindow (mixed in; see main_window.py)."""

    def setup_map_tab(self):
        """Set up the map view tab."""
        map_tab = QWidget()
        map_layout = QVBoxLayout(map_tab)

        # Create map widget
        self.map_widget = MapWidget()
        map_layout.addWidget(self.map_widget)

        # Location controls
        controls_layout = QHBoxLayout()

        # Center on location button
        center_btn = QPushButton("Center On:")
        controls_layout.addWidget(center_btn)

        # Location source combo
        self.location_source = QComboBox()
        self.location_source.addItems(["GPS", "IP Location", "Raster View"])
        controls_layout.addWidget(self.location_source)

        # Center button
        center_action = QPushButton("Go")
        center_action.clicked.connect(self.center_on_selected)
        controls_layout.addWidget(center_action)

        # Auto-follow checkbox
        self.auto_follow = QCheckBox("Auto-follow")
        self.auto_follow.stateChanged.connect(self.toggle_auto_follow)
        controls_layout.addWidget(self.auto_follow)

        # Add capture controls
        # Add to map tab controls
        capture_btn = QPushButton("Capture as Layer")
        capture_btn.clicked.connect(self.capture_map_view)

        # Add to map tab layout
        controls_layout.addWidget(capture_btn)


        map_layout.addLayout(controls_layout)

        # Add GPS controls
        gps_layout = QHBoxLayout()

        # GPS status
        self.gps_status = QLabel("GPS: Not Connected")
        gps_layout.addWidget(self.gps_status)

        # Port selection
        self.port_combo = QComboBox()
        self.refresh_ports()
        gps_layout.addWidget(QLabel("Port:"))
        gps_layout.addWidget(self.port_combo)

        # Refresh ports button
        refresh_btn = QPushButton("Refresh Ports")
        refresh_btn.clicked.connect(self.refresh_ports)
        gps_layout.addWidget(refresh_btn)

        # Connect/Disconnect button
        self.connect_btn = QPushButton("Connect GPS")
        self.connect_btn.clicked.connect(self.toggle_gps)
        gps_layout.addWidget(self.connect_btn)

        map_layout.addLayout(gps_layout)

        # Add tab
        self.tab_widget.addTab(map_tab, "Map View")

        # Initialize providers
        self.gps_provider = None
        self.ip_provider = IPLocationProvider()

        # Initialize timers
        self.gps_timer = QTimer()
        self.gps_timer.timeout.connect(self.update_gps)

        # Track state
        self.track_points = []
        self.auto_following = False

    def refresh_ports(self):
        """Refresh available serial ports."""
        self.port_combo.clear()

        if platform.system() == 'Darwin':  # macOS
            import glob
            ports = glob.glob('/dev/tty.*')
        elif platform.system() == 'Linux':
            import glob
            ports = glob.glob('/dev/tty[A-Za-z]*')
        else:  # Windows
            import serial.tools.list_ports
            ports = [port.device for port in serial.tools.list_ports.comports()]

        self.port_combo.addItems(ports)

    def toggle_gps(self):
        """Connect or disconnect GPS."""
        if not self.gps_provider:
            port = self.port_combo.currentText()
            try:
                self.gps_provider = SerialGPSProvider(port)
                self.connect_btn.setText("Disconnect GPS")
                self.gps_status.setText("GPS: Connected")
                self.gps_timer.start(1000)  # Update every second
            except Exception as e:
                QMessageBox.critical(self, "GPS Error", f"Failed to connect: {str(e)}")
        else:
            self.gps_provider.close()
            self.gps_provider = None
            self.connect_btn.setText("Connect GPS")
            self.gps_status.setText("GPS: Not Connected")
            self.gps_timer.stop()

    def update_gps(self):
        """Update GPS data and map display."""
        if self.gps_provider:
            try:
                data = self.gps_provider.get_location()
                if data and data.get('latitude') and data.get('longitude'):
                    # Update track
                    self.track_points.append((data['latitude'], data['longitude']))
                    if len(self.track_points) > 1000:  # Keep last 1000 points
                        self.track_points.pop(0)

                    # Update map
                    self.map_widget.update_track(self.track_points)
                    self.map_widget.set_center(data['latitude'], data['longitude'])

                    # Update status
                    if data.get('satellites'):
                        self.gps_status.setText(f"GPS: {data['satellites']} satellites")

            except Exception as e:
                print(f"GPS update error: {e}")
                self.gps_status.setText("GPS: Error")

    def center_on_selected(self):
        """Center map on selected location source."""
        try:
            source = self.location_source.currentText()

            if source == "GPS" and hasattr(self, 'gps_provider') and self.gps_provider:
                data = self.gps_provider.get_location()
                if data and data.get('latitude') and data.get('longitude'):
                    self.map_widget.set_center(data['latitude'], data['longitude'])

            elif source == "IP Location":
                if not hasattr(self, 'ip_provider'):
                    self.ip_provider = IPLocationProvider()
                data = self.ip_provider.get_location()
                if data and data.get('latitude') and data.get('longitude'):
                    self.map_widget.set_center(data['latitude'], data['longitude'])

            elif source == "Raster View":
                # Get active layer from layer manager
                active_layer = self.layer_manager.get_active_layer()
                if active_layer and isinstance(active_layer, RasterLayer):
                    try:
                        with rasterio.open(active_layer.path) as src:
                            bounds = src.bounds
                            center_lat = (bounds.bottom + bounds.top) / 2
                            center_lon = (bounds.left + bounds.right) / 2
                            self.map_widget.set_center(center_lat, center_lon)

                            # Calculate zoom level to fit raster
                            width = bounds.right - bounds.left
                            height = bounds.top - bounds.bottom
                            zoom = self.calculate_zoom_level(width, height)
                            self.map_widget.zoom_level = zoom
                            self.map_widget.update()
                    except Exception as e:
                        logger.error(f"Error centering on raster: {e}")

        except Exception as e:
            logger.error("Error in center_on_selected")
            logger.exception(e)

    def calculate_zoom_level(self, width: float, height: float) -> int:
        """Calculate appropriate zoom level to fit given dimensions."""
        try:
            # Convert dimensions to screen pixels
            screen_width = self.map_widget.width()
            screen_height = self.map_widget.height()

            # Calculate required scaling
            scale_x = width / screen_width
            scale_y = height / screen_height
            scale = max(scale_x, scale_y)

            # Convert scale to zoom level
            zoom = int(math.log2(360 / scale))
            return min(max(zoom, 1), 19)  # Clamp between 1 and 19

        except Exception as e:
            logger.error("Error calculating zoom level")
            logger.exception(e)
            return 15  # Return default zoom level if calculation fails

    def toggle_auto_follow(self, state):
        """Toggle auto-following of selected location source."""
        try:
            self.auto_following = state == Qt.CheckState.Checked

            # If enabled, immediately center on current location
            if self.auto_following:
                self.center_on_selected()

        except Exception as e:
            logger.error("Error in toggle_auto_follow")
            logger.exception(e)

    def capture_map_view(self):
        """Capture current map view as a georeferenced layer."""
        try:
            # Get current map view bounds
            center_lat = self.map_widget.center_lat
            center_lon = self.map_widget.center_lon
            zoom = self.map_widget.zoom_level

            # Calculate viewport bounds in world coordinates
            n = 2.0 ** zoom
            lat_rad = math.radians(center_lat)
            x = ((center_lon + 180.0) / 360.0 * n)
            y = ((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)

            # Get map widget size
            width = self.map_widget.width()
            height = self.map_widget.height()

            # Create QImage of current view
            image = QImage(width, height, QImage.Format.Format_RGB32)
            painter = QPainter(image)
            self.map_widget.render(painter)
            painter.end()

            # Save temporary image
            temp_dir = os.path.join(os.path.dirname(__file__), "temp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, "map_capture.tif")

            # Convert QImage to numpy array
            ptr = image.bits()
            ptr.setsize(height * width * 4)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
            arr = arr[:, :, :3]  # Remove alpha channel

            # Calculate world coordinates for corners
            x_min = center_lon - (width / 2) * (360 / (256 * n))
            x_max = center_lon + (width / 2) * (360 / (256 * n))
            y_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + height/2/256) / n))))
            y_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y - height/2/256) / n))))

            # Create GeoTIFF
            transform = rasterio.transform.from_bounds(
                x_min, y_min, x_max, y_max, width, height)

            with rasterio.open(
                temp_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=3,
                dtype=arr.dtype,
                crs='EPSG:4326',
                transform=transform,
            ) as dst:
                for i in range(3):
                    dst.write(arr[:, :, i], i + 1)

            # Get raster info
            info = get_raster_info(temp_path)

            # Create and add new layer
            layer_name = f"MapCapture_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            layer = RasterLayer(layer_name, temp_path, info)

            # Add to layer manager and map canvas
            self.layer_manager.add_layer(layer)
            self.map_canvas.add_layer(layer)

            QMessageBox.information(
                self,
                "Success",
                "Map view captured and added as new layer"
            )

        except Exception as e:
            logger.error("Error capturing map view")
            logger.exception(e)
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to capture map view: {str(e)}"
            )
