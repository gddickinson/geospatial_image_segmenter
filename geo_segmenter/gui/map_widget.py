"""Map display components with GPS tracking support."""
import os
import time
import math
import sqlite3
import requests
from datetime import datetime
from typing import Optional, Dict
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                           QComboBox, QPushButton)
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF
from PyQt6.QtGui import QPainter, QPen, QColor, QPixmap, QPainterPath, QFont, QBrush

class TileManager:
    """Manages map tiles with local caching."""

    def __init__(self, cache_dir="map_cache"):
        self.cache_dir = cache_dir
        self.tile_servers = {
            'OpenStreetMap': {
                'url': 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                'user_agent': 'GeoSegmenter/1.0'
            },
            'OpenTopoMap': {
                'url': 'https://tile.opentopomap.org/{z}/{x}/{y}.png',
                'user_agent': 'GeoSegmenter/1.0'
            },
            'Satellite': {
                'url': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                'user_agent': 'GeoSegmenter/1.0'
            },
            'Hybrid': {
                'url': 'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
                'user_agent': 'GeoSegmenter/1.0'
            },
            'Elevation (Hillshade)': {
                'url': 'https://server.arcgisonline.com/arcgis/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}',
                'user_agent': 'GeoSegmenter/1.0'
            },
            'NAIP': {
                'url':'https://gis.apfo.usda.gov/arcgis/rest/services/NAIP/USDA_CONUS_PRIME/ImageServer/tile/{z}/{y}/{x}',
                'user_agent': 'GeoSegmenter/1.0'
            }

        }
        self.current_server = 'OpenStreetMap'
        self._init_cache()

    def _init_cache(self):
        """Initialize tile cache."""
        os.makedirs(self.cache_dir, exist_ok=True)
        self.db_path = os.path.join(self.cache_dir, 'tile_cache.db')

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS tiles
                          (key TEXT PRIMARY KEY, data BLOB, timestamp TEXT)''')

    def get_tile(self, x: int, y: int, zoom: int) -> QPixmap:
        """Get a map tile, either from cache or download."""
        tile_key = f"{self.current_server}/{zoom}/{x}/{y}"

        # Check cache first
        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute('SELECT data FROM tiles WHERE key = ?',
                                    (tile_key,)).fetchone()
                if result:
                    pixmap = QPixmap()
                    pixmap.loadFromData(result[0])
                    if not pixmap.isNull():
                        return pixmap
        except sqlite3.Error as e:
            print(f"Cache error: {e}")

        # Download if not in cache
        server_info = self.tile_servers[self.current_server]
        url = server_info['url'].format(z=zoom, x=x, y=y)
        headers = {'User-Agent': server_info['user_agent']}

        try:
            response = requests.get(url, headers=headers, timeout=3)
            if response.status_code == 200:
                # Cache the tile
                try:
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute('INSERT OR REPLACE INTO tiles VALUES (?, ?, ?)',
                                   (tile_key, response.content,
                                    datetime.now().isoformat()))
                except sqlite3.Error as e:
                    print(f"Cache write error: {e}")

                pixmap = QPixmap()
                if pixmap.loadFromData(response.content):
                    return pixmap

        except Exception as e:
            print(f"Download error: {e}")

        # Return default tile if all else fails
        return self._get_default_tile()

    def _get_default_tile(self) -> QPixmap:
        """Create a default tile for when loading fails."""
        pixmap = QPixmap(256, 256)
        pixmap.fill(QColor(240, 240, 240))
        return pixmap

class MapWidget(QWidget):
    """Interactive map widget with GPS tracking support."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)

        # Initialize map state
        self.zoom_level = 15
        self.center_lat = 0
        self.center_lon = 0
        self.track_points = []
        self.dragging = False
        self.last_pos = None

        # Initialize components
        self.tile_manager = TileManager()

        # Setup UI
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Controls
        controls = QHBoxLayout()

        # Map type selector
        self.map_type = QComboBox()
        self.map_type.addItems(self.tile_manager.tile_servers.keys())
        self.map_type.currentTextChanged.connect(self.change_map_type)
        controls.addWidget(QLabel("Map Type:"))
        controls.addWidget(self.map_type)

        # Zoom controls
        zoom_in = QPushButton("+")
        zoom_out = QPushButton("-")
        zoom_in.clicked.connect(self.zoom_in)
        zoom_out.clicked.connect(self.zoom_out)
        controls.addWidget(zoom_in)
        controls.addWidget(zoom_out)

        controls.addStretch()
        layout.addLayout(controls)

        # Status bar
        self.status_bar = QLabel()
        layout.addWidget(self.status_bar)

    def set_center(self, lat: float, lon: float):
        """Set the map center coordinates."""
        self.center_lat = lat
        self.center_lon = lon
        self.update()

    def update_track(self, points):
        """Update GPS track points."""
        self.track_points = points
        self.update()

    def latlon_to_pixel(self, lat: float, lon: float):
        """Convert latitude/longitude to pixel coordinates."""
        n = 2.0 ** self.zoom_level
        lat_rad = math.radians(lat)
        x = ((lon + 180.0) / 360.0 * n)
        y = ((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
        return x, y

    def pixel_to_latlon(self, x: float, y: float):
        """Convert pixel coordinates to latitude/longitude."""
        n = 2.0 ** self.zoom_level
        lon_deg = x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat_deg = math.degrees(lat_rad)
        return lat_deg, lon_deg

    def paintEvent(self, event):
        """Draw the map and overlays."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Calculate tile coordinates
        center_x, center_y = self.latlon_to_pixel(self.center_lat, self.center_lon)
        tile_x = int(center_x)
        tile_y = int(center_y)

        # Calculate pixel offsets
        pixel_x = int((center_x - tile_x) * 256)
        pixel_y = int((center_y - tile_y) * 256)

        # Draw visible tiles
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                x = tile_x + dx
                y = tile_y + dy

                if 0 <= x < 2**self.zoom_level and 0 <= y < 2**self.zoom_level:
                    tile = self.tile_manager.get_tile(x, y, self.zoom_level)
                    draw_x = int(self.width()/2 + (dx*256) - pixel_x)
                    draw_y = int(self.height()/2 + (dy*256) - pixel_y)
                    painter.drawPixmap(draw_x, draw_y, tile)

        # Draw track
        if self.track_points:
            painter.setPen(QPen(QColor(255, 0, 0, 180), 3))
            path = QPainterPath()
            first = True

            for lat, lon in self.track_points:
                x, y = self.latlon_to_pixel(lat, lon)
                screen_x = int(self.width()/2 + (x - center_x) * 256)
                screen_y = int(self.height()/2 + (y - center_y) * 256)

                if first:
                    path.moveTo(screen_x, screen_y)
                    first = False
                else:
                    path.lineTo(screen_x, screen_y)

            painter.drawPath(path)

            # Draw current position
            last_lat, last_lon = self.track_points[-1]
            x, y = self.latlon_to_pixel(last_lat, last_lon)
            screen_x = int(self.width()/2 + (x - center_x) * 256)
            screen_y = int(self.height()/2 + (y - center_y) * 256)

            painter.setBrush(QBrush(QColor(255, 0, 0)))
            painter.drawEllipse(QPointF(screen_x, screen_y), 5, 5)

        # Update status bar
        self.status_bar.setText(
            f"Center: {self.center_lat:.6f}, {self.center_lon:.6f} | "
            f"Zoom: {self.zoom_level}"
        )

    def zoom_in(self):
        """Increase zoom level."""
        self.zoom_level = min(19, self.zoom_level + 1)
        self.update()

    def zoom_out(self):
        """Decrease zoom level."""
        self.zoom_level = max(1, self.zoom_level - 1)
        self.update()

    def change_map_type(self, map_type):
        """Switch between different map tile sources."""
        self.tile_manager.current_server = map_type
        self.update()

    def mousePressEvent(self, event):
        """Handle mouse press for map dragging."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.last_pos = event.pos()

    def mouseReleaseEvent(self, event):
        """Handle mouse release for map dragging."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False

    def mouseMoveEvent(self, event):
        """Handle mouse movement for map dragging."""
        if self.dragging and self.last_pos:
            dx = event.pos().x() - self.last_pos.x()
            dy = event.pos().y() - self.last_pos.y()

            # Convert pixel movement to lat/lon
            n = 2.0 ** self.zoom_level
            dlat = dy / 256.0 / n * 360.0
            dlon = -dx / 256.0 / n * 360.0

            self.center_lat += dlat
            self.center_lon += dlon
            self.last_pos = event.pos()
            self.update()

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_in()
        elif delta < 0:
            self.zoom_out()
