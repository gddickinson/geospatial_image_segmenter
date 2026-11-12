"""MainWindow project actions and data import (raster, vector, LIDAR)."""
from pathlib import Path
from PyQt6.QtWidgets import QFileDialog, QMessageBox
import geopandas as gpd
from typing import Optional
from ..utils.logger import setup_logger
from ..utils.geo_utils import get_raster_info
from ..utils import lidar_utils
from .layer_manager import RasterLayer
from .. import config

logger = setup_logger(__name__)


class MainWindowIOMixin:
    """Methods of MainWindow (mixed in; see main_window.py)."""

    def new_project(self):
        """Create a new project."""
        try:
            # TODO: Implement project creation
            pass
        except Exception as e:
            logger.error("Error creating new project")
            logger.exception(e)
            QMessageBox.critical(self, "Error", str(e))

    def open_project(self):
        """Open an existing project."""
        try:
            # TODO: Implement project loading
            pass
        except Exception as e:
            logger.error("Error opening project")
            logger.exception(e)
            QMessageBox.critical(self, "Error", str(e))

    def save_project(self):
        """Save the current project."""
        try:
            # TODO: Implement project saving
            pass
        except Exception as e:
            logger.error("Error saving project")
            logger.exception(e)
            QMessageBox.critical(self, "Error", str(e))

    def import_raster(self, file_path: Optional[str] = None):
        """Import a raster dataset.

        Args:
            file_path: Optional path to raster file. If not provided, shows file dialog.
        """
        try:
            logger.debug("import_raster called")  # Debug log
            if file_path is None:
                file_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Import Raster",
                    "",
                    "GeoTIFF (*.tif *.tiff);;All Files (*.*)"
                )
                logger.debug(f"File dialog result: {file_path}")  # Debug log

            if file_path:
                logger.debug(f"Importing raster: {file_path}")
                info = get_raster_info(file_path)
                logger.debug(f"Raster info: {info}")

                # Create layer
                layer = RasterLayer(Path(file_path).stem, file_path, info)

                # Add to layer manager
                self.layer_manager.add_layer(layer)

                # Add directly to map canvas
                self.map_canvas.add_layer(layer)
                logger.debug(f"Added layer to map canvas, total layers: {len(self.map_canvas._layers)}")

                # Force canvas update
                self.map_canvas.update()
                logger.debug("Requested canvas update after raster import")

                logger.info(f"Imported raster: {file_path}")

        except Exception as e:
            logger.error(f"Error importing raster: {str(e)}")
            logger.exception(e)
            QMessageBox.critical(self, "Error", str(e))

    def import_vector(self):
        """Import a vector dataset."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Import Vector",
                "",
                "Shapefiles (*.shp);;GeoJSON (*.geojson);;All Files (*.*)"
            )

            if file_path:
                gdf = gpd.read_file(file_path)
                self.layer_manager.add_vector_layer(file_path, gdf)
                logger.info(f"Imported vector: {file_path}")

        except Exception as e:
            logger.error("Error importing vector")
            logger.exception(e)
            QMessageBox.critical(self, "Error", str(e))

    def import_lidar(self):
        """Import a LIDAR dataset."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Import LIDAR",
                "",
                "LAS/LAZ Files (*.las *.laz);;All Files (*.*)"
            )

            if file_path:
                point_cloud, metadata = lidar_utils.read_las_file(
                    file_path,
                    max_points=config.LIDAR_MAX_POINTS
                )
                self.layer_manager.add_lidar_layer(file_path, point_cloud, metadata)
                logger.info(f"Imported LIDAR: {file_path}")

        except Exception as e:
            logger.error("Error importing LIDAR")
            logger.exception(e)
            QMessageBox.critical(self, "Error", str(e))
