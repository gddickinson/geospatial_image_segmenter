"""Main window implementation for the geospatial segmentation application."""
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLabel, QFileDialog, QMessageBox,
                           QDockWidget, QToolBar, QStatusBar, QComboBox,
                           QProgressBar, QMenu, QMenuBar, QStyle, QApplication, QProgressDialog)
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QAction, QIcon, QActionGroup

import rasterio
import geopandas as gpd
from typing import Optional, List, Dict

from ..utils.logger import setup_logger
from ..utils.geo_utils import get_raster_info
from ..utils import lidar_utils
from .map_canvas import MapCanvas
from .layer_manager import LayerManager, RasterLayer, SegmentationLayer
from .. import config
from .dialogs.label_dialog import LabelingDialog
from .dialogs.training_dialog import  TrainingDialog
from .paint_tool import TrainingPaintTool
from ..models.random_forest import RandomForestModel

logger = setup_logger(__name__)

class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        self.setWindowTitle("Geospatial Segmentation Tool")
        self.resize(1200, 800)

        # Initialize state
        self.current_project = None
        self.settings = QSettings('GeoSegmenter', 'GeoSegmentationTool')

        try:
            self.setup_ui()
            self.restore_settings()
            logger.info("MainWindow initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing MainWindow: {str(e)}")
            logger.exception(e)
            raise

    def setup_ui(self):
        """Set up the user interface."""
        try:
            # Set up menu bar
            self.setup_menu_bar()

            # Set up toolbar
            self.setup_toolbar()

            # Set up status bar
            self.statusBar = QStatusBar()
            self.setStatusBar(self.statusBar)

            # Set up central widget (map canvas)
            self.map_canvas = MapCanvas()
            self.setCentralWidget(self.map_canvas)

            # Set up layer manager dock
            self.setup_layer_manager()

            # Set up analysis dock
            self.setup_analysis_panel()

            logger.debug("UI setup completed")

            # Create labeling dialog
            self.label_dialog = LabelingDialog(self)
            self.label_dialog.labels_changed.connect(self.update_label_overlay)

            # Create training paint tool
            self.paint_tool = TrainingPaintTool()
            self.paint_tool.label_dialog = self.label_dialog
            self.paint_tool.selection_changed.connect(self.update_label_overlay)

            # Add Label Management to View menu
            view_label_manager = QAction("Label Manager", self)
            view_label_manager.setCheckable(True)
            view_label_manager.setChecked(False)
            view_label_manager.triggered.connect(
                lambda checked: self.label_dialog.setVisible(checked))
            self.view_menu.addAction(view_label_manager)

            # Store view menu as attribute in setup_menu_bar
            self.view_menu = self.menuBar.addMenu("&View")

            # Add label manager to view menu
            view_label_manager = QAction("Label Manager", self)
            view_label_manager.setCheckable(True)
            view_label_manager.setChecked(False)
            view_label_manager.triggered.connect(
                lambda checked: self.label_dialog.setVisible(checked))
            self.view_menu.addAction(view_label_manager)

            # Add training mode action to toolbar
            self.training_action = QAction("Training Mode", self)
            self.training_action.setCheckable(True)
            self.training_action.setToolTip("Enter Training Mode")
            self.training_action.triggered.connect(self.toggle_training_mode)
            self.toolbar.addAction(self.training_action)

        except Exception as e:
            logger.error("Error setting up UI")
            logger.exception(e)
            raise

    def setup_menu_bar(self):
        """Set up the application menu bar."""
        self.menuBar = QMenuBar()
        self.setMenuBar(self.menuBar)

        # File menu
        self.file_menu = self.menuBar.addMenu("&File")

        new_project = QAction("New Project", self)
        new_project.setShortcut("Ctrl+N")
        new_project.triggered.connect(self.new_project)
        self.file_menu.addAction(new_project)

        open_project = QAction("Open Project", self)
        open_project.setShortcut("Ctrl+O")
        open_project.triggered.connect(self.open_project)
        self.file_menu.addAction(open_project)

        save_project = QAction("Save Project", self)
        save_project.setShortcut("Ctrl+S")
        save_project.triggered.connect(self.save_project)
        self.file_menu.addAction(save_project)

        self.file_menu.addSeparator()

        # Import submenu
        import_menu = self.file_menu.addMenu("Import")

        # Import Raster action
        import_raster = QAction("Import Raster...", self)
        import_raster.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))
        import_raster.triggered.connect(lambda: self.import_raster())
        import_menu.addAction(import_raster)

        # Import Vector action
        import_vector = QAction("Import Vector...", self)
        import_vector.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon))
        import_vector.triggered.connect(lambda: self.import_vector())
        import_menu.addAction(import_vector)

        # Import LIDAR action
        import_lidar = QAction("Import LIDAR...", self)
        import_lidar.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DriveNetIcon))
        import_lidar.triggered.connect(lambda: self.import_lidar())
        import_menu.addAction(import_lidar)

        self.file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        self.file_menu.addAction(exit_action)

        # View menu
        self.view_menu = self.menuBar.addMenu("&View")

        view_layer_manager = QAction("Layer Manager", self)
        view_layer_manager.setCheckable(True)
        view_layer_manager.setChecked(True)
        view_layer_manager.triggered.connect(
            lambda checked: self.layer_manager_dock.setVisible(checked))
        self.view_menu.addAction(view_layer_manager)

        view_analysis_panel = QAction("Analysis Panel", self)
        view_analysis_panel.setCheckable(True)
        view_analysis_panel.setChecked(True)
        view_analysis_panel.triggered.connect(
            lambda checked: self.analysis_dock.setVisible(checked))
        self.view_menu.addAction(view_analysis_panel)

        # Analysis menu
        self.analysis_menu = self.menuBar.addMenu("&Analysis")

        segment_action = QAction("Segment Region...", self)
        segment_action.triggered.connect(self.start_segmentation)
        self.analysis_menu.addAction(segment_action)

        classify_action = QAction("Classify Features...", self)
        classify_action.triggered.connect(self.start_classification)
        self.analysis_menu.addAction(classify_action)

        self.analysis_menu.addSeparator()

        calculate_indices = QAction("Calculate Indices...", self)
        calculate_indices.triggered.connect(self.calculate_spectral_indices)
        self.analysis_menu.addAction(calculate_indices)

        # Help menu
        self.help_menu = self.menuBar.addMenu("&Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        self.help_menu.addAction(about_action)

    def setup_toolbar(self):
        """Set up the main toolbar."""
        self.toolbar = QToolBar()
        self.toolbar.setObjectName("main_toolbar")
        self.addToolBar(self.toolbar)

        # Import actions group
        import_group = QActionGroup(self)

        # Import Raster action
        raster_action = QAction("Import Raster", self)
        raster_action.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))
        raster_action.setToolTip("Import Raster Layer")
        raster_action.triggered.connect(lambda: self.import_raster())
        self.toolbar.addAction(raster_action)
        import_group.addAction(raster_action)

        # Import Vector action
        vector_action = QAction("Import Vector", self)
        vector_action.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon))
        vector_action.setToolTip("Import Vector Layer")
        vector_action.triggered.connect(lambda: self.import_vector())
        self.toolbar.addAction(vector_action)
        import_group.addAction(vector_action)

        # Import LIDAR action
        lidar_action = QAction("Import LIDAR", self)
        lidar_action.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DriveNetIcon))
        lidar_action.setToolTip("Import LIDAR Data")
        lidar_action.triggered.connect(lambda: self.import_lidar())
        self.toolbar.addAction(lidar_action)
        import_group.addAction(lidar_action)

        self.toolbar.addSeparator()

        # Navigation tools
        nav_group = QActionGroup(self)
        nav_group.setExclusive(True)

        # Pan tool
        self.pan_action = QAction("Pan", self)
        self.pan_action.setCheckable(True)
        self.pan_action.setToolTip("Pan Tool")
        self.pan_action.triggered.connect(lambda: self.set_map_tool("pan"))
        self.toolbar.addAction(self.pan_action)
        nav_group.addAction(self.pan_action)

        # Zoom tool
        self.zoom_action = QAction("Zoom", self)
        self.zoom_action.setCheckable(True)
        self.zoom_action.setToolTip("Zoom Tool")
        self.zoom_action.triggered.connect(lambda: self.set_map_tool("zoom"))
        self.toolbar.addAction(self.zoom_action)
        nav_group.addAction(self.zoom_action)

        self.toolbar.addSeparator()

        # Selection tools
        select_group = QActionGroup(self)
        select_group.setExclusive(True)

        self.select_action = QAction("Select", self)
        self.select_action.setCheckable(True)
        self.select_action.setToolTip("Select Features")
        self.select_action.triggered.connect(lambda: self.set_map_tool("select"))
        self.toolbar.addAction(self.select_action)
        select_group.addAction(self.select_action)

        self.polygon_select_action = QAction("Polygon Select", self)
        self.polygon_select_action.setCheckable(True)
        self.polygon_select_action.setToolTip("Select Features by Polygon")
        self.polygon_select_action.triggered.connect(
            lambda: self.set_map_tool("polygon_select"))
        self.toolbar.addAction(self.polygon_select_action)
        select_group.addAction(self.polygon_select_action)

        self.toolbar.addSeparator()

        # Measurement tools
        measure_group = QActionGroup(self)
        measure_group.setExclusive(True)

        self.measure_distance_action = QAction("Measure Distance", self)
        self.measure_distance_action.setCheckable(True)
        self.measure_distance_action.setToolTip("Measure Distance")
        self.measure_distance_action.triggered.connect(
            lambda: self.set_map_tool("measure_distance"))
        self.toolbar.addAction(self.measure_distance_action)
        measure_group.addAction(self.measure_distance_action)

        self.measure_area_action = QAction("Measure Area", self)
        self.measure_area_action.setCheckable(True)
        self.measure_area_action.setToolTip("Measure Area")
        self.measure_area_action.triggered.connect(
            lambda: self.set_map_tool("measure_area"))
        self.toolbar.addAction(self.measure_area_action)
        measure_group.addAction(self.measure_area_action)

        logger.debug("Toolbar setup completed")


    def setup_layer_manager(self):
        """Set up the layer manager dock widget."""
        self.layer_manager_dock = QDockWidget("Layer Manager", self)
        self.layer_manager_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea |
            Qt.DockWidgetArea.RightDockWidgetArea)

        self.layer_manager = LayerManager(self)
        self.layer_manager_dock.setWidget(self.layer_manager)

        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea,
                          self.layer_manager_dock)

    def setup_analysis_panel(self):
        """Set up the analysis panel dock widget."""
        self.analysis_dock = QDockWidget("Analysis", self)
        self.analysis_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea |
            Qt.DockWidgetArea.RightDockWidgetArea)

        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout(analysis_widget)

        # Model selection
        model_group = QWidget()
        model_layout = QHBoxLayout(model_group)
        model_layout.addWidget(QLabel("Model:"))

        self.model_combo = QComboBox()
        self.model_combo.addItems(["Random Forest", "CNN"])
        model_layout.addWidget(self.model_combo)

        analysis_layout.addWidget(model_group)

        # Training controls
        train_btn = QPushButton("Train Model")
        train_btn.clicked.connect(self.train_model)
        analysis_layout.addWidget(train_btn)

        segment_btn = QPushButton("Run Segmentation")
        segment_btn.clicked.connect(self.run_segmentation)
        analysis_layout.addWidget(segment_btn)

        analysis_layout.addStretch()

        self.analysis_dock.setWidget(analysis_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,
                          self.analysis_dock)

    def set_map_tool(self, tool: str):
        """Set the active map tool.

        Args:
            tool: Tool identifier
        """
        try:
            # Uncheck all tool actions
            for action in self.toolbar.actions():
                if action.isCheckable():
                    action.setChecked(False)

            # Set the selected tool
            if tool == "pan":
                self.pan_action.setChecked(True)
                self.map_canvas.set_tool("pan")
            elif tool == "zoom":
                self.zoom_action.setChecked(True)
                self.map_canvas.set_tool("zoom")
            elif tool == "select":
                self.select_action.setChecked(True)
                self.map_canvas.set_tool("select")
            elif tool == "polygon_select":
                self.polygon_select_action.setChecked(True)
                self.map_canvas.set_tool("polygon_select")
            elif tool == "measure_distance":
                self.measure_distance_action.setChecked(True)
                self.map_canvas.set_tool("measure_distance")
            elif tool == "measure_area":
                self.measure_area_action.setChecked(True)
                self.map_canvas.set_tool("measure_area")

            logger.debug(f"Set map tool to: {tool}")

        except Exception as e:
            logger.error(f"Error setting map tool: {str(e)}")
            logger.exception(e)

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

    def start_segmentation(self):
        """Start the segmentation workflow."""
        try:
            if not self.layer_manager.get_active_layer():
                QMessageBox.warning(self, "Warning",
                                  "Please select a layer to segment")
                return

            # TODO: Implement segmentation workflow
            logger.info("Started segmentation workflow")

        except Exception as e:
            logger.error("Error starting segmentation")
            logger.exception(e)
            QMessageBox.critical(self, "Error", str(e))

    def start_classification(self):
        """Start the classification workflow."""
        try:
            if not self.layer_manager.get_active_layer():
                QMessageBox.warning(self, "Warning",
                                  "Please select a layer to classify")
                return

            # TODO: Implement classification workflow
            logger.info("Started classification workflow")

        except Exception as e:
            logger.error("Error starting classification")
            logger.exception(e)
            QMessageBox.critical(self, "Error", str(e))

    def calculate_spectral_indices(self):
        """Calculate spectral indices for the active layer."""
        try:
            active_layer = self.layer_manager.get_active_layer()
            if not active_layer or active_layer.layer_type != "raster":
                QMessageBox.warning(self, "Warning",
                                  "Please select a raster layer")
                return

            # TODO: Implement spectral indices calculation
            logger.info("Started spectral indices calculation")

        except Exception as e:
            logger.error("Error calculating spectral indices")
            logger.exception(e)
            QMessageBox.critical(self, "Error", str(e))

    def train_model(self):
        """Train the selected model."""
        try:
            # Get active layer
            active_layer = self.layer_manager.get_active_layer()
            if not active_layer or not isinstance(active_layer, RasterLayer):
                QMessageBox.warning(self, "Warning",
                                  "Please select a raster layer for training")
                return

            # Get training data
            label_masks = self.label_dialog.get_label_masks()
            if not label_masks:
                QMessageBox.warning(self, "Warning",
                                  "Please create training samples first")
                return

            # Show training dialog early
            dialog = TrainingDialog("Random Forest", self)
            dialog.show()

            # Update progress
            dialog.progress_label.setText("Loading image data...")
            QApplication.processEvents()

            # Load image data
            with rasterio.open(active_layer.path) as src:
                image_data = src.read()
                # Move bands to last dimension for processing
                image_data = np.moveaxis(image_data, 0, -1)

            # Initialize model
            dialog.progress_label.setText("Initializing model...")
            QApplication.processEvents()

            model = RandomForestModel()

            # Add feature extractors
            if image_data.shape[-1] >= 3:
                dialog.progress_label.setText("Setting up spectral features...")
                QApplication.processEvents()

                from ..models.features.spectral import SpectralFeatureExtractor
                spectral = SpectralFeatureExtractor()
                band_map = {'red': 0, 'green': 1, 'blue': 2}
                if image_data.shape[-1] > 3:
                    band_map['nir'] = 3
                spectral.set_band_map(band_map)
                model.add_feature_extractor('spectral', spectral)

            # Add texture features
            dialog.progress_label.setText("Setting up texture features...")
            QApplication.processEvents()

            from ..models.features.texture import TextureFeatureExtractor
            texture = TextureFeatureExtractor()
            model.add_feature_extractor('texture', texture)

            try:
                dialog.progress_label.setText("Training model...")
                dialog.progress_bar.setValue(0)
                QApplication.processEvents()

                # Train model
                metrics = model.train(
                    image_data,
                    label_masks,
                    validation_split=0.2
                )

                dialog.progress_bar.setValue(100)

                # Update dialog with results
                if 'feature_importances' in metrics:
                    dialog.update_feature_importance(metrics['feature_importances'])

                if 'training_score' in metrics:
                    dialog.update_progress(
                        1, 1,  # RF trains in one step
                        {
                            'train_accuracy': [metrics['training_score']],
                            'val_accuracy': [metrics.get('validation_score', 0)]
                        }
                    )

                # Store model for later use
                self.current_model = model
                self.current_model_metrics = metrics

                dialog.training_finished(True)

                # Show results
                QMessageBox.information(
                    self,
                    "Training Complete",
                    f"Training accuracy: {metrics['training_score']:.2%}\n" +
                    (f"Validation accuracy: {metrics['validation_score']:.2%}"
                     if 'validation_score' in metrics else "")
                )

            except Exception as e:
                dialog.show_error(str(e))
                raise

        except Exception as e:
            logger.error("Error training model")
            logger.exception(e)
            QMessageBox.critical(self, "Error", str(e))

    def run_segmentation(self):
        """Run segmentation using the trained model."""
        try:
            if not hasattr(self, 'current_model') or not self.current_model:
                QMessageBox.warning(self, "Warning",
                                  "Please train a model first")
                return

            # Get active layer
            active_layer = self.layer_manager.get_active_layer()
            if not active_layer or not isinstance(active_layer, RasterLayer):
                QMessageBox.warning(self, "Warning",
                                  "Please select a raster layer for segmentation")
                return

            # Show progress dialog
            progress = QProgressDialog("Running segmentation...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()
            QApplication.processEvents()

            try:
                # Load image data
                with rasterio.open(active_layer.path) as src:
                    image_data = src.read()
                    image_data = np.moveaxis(image_data, 0, -1)
                    transform = src.transform
                    extent = (
                        src.bounds.left,
                        src.bounds.bottom,
                        src.bounds.right,
                        src.bounds.top
                    )

                progress.setValue(25)
                QApplication.processEvents()

                # Run prediction
                prediction = self.current_model.predict(image_data)

                progress.setValue(75)
                QApplication.processEvents()

                # Create color mapping from training labels
                class_colors = {
                    label.name: label.color
                    for label in self.label_dialog.labels.values()
                }

                # Create segmentation layer
                layer_name = f"{active_layer.name}_segmentation"
                seg_layer = SegmentationLayer(
                    layer_name,
                    prediction,
                    class_colors,
                    extent
                )

                # Add to layer manager and map
                self.layer_manager.add_layer(seg_layer)
                self.map_canvas.add_layer(seg_layer)

                progress.setValue(100)
                QMessageBox.information(self, "Segmentation Complete",
                                    "Segmentation completed successfully!")

            finally:
                progress.close()

        except Exception as e:
            logger.error("Error running segmentation")
            logger.exception(e)
            QMessageBox.critical(self, "Error", str(e))

    def show_about(self):
        """Show the about dialog."""
        QMessageBox.about(
            self,
            "About Geospatial Segmentation Tool",
            """<h3>Geospatial Segmentation Tool</h3>
            <p>A PyQt-based application for segmenting and analyzing
            satellite and aerial imagery.</p>
            <p>Version 1.0</p>"""
        )

    def save_settings(self):
        """Save application settings."""
        try:
            self.settings.setValue("geometry", self.saveGeometry())
            self.settings.setValue("windowState", self.saveState())
            logger.debug("Saved application settings")

        except Exception as e:
            logger.error("Error saving settings")
            logger.exception(e)

    def restore_settings(self):
        """Restore application settings."""
        try:
            if self.settings.value("geometry"):
                self.restoreGeometry(self.settings.value("geometry"))
            if self.settings.value("windowState"):
                self.restoreState(self.settings.value("windowState"))
            logger.debug("Restored application settings")

        except Exception as e:
            logger.error("Error restoring settings")
            logger.exception(e)

    def closeEvent(self, event):
        """Handle application closure."""
        try:
            # Save settings
            self.save_settings()

            # Clean up resources
            self.map_canvas.cleanup()

            logger.info("Application closing")
            event.accept()

        except Exception as e:
            logger.error("Error during application closure")
            logger.exception(e)
            event.accept()  # Still close even if there's an error

    def toggle_training_mode(self, enabled: bool):
        """Toggle training mode."""
        # Set training mode flag on map canvas
        self.map_canvas.training_mode = enabled

        if enabled:
            # Show label dialog
            self.label_dialog.show()

            # Set up paint tool
            if self.layer_manager.get_active_layer():
                layer = self.layer_manager.get_active_layer()
                if isinstance(layer, RasterLayer):
                    # Get image shape and transform from layer
                    with rasterio.open(layer.path) as src:
                        shape = src.shape
                        transform = src.transform
                    self.paint_tool.set_image_shape(shape)
                    self.paint_tool.set_transform(transform)

                    # Set up paint tool
                    self.paint_tool.set_brush_size(self.label_dialog.brush_size)
                    self.paint_tool.setParent(self.map_canvas.map_view)
                    self.paint_tool.show()
                else:
                    QMessageBox.warning(self, "Warning",
                                      "Please select a raster layer for training")
                    self.training_action.setChecked(False)
                    return
            else:
                QMessageBox.warning(self, "Warning",
                                  "Please select a layer for training")
                self.training_action.setChecked(False)
                return
        else:
            # Hide label dialog and paint tool
            self.label_dialog.hide()
            if hasattr(self, 'paint_tool'):
                self.paint_tool.hide()
                self.paint_tool.setParent(None)

    def update_label_overlay(self):
        """Update the label overlay on the map."""
        # Force map canvas to redraw
        self.map_canvas.update()

    def get_training_data(self) -> Dict[str, np.ndarray]:
        """Get training data from labels.

        Returns:
            dict: Dictionary mapping label names to boolean masks
        """
        return self.label_dialog.get_label_masks()
