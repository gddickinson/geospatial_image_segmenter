"""Main window implementation for the geospatial segmentation application."""
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLabel, QFileDialog, QMessageBox,
                           QDockWidget, QToolBar, QStatusBar, QComboBox,
                           QProgressBar, QMenu, QMenuBar, QStyle, QApplication, QProgressDialog,
                           QFormLayout, QSpinBox, QDoubleSpinBox, QTabWidget, QCheckBox)
from PyQt6.QtCore import Qt, QSettings, QTimer
from PyQt6.QtGui import QAction, QIcon, QActionGroup, QImage, QPainter

import os
from datetime import datetime
import rasterio
import rasterio.transform
import geopandas as gpd
import platform
import serial
import math

from typing import Optional, List, Dict

from ..utils.logger import setup_logger
from ..utils.geo_utils import get_raster_info
from ..utils import lidar_utils
from .map_canvas import MapCanvas
from .layer_manager import LayerManager, RasterLayer, SegmentationLayer, TrainingLabelsLayer
from .. import config
from .dialogs.label_dialog import LabelingDialog
from .dialogs.training_dialog import  TrainingDialog
from .paint_tool import TrainingPaintTool
from ..models.random_forest import RandomForestModel
from ..models.cnn import CNNModel
from ..gui.map_widget import MapWidget
from ..data.providers.gps_providers import SerialGPSProvider, IPLocationProvider

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

        self.auto_following = False

        try:
            self.setup_ui()
            self.restore_settings()

            self.check_layer_references()

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

            # Create main central widget and layout
            main_widget = QWidget()
            self.setCentralWidget(main_widget)
            main_layout = QVBoxLayout(main_widget)

            # Create map canvas
            self.map_canvas = MapCanvas()
            main_layout.addWidget(self.map_canvas)

            # Create tab widget
            self.tab_widget = QTabWidget()
            main_layout.addWidget(self.tab_widget)

            # Set up layer manager dock
            self.setup_layer_manager()

            # Set up analysis dock
            self.setup_analysis_panel()

            # Create labeling dialog
            self.label_dialog = LabelingDialog(self)
            self.label_dialog.labels_changed.connect(self.update_label_overlay)

            # Create training paint tool
            self.paint_tool = TrainingPaintTool()
            self.paint_tool.label_dialog = self.label_dialog
            self.paint_tool.selection_changed.connect(self.update_label_overlay)

            # Store view menu as attribute in setup_menu_bar
            self.view_menu = self.menuBar.addMenu("&View")

            # Add Label Management to View menu
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

            # Add Map View tab
            self.setup_map_tab()

            # Initialize training layer (after first layer is loaded)
            active_layer = self.layer_manager.get_active_layer()
            if active_layer and isinstance(active_layer, RasterLayer):
                with rasterio.open(active_layer.path) as src:
                    shape = src.shape
                    extent = (src.bounds.left, src.bounds.bottom,
                             src.bounds.right, src.bounds.top)

                self.training_layer = TrainingLabelsLayer("Training Labels", shape)
                self.training_layer.extent = extent

                # Add to layer manager
                self.layer_manager.add_layer(self.training_layer)

                # Connect label dialog changes to update layer
                self.label_dialog.labels_changed.connect(self.update_training_layer)


            # Add this at the end to connect the visibility signal
            self.connect_visibility_signals()

            logger.debug("UI setup completed")

        except Exception as e:
            logger.error("Error setting up UI")
            logger.exception(e)
            raise

    def connect_visibility_signals(self):
        """Connect to layer visibility signals for direct UI updates."""
        try:
            # Get layer manager
            if hasattr(self, 'layer_manager') and hasattr(self.layer_manager, 'layer_tree'):
                # Connect to itemChanged signal
                self.layer_manager.layer_tree.itemChanged.connect(self.force_display_update)
                logger.debug("Connected to layer tree itemChanged signal")
        except Exception as e:
            logger.error(f"Error connecting visibility signals: {e}")

    def force_display_update(self, item, column):
        """Force immediate display update when layers change."""
        try:
            if column == 0:  # Visibility checkbox
                layer = item.data(0, Qt.ItemDataRole.UserRole)
                if not layer:
                    return

                # Get checkbox state
                is_checked = item.checkState(0) == Qt.CheckState.Checked

                # Update layer visibility (this will happen in the normal handler too)
                layer._visible = is_checked

                # CRITICAL: Wait a tiny bit to let normal handler run, then force update
                def delayed_update():
                    try:
                        logger.debug(f"FORCING UPDATE after visibility change to {is_checked}")

                        # Force map view to repaint itself
                        if hasattr(self, 'map_canvas') and hasattr(self.map_canvas, 'map_view'):
                            self.map_canvas.map_view.repaint()

                        # Also force map canvas to repaint
                        if hasattr(self, 'map_canvas'):
                            self.map_canvas.repaint()
                    except Exception as e:
                        logger.error(f"Error in delayed update: {e}")

                # Schedule updates - use multiple timers with different delays
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(10, delayed_update)  # Almost immediate
                QTimer.singleShot(50, delayed_update)  # Short delay
                QTimer.singleShot(200, delayed_update)  # Longer delay as fallback

        except Exception as e:
            logger.error(f"Error in force_display_update: {e}")

    def check_layer_references(self):
        """Debug function to check layer references."""
        logger.debug("\n==== CHECKING LAYER REFERENCES ====")

        # Check layers in layer manager
        if hasattr(self, 'layer_manager'):
            logger.debug(f"Layer manager has {len(self.layer_manager.layers)} layers")
            for i, layer in enumerate(self.layer_manager.layers):
                logger.debug(f"Manager layer[{i}]: {layer.name}, id={id(layer)}, visible={layer.visible}")

                # Check if found in tree widget
                found_in_tree = False
                root = self.layer_manager.layer_tree.invisibleRootItem()
                for j in range(root.childCount()):
                    item = root.child(j)
                    tree_layer = item.data(0, Qt.ItemDataRole.UserRole)
                    if tree_layer == layer:
                        found_in_tree = True
                        logger.debug(f"  Found in tree at index {j}, checkState={item.checkState(0)}")
                        break
                if not found_in_tree:
                    logger.debug("  NOT found in layer tree!")

        # Check layers in map canvas
        if hasattr(self, 'map_canvas'):
            logger.debug(f"Map canvas has {len(self.map_canvas._layers)} layers")
            for i, layer in enumerate(self.map_canvas._layers):
                logger.debug(f"Canvas layer[{i}]: {layer.name}, id={id(layer)}, visible={layer.visible}")

                # Check if found in layer manager
                if hasattr(self, 'layer_manager'):
                    found_in_manager = layer in self.layer_manager.layers
                    logger.debug(f"  Found in layer manager: {found_in_manager}")

                    # Check if it's the exact same object
                    for j, mgr_layer in enumerate(self.layer_manager.layers):
                        if mgr_layer.name == layer.name:
                            if mgr_layer is layer:
                                logger.debug(f"  Same object as manager layer[{j}], id={id(mgr_layer)}")
                            else:
                                logger.debug(f"  DIFFERENT object from manager layer[{j}], id={id(mgr_layer)} vs id={id(layer)}")

        logger.debug("==== LAYER REFERENCE CHECK COMPLETE ====\n")


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
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo)

        analysis_layout.addWidget(model_group)

        # Model parameters group
        self.params_group = QWidget()
        params_layout = QFormLayout(self.params_group)
        self.params_group.setLayout(params_layout)
        analysis_layout.addWidget(self.params_group)

        # Update parameters for current model
        self.update_model_parameters()

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

            logger.debug(f"Got {len(label_masks)} label masks")
            for name, mask in label_masks.items():
                if mask is not None:
                    logger.debug(f"Label '{name}' has {np.sum(mask)} pixels")

            # Filter out empty masks
            label_masks = {name: mask for name, mask in label_masks.items()
                         if mask is not None and np.any(mask)}

            # Load image data
            with rasterio.open(active_layer.path) as src:
                image_data = src.read()
                # Move bands to last dimension for processing
                image_data = np.moveaxis(image_data, 0, -1)

            # Initialize model based on selection
            model_type = self.model_combo.currentText()
            if model_type == "Random Forest":
                model = RandomForestModel(
                    n_estimators=self.rf_n_estimators.value(),
                    max_depth=self.rf_max_depth.value()
                )
            else:  # CNN
                model = CNNModel(
                    patch_size=self.cnn_patch_size.value(),
                    batch_size=self.cnn_batch_size.value(),
                    learning_rate=self.cnn_learning_rate.value(),
                    n_epochs=self.cnn_epochs.value()
                )

            # Add feature extractors
            if image_data.shape[-1] >= 3:  # If we have enough bands
                from ..models.features.spectral import SpectralFeatureExtractor
                spectral = SpectralFeatureExtractor()
                # Set band mapping based on data
                band_map = {'red': 0, 'green': 1, 'blue': 2}
                if image_data.shape[-1] > 3:
                    band_map['nir'] = 3
                spectral.set_band_map(band_map)
                model.add_feature_extractor('spectral', spectral)

            # Add texture features
            from ..models.features.texture import TextureFeatureExtractor
            texture = TextureFeatureExtractor()
            model.add_feature_extractor('texture', texture)

            # Show training dialog
            dialog = TrainingDialog(model_type, self)
            dialog.show()

            try:
                # Train model
                metrics = model.train(
                    image_data,
                    label_masks,
                    validation_split=0.2
                )

                # Update dialog with results
                if model_type == "Random Forest":
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
                else:  # CNN
                    # CNN training updates progress during training via callbacks
                    pass

                # Store model for later use
                self.current_model = model
                self.current_model_metrics = metrics

                dialog.training_finished(True)

                # Show results
                if model_type == "Random Forest":
                    msg = (f"Training accuracy: {metrics['training_score']:.2%}\n" +
                          (f"Validation accuracy: {metrics['validation_score']:.2%}"
                           if 'validation_score' in metrics else ""))
                else:
                    msg = (f"Final validation accuracy: {metrics['val_accuracy'][-1]:.2%}\n" +
                          f"Training loss: {metrics['train_loss'][-1]:.4f}")

                QMessageBox.information(self, "Training Complete", msg)

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

            # Clean up GPS if connected
            if self.gps_provider:
                self.gps_provider.close()

            # Call parent closeEvent
            super().closeEvent(event)

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
                        bounds = src.bounds
                        extent = (bounds.left, bounds.bottom, bounds.right, bounds.top)
                        logger.debug(f"Creating training layer with shape {shape} and extent {extent}")

                    # Create training layer if it doesn't exist
                    if not hasattr(self, 'training_layer') or self.training_layer not in self.layer_manager.layers:
                        self.training_layer = TrainingLabelsLayer("Training Labels", shape)
                        self.training_layer.extent = extent
                        self.training_layer.set_transform(transform)  # Set transform for consistent coordinates
                        # Add to both layer manager and map canvas
                        self.layer_manager.add_layer(self.training_layer)
                        self.map_canvas._layers.append(self.training_layer)
                        logger.debug(f"Added training layer to map canvas. Total layers: {len(self.map_canvas._layers)}")

                    # Setup paint tool for consistent coordinate transformation
                    self.paint_tool.set_image_shape(shape)
                    self.paint_tool.set_transform(transform)
                    self.paint_tool.set_extent(extent)  # Set extent for consistent coordinates
                    self.paint_tool.setParent(self.map_canvas.map_view)
                    self.paint_tool.show()
        else:
            # Hide the paint tool when not in training mode
            if hasattr(self, 'paint_tool'):
                self.paint_tool.hide()

    def update_label_overlay(self):
        """Update the label overlay when labels change."""
        try:
            if hasattr(self, 'training_layer'):
                logger.debug("Updating training layer")

                # Debug log the labels
                for name, label in self.label_dialog.labels.items():
                    if label.mask is not None:
                        logger.debug(f"Label {name} has {np.sum(label.mask)} pixels set")
                    else:
                        logger.debug(f"Label {name} has no mask")

                # Update the training layer with the current labels
                self.training_layer.update_labels(self.label_dialog.labels)

                # Force map canvas update
                self.map_canvas.update()

                # Also schedule a delayed update to ensure rendering
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(50, self.map_canvas.update)
                QTimer.singleShot(50, self.map_canvas.map_view.update)

                logger.debug("Training layer updated, canvas update requested")
        except Exception as e:
            logger.error(f"Error updating label overlay: {e}")
            logger.exception(e)

    def get_training_data(self) -> Dict[str, np.ndarray]:
        """Get training data from labels.

        Returns:
            dict: Dictionary mapping label names to boolean masks
        """
        return self.label_dialog.get_label_masks()

    def on_model_changed(self, model_name: str):
        """Handle model selection change."""
        self.update_model_parameters()

    def update_model_parameters(self):
        """Update parameter widgets based on selected model."""
        # Clear existing parameters
        layout = self.params_group.layout()
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add parameters based on selected model
        if self.model_combo.currentText() == "Random Forest":
            self.rf_n_estimators = QSpinBox()
            self.rf_n_estimators.setRange(10, 1000)
            self.rf_n_estimators.setValue(100)
            self.rf_n_estimators.setSingleStep(10)
            layout.addRow("Number of Trees:", self.rf_n_estimators)

            self.rf_max_depth = QSpinBox()
            self.rf_max_depth.setRange(1, 100)
            self.rf_max_depth.setValue(10)
            layout.addRow("Max Depth:", self.rf_max_depth)

        else:  # CNN
            self.cnn_patch_size = QSpinBox()
            self.cnn_patch_size.setRange(32, 512)
            self.cnn_patch_size.setValue(256)
            self.cnn_patch_size.setSingleStep(32)
            layout.addRow("Patch Size:", self.cnn_patch_size)

            self.cnn_batch_size = QSpinBox()
            self.cnn_batch_size.setRange(1, 32)
            self.cnn_batch_size.setValue(4)
            layout.addRow("Batch Size:", self.cnn_batch_size)

            self.cnn_epochs = QSpinBox()
            self.cnn_epochs.setRange(1, 200)
            self.cnn_epochs.setValue(50)
            layout.addRow("Epochs:", self.cnn_epochs)

            self.cnn_learning_rate = QDoubleSpinBox()
            self.cnn_learning_rate.setRange(0.0001, 0.1)
            self.cnn_learning_rate.setValue(0.001)
            self.cnn_learning_rate.setDecimals(4)
            self.cnn_learning_rate.setSingleStep(0.0001)
            layout.addRow("Learning Rate:", self.cnn_learning_rate)

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

    def update_training_layer(self):
        """Update training labels layer when labels change."""
        try:
            self.training_layer.update_labels(self.label_dialog.labels)
            self.map_canvas.update()
        except Exception as e:
            logger.error("Error updating training layer")
            logger.exception(e)
