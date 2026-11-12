"""Main window implementation for the geospatial segmentation application.

MainWindow keeps the window layout, layer-visibility plumbing, settings and close handling; its other methods come
from mixins: main_window_menus (menu bar, toolbar, docks), main_window_io (projects, imports), main_window_training
(training mode, models, segmentation) and main_window_map_tab (base map, GPS, map capture).
"""
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QMessageBox, QStatusBar, QTabWidget
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QAction

import rasterio
import rasterio.transform

from ..utils.logger import setup_logger
from .map_canvas import MapCanvas
from .layer_manager import RasterLayer, TrainingLabelsLayer
from .dialogs.label_dialog import LabelingDialog
from .paint_tool import TrainingPaintTool
from .main_window_menus import MainWindowMenusMixin
from .main_window_io import MainWindowIOMixin
from .main_window_training import MainWindowTrainingMixin
from .main_window_map_tab import MainWindowMapTabMixin

logger = setup_logger(__name__)

class MainWindow(MainWindowMenusMixin, MainWindowIOMixin, MainWindowTrainingMixin, MainWindowMapTabMixin, QMainWindow):
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
















