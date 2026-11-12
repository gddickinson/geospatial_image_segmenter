"""MainWindow menu bar, toolbar, layer-manager dock, analysis panel and map tool selection."""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QDockWidget, QToolBar,
                             QComboBox, QMenuBar, QStyle, QFormLayout)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QActionGroup
from ..utils.logger import setup_logger
from .layer_manager import LayerManager

logger = setup_logger(__name__)


class MainWindowMenusMixin:
    """Methods of MainWindow (mixed in; see main_window.py)."""

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
