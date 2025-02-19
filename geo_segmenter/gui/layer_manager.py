"""Layer management interface for organizing and controlling map layers."""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
                           QPushButton, QHBoxLayout, QMenu, QInputDialog,
                           QMessageBox, QStyle, QFormLayout, QDialog, QComboBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QAction

from pathlib import Path
from typing import Optional, List, Dict

import rasterio

from ..data.layer import Layer, RasterLayer, VectorLayer, LidarLayer, SegmentationLayer
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class LayerManager(QWidget):
    """Widget for managing map layers."""

    # Signals
    layer_added = pyqtSignal(Layer)
    layer_removed = pyqtSignal(Layer)
    layer_changed = pyqtSignal(Layer)
    active_layer_changed = pyqtSignal(Layer)

    def __init__(self, parent=None):
        """Initialize layer manager.

        Args:
            parent: Parent widget (should be MainWindow)
        """
        super().__init__()  # Don't pass parent to QWidget
        self.main_window = parent  # Store MainWindow reference
        self.layers = []
        self.active_layer = None
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Layer tree
        self.layer_tree = QTreeWidget()
        self.layer_tree.setHeaderLabels(["Layers"])
        self.layer_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.layer_tree.customContextMenuRequested.connect(self.show_context_menu)
        self.layer_tree.itemSelectionChanged.connect(self.on_selection_changed)
        layout.addWidget(self.layer_tree)

        # Buttons
        button_layout = QHBoxLayout()

        add_btn = QPushButton("Add Layer")
        add_btn.clicked.connect(self.show_add_menu)
        button_layout.addWidget(add_btn)

        remove_btn = QPushButton("Remove Layer")
        remove_btn.clicked.connect(self.remove_selected_layer)
        button_layout.addWidget(remove_btn)

        layout.addLayout(button_layout)

    def show_add_menu(self):
        """Show menu for adding different layer types."""
        menu = QMenu(self)

        add_raster = QAction("Add Raster Layer", self)
        add_raster.triggered.connect(self._import_raster)
        menu.addAction(add_raster)

        add_vector = QAction("Add Vector Layer", self)
        add_vector.triggered.connect(self._import_vector)
        menu.addAction(add_vector)

        add_lidar = QAction("Add LIDAR Layer", self)
        add_lidar.triggered.connect(self._import_lidar)
        menu.addAction(add_lidar)

        menu.exec(self.mapToGlobal(self.rect().bottomLeft()))

    def add_raster_layer(self, path: str, info: Dict) -> None:
        """Add a raster layer to the manager.

        Args:
            path: Path to raster file
            info: Raster information dictionary
        """
        try:
            name = Path(path).stem
            layer = RasterLayer(name, path, info)

            # Add to layer manager
            self.add_layer(layer)
            logger.debug(f"Added layer to layer manager: {layer.name}")

            # Add to map canvas
            main_window = self.parent()
            if main_window and hasattr(main_window, 'map_canvas'):
                logger.debug(f"Adding layer {name} to map canvas")
                main_window.map_canvas.add_layer(layer)
                # Zoom to layer
                main_window.map_canvas.zoom_to_layer(layer)

            logger.debug(f"Added raster layer: {path}")

        except Exception as e:
            logger.error("Error adding raster layer")
            logger.exception(e)
            raise

    def add_vector_layer(self, path: str, data) -> None:
        """Add a vector layer to the manager.

        Args:
            path: Path to vector file
            data: GeoDataFrame with vector data
        """
        try:
            name = Path(path).stem
            layer = VectorLayer(name, path, data)
            self.add_layer(layer)

        except Exception as e:
            logger.error("Error adding vector layer")
            logger.exception(e)
            raise

    def add_lidar_layer(self, path: str, point_cloud, metadata: Dict) -> None:
        """Add a LIDAR layer to the manager.

        Args:
            path: Path to LIDAR file
            point_cloud: Point cloud array
            metadata: LIDAR metadata dictionary
        """
        try:
            name = Path(path).stem
            layer = LidarLayer(name, path, point_cloud, metadata)
            self.add_layer(layer)

        except Exception as e:
            logger.error("Error adding LIDAR layer")
            logger.exception(e)
            raise

    def add_layer(self, layer: Layer) -> None:
        """Add a layer to the manager.

        Args:
            layer: Layer object to add
        """
        try:
            # Add to internal list
            self.layers.append(layer)

            # Create tree item
            item = QTreeWidgetItem([layer.name])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Checked)

            # Add icon based on layer type
            if isinstance(layer, RasterLayer):
                icon = self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon)
            elif isinstance(layer, VectorLayer):
                icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon)
            elif isinstance(layer, LidarLayer):
                icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DriveNetIcon)
            elif isinstance(layer, SegmentationLayer):
                icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
            else:
                icon = self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon)
            item.setIcon(0, icon)

            # Store layer reference
            item.setData(0, Qt.ItemDataRole.UserRole, layer)

            # Add to tree
            self.layer_tree.addTopLevelItem(item)

            # Set as active layer
            self.layer_tree.setCurrentItem(item)
            self.active_layer = layer
            self.active_layer_changed.emit(layer)

            # Emit signal
            self.layer_added.emit(layer)

            logger.debug(f"Added layer: {layer.name}")

        except Exception as e:
            logger.error("Error adding layer")
            logger.exception(e)
            raise

    def remove_layer(self, layer: Layer) -> None:
        """Remove a layer from the manager.

        Args:
            layer: Layer to remove
        """
        try:
            # Remove from internal list
            if layer in self.layers:
                self.layers.remove(layer)

            # Remove from tree
            root = self.layer_tree.invisibleRootItem()
            for i in range(root.childCount()):
                item = root.child(i)
                if item.data(0, Qt.ItemDataRole.UserRole) == layer:
                    root.removeChild(item)
                    break

            # Update active layer
            if self.active_layer == layer:
                self.active_layer = None
                self.active_layer_changed.emit(None)

            # Emit signal
            self.layer_removed.emit(layer)

            # Remove from map canvas
            if hasattr(self.parent(), 'map_canvas'):
                self.parent().map_canvas.remove_layer(layer)

            logger.debug(f"Removed layer: {layer.name}")

        except Exception as e:
            logger.error("Error removing layer")
            logger.exception(e)
            raise

    def remove_selected_layer(self) -> None:
        """Remove the currently selected layer."""
        try:
            items = self.layer_tree.selectedItems()
            if not items:
                return

            layer = items[0].data(0, Qt.ItemDataRole.UserRole)
            if layer:
                reply = QMessageBox.question(
                    self,
                    "Remove Layer",
                    f"Remove layer '{layer.name}'?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )

                if reply == QMessageBox.StandardButton.Yes:
                    self.remove_layer(layer)

        except Exception as e:
            logger.error("Error removing selected layer")
            logger.exception(e)

    def get_active_layer(self) -> Optional[Layer]:
        """Get the currently active layer.

        Returns:
            Layer: Active layer or None
        """
        return self.active_layer

    def get_visible_layers(self) -> List[Layer]:
        """Get list of currently visible layers.

        Returns:
            list: List of visible layers
        """
        visible = []
        root = self.layer_tree.invisibleRootItem()

        for i in range(root.childCount()):
            item = root.child(i)
            if item.checkState(0) == Qt.CheckState.Checked:
                layer = item.data(0, Qt.ItemDataRole.UserRole)
                if layer:
                    visible.append(layer)

        return visible

    def on_selection_changed(self):
        """Handle layer selection changes."""
        try:
            items = self.layer_tree.selectedItems()
            if items:
                layer = items[0].data(0, Qt.ItemDataRole.UserRole)
                if layer != self.active_layer:
                    self.active_layer = layer
                    self.active_layer_changed.emit(layer)
                    logger.debug(f"Active layer changed to: {layer.name}")
            else:
                self.active_layer = None
                self.active_layer_changed.emit(None)

        except Exception as e:
            logger.error("Error handling selection change")
            logger.exception(e)

    def show_context_menu(self, position):
        """Show context menu for layer items.

        Args:
            position: Menu position
        """
        try:
            item = self.layer_tree.itemAt(position)
            if not item:
                return

            layer = item.data(0, Qt.ItemDataRole.UserRole)
            if not layer:
                return

            menu = QMenu()

            # Rename action
            rename_action = QAction("Rename", self)
            rename_action.triggered.connect(lambda: self.rename_layer(layer))
            menu.addAction(rename_action)

            # Zoom to layer action
            zoom_action = QAction("Zoom to Layer", self)
            zoom_action.triggered.connect(lambda: self.zoom_to_layer(layer))
            menu.addAction(zoom_action)

            # Layer-specific actions
            if isinstance(layer, RasterLayer):
                self.add_raster_actions(menu, layer)
            elif isinstance(layer, VectorLayer):
                self.add_vector_actions(menu, layer)
            elif isinstance(layer, LidarLayer):
                self.add_lidar_actions(menu, layer)

            menu.addSeparator()

            # Remove action
            remove_action = QAction("Remove", self)
            remove_action.triggered.connect(lambda: self.remove_layer(layer))
            menu.addAction(remove_action)

            menu.exec(self.layer_tree.viewport().mapToGlobal(position))

        except Exception as e:
            logger.error("Error showing context menu")
            logger.exception(e)

    def rename_layer(self, layer: Layer):
        """Rename a layer.

        Args:
            layer: Layer to rename
        """
        try:
            name, ok = QInputDialog.getText(
                self,
                "Rename Layer",
                "Enter new name:",
                text=layer.name
            )

            if ok and name:
                layer.name = name
                # Update tree item text
                root = self.layer_tree.invisibleRootItem()
                for i in range(root.childCount()):
                    item = root.child(i)
                    if item.data(0, Qt.ItemDataRole.UserRole) == layer:
                        item.setText(0, name)
                        break

                self.layer_changed.emit(layer)
                logger.debug(f"Renamed layer to: {name}")

        except Exception as e:
            logger.error("Error renaming layer")
            logger.exception(e)

    def zoom_to_layer(self, layer: Layer):
        """Zoom the map to show a layer.

        Args:
            layer: Layer to zoom to
        """
        try:
            if hasattr(self.parent(), 'map_canvas'):
                self.parent().map_canvas.zoom_to_layer(layer)

        except Exception as e:
            logger.error("Error zooming to layer")
            logger.exception(e)

    def add_raster_actions(self, menu: QMenu, layer: RasterLayer):
        """Add raster-specific context menu actions.

        Args:
            menu: Context menu
            layer: Raster layer
        """
        # Stretch action
        stretch_action = QAction("Stretch...", self)
        stretch_action.triggered.connect(lambda: self.show_stretch_dialog(layer))
        menu.addAction(stretch_action)

        # Band combination action
        bands_action = QAction("Band Combination...", self)
        bands_action.triggered.connect(lambda: self.show_bands_dialog(layer))
        menu.addAction(bands_action)

    def add_vector_actions(self, menu: QMenu, layer: VectorLayer):
        """Add vector-specific context menu actions.

        Args:
            menu: Context menu
            layer: Vector layer
        """
        # Style action
        style_action = QAction("Style...", self)
        style_action.triggered.connect(lambda: self.show_style_dialog(layer))
        menu.addAction(style_action)

        # Label action
        label_action = QAction("Labels...", self)
        label_action.triggered.connect(lambda: self.show_label_dialog(layer))
        menu.addAction(label_action)

    def add_lidar_actions(self, menu: QMenu, layer: LidarLayer):
        """Add LIDAR-specific context menu actions.

        Args:
            menu: Context menu
            layer: LIDAR layer
        """
        # Point classification action
        classify_action = QAction("Classification...", self)
        classify_action.triggered.connect(lambda: self.show_classification_dialog(layer))
        menu.addAction(classify_action)

        # Color by elevation action
        elevation_action = QAction("Color by Elevation...", self)
        elevation_action.triggered.connect(lambda: self.show_elevation_dialog(layer))
        menu.addAction(elevation_action)

    def get_training_data(self) -> Dict:
        """Get training data from labeled layers.

        Returns:
            dict: Dictionary mapping layer names to training data
        """
        # TODO: Implement training data collection
        return {}

    def show_bands_dialog(self, layer: RasterLayer):
        """Show dialog for band selection.

        Args:
            layer: Raster layer
        """
        try:
            with rasterio.open(layer.path) as src:
                band_count = src.count

                dialog = QDialog(self)
                dialog.setWindowTitle("Band Selection")
                layout = QVBoxLayout(dialog)

                # Band selection
                form = QFormLayout()

                # RGB band selection
                rgb_bands = []
                for band_name in ["Red", "Green", "Blue"]:
                    combo = QComboBox()
                    combo.addItems([f"Band {i+1}" for i in range(band_count)])
                    form.addRow(f"{band_name} Band:", combo)
                    rgb_bands.append(combo)

                # Set current values
                for i, combo in enumerate(rgb_bands):
                    combo.setCurrentIndex(layer.band_indices[i] - 1)

                layout.addLayout(form)

                # Buttons
                button_box = QHBoxLayout()

                apply_btn = QPushButton("Apply")
                apply_btn.clicked.connect(lambda: self.apply_band_selection(layer, rgb_bands))
                button_box.addWidget(apply_btn)

                cancel_btn = QPushButton("Cancel")
                cancel_btn.clicked.connect(dialog.reject)
                button_box.addWidget(cancel_btn)

                layout.addLayout(button_box)

                dialog.exec()

        except Exception as e:
            logger.error("Error showing band dialog")
            logger.exception(e)
            QMessageBox.critical(self, "Error", str(e))

    def apply_band_selection(self, layer: RasterLayer, band_combos: List[QComboBox]):
        """Apply band selection to layer.

        Args:
            layer: Raster layer
            band_combos: List of band selection combos
        """
        try:
            # Get selected bands (convert to 1-based indices)
            bands = [combo.currentIndex() + 1 for combo in band_combos]

            # Update layer
            layer.band_indices = bands

            # Refresh display
            if hasattr(self.parent(), 'map_canvas'):
                self.parent().map_canvas.update()

            logger.debug(f"Updated band selection for {layer.name}: {bands}")

        except Exception as e:
            logger.error("Error applying band selection")
            logger.exception(e)
            QMessageBox.critical(self, "Error", str(e))

    def _import_raster(self):
        """Handle raster import action."""
        if self.main_window:
            self.main_window.import_raster()

    def _import_vector(self):
        """Handle vector import action."""
        if self.main_window:
            self.main_window.import_vector()

    def _import_lidar(self):
        """Handle LIDAR import action."""
        if self.main_window:
            self.main_window.import_lidar()
