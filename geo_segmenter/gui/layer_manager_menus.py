"""LayerManager context menu and per-layer actions: rename, zoom, stretch, band selection, vector/LIDAR actions."""
from PyQt6.QtWidgets import QMenu, QInputDialog, QMessageBox, QComboBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from typing import List
from ..data.layer import Layer, RasterLayer, VectorLayer, LidarLayer
from ..utils.logger import setup_logger
from ..gui.dialogs.stretch_dialog import StretchDialog
from ..gui.dialogs.band_combo_dialog import BandComboDialog

logger = setup_logger(__name__)


class LayerManagerMenusMixin:
    """Methods of LayerManager (mixed in; see layer_manager.py)."""

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
                extent = layer.extent()
                if extent:
                    # Get layer extent with padding
                    x_min, y_min, x_max, y_max = extent
                    x_pad = (x_max - x_min) * 0.05
                    y_pad = (y_max - y_min) * 0.05

                    padded_extent = (
                        x_min - x_pad,
                        y_min - y_pad,
                        x_max + x_pad,
                        y_max + y_pad
                    )

                    self.parent().map_canvas.zoom_to_extent(padded_extent)
                    logger.debug(f"Zoomed to layer: {layer.name}")
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

    def show_stretch_dialog(self, layer: RasterLayer):
        """Show stretch parameters dialog.

        Args:
            layer: Raster layer
        """
        try:
            dialog = StretchDialog(layer, self)
            dialog.exec()
        except Exception as e:
            logger.error("Error showing stretch dialog")
            logger.exception(e)

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

    def show_bands_dialog(self, layer: RasterLayer):
        """Show band combination dialog.

        Args:
            layer: Raster layer
        """
        try:
            dialog = BandComboDialog(layer, self)
            dialog.exec()
        except Exception as e:
            logger.error("Error showing bands dialog")
            logger.exception(e)

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
