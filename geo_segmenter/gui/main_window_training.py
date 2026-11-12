"""MainWindow training and classification: training mode, label overlay, model parameters, train_model, run_segmentation."""
import numpy as np
from PyQt6.QtWidgets import QMessageBox, QApplication, QProgressDialog, QSpinBox, QDoubleSpinBox
from PyQt6.QtCore import Qt, QTimer
import rasterio
import rasterio.transform
from typing import Dict
from ..utils.logger import setup_logger
from .layer_manager import RasterLayer, SegmentationLayer, TrainingLabelsLayer
from .dialogs.training_dialog import TrainingDialog
from ..models.random_forest import RandomForestModel
from ..models.cnn import CNNModel

logger = setup_logger(__name__)


class MainWindowTrainingMixin:
    """Methods of MainWindow (mixed in; see main_window.py)."""

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

            # Add feature extractors: spectral (>= 3 bands, R/G/B/NIR band map) and texture
            from ..models.pipeline import add_default_extractors
            add_default_extractors(model, image_data.shape[-1])

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

    def update_training_layer(self):
        """Update training labels layer when labels change."""
        try:
            self.training_layer.update_labels(self.label_dialog.labels)
            self.map_canvas.update()
        except Exception as e:
            logger.error("Error updating training layer")
            logger.exception(e)
