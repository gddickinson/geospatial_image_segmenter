"""Dialog for monitoring model training progress."""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                           QPushButton, QProgressBar, QTabWidget, QWidget,
                           QTextEdit, QTableWidget, QTableWidgetItem, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from typing import Dict, Any, Optional

from ...utils.logger import setup_logger

logger = setup_logger(__name__)

class TrainingDialog(QDialog):
    """Dialog for monitoring training progress."""

    # Signals
    training_cancelled = pyqtSignal()

    def __init__(self, model_name: str, parent=None):
        """Initialize dialog.

        Args:
            model_name: Name of model being trained
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle(f"Training {model_name}")
        self.setModal(True)
        self.resize(800, 600)

        self.model_name = model_name  # Store model name as instance variable
        self.metrics = {}
        self.current_epoch = 0
        self.total_epochs = 0

        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Progress section
        progress_layout = QVBoxLayout()

        self.progress_label = QLabel("Initializing training...")
        progress_layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.progress_bar)

        layout.addLayout(progress_layout)

        # Tabs for different views
        self.tab_widget = QTabWidget()

        # Metrics tab
        self.setup_metrics_tab()

        # Log tab
        self.setup_log_tab()

        # Feature importance tab (for Random Forest)
        if self.model_name == "Random Forest":
            self.setup_importance_tab()

        layout.addWidget(self.tab_widget)

        # Control buttons
        button_layout = QHBoxLayout()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_training)
        button_layout.addWidget(self.cancel_btn)

        button_layout.addStretch()

        self.close_btn = QPushButton("Close")
        self.close_btn.setEnabled(False)
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)

        layout.addLayout(button_layout)

    def setup_metrics_tab(self):
        """Set up the metrics visualization tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Create figure for plots
        self.figure, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 8))
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)

        # Metrics table
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.metrics_table)

        self.tab_widget.addTab(tab, "Metrics")

    def setup_log_tab(self):
        """Set up the training log tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        self.tab_widget.addTab(tab, "Log")

    def setup_importance_tab(self):
        """Set up the feature importance tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Create figure for importance plot
        self.importance_figure, self.importance_ax = plt.subplots(figsize=(8, 6))
        self.importance_canvas = FigureCanvasQTAgg(self.importance_figure)
        layout.addWidget(self.importance_canvas)

        # Importance table
        self.importance_table = QTableWidget()
        self.importance_table.setColumnCount(2)
        self.importance_table.setHorizontalHeaderLabels(
            ["Feature", "Importance"]
        )
        self.importance_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.importance_table)

        self.tab_widget.addTab(tab, "Feature Importance")

    def update_progress(self, epoch: int, total_epochs: int, metrics: Dict[str, Any]):
        """Update training progress.

        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs
            metrics: Training metrics
        """
        try:
            self.current_epoch = epoch
            self.total_epochs = total_epochs

            # Update progress bar
            progress = (epoch / total_epochs) * 100
            self.progress_bar.setValue(int(progress))

            # Update progress label
            self.progress_label.setText(
                f"Training epoch {epoch}/{total_epochs}"
            )

            # Update metrics
            self.metrics = metrics
            self.update_plots()
            self.update_metrics_table()

            # Add to log
            self._log_metrics(epoch, metrics)

            # Update display
            self.canvas.draw()

        except Exception as e:
            logger.error("Error updating progress")
            logger.exception(e)

    def update_plots(self):
        """Update metric plots."""
        try:
            self.ax1.clear()
            self.ax2.clear()

            # Training/validation loss
            if 'train_loss' in self.metrics:
                epochs = range(1, len(self.metrics['train_loss']) + 1)
                self.ax1.plot(epochs, self.metrics['train_loss'],
                            'b-', label='Training Loss')
                if 'val_loss' in self.metrics:
                    self.ax1.plot(epochs, self.metrics['val_loss'],
                                'r-', label='Validation Loss')
                self.ax1.set_xlabel('Epoch')
                self.ax1.set_ylabel('Loss')
                self.ax1.legend()
                self.ax1.grid(True)

            # Accuracy/custom metrics
            if 'val_accuracy' in self.metrics:
                epochs = range(1, len(self.metrics['val_accuracy']) + 1)
                self.ax2.plot(epochs, self.metrics['val_accuracy'],
                            'g-', label='Validation Accuracy')

            self.ax2.set_xlabel('Epoch')
            self.ax2.set_ylabel('Accuracy (%)')
            self.ax2.legend()
            self.ax2.grid(True)

            self.figure.tight_layout()

        except Exception as e:
            logger.error("Error updating plots")
            logger.exception(e)

    def update_metrics_table(self):
        """Update metrics table with current values."""
        try:
            self.metrics_table.setRowCount(0)

            # Add final/current values for each metric
            for name, values in self.metrics.items():
                if isinstance(values, list) and values:
                    row = self.metrics_table.rowCount()
                    self.metrics_table.insertRow(row)
                    self.metrics_table.setItem(row, 0,
                        QTableWidgetItem(name.replace('_', ' ').title()))
                    self.metrics_table.setItem(row, 1,
                        QTableWidgetItem(f"{values[-1]:.4f}"))

            self.metrics_table.resizeColumnsToContents()

        except Exception as e:
            logger.error("Error updating metrics table")
            logger.exception(e)

    def update_feature_importance(self, importance_dict: Dict[str, float]):
        """Update feature importance visualization.

        Args:
            importance_dict: Dictionary mapping feature names to importance values
        """
        try:
            if not hasattr(self, 'importance_ax'):
                return

            self.importance_ax.clear()

            # Sort features by importance
            sorted_features = sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )

            features, importances = zip(*sorted_features)

            # Plot bar chart
            y_pos = np.arange(len(features))
            self.importance_ax.barh(y_pos, importances)
            self.importance_ax.set_yticks(y_pos)
            self.importance_ax.set_yticklabels(features)
            self.importance_ax.set_xlabel('Feature Importance')
            self.importance_ax.set_title('Feature Importance')

            self.importance_figure.tight_layout()
            self.importance_canvas.draw()

            # Update table
            self.importance_table.setRowCount(0)
            for feature, importance in sorted_features:
                row = self.importance_table.rowCount()
                self.importance_table.insertRow(row)
                self.importance_table.setItem(row, 0,
                    QTableWidgetItem(feature))
                self.importance_table.setItem(row, 1,
                    QTableWidgetItem(f"{importance:.4f}"))

            self.importance_table.resizeColumnsToContents()

        except Exception as e:
            logger.error("Error updating feature importance")
            logger.exception(e)

    def _log_metrics(self, epoch: int, metrics: Dict[str, Any]):
        """Add metrics to log.

        Args:
            epoch: Current epoch
            metrics: Metric dictionary
        """
        try:
            log_text = f"Epoch {epoch}/{self.total_epochs}:\n"

            for name, values in metrics.items():
                if isinstance(values, list) and values:
                    log_text += f"  {name}: {values[-1]:.4f}\n"

            self.log_text.append(log_text)

        except Exception as e:
            logger.error("Error logging metrics")
            logger.exception(e)

    def training_finished(self, success: bool):
        """Handle training completion.

        Args:
            success: Whether training completed successfully
        """
        try:
            if success:
                self.progress_label.setText("Training completed successfully")
                self.log_text.append("\nTraining completed successfully")
            else:
                self.progress_label.setText("Training cancelled")
                self.log_text.append("\nTraining cancelled")

            self.progress_bar.setValue(100 if success else 0)
            self.cancel_btn.setEnabled(False)
            self.close_btn.setEnabled(True)

        except Exception as e:
            logger.error("Error handling training completion")
            logger.exception(e)

    def cancel_training(self):
        """Handle training cancellation."""
        try:
            reply = QMessageBox.question(
                self,
                "Cancel Training",
                "Are you sure you want to cancel training?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.training_cancelled.emit()

        except Exception as e:
            logger.error("Error handling training cancellation")
            logger.exception(e)

    def show_error(self, message: str):
        """Show error message.

        Args:
            message: Error message
        """
        try:
            self.progress_label.setText("Training failed")
            self.log_text.append(f"\nError: {message}")
            self.cancel_btn.setEnabled(False)
            self.close_btn.setEnabled(True)

            QMessageBox.critical(self, "Training Error", message)

        except Exception as e:
            logger.error("Error showing error message")
            logger.exception(e)
