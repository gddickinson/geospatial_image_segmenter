"""Training label management and pixel selection system."""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                           QListWidget, QLabel, QColorDialog, QInputDialog,
                           QMessageBox)
from PyQt6.QtGui import QColor, QPainter, QBrush
from PyQt6.QtCore import Qt, pyqtSignal
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class TrainingLabel:
    """Class to store label information and mask."""
    name: str
    color: QColor
    mask: Optional[np.ndarray] = None

    def clear_mask(self):
        """Clear the label mask."""
        if self.mask is not None:
            self.mask.fill(0)

class LabelingDialog(QDialog):
    """Dialog for managing labels and pixel selection."""

    # Signal emitted when labels change
    labels_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Training Label Management")
        self.setModal(False)  # Allow interaction with main window

        # Store labels and current selection
        self.labels: Dict[str, TrainingLabel] = {}
        self.active_label: Optional[TrainingLabel] = None

        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Label list
        list_layout = QVBoxLayout()
        list_layout.addWidget(QLabel("Labels:"))

        self.label_list = QListWidget()
        self.label_list.itemSelectionChanged.connect(self.on_selection_changed)
        list_layout.addWidget(self.label_list)

        # Buttons for label management
        button_layout = QHBoxLayout()

        add_btn = QPushButton("Add Label")
        add_btn.clicked.connect(self.add_label)
        button_layout.addWidget(add_btn)

        remove_btn = QPushButton("Remove Label")
        remove_btn.clicked.connect(self.remove_label)
        button_layout.addWidget(remove_btn)

        clear_btn = QPushButton("Clear Selection")
        clear_btn.clicked.connect(self.clear_current_selection)
        button_layout.addWidget(clear_btn)

        list_layout.addLayout(button_layout)
        layout.addLayout(list_layout)

        # Painting controls
        paint_layout = QVBoxLayout()
        paint_layout.addWidget(QLabel("Painting:"))
        paint_layout.addWidget(QLabel("Hold Shift + Left Click to paint"))

        brush_layout = QHBoxLayout()
        brush_layout.addWidget(QLabel("Brush Size:"))

        self.brush_size = 5
        decrease_btn = QPushButton("-")
        decrease_btn.clicked.connect(self.decrease_brush)
        brush_layout.addWidget(decrease_btn)

        self.brush_label = QLabel(str(self.brush_size))
        brush_layout.addWidget(self.brush_label)

        increase_btn = QPushButton("+")
        increase_btn.clicked.connect(self.increase_brush)
        brush_layout.addWidget(increase_btn)

        paint_layout.addLayout(brush_layout)
        layout.addLayout(paint_layout)

    def add_label(self):
        """Add a new label."""
        name, ok = QInputDialog.getText(
            self, "Add Label", "Enter label name:")

        if ok and name:
            if name in self.labels:
                QMessageBox.warning(
                    self, "Warning", "Label already exists!")
                return

            # Let user pick a color
            color = QColorDialog.getColor(
                QColor(Qt.GlobalColor.red), self, "Select Label Color")

            if color.isValid():
                label = TrainingLabel(name=name, color=color)
                self.labels[name] = label

                # Add to list widget
                self.label_list.addItem(name)
                self.labels_changed.emit()

    def remove_label(self):
        """Remove the selected label."""
        current = self.label_list.currentItem()
        if current is None:
            return

        name = current.text()
        reply = QMessageBox.question(
            self, "Remove Label",
            f"Remove label '{name}'? This will delete all selections for this label.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            del self.labels[name]
            self.label_list.takeItem(self.label_list.row(current))

            if self.active_label and self.active_label.name == name:
                self.active_label = None

            self.labels_changed.emit()

    def on_selection_changed(self):
        """Handle label selection change."""
        current = self.label_list.currentItem()
        if current is None:
            self.active_label = None
        else:
            self.active_label = self.labels[current.text()]

    def clear_current_selection(self):
        """Clear selection for current label."""
        if self.active_label:
            self.active_label.clear_mask()
            self.labels_changed.emit()

    def decrease_brush(self):
        """Decrease brush size."""
        self.brush_size = max(1, self.brush_size - 2)
        self.brush_label.setText(str(self.brush_size))
        if hasattr(self.parent(), 'paint_tool'):
            self.parent().paint_tool.set_brush_size(self.brush_size)

    def increase_brush(self):
        """Increase brush size."""
        self.brush_size = min(50, self.brush_size + 2)
        self.brush_label.setText(str(self.brush_size))
        if hasattr(self.parent(), 'paint_tool'):
            self.parent().paint_tool.set_brush_size(self.brush_size)

    def initialize_masks(self, shape: tuple):
        """Initialize masks for all labels.

        Args:
            shape: Shape of the image (height, width)
        """
        for label in self.labels.values():
            if label.mask is None or label.mask.shape != shape:
                label.mask = np.zeros(shape, dtype=bool)

    def get_label_masks(self) -> Dict[str, np.ndarray]:
        """Get dictionary of all label masks.

        Returns:
            dict: Dictionary mapping label names to boolean masks
        """
        return {name: label.mask.copy() if label.mask is not None else None
                for name, label in self.labels.items()}
