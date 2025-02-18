"""Base class for model-related dialogs."""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                           QPushButton, QWidget, QTabWidget, QGroupBox,
                           QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox,
                           QCheckBox, QProgressBar)
from PyQt6.QtCore import Qt, pyqtSignal

from ...utils.logger import setup_logger

logger = setup_logger(__name__)

class ModelDialogBase(QDialog):
    """Base class for model configuration dialogs."""
    
    # Signals
    parameters_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None, title="Model Configuration"):
        """Initialize dialog.
        
        Args:
            parent: Parent widget
            title: Dialog title
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(600, 400)
        
        self.parameters = {}
        self.controls = {}
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Create tabs
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Add basic tabs
        self.setup_parameters_tab()
        self.setup_features_tab()
        self.setup_advanced_tab()
        
        # Add buttons
        button_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self.reset_parameters)
        button_layout.addWidget(self.reset_btn)
        
        button_layout.addStretch()
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_btn)
        
        layout.addLayout(button_layout)
    
    def setup_parameters_tab(self):
        """Set up the parameters tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Placeholder for model-specific parameters
        params_group = QGroupBox("Model Parameters")
        params_layout = QFormLayout()
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Parameters")
    
    def setup_features_tab(self):
        """Set up the features tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Feature selection
        features_group = QGroupBox("Feature Selection")
        features_layout = QVBoxLayout()
        
        self.feature_checkboxes = {}
        for name in ["Spectral", "Texture", "Terrain"]:
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.stateChanged.connect(self.on_feature_changed)
            self.feature_checkboxes[name] = cb
            features_layout.addWidget(cb)
        
        features_group.setLayout(features_layout)
        layout.addWidget(features_group)
        
        # Feature parameters
        params_group = QGroupBox("Feature Parameters")
        params_layout = QFormLayout()
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Features")
    
    def setup_advanced_tab(self):
        """Set up the advanced tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Placeholder for advanced options
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QFormLayout()
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Advanced")
    
    def add_parameter(
        self,
        name: str,
        widget: QWidget,
        group: str = "Parameters",
        label: str = None
    ):
        """Add a parameter control.
        
        Args:
            name: Parameter name
            widget: Control widget
            group: Parameter group
            label: Optional display label
        """
        self.controls[name] = widget
        
        # Add to appropriate group
        if group == "Parameters":
            layout = self.tab_widget.widget(0).layout()
            group_widget = layout.itemAt(0).widget()
            form_layout = group_widget.layout()
        elif group == "Features":
            layout = self.tab_widget.widget(1).layout()
            group_widget = layout.itemAt(1).widget()
            form_layout = group_widget.layout()
        else:  # Advanced
            layout = self.tab_widget.widget(2).layout()
            group_widget = layout.itemAt(0).widget()
            form_layout = group_widget.layout()
        
        form_layout.addRow(label or name.replace('_', ' ').title(), widget)
    
    def get_parameters(self) -> dict:
        """Get current parameter values.
        
        Returns:
            dict: Parameter dictionary
        """
        parameters = {}
        
        # Get control values
        for name, control in self.controls.items():
            if isinstance(control, (QSpinBox, QDoubleSpinBox)):
                parameters[name] = control.value()
            elif isinstance(control, QComboBox):
                parameters[name] = control.currentText()
            elif isinstance(control, QCheckBox):
                parameters[name] = control.isChecked()
        
        # Get feature selection
        parameters['enabled_features'] = [
            name for name, cb in self.feature_checkboxes.items()
            if cb.isChecked()
        ]
        
        return parameters
    
    def set_parameters(self, parameters: dict):
        """Set parameter values.
        
        Args:
            parameters: Parameter dictionary
        """
        # Update controls
        for name, value in parameters.items():
            if name in self.controls:
                control = self.controls[name]
                if isinstance(control, (QSpinBox, QDoubleSpinBox)):
                    control.setValue(value)
                elif isinstance(control, QComboBox):
                    index = control.findText(str(value))
                    if index >= 0:
                        control.setCurrentIndex(index)
                elif isinstance(control, QCheckBox):
                    control.setChecked(value)
        
        # Update feature selection
        if 'enabled_features' in parameters:
            for name in self.feature_checkboxes:
                self.feature_checkboxes[name].setChecked(
                    name in parameters['enabled_features']
                )
    
    def reset_parameters(self):
        """Reset parameters to defaults."""
        raise NotImplementedError("Subclasses must implement reset_parameters")
    
    def on_feature_changed(self):
        """Handle feature selection changes."""
        self.parameters_changed.emit(self.get_parameters())
    
    def accept(self):
        """Handle dialog acceptance."""
        self.parameters = self.get_parameters()
        super().accept()
