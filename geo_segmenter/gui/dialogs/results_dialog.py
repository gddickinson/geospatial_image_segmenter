"""Dialog for visualizing and exporting segmentation results."""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
                           QPushButton, QGroupBox, QLabel, QSpinBox,
                           QComboBox, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import rasterio
from rasterio.transform import from_origin
import json
from pathlib import Path

from ...utils.logger import setup_logger

logger = setup_logger(__name__)

class ResultsDialog(QDialog):
    """Dialog for visualizing and exporting results."""
    
    def __init__(self, image, prediction, probabilities, class_names, parent=None):
        """Initialize dialog.
        
        Args:
            image: Original image
            prediction: Predicted segmentation
            probabilities: Class probabilities
            class_names: List of class names
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Segmentation Results")
        self.setModal(True)
        self.resize(1200, 800)
        
        self.image = image
        self.prediction = prediction
        self.probabilities = probabilities
        self.class_names = class_names
        
        self.setup_ui()
        self.update_visualization()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Create tabs
        tabs = QTabWidget()
        
        # Results visualization tab
        viz_tab = self.create_visualization_tab()
        tabs.addTab(viz_tab, "Visualization")
        
        # Statistics tab
        stats_tab = self.create_statistics_tab()
        tabs.addTab(stats_tab, "Statistics")
        
        # Probability maps tab
        if self.probabilities is not None:
            prob_tab = self.create_probability_tab()
            tabs.addTab(prob_tab, "Probability Maps")
        
        layout.addWidget(tabs)
        
        # Export buttons
        button_layout = QHBoxLayout()
        
        export_geotiff = QPushButton("Export as GeoTIFF")
        export_geotiff.clicked.connect(self.export_geotiff)
        button_layout.addWidget(export_geotiff)
        
        export_stats = QPushButton("Export Statistics")
        export_stats.clicked.connect(self.export_statistics)
        button_layout.addWidget(export_stats)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def create_visualization_tab(self):
        """Create visualization tab.
        
        Returns:
            QWidget: Tab widget
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Control panel
        control_layout = QHBoxLayout()
        
        # Overlay opacity
        control_layout.addWidget(QLabel("Overlay Opacity:"))
        self.opacity_spin = QSpinBox()
        self.opacity_spin.setRange(0, 100)
        self.opacity_spin.setValue(50)
        self.opacity_spin.valueChanged.connect(self.update_visualization)
        control_layout.addWidget(self.opacity_spin)
        
        # Color scheme
        control_layout.addWidget(QLabel("Color Scheme:"))
        self.color_combo = QComboBox()
        self.color_combo.addItems(['viridis', 'jet', 'tab20'])
        self.color_combo.currentTextChanged.connect(self.update_visualization)
        control_layout.addWidget(self.color_combo)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # Matplotlib figure
        self.figure, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)
        
        return tab
    
    def create_statistics_tab(self):
        """Create statistics tab.
        
        Returns:
            QWidget: Tab widget
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create figure for statistics
        self.stats_figure, self.stats_ax = plt.subplots(figsize=(10, 6))
        self.stats_canvas = FigureCanvasQTAgg(self.stats_figure)
        layout.addWidget(self.stats_canvas)
        
        self.update_statistics()
        return tab
    
    def create_probability_tab(self):
        """Create probability maps tab.
        
        Returns:
            QWidget: Tab widget
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Class selection
        layout.addWidget(QLabel("Select Class:"))
        self.class_combo = QComboBox()
        self.class_combo.addItems(self.class_names)
        self.class_combo.currentIndexChanged.connect(self.update_probability_map)
        layout.addWidget(self.class_combo)
        
        # Matplotlib figure
        self.prob_figure, self.prob_ax = plt.subplots(figsize=(10, 6))
        self.prob_canvas = FigureCanvasQTAgg(self.prob_figure)
        layout.addWidget(self.prob_canvas)
        
        self.update_probability_map()
        return tab
    
    def update_visualization(self):
        """Update result visualization."""
        try:
            self.ax1.clear()
            self.ax2.clear()
            
            # Show original image
            self.ax1.imshow(self.image)
            self.ax1.set_title("Original Image")
            self.ax1.axis('off')
            
            # Show segmentation result
            opacity = self.opacity_spin.value() / 100.0
            cmap = self.color_combo.currentText()
            
            # Create segmentation overlay
            self.ax2.imshow(self.image)
            self.ax2.imshow(self.prediction, cmap=cmap, alpha=opacity)
            self.ax2.set_title("Segmentation Result")
            self.ax2.axis('off')
            
            # Add colorbar
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(self.ax2)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(
                plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap)),
                cax=cax,
                ticks=range(len(self.class_names)),
                label='Classes'
            )
            cax.set_yticklabels(self.class_names)
            
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            logger.error("Error updating visualization")
            logger.exception(e)
    
    def update_statistics(self):
        """Update statistics visualization."""
        try:
            self.stats_ax.clear()
            
            # Calculate class distribution
            unique, counts = np.unique(self.prediction, return_counts=True)
            percentages = counts / counts.sum() * 100
            
            # Create bar chart
            bars = self.stats_ax.bar(self.class_names, percentages)
            
            # Add percentage labels on top of bars
            for bar in bars:
                height = bar.get_height()
                self.stats_ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:.1f}%',
                    ha='center',
                    va='bottom'
                )
            
            self.stats_ax.set_title("Class Distribution")
            self.stats_ax.set_ylabel("Percentage")
            plt.xticks(rotation=45)
            
            self.stats_figure.tight_layout()
            self.stats_canvas.draw()
            
        except Exception as e:
            logger.error("Error updating statistics")
            logger.exception(e)
    
    def update_probability_map(self):
        """Update probability map visualization."""
        try:
            if self.probabilities is None:
                return
                
            self.prob_ax.clear()
            
            # Get selected class
            class_idx = self.class_combo.currentIndex()
            class_name = self.class_names[class_idx]
            
            # Show probability map
            im = self.prob_ax.imshow(
                self.probabilities[class_idx],
                cmap='viridis',
                vmin=0,
                vmax=1
            )
            self.prob_ax.set_title(f"Probability Map - {class_name}")
            
            # Add colorbar
            plt.colorbar(im, ax=self.prob_ax, label='Probability')
            
            self.prob_ax.axis('off')
            self.prob_figure.tight_layout()
            self.prob_canvas.draw()
            
        except Exception as e:
            logger.error("Error updating probability map")
            logger.exception(e)
    
    def export_geotiff(self):
        """Export results as GeoTIFF."""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Export GeoTIFF",
                "",
                "GeoTIFF files (*.tif)"
            )
            
            if filename:
                # Ensure .tif extension
                if not filename.endswith('.tif'):
                    filename += '.tif'
                
                # Write GeoTIFF
                with rasterio.open(
                    filename,
                    'w',
                    driver='GTiff',
                    height=self.prediction.shape[0],
                    width=self.prediction.shape[1],
                    count=1,
                    dtype=self.prediction.dtype,
                    crs='EPSG:4326',  # Default to WGS84
                    transform=from_origin(0, 0, 1, 1)  # Default transform
                ) as dst:
                    dst.write(self.prediction, 1)
                    
                    # Write class names as metadata
                    dst.update_tags(classes=','.join(self.class_names))
                
                QMessageBox.information(
                    self,
                    "Export Complete",
                    "Results exported successfully!"
                )
                
        except Exception as e:
            logger.error("Error exporting GeoTIFF")
            logger.exception(e)
            QMessageBox.critical(
                self,
                "Export Error",
                f"Error exporting results: {str(e)}"
            )
    
    def export_statistics(self):
        """Export statistics as JSON."""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Export Statistics",
                "",
                "JSON files (*.json)"
            )
            
            if filename:
                # Ensure .json extension
                if not filename.endswith('.json'):
                    filename += '.json'
                
                # Calculate statistics
                unique, counts = np.unique(self.prediction, return_counts=True)
                percentages = counts / counts.sum() * 100
                
                stats = {
                    'classes': self.class_names,
                    'pixel_counts': counts.tolist(),
                    'percentages': percentages.tolist(),
                    'total_pixels': int(counts.sum())
                }
                
                # Calculate confusion matrix if available
                if hasattr(self, 'confusion_matrix'):
                    stats['confusion_matrix'] = self.confusion_matrix.tolist()
                
                # Write JSON
                with open(filename, 'w') as f:
                    json.dump(stats, f, indent=4)
                
                QMessageBox.information(
                    self,
                    "Export Complete",
                    "Statistics exported successfully!"
                )
                
        except Exception as e:
            logger.error("Error exporting statistics")
            logger.exception(e)
            QMessageBox.critical(
                self,
                "Export Error",
                f"Error exporting statistics: {str(e)}"
            )