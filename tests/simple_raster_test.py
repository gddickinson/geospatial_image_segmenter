import sys
from pathlib import Path
import numpy as np
import rasterio
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QImage

class SimpleRasterViewer(QMainWindow):
    def __init__(self, raster_path):
        super().__init__()
        self.setWindowTitle("Simple Raster Viewer")
        self.resize(800, 600)

        # Load raster data
        with rasterio.open(raster_path) as src:
            # Read all bands
            self.data = src.read()
            print(f"Raster shape: {self.data.shape}")
            print(f"Data type: {self.data.dtype}")
            print(f"Bounds: {src.bounds}")

            # Basic statistics
            for band in range(self.data.shape[0]):
                band_data = self.data[band]
                print(f"Band {band + 1}:")
                print(f"  Min: {band_data.min()}")
                print(f"  Max: {band_data.max()}")
                print(f"  Mean: {band_data.mean()}")
                print(f"  Has NaN: {np.any(np.isnan(band_data))}")

            # Convert to RGB display format
            self.display_data = self._prepare_for_display(self.data)
            print(f"Display data shape: {self.display_data.shape}")
            print(f"Display data type: {self.display_data.dtype}")

    def _prepare_for_display(self, data):
        """Prepare raster data for display."""
        try:
            # Handle different band counts
            if data.shape[0] >= 3:
                # Use first three bands as RGB
                rgb = data[:3, :, :]
            else:
                # Replicate single band to RGB
                rgb = np.repeat(data[:1, :, :], 3, axis=0)

            # Normalize each band to 0-255 range using percentile stretching
            normalized = np.zeros_like(rgb, dtype=np.uint8)
            for i in range(3):
                band = rgb[i].astype(float)
                # Use 2nd and 98th percentiles for better contrast
                p2, p98 = np.percentile(band[~np.isnan(band)], (2, 98))
                normalized[i] = np.clip((band - p2) * 255 / (p98 - p2), 0, 255)

            # Transpose to height, width, channels and ensure contiguous memory
            return np.ascontiguousarray(normalized.transpose(1, 2, 0))

        except Exception as e:
            print(f"Error preparing display data: {str(e)}")
            raise

    def paintEvent(self, event):
        try:
            painter = QPainter(self)

            # Convert numpy array to QImage
            height, width, channels = self.display_data.shape
            bytes_per_line = channels * width

            # Create QImage from numpy array
            image = QImage(
                self.display_data.tobytes(),  # Convert to bytes
                width,
                height,
                bytes_per_line,
                QImage.Format.Format_RGB888
            )

            # Scale image to fit window while maintaining aspect ratio
            scaled_image = image.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            # Center image in window
            x = (self.width() - scaled_image.width()) // 2
            y = (self.height() - scaled_image.height()) // 2

            # Draw image
            painter.drawImage(x, y, scaled_image)

        except Exception as e:
            print(f"Error painting: {str(e)}")
            raise

def main():
    try:
        # Initialize Qt application
        app = QApplication(sys.argv)

        # Get test file path
        test_path = Path(__file__).parent / "test_data" / "sample.tif"
        if not test_path.exists():
            print(f"Test file not found: {test_path}")
            return

        # Create and show viewer
        viewer = SimpleRasterViewer(str(test_path))
        viewer.show()

        return app.exec()

    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main()
