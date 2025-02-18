"""Test script for raster visualization."""
import sys
from pathlib import Path
import numpy as np
import rasterio
from PyQt6.QtWidgets import QApplication

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from geo_segmenter.gui.main_window import MainWindow
from geo_segmenter.data.providers.raster import RasterProvider
from geo_segmenter.utils.logger import setup_logger

logger = setup_logger(__name__)



def test_raster_display():
    """Test raster display functionality."""
    try:
        # Initialize application
        app = QApplication(sys.argv)
        window = MainWindow()

        # Get test file path
        test_path = Path(__file__).parent.parent / "test_data" / "sample.tif"
        if not test_path.exists():
            logger.error(f"Test file not found: {test_path}")
            return

        logger.info(f"Testing raster: {test_path}")

        # Read raster info
        with rasterio.open(test_path) as src:
            logger.info(f"Profile: {src.profile}")
            logger.info(f"Bounds: {src.bounds}")
            logger.info(f"Transform: {src.transform}")

            # Read and check data
            data = src.read()
            logger.info(f"Data shape: {data.shape}")
            logger.info(f"Data range: {data.min()} to {data.max()}")

            # Check for NaN or infinite values
            if np.any(np.isnan(data)):
                logger.warning("Data contains NaN values")
            if np.any(np.isinf(data)):
                logger.warning("Data contains infinite values")

        # Import raster
        window.import_raster(str(test_path))

        # Verify layer was added
        logger.info(f"Layer manager layers: {len(window.layer_manager.layers)}")
        logger.info(f"Map canvas layers: {len(window.map_canvas._layers)}")

        if window.map_canvas._layers:
            logger.info(f"First layer name: {window.map_canvas._layers[0].name}")
            logger.info(f"First layer extent: {window.map_canvas._layers[0].extent()}")
            logger.info(f"Canvas viewport bounds: {window.map_canvas.viewport_bounds}")

        # Show window
        window.resize(1024, 768)
        window.show()

        return app.exec()

    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        logger.exception(e)
        raise

if __name__ == '__main__':
    test_raster_display()
