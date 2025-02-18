"""Basic test script for the geo-segmenter application."""
import sys
from pathlib import Path
import numpy as np
from PyQt6.QtWidgets import QApplication

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from geo_segmenter.gui.main_window import MainWindow
from geo_segmenter.data.providers.raster import RasterProvider
from geo_segmenter.models.features import (
    SpectralFeatureExtractor,
    TextureFeatureExtractor,
    TerrainFeatureExtractor
)
from geo_segmenter.models import RandomForestModel, CNNModel
from geo_segmenter.utils import normalize_image

def test_basic_functionality():
    """Test basic functionality of the application."""
    try:
        # Initialize application
        app = QApplication(sys.argv)
        window = MainWindow()

        # Test data loading
        raster_provider = RasterProvider()

        # Create test image (4-band: RGB + NIR)
        test_image = np.random.rand(500, 500, 4)
        test_image = normalize_image(test_image)  # Normalize to [0,1]

        # Test feature extraction
        spectral = SpectralFeatureExtractor()
        texture = TextureFeatureExtractor()
        terrain = TerrainFeatureExtractor()

        print("Extracting features...")

        # Set band mapping for spectral features
        spectral.set_parameters({
            'band_map': {
                'blue': 0,
                'green': 1,
                'red': 2,
                'nir': 3
            }
        })

        spectral_features = spectral.extract_features(test_image)
        print(f"Spectral features shape: {spectral_features.shape}")

        # Convert to single band for texture
        gray_image = np.mean(test_image[:, :, :3], axis=2)
        texture_features = texture.extract_features(gray_image)
        print(f"Texture features shape: {texture_features.shape}")

        # Use first band for terrain (as elevation)
        terrain_features = terrain.extract_features(test_image[:, :, 0])
        print(f"Terrain features shape: {terrain_features.shape}")

        # Test model initialization
        print("\nInitializing models...")
        rf_model = RandomForestModel()
        cnn_model = CNNModel()
        print("Models initialized successfully")

        # Show window
        window.show()
        return app.exec()

    except Exception as e:
        print(f"Error during testing: {str(e)}")
        raise

if __name__ == '__main__':
    test_basic_functionality()
