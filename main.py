"""Main entry point for the geospatial segmentation application."""
import sys
from PyQt6.QtWidgets import QApplication
from geo_segmenter.gui.main_window import MainWindow
from geo_segmenter.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    """Initialize and run the application."""
    try:
        # Create application
        app = QApplication(sys.argv)

        # Set application style
        app.setStyle("Fusion")

        # Create and show main window
        window = MainWindow()
        window.show()

        logger.info("Application started")

        # Start event loop
        sys.exit(app.exec())

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.exception(e)
        sys.exit(1)

if __name__ == "__main__":
    main()