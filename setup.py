from setuptools import setup, find_packages

setup(
    name="geo-segmenter",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "PyQt6>=6.2.0",
        "rasterio>=1.2.0",  # GeoTIFF handling
        "pyproj>=3.0.0",    # Coordinate transformations
        "folium>=0.12.0",   # Map visualization
        "geopandas>=0.9.0", # Geospatial data handling
        "scikit-learn>=0.24.0",
        "torch>=1.9.0",
        "laspy>=2.0.0",     # LIDAR data handling
        "pdal>=2.4.0",      # Point cloud processing
        "scipy>=1.7.0",
        "shapely>=1.8.0",   # Geometric operations
        "pyqtlet2>=0.8.0",  # Leaflet maps integration
        "scikit-image>=0.18.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "isort>=5.0.0",
            "flake8>=3.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "geo-segmenter=geo_segmenter.main:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for segmenting and analyzing aerial and satellite imagery",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="gis remote-sensing image-segmentation machine-learning",
    url="https://github.com/yourusername/geo-segmenter",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.8",
)