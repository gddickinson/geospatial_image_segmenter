from setuptools import setup, find_packages

# PDAL (LIDAR ground classification only) is not in install_requires: pip cannot build it without the PDAL C++
# library. Install it with conda (conda install -c conda-forge pdal python-pdal), or `pip install .[lidar]` where
# the PDAL library is already installed. The rest of the app runs without it.
setup(
    name="geo-segmenter",
    version="0.1.0",
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    package_data={"geo_segmenter.web": ["static/*"]},
    install_requires=[
        "numpy>=1.20.0",
        "PyQt6>=6.2.0",
        "rasterio>=1.2.0",  # GeoTIFF handling
        "pyproj>=3.0.0",    # Coordinate transformations
        "geopandas>=0.9.0", # Vector data
        "shapely>=1.8.0",   # Geometric operations
        "scikit-learn>=0.24.0",
        "joblib>=1.0.0",    # model save/load
        "torch>=1.9.0",     # CNN model
        "laspy>=2.0.0",     # LIDAR data handling
        "scipy>=1.7.0",
        "scikit-image>=0.18.0",
        "matplotlib>=3.3.0",
        "Pillow>=8.0.0",
        "requests>=2.25.0", # map tiles, IP location
        "pyserial>=3.5",    # GPS receiver
        "pynmea2>=1.18.0",  # GPS NMEA sentences
    ],
    extras_require={
        "lidar": ["pdal>=3.0.0"],  # needs the PDAL C++ library (see the note above)
        "web": ["fastapi>=0.110", "uvicorn>=0.25"],  # pinned versions: requirements-web.txt
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
    author="George Dickinson",
    description="A tool for segmenting and analyzing aerial and satellite imagery",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="gis remote-sensing image-segmentation machine-learning",
    url="https://github.com/gddickinson/geospatial_image_segmenter",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.8",
)
