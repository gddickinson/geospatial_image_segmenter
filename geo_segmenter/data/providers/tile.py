"""Provider for loading and caching map tiles."""
import os
import hashlib
import requests
from typing import Optional, Dict, Tuple
from pathlib import Path
from PIL import Image
from io import BytesIO
import numpy as np

from ...utils.logger import setup_logger
from ...utils.geo_utils import calculate_tile_bounds
from ... import config

logger = setup_logger(__name__)

class TileProvider:
    """Provider for map tiles from various services."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize tile provider.
        
        Args:
            cache_dir: Directory for caching tiles
        """
        self.cache_dir = cache_dir or config.CACHE_DIR
        self.services = config.AVAILABLE_BASE_MAPS
        self.current_service = list(self.services.keys())[0]
        
        # Create cache directory if needed
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Track memory usage
        self.cache_size = 0  # bytes
        self.cache = {}  # In-memory cache
        
    def get_tile(self, x: int, y: int, z: int) -> Optional[Image.Image]:
        """Get a map tile.
        
        Args:
            x: Tile X coordinate
            y: Tile Y coordinate
            z: Zoom level
            
        Returns:
            PIL.Image: Tile image or None if not available
        """
        try:
            # Check memory cache first
            cache_key = f"{self.current_service}_{z}_{x}_{y}"
            if cache_key in self.cache:
                logger.debug(f"Tile found in memory cache: {cache_key}")
                return self.cache[cache_key]
            
            # Check disk cache
            cache_path = self._get_cache_path(x, y, z)
            if cache_path.exists():
                try:
                    image = Image.open(cache_path)
                    self._add_to_memory_cache(cache_key, image)
                    logger.debug(f"Tile loaded from disk cache: {cache_key}")
                    return image
                except Exception as e:
                    logger.warning(f"Error loading cached tile: {str(e)}")
            
            # Download tile
            image = self._download_tile(x, y, z)
            if image:
                # Save to disk cache
                image.save(cache_path)
                self._add_to_memory_cache(cache_key, image)
                logger.debug(f"Tile downloaded and cached: {cache_key}")
                return image
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting tile: {str(e)}")
            logger.exception(e)
            return None
    
    def set_service(self, service_name: str) -> bool:
        """Set the current map service.
        
        Args:
            service_name: Name of service to use
            
        Returns:
            bool: True if service was changed successfully
        """
        if service_name in self.services:
            self.current_service = service_name
            self.cache.clear()  # Clear memory cache
            logger.info(f"Changed map service to: {service_name}")
            return True
        return False
    
    def get_bounds(self, x: int, y: int, z: int) -> Tuple[float, float, float, float]:
        """Get geographic bounds of a tile.
        
        Args:
            x: Tile X coordinate
            y: Tile Y coordinate
            z: Zoom level
            
        Returns:
            tuple: (min_lon, min_lat, max_lon, max_lat)
        """
        return calculate_tile_bounds(z, x, y)
    
    def clear_cache(self) -> None:
        """Clear all cached tiles."""
        try:
            # Clear memory cache
            self.cache.clear()
            self.cache_size = 0
            
            # Clear disk cache
            for file in self.cache_dir.glob("*.png"):
                file.unlink()
            
            logger.info("Cleared tile cache")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            logger.exception(e)
    
    def _get_cache_path(self, x: int, y: int, z: int) -> Path:
        """Get file path for cached tile.
        
        Args:
            x: Tile X coordinate
            y: Tile Y coordinate
            z: Zoom level
            
        Returns:
            Path: Cache file path
        """
        # Create unique filename
        filename = f"{self.current_service}_{z}_{x}_{y}.png"
        return self.cache_dir / filename
    
    def _download_tile(self, x: int, y: int, z: int) -> Optional[Image.Image]:
        """Download a tile from the current service.
        
        Args:
            x: Tile X coordinate
            y: Tile Y coordinate
            z: Zoom level
            
        Returns:
            PIL.Image: Downloaded tile image or None if failed
        """
        try:
            url = self.services[self.current_service]
            
            # Replace placeholders in URL
            url = url.replace("{z}", str(z))
            url = url.replace("{x}", str(x))
            url = url.replace("{y}", str(y))
            url = url.replace("{s}", "a")  # Use first subdomain
            
            # Download tile
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content))
            else:
                logger.warning(f"Failed to download tile: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading tile: {str(e)}")
            logger.exception(e)
            return None
    
    def _add_to_memory_cache(self, key: str, image: Image.Image) -> None:
        """Add a tile to the memory cache.
        
        Args:
            key: Cache key
            image: Tile image
        """
        try:
            # Calculate image size in bytes
            img_size = image.size[0] * image.size[1] * len(image.getbands())
            
            # Check if adding would exceed cache limit
            while (self.cache_size + img_size) > config.CACHE_SIZE_LIMIT * 1024 * 1024:
                # Remove oldest item
                if self.cache:
                    oldest_key = next(iter(self.cache))
                    oldest_img = self.cache.pop(oldest_key)
                    self.cache_size -= (oldest_img.size[0] * 
                                      oldest_img.size[1] * 
                                      len(oldest_img.getbands()))
                else:
                    break
            
            # Add new item
            self.cache[key] = image
            self.cache_size += img_size
            
        except Exception as e:
            logger.error(f"Error adding to memory cache: {str(e)}")
            logger.exception(e)