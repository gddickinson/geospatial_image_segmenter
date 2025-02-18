"""Module for managing segmentation labels and masks for geospatial data."""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional
from PyQt6.QtGui import QColor
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Polygon, mapping
import json
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class Label:
    """Class to store label information and masks.
    
    Attributes:
        name: Label name
        color: QColor for display
        layer_id: ID of the layer this label is associated with
        masks: Dictionary mapping layer IDs to boolean masks
        metadata: Dictionary of additional metadata
    """
    name: str
    color: QColor
    layer_id: str
    masks: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    
    def add_mask(self, layer_id: str, mask: np.ndarray) -> None:
        """Add or update mask for a specific layer.
        
        Args:
            layer_id: Layer identifier
            mask: Boolean mask array
        """
        try:
            if not isinstance(mask, np.ndarray):
                raise ValueError("Mask must be a numpy array")
            if mask.dtype != bool:
                mask = mask.astype(bool)
            self.masks[layer_id] = mask
            logger.debug(f"Added mask for layer {layer_id} to label '{self.name}'")
        except Exception as e:
            logger.error(f"Error adding mask to label '{self.name}': {str(e)}")
            raise
    
    def get_mask(self, layer_id: str) -> Optional[np.ndarray]:
        """Get mask for a specific layer.
        
        Args:
            layer_id: Layer identifier
            
        Returns:
            numpy.ndarray: Boolean mask array or None if not found
        """
        try:
            return self.masks.get(layer_id, None)
        except Exception as e:
            logger.error(f"Error retrieving mask for layer {layer_id}: {str(e)}")
            raise
    
    def merge_mask(self, layer_id: str, new_mask: np.ndarray) -> None:
        """Merge new mask with existing mask using OR operation.
        
        Args:
            layer_id: Layer identifier
            new_mask: Boolean mask array to merge
        """
        try:
            if layer_id in self.masks:
                self.masks[layer_id] = np.logical_or(self.masks[layer_id], new_mask)
            else:
                self.add_mask(layer_id, new_mask)
            logger.debug(f"Merged mask for layer {layer_id} in label '{self.name}'")
        except Exception as e:
            logger.error(f"Error merging mask for label '{self.name}': {str(e)}")
            raise
    
    def clear_mask(self, layer_id: str) -> None:
        """Clear mask for a specific layer.
        
        Args:
            layer_id: Layer identifier
        """
        try:
            if layer_id in self.masks:
                del self.masks[layer_id]
                logger.debug(f"Cleared mask for layer {layer_id} in label '{self.name}'")
        except Exception as e:
            logger.error(f"Error clearing mask for label '{self.name}': {str(e)}")
            raise
    
    def export_geojson(self, transform, crs) -> dict:
        """Export label masks as GeoJSON.
        
        Args:
            transform: Rasterio transform object
            crs: Coordinate reference system
            
        Returns:
            dict: GeoJSON feature collection
        """
        try:
            features = []
            
            for layer_id, mask in self.masks.items():
                # Convert mask to polygons
                from rasterio import features
                shapes = features.shapes(
                    mask.astype(np.uint8),
                    transform=transform
                )
                
                # Convert shapes to GeoJSON features
                for shape, value in shapes:
                    if value == 1:  # Only include masked areas
                        feature = {
                            "type": "Feature",
                            "geometry": shape,
                            "properties": {
                                "label_name": self.name,
                                "layer_id": layer_id,
                                "color": {
                                    "r": self.color.red(),
                                    "g": self.color.green(),
                                    "b": self.color.blue(),
                                    "a": self.color.alpha()
                                },
                                "metadata": self.metadata
                            }
                        }
                        features.append(feature)
            
            geojson = {
                "type": "FeatureCollection",
                "features": features,
                "crs": {
                    "type": "name",
                    "properties": {
                        "name": crs.to_string()
                    }
                }
            }
            
            logger.debug(f"Exported {len(features)} features to GeoJSON")
            return geojson
            
        except Exception as e:
            logger.error(f"Error exporting to GeoJSON: {str(e)}")
            raise
    
    def import_geojson(self, geojson: dict, transform, shape: tuple) -> None:
        """Import label masks from GeoJSON.
        
        Args:
            geojson: GeoJSON feature collection
            transform: Rasterio transform object
            shape: Output mask shape (height, width)
        """
        try:
            if geojson["type"] != "FeatureCollection":
                raise ValueError("Input must be a GeoJSON FeatureCollection")
            
            # Create empty mask
            mask = np.zeros(shape, dtype=bool)
            
            # Convert features to geometries
            geometries = []
            for feature in geojson["features"]:
                if feature["properties"].get("label_name") == self.name:
                    geometries.append(shape(feature["geometry"]))
            
            if geometries:
                # Rasterize geometries
                mask = rasterize(
                    geometries,
                    out_shape=shape,
                    transform=transform,
                    fill=0,
                    default_value=1,
                    dtype=np.uint8
                ).astype(bool)
            
            # Get layer ID from first matching feature
            layer_id = None
            for feature in geojson["features"]:
                if feature["properties"].get("label_name") == self.name:
                    layer_id = feature["properties"].get("layer_id")
                    break
            
            if layer_id:
                self.add_mask(layer_id, mask)
                logger.debug(f"Imported mask for layer {layer_id} from GeoJSON")
            
        except Exception as e:
            logger.error(f"Error importing from GeoJSON: {str(e)}")
            raise
    
    def save_to_file(self, path: str) -> None:
        """Save label data to file.
        
        Args:
            path: Save path
        """
        try:
            # Convert masks to lists for JSON serialization
            mask_dict = {
                layer_id: mask.tolist()
                for layer_id, mask in self.masks.items()
            }
            
            data = {
                "name": self.name,
                "color": {
                    "r": self.color.red(),
                    "g": self.color.green(),
                    "b": self.color.blue(),
                    "a": self.color.alpha()
                },
                "layer_id": self.layer_id,
                "masks": mask_dict,
                "metadata": self.metadata
            }
            
            with open(path, 'w') as f:
                json.dump(data, f)
                
            logger.debug(f"Saved label data to {path}")
            
        except Exception as e:
            logger.error(f"Error saving label data: {str(e)}")
            raise
    
    @classmethod
    def load_from_file(cls, path: str) -> 'Label':
        """Load label data from file.
        
        Args:
            path: Load path
            
        Returns:
            Label: New Label instance
        """
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Reconstruct color
            color = QColor(
                data["color"]["r"],
                data["color"]["g"],
                data["color"]["b"],
                data["color"]["a"]
            )
            
            # Create label instance
            label = cls(
                name=data["name"],
                color=color,
                layer_id=data["layer_id"]
            )
            
            # Load masks
            for layer_id, mask_list in data["masks"].items():
                label.masks[layer_id] = np.array(mask_list, dtype=bool)
            
            # Load metadata
            label.metadata = data["metadata"]
            
            logger.debug(f"Loaded label data from {path}")
            return label
            
        except Exception as e:
            logger.error(f"Error loading label data: {str(e)}")
            raise
                                