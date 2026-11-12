"""The classification pipeline as the desktop app sets it up (Qt-free, shared by the GUI and the web interface).

main_window.train_model loads a raster as (height, width, bands), maps bands 1-3 to red, green, blue (and band 4 to
NIR when present), adds the spectral extractor when there are at least three bands and always adds the texture
extractor, then trains a RandomForestModel on the painted label masks.
"""
from typing import Dict, Optional

from .features.spectral import SpectralFeatureExtractor
from .features.texture import TextureFeatureExtractor


def default_band_map(n_bands: int) -> Dict[str, int]:
    """Band roles assumed by the desktop app: R, G, B in bands 1-3 and NIR in band 4 (0-based indices)."""
    band_map = {'red': 0, 'green': 1, 'blue': 2}
    if n_bands > 3:
        band_map['nir'] = 3
    return band_map


def add_default_extractors(model, n_bands: int, band_map: Dict[str, int] = None,
                           texture_local_mean: Optional[bool] = None) -> None:
    """Add the spectral (>= 3 bands) and texture extractors to a GeospatialModel, as main_window.train_model does.

    texture_local_mean: add the local mean grey level per texture window (None = config.TEXTURE_LOCAL_MEAN).
    """
    if n_bands >= 3:
        spectral = SpectralFeatureExtractor()
        spectral.set_band_map(dict(band_map) if band_map is not None else default_band_map(n_bands))
        model.add_feature_extractor('spectral', spectral)
    texture = TextureFeatureExtractor()
    if texture_local_mean is not None:
        texture.set_parameters({'local_mean': bool(texture_local_mean)})
    model.add_feature_extractor('texture', texture)
