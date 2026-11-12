"""Band combination and contrast stretch used to display a raster (numpy only; RasterLayer.prepare_display_data).

stretch_params is None (min-max per band) or {'type': 'Linear' | 'Linear 2%' | 'Linear 5%' | 'Standard Deviation',
'min': float, 'max': float} as set by the Stretch dialog ('min'/'max' are used by plain 'Linear' only).
"""
from typing import Dict, Optional, Sequence

import numpy as np


def _scale(band: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    return np.clip((band - min_val) * 255 / (max_val - min_val), 0, 255)


def prepare_display(data: np.ndarray, band_indices: Sequence[int] = (1, 2, 3),
                    stretch_params: Optional[Dict] = None) -> np.ndarray:
    """Stretch bands (1-based band_indices) of a (bands, height, width) array to an RGB uint8 (height, width, 3)."""
    if data.shape[0] >= 3:
        rgb = np.stack([data[i - 1] for i in band_indices[:3]])
    else:
        rgb = np.repeat(data[:1], 3, axis=0)  # single band -> grey

    normalized = np.zeros_like(rgb, dtype=np.uint8)
    stretch_type = stretch_params['type'] if stretch_params else None
    for i in range(3):
        band = rgb[i].astype(float)
        if stretch_type and stretch_type.startswith("Linear"):
            if "2%" in stretch_type:
                p_min, p_max = 2, 98
            elif "5%" in stretch_type:
                p_min, p_max = 5, 95
            else:
                p_min, p_max = 0, 100
            if p_min == 0 and p_max == 100:
                min_val, max_val = stretch_params['min'], stretch_params['max']
            else:
                min_val, max_val = np.nanpercentile(band[~np.isnan(band)], (p_min, p_max))
            if max_val > min_val:
                normalized[i] = _scale(band, min_val, max_val)
        elif stretch_type == "Standard Deviation":
            valid = ~np.isnan(band)
            mean, std = np.mean(band[valid]), np.std(band[valid])
            if std > 0:
                normalized[i] = _scale(band, mean - 2 * std, mean + 2 * std)
        else:  # default (also 'None'): min-max
            min_val, max_val = np.nanmin(band), np.nanmax(band)
            if max_val > min_val:
                normalized[i] = _scale(band, min_val, max_val)

    return np.ascontiguousarray(normalized.transpose(1, 2, 0))
