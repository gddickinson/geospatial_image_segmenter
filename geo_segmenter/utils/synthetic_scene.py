"""Seeded synthetic 4-band scene (red, green, blue, NIR) for the web demo and the tests.

This is NOT real imagery: the land-cover layout is drawn from random fields and the spectra are generic textbook
shapes (water dark in NIR, vegetation bright in NIR, soil and roofs bright in the visible), scaled like surface
reflectance x 10000 and stored as uint16. The band order matches the desktop app's default band map
(models/pipeline.default_band_map): band 1 red, 2 green, 3 blue, 4 NIR. Pixels are 10 m on a side in a local,
user-defined metric grid (no real location).

make_scene(size, seed) -> dict(image (H, W, 4) uint16, truth (H, W) uint8 class index, classes, strokes
{class name: (H, W) bool example training strokes}, pixel_size_m)
"""
from typing import Dict

import numpy as np
from scipy import ndimage

CLASSES = [  # name, display colour, mean reflectance x 10000 (red, green, blue, NIR)
    ("Water", "#2f7bff", (300, 520, 640, 180)),
    ("Forest", "#1f9e4a", (280, 520, 300, 3300)),
    ("Cropland", "#b7e05a", (520, 820, 420, 4100)),
    ("Bare soil", "#c98b4f", (1900, 1550, 1200, 2500)),
    ("Built-up", "#ff4fa0", (1500, 1450, 1400, 1900)),
]
PIXEL_SIZE_M = 10.0


def _field(rng, shape, sigma):
    """Smooth random field with zero mean and unit standard deviation."""
    f = ndimage.gaussian_filter(rng.standard_normal(shape), sigma, mode="reflect")
    return (f - f.mean()) / f.std()


def _layout(rng, n):
    """Class index per pixel: farmland background, forest patches, a town, a lake and a river."""
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    truth = np.full((n, n), 2, dtype=np.uint8)  # cropland
    # bare-soil parcels: a jittered grid of fields, about one in four left bare
    cell = n // 8
    parcels = (yy // cell).astype(int) * 8 + (xx // cell).astype(int)
    bare_ids = rng.permutation(64)[:16]
    truth[np.isin(parcels, bare_ids)] = 3
    # forest patches from a smooth random field, densest towards the top-left
    forest = _field(rng, (n, n), n / 18) + 1.2 * (1 - (xx + yy) / n) > 1.0
    truth[forest] = 1
    # town: a block in the lower right with a street grid (streets are part of the built-up class)
    town = (xx > 0.58 * n) & (yy > 0.55 * n) & (_field(rng, (n, n), n / 25) > -0.9)
    truth[town] = 4
    # lake (ellipse with a wobbly shore) and a meandering river crossing the scene
    r = np.hypot((xx - 0.3 * n) / (0.16 * n), (yy - 0.68 * n) / (0.11 * n)) + 0.18 * _field(rng, (n, n), n / 30)
    truth[r < 1] = 0
    river_x = 0.72 * n + 0.07 * n * np.sin(yy / n * 3 * np.pi) + 0.03 * n * np.sin(yy / n * 7.3 * np.pi + 1)
    truth[np.abs(xx - river_x) < 2.6 + 0.8 * np.sin(yy / 9)] = 0
    return truth, parcels, town


def _spectra(rng, truth, parcels, town):
    """Reflectance x 10000 per band with class-specific texture, sensor blur and noise."""
    n = truth.shape[0]
    img = np.zeros((n, n, 4))
    for k, (_, _, mean) in enumerate(CLASSES):
        img[truth == k] = mean
    canopy = 1 + 0.18 * _field(rng, (n, n), 0.8)  # forest: rough canopy texture
    img[truth == 1] *= canopy[truth == 1][:, None]
    vigour = rng.uniform(0.65, 1.15, size=parcels.max() + 1)[parcels]  # cropland: one crop state per field
    rows = 1 + 0.05 * np.sin(np.arange(n) * 2 * np.pi / 3)[None, :]  # faint crop rows
    crop = truth == 2
    img[crop, 3] *= (vigour * rows)[crop]
    img[crop, :3] *= (1.25 - 0.4 * vigour)[crop][:, None]
    moist = 1 + 0.12 * _field(rng, (n, n), 3)  # bare soil: moisture patches
    img[truth == 3] *= moist[truth == 3][:, None]
    sediment = 1 + 0.25 * _field(rng, (n, n), 6)  # water: sediment plumes in the visible bands
    img[truth == 0, :3] *= sediment[truth == 0][:, None]
    # built-up: roofs of varying brightness on 4-pixel blocks, darker streets every 9 pixels
    yy, xx = np.mgrid[0:n, 0:n]
    roofs = rng.uniform(0.6, 1.7, size=((n + 3) // 4, (n + 3) // 4))[yy // 4, xx // 4]
    street = (yy % 9 == 0) | (xx % 9 == 0)
    b = (truth == 4)
    img[b] *= np.where(street, 0.7, roofs)[b][:, None]
    img = ndimage.gaussian_filter(img, sigma=(0.6, 0.6, 0), mode="reflect")  # sensor point-spread
    img += rng.normal(0, 40, size=img.shape)  # sensor noise
    return np.clip(np.rint(img), 1, 10000).astype(np.uint16)


def _strokes(rng, truth, per_class=5, radius=3) -> Dict[str, np.ndarray]:
    """Example training strokes: a few short brush strokes well inside each class."""
    n = truth.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    out = {}
    for k, (name, _, _) in enumerate(CLASSES):
        inner = ndimage.binary_erosion(truth == k, iterations=radius + 3)
        cand = np.argwhere(inner)
        mask = np.zeros((n, n), bool)
        for i in rng.choice(len(cand), size=min(per_class, len(cand)), replace=False):
            y, x = cand[i]
            for t in np.linspace(-1, 1, 5):  # a short horizontal stroke made of brush stamps
                cx = int(np.clip(x + 3 * t, 0, n - 1))
                mask |= ((yy - y) ** 2 + (xx - cx) ** 2 <= radius ** 2)
        out[name] = mask & (truth == k)
    return out


def make_scene(size: int = 256, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    truth, parcels, town = _layout(rng, size)
    image = _spectra(rng, truth, parcels, town)
    return dict(image=image, truth=truth, strokes=_strokes(rng, truth),
                classes=[dict(name=n, color=c) for n, c, _ in CLASSES], pixel_size_m=PIXEL_SIZE_M)
