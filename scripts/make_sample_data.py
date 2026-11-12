"""Write test_data/sample.tif: a seeded, synthetic 3-band GeoTIFF for trying the desktop app and the GUI scripts.

The image is NOT real imagery. It is the red, green and blue bands of utils/synthetic_scene.make_scene (generic
land-cover spectra on random-field layouts, reflectance x 10000, uint16) at 1001 x 1001 pixels, so it is free of any
third-party licence (see DATA_SOURCES.md). It has the same layout as the file it replaces: 3 bands, uint16,
pixel-interleaved, uncompressed, 10 m pixels, WGS 84 / UTM zone 31N (EPSG:32631) with a north-up geotransform.
The CRS and origin only give the file a valid georeference; the scene does not depict that place.

Run: python scripts/make_sample_data.py [--out PATH] [--size N] [--seed S]
(tests/test_sample_data.py checks the written file's layout.)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from geo_segmenter.utils.synthetic_scene import PIXEL_SIZE_M, make_scene  # noqa: E402

SIZE, SEED = 1001, 2026
EPSG = 32631  # WGS 84 / UTM zone 31N
ORIGIN = (590520.0, 5790630.0)  # upper-left corner (easting, northing), metres
CITATION = "WGS 84 / UTM zone 31N|WGS 84|"
DESCRIPTION = ("Synthetic sample scene (geo_segmenter scripts/make_sample_data.py, seed {seed}); not real imagery; "
               "bands: red, green, blue (reflectance x 10000)")


def geotiff_tags(epsg: int = EPSG, origin=ORIGIN, pixel: float = PIXEL_SIZE_M, citation: str = CITATION):
    """tifffile extratags for a north-up projected GeoTIFF (pixel scale, tiepoint, GeoKey directory, ASCII params)."""
    ascii_len = len(citation)
    keys = [1, 1, 0, 7,
            1024, 0, 1, 1,                      # GTModelType = projected
            1025, 0, 1, 1,                      # GTRasterType = PixelIsArea
            1026, 34737, ascii_len - len("WGS 84|"), 0,  # GTCitation
            2049, 34737, len("WGS 84|"), ascii_len - len("WGS 84|"),  # GeogCitation
            2054, 0, 1, 9102,                   # GeogAngularUnits = degree
            3072, 0, 1, epsg,                   # ProjectedCSType
            3076, 0, 1, 9001]                   # ProjLinearUnits = metre
    return [(33550, 12, 3, (pixel, pixel, 0.0), True),
            (33922, 12, 6, (0.0, 0.0, 0.0, origin[0], origin[1], 0.0), True),
            (34735, 3, len(keys), tuple(keys), True),
            (34737, 2, 0, citation, True)]


def make_sample(size: int = SIZE, seed: int = SEED) -> np.ndarray:
    """(size, size, 3) uint16 red, green, blue bands of the synthetic scene."""
    return np.ascontiguousarray(make_scene(size=size, seed=seed)["image"][..., :3])


def write_sample(path: Path, size: int = SIZE, seed: int = SEED) -> Path:
    import tifffile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, make_sample(size, seed), photometric="minisblack", planarconfig="contig",
                     description=DESCRIPTION.format(seed=seed), metadata=None, extratags=geotiff_tags())
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "test_data" / "sample.tif")
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    out = write_sample(a.out, a.size, a.seed)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
