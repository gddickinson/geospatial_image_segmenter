"""test_data/sample.tif is the synthetic scene written by scripts/make_sample_data.py (no third-party data)."""
import sys
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import make_sample_data as msd  # noqa: E402

SAMPLE = ROOT / "test_data" / "sample.tif"


def test_sample_layout_matches_the_replaced_file():
    with tifffile.TiffFile(SAMPLE) as f:
        page = f.pages[0]
        assert page.shape == (1001, 1001, 3) and page.dtype == np.uint16
        assert page.planarconfig == 1 and page.compression == 1
        tags = {t.name: t.value for t in page.tags.values()}
    assert tags["ModelPixelScaleTag"] == (10.0, 10.0, 0.0)
    assert tags["ModelTiepointTag"] == (0.0, 0.0, 0.0, 590520.0, 5790630.0, 0.0)
    keys = tags["GeoKeyDirectoryTag"]
    assert keys[keys.index(3072) + 3] == 32631  # ProjectedCSType: WGS 84 / UTM 31N
    assert "Synthetic" in tags["ImageDescription"]


def test_sample_is_current(tmp_path):
    """Regenerating gives the committed pixels (allowing 1 count of float rounding across platforms)."""
    new = tifffile.imread(msd.write_sample(tmp_path / "s.tif"))
    old = tifffile.imread(SAMPLE)
    assert np.abs(new.astype(int) - old.astype(int)).max() <= 1
