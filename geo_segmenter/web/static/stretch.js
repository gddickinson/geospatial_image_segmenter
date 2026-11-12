// Display of a multi-band raster: port of utils/display_stretch.prepare_display (RasterLayer.prepare_display_data).
// Three bands (1-based band indices; one band -> grey) are stretched to 0-255 and truncated as numpy's uint8 cast:
//   null (default)       min-max per band
//   'Linear'             the given min / max
//   'Linear 2%' / '5%'   np.nanpercentile (linear interpolation) 2-98 % / 5-95 %
//   'Standard Deviation' mean +- 2 std
// Tested against Python in stretch.test.mjs.
import { npSum } from './filters.js';

export const STRETCHES = ['None', 'Linear', 'Linear 2%', 'Linear 5%', 'Standard Deviation'];  // StretchDialog

// np.percentile(a, q) with the default 'linear' method on finite values.
export function percentile(values, q) {
  const v = Float64Array.from(values).filter(x => !Number.isNaN(x)).sort(), n = v.length;
  const qq = q / 100, vi = n * qq + (1 + qq * (1 - 1 - 1)) - 1;
  const lo = Math.min(Math.max(Math.floor(vi), 0), n - 1), hi = Math.min(lo + 1, n - 1), t = vi - Math.floor(vi);
  const a = v[lo], b = v[hi], d = b - a;
  return t >= 0.5 ? b - d * (1 - t) : a + d * t;
}

function minmax(band) {
  let lo = Infinity, hi = -Infinity;
  for (const x of band) { if (x < lo) lo = x; if (x > hi) hi = x; }
  return [lo, hi];
}

// bands: array of typed arrays (length N). Returns Uint8Array N*3 (RGB interleaved).
export function prepareDisplay(bands, N, bandIndices = [1, 2, 3], stretch = null) {
  const rgb = bands.length >= 3 ? bandIndices.slice(0, 3).map(i => bands[i - 1]) : [bands[0], bands[0], bands[0]];
  const out = new Uint8Array(N * 3), type = stretch && stretch.type !== 'None' ? stretch.type : null;
  rgb.forEach((band, c) => {
    let lo, hi;
    if (type && type.startsWith('Linear')) {
      const p = type.includes('2%') ? [2, 98] : type.includes('5%') ? [5, 95] : null;
      [lo, hi] = p ? [percentile(band, p[0]), percentile(band, p[1])] : [stretch.min, stretch.max];
      if (!(hi > lo)) return;
    } else if (type === 'Standard Deviation') {
      const f = Float64Array.from(band), mean = npSum(f) / N, d = f.map(x => Math.abs(x - mean) ** 2);
      const std = Math.sqrt(npSum(d) / N);
      if (!(std > 0)) return;
      [lo, hi] = [mean - 2 * std, mean + 2 * std];
    } else {
      [lo, hi] = minmax(band);
      if (!(hi > lo)) return;
    }
    for (let i = 0; i < N; i++) {
      const v = Math.min(Math.max((band[i] - lo) * 255 / (hi - lo), 0), 255);
      out[3 * i + c] = Math.trunc(v);
    }
  });
  return out;
}
