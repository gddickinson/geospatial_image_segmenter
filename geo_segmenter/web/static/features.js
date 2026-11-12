// JS port of the feature pipeline the desktop app trains on (models/pipeline.py):
//   spectral (SpectralFeatureExtractor, when the image has >= 3 bands): the indices of config.SPECTRAL_INDICES whose
//     bands are mapped, then RG_ratio and NIR_ratio;
//   texture (TextureFeatureExtractor): grey = 0.2989 R + 0.5870 G + 0.1140 B of bands 1-3 (band 1 alone for fewer
//     bands), stretched to 0-255 and truncated to uint8, then per window size (config.TEXTURE_WINDOWS):
//     local standard deviation, local range (max - min), Gaussian gradient magnitude with sigma = window / 3.
// Bands are typed arrays of length H*W (any numeric type). Tested against Python in features.test.mjs.
import { boxSum, gaussianGradientMagnitude, rankFilter } from './filters.js';

export const SPECTRAL_INDICES = ['NDVI', 'NDWI', 'SAVI', 'EVI'];  // config.SPECTRAL_INDICES
export const TEXTURE_WINDOWS = [3, 5, 7];                         // config.TEXTURE_WINDOWS
const EPS = 1e-8;
const NEEDS = { NDVI: ['red', 'nir'], NDWI: ['green', 'nir'], SAVI: ['red', 'nir'], EVI: ['blue', 'red', 'nir'],
  ARVI: ['blue', 'red', 'nir'], SIPI: ['blue', 'red', 'nir'] };

// models/pipeline.default_band_map: bands 1-3 = R, G, B, band 4 = NIR (0-based).
export function defaultBandMap(nBands) {
  const m = { red: 0, green: 1, blue: 2 };
  if (nBands > 3) m.nir = 3;
  return m;
}

const has = (map, roles) => roles.every(r => map[r] !== undefined && map[r] !== null);

export function spectralNames(bandMap, indices = SPECTRAL_INDICES) {
  const names = indices.filter(ix => NEEDS[ix] && has(bandMap, NEEDS[ix]));
  if (has(bandMap, ['red', 'green'])) names.push('RG_ratio');
  if (has(bandMap, ['nir', 'red'])) names.push('NIR_ratio');
  return names;
}

export function spectralFeatures(bands, N, bandMap, indices = SPECTRAL_INDICES) {
  const b = role => bands[bandMap[role]], out = [];
  const map = f => { const o = new Float64Array(N); for (let i = 0; i < N; i++) o[i] = f(i); out.push(o); };
  for (const ix of indices) {
    if (!NEEDS[ix] || !has(bandMap, NEEDS[ix])) continue;
    const R = b('red'), G = b('green'), B = b('blue'), NIR = b('nir');
    if (ix === 'NDVI') map(i => (NIR[i] - R[i]) / (NIR[i] + R[i] + EPS));
    else if (ix === 'NDWI') map(i => (G[i] - NIR[i]) / (G[i] + NIR[i] + EPS));
    else if (ix === 'SAVI') map(i => ((NIR[i] - R[i]) * (1 + 0.5)) / (NIR[i] + R[i] + 0.5 + EPS));
    else if (ix === 'EVI') map(i => 2.5 * (NIR[i] - R[i]) / (NIR[i] + 6.0 * R[i] - 7.5 * B[i] + 1.0 + EPS));
    else if (ix === 'ARVI') map(i => { const rb = R[i] - (R[i] - B[i]); return (NIR[i] - rb) / (NIR[i] + rb + EPS); });
    else if (ix === 'SIPI') map(i => (NIR[i] - B[i]) / (NIR[i] - R[i] + EPS));
  }
  if (has(bandMap, ['red', 'green'])) { const R = b('red'), G = b('green'); map(i => R[i] / (G[i] + EPS)); }
  if (has(bandMap, ['nir', 'red'])) { const R = b('red'), NIR = b('nir'); map(i => NIR[i] / (R[i] + EPS)); }
  return out;
}

export const textureNames = (windows = TEXTURE_WINDOWS) =>
  windows.flatMap(w => [`StdDev_${w}px`, `Range_${w}px`, `GradientMagnitude_${w}px`]);

// Grey levels 0-255 (uint8 semantics) as in TextureFeatureExtractor.extract_features.
export function grayLevels(bands, N) {
  const g = new Float64Array(N);
  if (bands.length >= 3) {
    const [R, G, B] = bands;
    for (let i = 0; i < N; i++) g[i] = +R[i] * 0.2989 + +G[i] * 0.5870 + +B[i] * 0.1140;
  } else for (let i = 0; i < N; i++) g[i] = +bands[0][i];
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < N; i++) { if (g[i] < lo) lo = g[i]; if (g[i] > hi) hi = g[i]; }
  const span = hi - lo, out = new Float64Array(N);
  if (span > 0) for (let i = 0; i < N; i++) out[i] = Math.trunc(((g[i] - lo) / span) * 255);
  return out;
}

export function textureFeatures(bands, H, W, windows = TEXTURE_WINDOWS, onStep = () => {}) {
  const N = H * W, g = grayLevels(bands, N), sq = g.map(v => v * v), out = [];
  for (const w of windows) {
    const n = w * w, s1 = boxSum(g, H, W, w), s2 = boxSum(sq, H, W, w), std = new Float64Array(N);
    for (let i = 0; i < N; i++) std[i] = Math.sqrt(Math.max(n * s2[i] - s1[i] * s1[i], 0)) / n;
    out.push(std);
    onStep(`StdDev_${w}px`);
    const mx = rankFilter(g, H, W, w, true), mn = rankFilter(g, H, W, w, false), range = new Float64Array(N);
    for (let i = 0; i < N; i++) range[i] = mx[i] - mn[i];
    out.push(range);
    onStep(`Range_${w}px`);
    out.push(gaussianGradientMagnitude(g, H, W, w / 3));
    onStep(`GradientMagnitude_${w}px`);
  }
  return out;
}

// All feature maps in column order, with names '<extractor>_<feature>' (FeatureSet.enabled_feature_names).
// opts: { bandMap, spectral: true, texture: true } (the desktop app always uses both).
export function extractAll(bands, H, W, { bandMap = defaultBandMap(bands.length), spectral = true, texture = true,
  onStep = () => {} } = {}) {
  const maps = [], names = [];
  if (spectral && bands.length >= 3) {
    const sn = spectralNames(bandMap);
    if (sn.length) {
      maps.push(...spectralFeatures(bands, H * W, bandMap));
      names.push(...sn.map(n => `spectral_${n}`));
      onStep('spectral indices');
    }
  }
  if (texture) {
    maps.push(...textureFeatures(bands, H, W, TEXTURE_WINDOWS, onStep));
    names.push(...textureNames().map(n => `texture_${n}`));
  }
  return { maps, names };
}

// Human-readable descriptions for the feature page.
export const DESCRIBE = {
  NDVI: 'Normalised difference vegetation index (NIR − red) / (NIR + red)',
  NDWI: 'Normalised difference water index (green − NIR) / (green + NIR)',
  SAVI: 'Soil-adjusted vegetation index, L = 0.5',
  EVI: 'Enhanced vegetation index, G = 2.5, C1 = 6, C2 = 7.5, L = 1',
  RG_ratio: 'Red / green ratio', NIR_ratio: 'NIR / red ratio',
  StdDev: 'Local standard deviation of the grey image', Range: 'Local range (max − min) of the grey image',
  GradientMagnitude: 'Gaussian gradient magnitude of the grey image, σ = window / 3',
};
export function describe(name) {
  const short = name.replace(/^(spectral|texture)_/, ''), key = short.replace(/_\d+px$/, '');
  return DESCRIBE[short] || DESCRIBE[key] || short;
}
