// Session state shared by the pages: the loaded image, the classes with their painted training pixels, display
// settings and the classification result. Mirrors the desktop workflow: Label Manager classes, paint training areas,
// Train (Random Forest), Run Segmentation.
import { api, isStatic } from './core.js';
import { call } from './worker_client.js';
import { openImage } from './image_io.js';
import { defaultBandMap } from './features.js';

const DEMO_CRS = 'local metric grid (synthetic demo, user-defined)';
export const PALETTE = ['#2f7bff', '#1f9e4a', '#b7e05a', '#c98b4f', '#ff4fa0', '#ffd23f', '#9b7bff', '#3de0c8'];

export const S = {
  info: null,
  image: null,   // { name, W, H, bands, dtype, bandNames, geo, factor, source: 'demo'|'upload', truth? }
  classes: [],   // [{ name, color, mask: Uint8Array }]
  active: 0, brush: 5, opacity: 0.5,
  bandMap: null, featureNames: [],
  display: { bands: [1, 2, 3], stretch: 'Linear 2%' },
  result: null,  // { pred: Uint8Array (class index), engine, note, metrics, ... }
  history: [],
  listeners: new Set(),
};

export const emit = () => S.listeners.forEach(f => f());

export async function getInfo() {
  if (!S.info) S.info = await api('/info');
  return S.info;
}

export const b64bytes = s => { const bin = atob(s), u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; };
export function unpackMask(s, n) {
  const bits = b64bytes(s), m = new Uint8Array(n);
  for (let i = 0; i < n; i++) m[i] = (bits[i >> 3] >> (7 - (i & 7))) & 1;
  return m;
}
export function packMask(m) {
  const bytes = new Uint8Array(Math.ceil(m.length / 8));
  for (let i = 0; i < m.length; i++) if (m[i]) bytes[i >> 3] |= 1 << (7 - (i & 7));
  return bytesB64(bytes);
}
export function bytesB64(bytes) {
  let bin = '';
  for (let i = 0; i < bytes.length; i += 8192) bin += String.fromCharCode(...bytes.subarray(i, i + 8192));
  return btoa(bin);
}

// Pixel area in m² when the georeferencing gives metres (null for degrees or no georeferencing).
export function pixelArea(img = S.image) {
  const g = img?.geo;
  if (!g?.pixelScale || g.geographic || !g.units || !['metre', 'foot', 'US survey foot'].includes(g.units)) return null;
  const f = g.units === 'metre' ? 1 : g.units === 'foot' ? 0.3048 : 1200 / 3937;
  return Math.abs(g.pixelScale[0] * g.pixelScale[1]) * f * f;
}

async function setImage(img, classes, { bandMap = null, onProgress = () => {} } = {}) {
  S.image = img; S.result = null; S.history = []; S.active = 0;
  S.brush = S.info?.defaults.brush_size ?? 5;
  S.opacity = S.info?.defaults.overlay_opacity ?? 0.5;
  S.classes = classes;
  S.bandMap = bandMap || defaultBandMap(img.bands.length);
  S.display = { bands: img.bands.length >= 3 ? [1, 2, 3] : [1, 1, 1], stretch: 'Linear 2%' };
  S.featureNames = (await call('load', { bands: img.bands, H: img.H, W: img.W, bandMap: S.bandMap }, onProgress)).names;
}

export async function loadDemo(onProgress) {
  const [info, d] = await Promise.all([getInfo(), api('/demo')]);
  const { width: W, height: H } = d.summary, N = W * H;
  const bands = d.bands.map(s => new Uint16Array(b64bytes(s).buffer));
  const img = { name: 'Synthetic demo scene', W, H, bands, dtype: 'uint16', bandNames: d.band_names, source: 'demo',
    factor: 1, truth: b64bytes(d.truth), geo: { pixelScale: [d.summary.pixel_size_m, d.summary.pixel_size_m, 0],
      tiepoint: [0, 0, 0, 0, H * d.summary.pixel_size_m, 0], units: 'metre', epsg: null, geographic: false,
      crsName: DEMO_CRS,
      raw: { keys: [1, 1, 0, 5, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, 32767, 3073, 34737, DEMO_CRS.length + 1, 0,
        3076, 0, 1, 9001], doubles: null, ascii: `${DEMO_CRS}|` } } };
  const classes = d.classes.map((c, k) => ({ name: c.name, color: c.color, mask: unpackMask(d.strokes[k], N) }));
  await setImage(img, classes, { onProgress });
  const py = d.python;
  S.result = { pred: b64bytes(py.pred), engine: 'python', metrics: py.metrics, n_train: d.summary.stroke_pixels,
    note: `Python RandomForestModel (${info.defaults.n_estimators} trees, max depth ${info.defaults.max_depth}, random_state ${d.summary.seed}) trained on the example strokes when the site was built` };
  S.demoForest = py.forest;
  emit();
}

export async function loadFile(file, onProgress) {
  const t = await openImage(file);
  const names = t.bands.length === 3 && t.dtype === 'uint8' && !t.geo ? ['Red', 'Green', 'Blue'] : t.bands.map((_, i) => `Band ${i + 1}`);
  const img = { name: file.name, W: t.width, H: t.height, bands: t.bands, dtype: t.dtype, bandNames: names, geo: t.geo,
    factor: t.factor, source: 'upload', nodata: t.nodata };
  const N = t.width * t.height;
  const classes = ['Water', 'Vegetation', 'Other'].map((n, k) => ({ name: n, color: PALETTE[[0, 1, 3][k]], mask: new Uint8Array(N) }));
  await setImage(img, classes, { onProgress });
  emit();
}

export async function setBandMap(map, onProgress) {
  S.bandMap = map; S.result = null;
  S.featureNames = (await call('load', { bands: S.image.bands, H: S.image.H, W: S.image.W, bandMap: map }, onProgress)).names;
  emit();
}

export const painted = () => S.classes.filter(c => c.mask.some(v => v)).length;

// Train on the painted pixels and classify the whole image, with the Python server or the JS port.
export async function classify({ engine, n_estimators, max_depth, seed }, onProgress) {
  const img = S.image, masks = S.classes.map(c => c.mask), names = S.classes.map(c => c.name);
  if (engine === 'python') {
    if (isStatic()) throw new Error('The Python engine needs the local app');
    onProgress({ stage: 'Python: RandomForestModel.train + predict', done: 0, total: 1 });
    const bands = img.bands.map(b => bytesB64(new Uint8Array(Float32Array.from(b).buffer)));
    const r = await api('/classify', { body: { width: img.W, height: img.H, bands, band_map: S.bandMap, seed,
      n_estimators, max_depth, classes: S.classes.map(c => ({ name: c.name, mask: packMask(c.mask) })) } });
    S.result = { pred: b64bytes(r.pred), engine: 'python', metrics: r.metrics, n_train: r.n_train,
      note: `Python RandomForestModel (seed ${seed ?? 'none'}) trained on ${r.n_train.toLocaleString()} painted pixels in ${r.train_s.toFixed(1)} s, image classified in ${r.predict_s.toFixed(1)} s` };
  } else {
    const t0 = performance.now();
    const r = await call('train', { masks, classNames: names, n_estimators, max_depth, seed }, onProgress);
    S.result = { pred: r.pred, engine: 'js', metrics: r.metrics, n_train: r.n_train,
      note: `Browser (JS port): ${r.trees} trees, ${r.nodes.toLocaleString()} nodes, ${r.n_fit.toLocaleString()} of ${r.n_train.toLocaleString()} painted pixels used for fitting (${r.n_val.toLocaleString()} held out), seed ${seed}, ${((performance.now() - t0) / 1000).toFixed(1)} s` };
  }
  emit();
}

// Re-run the demo's Python forest in the browser (the same trees, evaluated by forest.js).
export async function rerunDemoForest() {
  const r = await call('forest', { forest: S.demoForest });
  return r.pred;
}
