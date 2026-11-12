// Canvas view: the raster as a stretched band combination (stretch.js, as RasterLayer displays it), with the
// classification (at S.opacity), the painted training pixels and, for the demo, the pixels that disagree with the
// known classes. The brush is a port of TrainingPaintTool (disk of radius brush/2 around the cursor, clipped at the
// edges).
import { S, emit } from './state.js';
import { prepareDisplay } from './stretch.js';

export const hex = c => [1, 3, 5].map(i => parseInt(c.slice(i, i + 2), 16));

let cache = { key: null, rgb: null };
export function baseRgb() {
  const { image: img, display: d } = S, key = `${img.name}|${img.W}|${d.bands.join(',')}|${d.stretch}`;
  if (cache.key !== key || cache.img !== img) {
    const stretch = d.stretch === 'Linear' ? { type: 'Linear', ...linearRange() } : d.stretch === 'None' ? null : { type: d.stretch };
    cache = { key, img, rgb: prepareDisplay(img.bands, img.W * img.H, d.bands, stretch) };
  }
  return cache.rgb;
}

// 'Linear' uses the data min / max of the whole raster, as the Stretch dialog's Reset sets them.
function linearRange() {
  let min = Infinity, max = -Infinity;
  for (const b of S.image.bands) for (const v of b) { if (v < min) min = v; if (v > max) max = v; }
  return { min, max };
}

// opts: { result, strokes, errors }
export function compose(opts) {
  const { W, H } = S.image, N = W * H, rgb = baseRgb(), img = new ImageData(W, H), d = img.data;
  for (let i = 0; i < N; i++) { d[4 * i] = rgb[3 * i]; d[4 * i + 1] = rgb[3 * i + 1]; d[4 * i + 2] = rgb[3 * i + 2]; d[4 * i + 3] = 255; }
  const pred = opts.result && S.result?.pred;
  if (pred) {
    const cols = S.classes.map(c => hex(c.color)), a = S.opacity;
    for (let i = 0; i < N; i++) { const c = cols[pred[i]]; if (c) for (let j = 0; j < 3; j++) d[4 * i + j] = d[4 * i + j] * (1 - a) + c[j] * a; }
  }
  if (opts.strokes) S.classes.forEach(cl => {
    const c = hex(cl.color), m = cl.mask;
    for (let i = 0; i < N; i++) if (m[i]) {
      const edge = !m[i - 1] || !m[i + 1] || !m[i - W] || !m[i + W];
      for (let j = 0; j < 3; j++) d[4 * i + j] = edge ? 255 : d[4 * i + j] * 0.2 + c[j] * 0.8;
    }
  });
  const truth = opts.errors && S.image.truth;
  if (truth && pred) for (let i = 0; i < N; i++) if (truth[i] !== pred[i]) { d[4 * i] = 255; d[4 * i + 1] = 255; d[4 * i + 2] = 255; }
  return img;
}

export function stamp(mask, H, W, row, col, size, value) {
  const r2 = (size / 2) ** 2;
  for (let dr = -size; dr <= size; dr++) for (let dc = -size; dc <= size; dc++) {
    const r = row + dr, c = col + dc;
    if (r < 0 || c < 0 || r >= H || c >= W || dr * dr + dc * dc > r2) continue;
    mask[r * W + c] = value;
  }
}

// Paint on a canvas. getMode() -> 'paint' | 'erase' | 'view'; redraw() repaints.
export function attachBrush(canvas, getMode, redraw) {
  let down = false, last = null;
  const pos = e => {
    const r = canvas.getBoundingClientRect();
    return [Math.floor((e.clientY - r.top) * S.image.H / r.height), Math.floor((e.clientX - r.left) * S.image.W / r.width)];
  };
  const paintTo = p => {
    const cl = S.classes[S.active];
    if (!cl) return;
    const { H, W } = S.image, value = getMode() === 'erase' ? 0 : 1;
    const [r0, c0] = last || p, n = Math.max(Math.abs(p[0] - r0), Math.abs(p[1] - c0), 1);
    for (let s = 0; s <= n; s++) {
      const r = Math.round(r0 + (p[0] - r0) * s / n), c = Math.round(c0 + (p[1] - c0) * s / n);
      stamp(cl.mask, H, W, r, c, S.brush, value);
      // web choice: a pixel belongs to one class, so painting removes it from the others (the desktop keeps
      // independent masks, where a pixel painted twice becomes two training samples)
      if (value) for (const o of S.classes) if (o !== cl) stamp(o.mask, H, W, r, c, S.brush, 0);
    }
    last = p;
    redraw();
  };
  canvas.addEventListener('pointerdown', e => {
    if (getMode() === 'view' || !S.classes[S.active]) return;
    e.preventDefault();
    canvas.setPointerCapture(e.pointerId);
    S.history.push(S.classes.map(c => c.mask.slice()));
    if (S.history.length > 30) S.history.shift();
    down = true; last = null;
    paintTo(pos(e));
  });
  canvas.addEventListener('pointermove', e => { if (down) paintTo(pos(e)); });
  const up = () => { if (down) { down = false; last = null; emit(); } };
  canvas.addEventListener('pointerup', up);
  canvas.addEventListener('pointercancel', up);
}

export function undo() {
  const h = S.history.pop();
  if (!h) return false;
  S.classes.forEach((c, i) => { if (h[i]) c.mask = h[i]; });
  emit();
  return true;
}

export const countPainted = cl => { let n = 0; for (const v of cl.mask) n += v; return n; };
