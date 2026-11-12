// Image filters with scipy.ndimage semantics, used by the texture features (features.js). Images are Float64Array
// (or any typed array) in row-major order, H rows x W columns. Boundary mode is scipy's 'reflect' (d c b a | a b c d).
// correlate1d follows scipy's C code (NI_Correlate1D), including its symmetric / antisymmetric shortcuts, so results
// agree with Python to the last few bits (features.test.mjs).

// Index into [0, n) for i outside it, mode 'reflect' (half-sample symmetric, period 2n).
export function reflectIndex(i, n) {
  if (i >= 0 && i < n) return i;
  const p = 2 * n, k = ((i % p) + p) % p;
  return k < n ? k : p - 1 - k;
}

// scipy.ndimage.correlate1d(img, w, axis, mode='reflect', origin=0). axis 0 = along columns (rows vary), 1 = along rows.
export function correlate1d(img, H, W, w, axis) {
  const n = w.length, s1 = n >> 1, s2 = n - s1 - 1, len = axis === 0 ? H : W, lines = axis === 0 ? W : H;
  const eps = 2.220446049250313e-16;
  let sym = 0;
  if (n & 1) {
    sym = 1;
    for (let i = 1; i <= s1; i++) if (Math.abs(w[s1 + i] - w[s1 - i]) > eps) { sym = 0; break; }
    if (!sym) {
      sym = -1;
      for (let i = 1; i <= s1; i++) if (Math.abs(w[s1 + i] + w[s1 - i]) > eps) { sym = 0; break; }
    }
  }
  const out = new Float64Array(H * W), line = new Float64Array(len + s1 + s2);
  for (let l = 0; l < lines; l++) {
    // extended input line: line[s1 + i] = in[i]
    for (let i = -s1; i < len + s2; i++) {
      const j = reflectIndex(i, len);
      line[i + s1] = axis === 0 ? img[j * W + l] : img[l * W + j];
    }
    for (let i = 0; i < len; i++) {
      const c = i + s1;  // centre of the window in `line`
      let v;
      if (sym > 0) {
        v = line[c] * w[s1];
        for (let j = -s1; j < 0; j++) v += (line[c + j] + line[c - j]) * w[s1 + j];
      } else if (sym < 0) {
        v = line[c] * w[s1];
        for (let j = -s1; j < 0; j++) v += (line[c + j] - line[c - j]) * w[s1 + j];
      } else {
        v = line[c + s2] * w[s1 + s2];
        for (let j = -s1; j < s2; j++) v += line[c + j] * w[s1 + j];
      }
      if (axis === 0) out[i * W + l] = v; else out[l * W + i] = v;
    }
  }
  return out;
}

// numpy's float64 sum (pairwise_sum: 8 partial sums for 8 <= n <= 128, plain loop below 8).
export function npSum(a) {
  const n = a.length;
  if (n < 8) { let s = -0; for (let i = 0; i < n; i++) s += a[i]; return s; }
  if (n > 128) { const h = (n >> 1) - ((n >> 1) % 8); return npSum(a.subarray(0, h)) + npSum(a.subarray(h)); }
  const r = Array.from(a.subarray(0, 8));
  let i = 8;
  for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] += a[i + j];
  let s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
  for (; i < n; i++) s += a[i];
  return s;
}

// scipy.ndimage._gaussian_kernel1d(sigma, order, radius) for order 0 or 1.
export function gaussianKernel1d(sigma, order, radius) {
  const n = 2 * radius + 1, phi = new Float64Array(n), sigma2 = sigma * sigma, a = -0.5 / sigma2;
  let sum = 0;
  for (let i = 0; i < n; i++) { const x = i - radius; phi[i] = Math.exp(a * (x * x)); }
  sum = npSum(phi);
  for (let i = 0; i < n; i++) phi[i] = phi[i] / sum;
  if (order === 0) return phi;
  if (order !== 1) throw new Error('only order 0 or 1');
  const q1 = 1 / -sigma2, out = new Float64Array(n);  // polynomial q(x) = x * (1 / -sigma^2)
  for (let i = 0; i < n; i++) out[i] = ((i - radius) * q1) * phi[i];
  return out;
}

// scipy.ndimage.gaussian_filter(img, sigma, order=[oy, ox], mode='reflect', truncate=4.0): axis 0 then axis 1.
export function gaussianFilter(img, H, W, sigma, order = [0, 0], truncate = 4) {
  const radius = Math.trunc(truncate * sigma + 0.5);
  let out = img;
  for (const axis of [0, 1]) {
    const w = gaussianKernel1d(sigma, order[axis], radius).reverse();
    out = correlate1d(out, H, W, w, axis);
  }
  return out;
}

// scipy.ndimage.gaussian_gradient_magnitude(img, sigma): sqrt(d/dy^2 + d/dx^2) of Gaussian derivatives.
export function gaussianGradientMagnitude(img, H, W, sigma) {
  const gy = gaussianFilter(img, H, W, sigma, [1, 0]), gx = gaussianFilter(img, H, W, sigma, [0, 1]);
  const out = new Float64Array(H * W);
  for (let i = 0; i < out.length; i++) out[i] = Math.sqrt(gy[i] * gy[i] + gx[i] * gx[i]);
  return out;
}

// Sum over a size x size window (ndimage.correlate with a box of ones, mode 'reflect'). Exact for integer input.
export function boxSum(img, H, W, size) {
  const ones = new Float64Array(size).fill(1);
  return correlate1d(correlate1d(img, H, W, ones, 0), H, W, ones, 1);
}

// ndimage.minimum_filter / maximum_filter with a size x size footprint (mode 'reflect'); separable for a box.
export function rankFilter(img, H, W, size, isMax) {
  const s1 = size >> 1, pass = (src, axis) => {
    const out = new Float64Array(H * W), len = axis === 0 ? H : W, lines = axis === 0 ? W : H;
    for (let l = 0; l < lines; l++) for (let i = 0; i < len; i++) {
      let v = isMax ? -Infinity : Infinity;
      for (let k = -s1; k < size - s1; k++) {
        const j = reflectIndex(i + k, len), x = axis === 0 ? src[j * W + l] : src[l * W + j];
        if (isMax ? x > v : x < v) v = x;
      }
      if (axis === 0) out[i * W + l] = v; else out[l * W + i] = v;
    }
    return out;
  };
  return pass(pass(img, 0), 1);
}
