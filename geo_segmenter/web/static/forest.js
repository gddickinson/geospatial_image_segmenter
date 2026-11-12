// Random forest, the classifier behind RandomForestModel (models/random_forest.py: scikit-learn
// RandomForestClassifier, n_estimators=100, max_depth=10, max_features='sqrt', class_weight='balanced').
//  - predictProba / predictForest: exact port of sklearn's predict for an exported forest (features cast to float32,
//    left when x <= threshold, average the per-tree class fractions, first maximum wins).
//  - trainForest: the same algorithm in JS (bootstrap, balanced class weights, sqrt(n_features) candidates per split,
//    Gini, midpoint thresholds, max_depth, impurity-based importances). Random draws differ from sklearn's, so a
//    JS-trained forest is a different forest of the same kind (agreement is measured in forest.test.mjs).
// A forest is { classes, n_features, trees: [{ feature, threshold, left, right, value }], importances? } where value
// holds n_classes fractions per node and left[i] === -1 marks a leaf.
import { rng as makeRng } from './core.js';

export const RF_DEFAULTS = { n_estimators: 100, max_depth: 10 };  // config.RF_N_ESTIMATORS, config.RF_MAX_DEPTH

const f32 = X => X.map(f => (f instanceof Float32Array ? f : Float32Array.from(f)));

// Per-pixel class probabilities (N * K, pixel-major). X: n_features maps of length N.
export function predictProba(forest, X, N) {
  if (X.length !== forest.n_features) throw new Error(`forest expects ${forest.n_features} features, got ${X.length}`);
  const K = forest.classes.length, proba = new Float64Array(N * K), Xf = f32(X);
  for (const t of forest.trees) {
    const { feature, threshold, left, right, value } = t;
    for (let i = 0; i < N; i++) {
      let n = 0;
      while (left[n] !== -1) n = Xf[feature[n]][i] <= threshold[n] ? left[n] : right[n];
      const o = i * K, v = n * K;
      for (let k = 0; k < K; k++) proba[o + k] += value[v + k];
    }
  }
  const T = forest.trees.length;
  for (let i = 0; i < proba.length; i++) proba[i] /= T;
  return proba;
}

export function argmaxLabels(forest, proba, N) {
  const K = forest.classes.length, out = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    let best = 0;
    for (let k = 1; k < K; k++) if (proba[i * K + k] > proba[i * K + best]) best = k;
    out[i] = forest.classes[best];
  }
  return out;
}

export const predictForest = (forest, X, N) => argmaxLabels(forest, predictProba(forest, X, N), N);

// X: n_features arrays over the n training samples; y: class index per sample.
export function trainForest(X, y, { n_estimators = RF_DEFAULTS.n_estimators, max_depth = RF_DEFAULTS.max_depth, seed = 1,
  classWeight = 'balanced', onTree = () => {} } = {}) {
  const F = X.length, n = y.length, Xf = f32(X);
  const classes = [...new Set(y)].sort((a, b) => a - b), K = classes.length;
  const yi = Int32Array.from(y, v => classes.indexOf(v));
  const cw = new Float64Array(K).fill(1);
  if (classWeight === 'balanced') {  // n_samples / (n_classes * count), over the whole training set
    const cnt = new Float64Array(K);
    for (const v of yi) cnt[v]++;
    for (let k = 0; k < K; k++) cw[k] = n / (K * cnt[k]);
  }
  const mtry = Math.max(1, Math.floor(Math.sqrt(F))), rnd = makeRng(seed), trees = [];
  const imp = new Float64Array(F);
  for (let t = 0; t < n_estimators; t++) {
    const w = new Float64Array(n);
    for (let i = 0; i < n; i++) w[Math.floor(rnd() * n)]++;  // bootstrap counts as sample weights
    const idx = [];
    for (let i = 0; i < n; i++) if (w[i] > 0) { w[i] *= cw[yi[i]]; idx.push(i); }
    const tree = growTree(Xf, yi, w, idx, K, mtry, max_depth, rnd);
    let s = 0;
    for (const v of tree.imp) s += v;
    if (s > 0) for (let f = 0; f < F; f++) imp[f] += tree.imp[f] / s;
    delete tree.imp;
    trees.push(tree);
    onTree(t + 1, n_estimators);
  }
  let s = 0;
  for (const v of imp) s += v;
  const importances = Array.from(imp, v => (s > 0 ? v / s : 0));
  return { classes, n_features: F, trees, importances };
}

function growTree(X, y, w, idx, K, mtry, maxDepth, rnd) {
  const feature = [], threshold = [], left = [], right = [], value = [];
  const F = X.length, order = new Int32Array(F), imp = new Float64Array(F);
  let rootW = 0;
  for (const i of idx) rootW += w[i];
  const node = (ids, depth) => {
    const id = feature.length;
    feature.push(-2); threshold.push(-2); left.push(-1); right.push(-1);
    const counts = new Float64Array(K);
    let tot = 0;
    for (const i of ids) { counts[y[i]] += w[i]; tot += w[i]; }
    for (let k = 0; k < K; k++) value.push(counts[k] / tot);
    let gini = 1;
    for (let k = 0; k < K; k++) gini -= (counts[k] / tot) ** 2;
    if (depth >= maxDepth || ids.length < 2 || gini <= 1e-12) return id;
    for (let f = 0; f < F; f++) order[f] = f;
    let best = null, tried = 0;
    for (let f = 0; f < F && tried < mtry; f++) {  // draw candidate features until mtry non-constant ones were tried
      const j = f + Math.floor(rnd() * (F - f));
      const tmp = order[f]; order[f] = order[j]; order[j] = tmp;
      const col = X[order[f]];
      const sorted = Int32Array.from(ids).sort((a, b) => col[a] - col[b]);
      if (col[sorted[sorted.length - 1]] <= col[sorted[0]] + 1e-7) continue;
      tried++;
      const lc = new Float64Array(K);
      let lw = 0;
      for (let p = 0; p < sorted.length - 1; p++) {
        const i = sorted[p];
        lc[y[i]] += w[i]; lw += w[i];
        const a = col[i], b = col[sorted[p + 1]];
        if (b <= a + 1e-7) continue;
        const rw = tot - lw;
        let gl = 1, gr = 1;
        for (let k = 0; k < K; k++) { gl -= (lc[k] / lw) ** 2; gr -= ((counts[k] - lc[k]) / rw) ** 2; }
        const im = (lw * gl + rw * gr) / tot;
        if (!best || im < best.imp - 1e-12) best = { imp: im, f: order[f], thr: a / 2 + b / 2 };
      }
    }
    if (!best || best.imp >= gini - 1e-12) return id;
    const col = X[best.f], L = [], R = [];
    for (const i of ids) (col[i] <= best.thr ? L : R).push(i);
    if (!L.length || !R.length) return id;
    imp[best.f] += (tot / rootW) * (gini - best.imp);  // weighted impurity decrease (MDI)
    feature[id] = best.f; threshold[id] = best.thr;
    left[id] = node(L, depth + 1);
    right[id] = node(R, depth + 1);
    return id;
  };
  node(idx, 0);
  return { feature: Int32Array.from(feature), threshold: Float64Array.from(threshold), left: Int32Array.from(left),
    right: Int32Array.from(right), value: Float64Array.from(value), imp };
}

// Plain typed-array forest from the JSON exported by the Python side (web/classify_service.export_forest).
export function forestFromJson(j) {
  return { classes: j.classes, n_features: j.n_features, importances: j.importances ?? null, trees: j.trees.map(t => ({
    feature: Int32Array.from(t.feature), threshold: Float64Array.from(t.threshold), left: Int32Array.from(t.left),
    right: Int32Array.from(t.right), value: Float64Array.from(t.value) })) };
}
