// Training metrics as RandomForestModel.train reports them: a stratified 80/20 split of the painted pixels
// (train_test_split(test_size=0.2, stratify=y); the JS split draws differ from numpy's), training and validation
// accuracy, sklearn's classification_report (precision, recall, F1, support per class, macro and weighted averages)
// and a confusion matrix; plus per-class area statistics of a classified image.
import { rng as makeRng } from './core.js';

// Indices of a stratified split: about `testSize` of each class goes to validation (at least one when possible).
export function stratifiedSplit(y, testSize = 0.2, seed = 42) {
  const rnd = makeRng(seed), byClass = new Map();
  y.forEach((c, i) => { if (!byClass.has(c)) byClass.set(c, []); byClass.get(c).push(i); });
  const train = [], test = [];
  for (const ids of byClass.values()) {
    for (let i = ids.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [ids[i], ids[j]] = [ids[j], ids[i]]; }
    const nt = ids.length > 1 ? Math.max(1, Math.round(ids.length * testSize)) : 0;
    test.push(...ids.slice(0, nt)); train.push(...ids.slice(nt));
  }
  return { train: train.sort((a, b) => a - b), test: test.sort((a, b) => a - b) };
}

export function confusion(yTrue, yPred, K) {
  const m = Array.from({ length: K }, () => new Array(K).fill(0));
  for (let i = 0; i < yTrue.length; i++) m[yTrue[i]][yPred[i]]++;
  return m;
}

export const accuracy = (yTrue, yPred) => { let ok = 0; for (let i = 0; i < yTrue.length; i++) ok += yTrue[i] === yPred[i]; return yTrue.length ? ok / yTrue.length : NaN; };

// sklearn.metrics.classification_report(output_dict=True) for classes 0..K-1 with the given names.
export function classificationReport(yTrue, yPred, names) {
  const K = names.length, cm = confusion(yTrue, yPred, K), rep = {};
  let macro = [0, 0, 0], weighted = [0, 0, 0], total = 0, present = 0;
  for (let k = 0; k < K; k++) {
    const tp = cm[k][k], support = cm[k].reduce((a, b) => a + b, 0), predicted = cm.reduce((a, r) => a + r[k], 0);
    if (!support && !predicted) continue;
    const precision = predicted ? tp / predicted : 0, recall = support ? tp / support : 0;
    const f1 = precision + recall ? 2 * precision * recall / (precision + recall) : 0;
    rep[names[k]] = { precision, recall, 'f1-score': f1, support };
    present++; total += support;
    macro = macro.map((v, i) => v + [precision, recall, f1][i]);
    weighted = weighted.map((v, i) => v + [precision, recall, f1][i] * support);
  }
  rep.accuracy = accuracy(yTrue, yPred);
  const avg = (v, d) => ({ precision: v[0] / d, recall: v[1] / d, 'f1-score': v[2] / d, support: total });
  rep['macro avg'] = avg(macro, present);
  rep['weighted avg'] = avg(weighted, total);
  return rep;
}

// Pixel count, share, area (when the pixel size is known in metres) and mean band values / NDVI per class.
export function classStats(pred, K, bands, { pixelArea = null, bandMap = null } = {}) {
  const N = pred.length, count = new Array(K).fill(0), sums = Array.from({ length: K }, () => new Array(bands.length).fill(0));
  const ndvi = new Array(K).fill(0), hasNdvi = bandMap && bandMap.nir !== undefined && bandMap.red !== undefined;
  for (let i = 0; i < N; i++) {
    const c = pred[i];
    count[c]++;
    for (let b = 0; b < bands.length; b++) sums[c][b] += +bands[b][i];
    if (hasNdvi) { const n = +bands[bandMap.nir][i], r = +bands[bandMap.red][i]; ndvi[c] += (n - r) / (n + r + 1e-8); }
  }
  return count.map((n, k) => ({
    k, pixels: n, share: n / N, area_ha: pixelArea ? n * pixelArea / 1e4 : null,
    means: sums[k].map(s => (n ? s / n : null)), ndvi: hasNdvi && n ? ndvi[k] / n : null,
  }));
}
