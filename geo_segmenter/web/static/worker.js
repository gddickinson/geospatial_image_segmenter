// Web Worker: feature extraction, forest training and prediction off the main thread (the desktop app trains in the
// GUI thread; the browser must not freeze). Messages { id, type, ... } -> { id, ok, result } / { id, error } /
// { id, progress }.
//   load    { bands, H, W, bandMap }        compute the feature maps (features.js) -> { names }
//   train   { masks: [Uint8Array per class], n_estimators, max_depth, seed }
//           -> { pred, metrics, importances, n_train, trees, nodes } (classes are the page's class indices)
//   forest  { forest }                      use a forest trained by Python (exported JSON) -> { pred }
//   feature { k }                           -> Float64Array feature map k
import { extractAll } from './features.js';
import { forestFromJson, predictForest, trainForest } from './forest.js';
import { accuracy, classificationReport, stratifiedSplit } from './metrics.js';

let H = 0, W = 0, maps = [], names = [];

const handlers = {
  load({ bands, H: h, W: w, bandMap }, report) {
    H = h; W = w;
    let step = 0;
    ({ maps, names } = extractAll(bands, H, W, { bandMap, onStep: s => report({ stage: `features: ${s}`, done: ++step, total: 10 }) }));
    return { names };
  },
  train({ masks, classNames, n_estimators, max_depth, seed }, report) {
    const N = H * W, painted = masks.map((m, k) => (m.some(v => v) ? k : -1)).filter(k => k >= 0);
    if (painted.length < 2) throw new Error('Paint training pixels for at least two classes');
    const pix = [], y = [];
    for (const k of painted) for (let i = 0; i < N; i++) if (masks[k][i]) { pix.push(i); y.push(k); }
    // RandomForestModel.train: stratified 80/20 split of the painted pixels (random_state=42 in Python)
    const { train, test } = stratifiedSplit(y, 0.2, 42);
    const col = ids => maps.map(m => Float32Array.from(ids, j => m[pix[j]]));
    const forest = trainForest(col(train), train.map(j => y[j]), { n_estimators, max_depth, seed,
      onTree: (t, n) => report({ stage: 'training trees', done: t, total: n }) });
    report({ stage: 'classifying the image', done: 0, total: 1 });
    const trPred = predictForest(forest, col(train), train.length), yTr = train.map(j => y[j]);
    const vaPred = predictForest(forest, col(test), test.length), yVa = test.map(j => y[j]);
    const pred = predictForest(forest, maps, N);
    const metrics = { training_score: accuracy(yTr, Array.from(trPred)), validation_score: accuracy(yVa, Array.from(vaPred)),
      classification_report: reportByIndex(yVa, Array.from(vaPred), classNames),
      feature_importances: Object.fromEntries(names.map((n, k) => [n, forest.importances[k]])) };
    return { pred, metrics, n_train: y.length, n_fit: train.length, n_val: test.length, trees: forest.trees.length,
      nodes: forest.trees.reduce((s, t) => s + t.left.length, 0) };
  },
  forest({ forest }) { return { pred: predictForest(forestFromJson(forest), maps, H * W) }; },
  feature({ k }) { return maps[k]; },
};

function reportByIndex(yTrue, yPred, classNames) {
  return classificationReport(yTrue, yPred, classNames);
}

self.onmessage = e => {
  const { id, type, ...args } = e.data;
  try {
    const result = handlers[type](args, progress => self.postMessage({ id, progress }));
    self.postMessage({ id, ok: true, result });
  } catch (err) {
    self.postMessage({ id, error: err.message || String(err) });
  }
};
