// node forest.test.mjs: (1) a forest trained by Python's RandomForestModel, evaluated in JS on the Python features,
// gives sklearn's labels and probabilities; (2) end to end (JS features + exported forest) = RandomForestModel.predict;
// (3) a JS-trained forest on the demo strokes agrees with the Python forest and the known classes on the crop;
// (4) the classification report matches hand-checked numbers.
import assert from 'node:assert/strict';
import { extractAll } from './features.js';
import { forestFromJson, predictForest, predictProba, trainForest } from './forest.js';
import { classificationReport, stratifiedSplit } from './metrics.js';
import { f64, fixture, relErr, u8 } from './fixture_util.mjs';

const fx = fixture('forest_case.json'), [H, W] = fx.shape, N = H * W;
const forest = forestFromJson(fx.forest), feats = f64(fx.features), F = forest.n_features;
const X = Array.from({ length: F }, (_, k) => feats.subarray(k * N, (k + 1) * N));
const want = u8(fx.pred), got = predictForest(forest, X, N);
assert.deepEqual(Array.from(got), Array.from(want));
const K = forest.classes.length, pp = predictProba(forest, X, N), wantP = f64(fx.proba);
const ppT = new Float64Array(N * K);  // Python: (K, H, W)
for (let i = 0; i < N; i++) for (let k = 0; k < K; k++) ppT[k * N + i] = pp[i * K + k];
assert.ok(relErr(ppT, wantP) < 1e-12, 'probabilities');
console.log(`ok exported Python forest (${forest.trees.length} trees) in JS: ${N} labels and probabilities identical`);

// end to end from the raw bands of the crop: rebuild them from the features fixture
const fc = fixture('features_case.json').cases.find(c => c.name === 'demo_crop');
const bands = fc.bands.map(f64), e2e = predictForest(forest, extractAll(bands, H, W).maps, N);
assert.deepEqual(Array.from(e2e), Array.from(want));
console.log('ok JS features + forest = RandomForestModel.predict on the crop');

// JS training on the Python features of the crop's own predictions (separable, 5 classes)
const y = Array.from(want), idx = stratifiedSplit(y, 0.2, 1);
const Xtr = X.map(f => Float32Array.from(idx.train, i => f[i])), ytr = idx.train.map(i => y[i]);
const js = trainForest(Xtr, ytr, { n_estimators: 30, max_depth: 10, seed: 3 });
const jsPred = predictForest(js, X, N);
let agree = 0;
for (let i = 0; i < N; i++) agree += jsPred[i] === want[i];
assert.ok(agree / N > 0.97, `JS forest agrees with the Python labels on ${(100 * agree / N).toFixed(1)}%`);
assert.equal(js.importances.length, F);
assert.ok(Math.abs(js.importances.reduce((a, b) => a + b, 0) - 1) < 1e-9);
console.log(`ok JS-trained forest (30 trees) reproduces ${(100 * agree / N).toFixed(1)}% of the Python labels; importances sum to 1`);

const rep = classificationReport([0, 0, 1, 1, 1, 2], [0, 1, 1, 1, 2, 2], ['a', 'b', 'c']);
assert.equal(rep.accuracy, 4 / 6);
assert.equal(rep.b.precision, 2 / 3); assert.equal(rep.b.recall, 2 / 3); assert.equal(rep.c.precision, 0.5);
assert.ok(Math.abs(rep['macro avg']['f1-score'] - (2 / 3 + 2 / 3 + 2 / 3) / 3) < 1e-12);
console.log('ok classification report');
