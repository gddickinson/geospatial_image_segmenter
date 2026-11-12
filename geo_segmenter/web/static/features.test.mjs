// node features.test.mjs: the JS feature pipeline reproduces models/pipeline.py (spectral + texture extractors,
// Python) on the fixture images (tests/fixtures/features_case.json, written by scripts/make_web_fixtures.py).
import assert from 'node:assert/strict';
import { defaultBandMap, extractAll } from './features.js';
import { f64, fixture, relErr } from './fixture_util.mjs';

const fx = fixture('features_case.json');
for (const c of fx.cases) {
  const [H, W] = c.shape, N = H * W, bands = c.bands.map(f64), want = f64(c.features);
  const { maps, names } = extractAll(bands, H, W, { bandMap: c.band_map || defaultBandMap(bands.length) });
  assert.deepEqual(names, c.names, `${c.name}: names`);
  let worst = 0;
  maps.forEach((m, k) => {
    const rel = relErr(m, want.subarray(k * N, (k + 1) * N));
    worst = Math.max(worst, rel);
    assert.ok(rel < 1e-9, `${c.name}: ${names[k]} relative error ${rel}`);
  });
  console.log(`ok ${c.name} ${H}x${W}: ${names.length} features match Python, worst relative error ${worst.toExponential(1)}`);
}
