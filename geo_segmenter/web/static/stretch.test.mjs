// node stretch.test.mjs: prepareDisplay reproduces utils/display_stretch.prepare_display for every stretch type.
import assert from 'node:assert/strict';
import { percentile, prepareDisplay } from './stretch.js';
import { f64, fixture, u8 } from './fixture_util.mjs';

const fx = fixture('stretch_case.json');
const [, H, W] = fx.shape, N = H * W, bands = fx.bands.map(f64);
for (const c of fx.cases) {
  const got = prepareDisplay(bands, N, c.band_indices, c.params), want = u8(c.rgb);
  let diff = 0;
  for (let i = 0; i < want.length; i++) diff += got[i] !== want[i];
  assert.equal(diff, 0, `${c.params?.type ?? 'default'}: ${diff} values differ`);
  console.log(`ok stretch ${c.params?.type ?? 'default (min-max)'} bands ${c.band_indices}: identical to Python`);
}
assert.equal(percentile([1, 2, 3, 4], 50), 2.5);
assert.equal(percentile([10, 0, 5], 98), 9.8);
console.log('ok percentile');
