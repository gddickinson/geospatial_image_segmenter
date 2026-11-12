// node geotiff.test.mjs: readGeoTiff reads GeoTIFFs written by tifffile (tests/fixtures/tiff_case.json) exactly,
// with their georeferencing; writeGeoTiff output reads back identically (pytest also reads it with tifffile).
import assert from 'node:assert/strict';
import { writeFileSync } from 'node:fs';
import { readGeoTiff, writeGeoTiff } from './geotiff.js';
import { f64, fixture } from './fixture_util.mjs';

const fx = fixture('tiff_case.json');
for (const f of fx.files) {
  const b = Buffer.from(f.tiff, 'base64'), [nb, H, W] = f.shape;
  const t = await readGeoTiff(b.buffer.slice(b.byteOffset, b.byteOffset + b.length));
  assert.equal(t.dtype, f.dtype, f.name); assert.equal(t.width, W); assert.equal(t.height, H); assert.equal(t.bands.length, nb);
  const want = f64(f.bands);
  t.bands.forEach((band, k) => { for (let i = 0; i < H * W; i++) assert.equal(band[i], want[k * H * W + i], `${f.name} band ${k} px ${i}`); });
  assert.deepEqual(t.geo.pixelScale, fx.pixel_scale); assert.deepEqual(t.geo.tiepoint, fx.tiepoint);
  assert.equal(t.geo.epsg, fx.epsg); assert.equal(t.geo.units, 'metre'); assert.equal(t.nodata, 0);
  console.log(`ok ${f.name}: ${nb} band(s) ${W}x${H} ${f.dtype}, EPSG:${t.geo.epsg}, 10 m pixels`);
}

// writer round trip: class map with the georeferencing of the first file
const src = Buffer.from(fx.files[0].tiff, 'base64'), g = (await readGeoTiff(src.buffer.slice(src.byteOffset, src.byteOffset + src.length))).geo;
const cls = Uint8Array.from({ length: 6 * 7 }, (_, i) => i % 5);
const out = writeGeoTiff([cls], 7, 6, { dtype: 'uint8', geo: g });
const back = await readGeoTiff(out.buffer);
assert.deepEqual(Array.from(back.bands[0]), Array.from(cls));
assert.equal(back.geo.epsg, 32633); assert.deepEqual(back.geo.pixelScale, fx.pixel_scale);
const multi = writeGeoTiff([Uint16Array.of(1, 2, 3, 4), Uint16Array.of(5, 6, 7, 8)], 2, 2, { dtype: 'uint16', geo: g });
const mb = await readGeoTiff(multi.buffer);
assert.deepEqual(mb.bands.map(b => Array.from(b)), [[1, 2, 3, 4], [5, 6, 7, 8]]);
if (process.env.GEOTIFF_OUT) { writeFileSync(process.env.GEOTIFF_OUT, out); writeFileSync(process.env.GEOTIFF_OUT + '.multi.tif', multi); }
console.log('ok writeGeoTiff: class map and 2-band uint16 read back with their georeferencing');
