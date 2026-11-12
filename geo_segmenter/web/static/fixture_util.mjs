// Shared helpers for the node tests: load a fixture from tests/fixtures/ and decode base64 arrays.
import { readFileSync } from 'node:fs';

export const fixture = name => JSON.parse(readFileSync(new URL(`../../../tests/fixtures/${name}`, import.meta.url)));
const bytes = s => { const b = Buffer.from(s, 'base64'); return b.buffer.slice(b.byteOffset, b.byteOffset + b.length); };
export const f64 = s => new Float64Array(bytes(s));
export const u8 = s => new Uint8Array(bytes(s));

// Max error of `got` against `want`, relative to the largest |want| (at least 1e-12).
export function relErr(got, want) {
  let err = 0, scale = 1e-12;
  for (let i = 0; i < want.length; i++) { err = Math.max(err, Math.abs(got[i] - want[i])); scale = Math.max(scale, Math.abs(want[i])); }
  return err / scale;
}
