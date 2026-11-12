// Small pure-JS GeoTIFF reader and writer (the desktop app reads rasters with rasterio/GDAL).
// readGeoTiff(arrayBuffer) -> { width, height, bands: [TypedArray], dtype, geo, nodata }
//   First image (IFD) only: overviews and masks are ignored. Classic TIFF (not BigTIFF), little/big-endian, strips or
//   tiles, chunky or planar samples, 8/16/32-bit unsigned/signed integers and 32/64-bit floats, compression none /
//   LZW / Deflate / PackBits, horizontal predictor. geo = { pixelScale, tiepoint, keys: {id: value}, epsg, units,
//   crsName } from the GeoTIFF tags when present (null otherwise).
// writeGeoTiff(bands, width, height, { dtype: 'uint8'|'uint16'|'float32', geo, nodata }) -> Uint8Array: uncompressed,
//   little-endian, chunky, copying ModelPixelScale / ModelTiepoint / GeoKeyDirectory / GeoDouble / GeoAscii tags.
// Tested against files written by tifffile in geotiff.test.mjs (and the writer is read back by tifffile in pytest).
const SIZE = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 16: 8 };
const T = { WIDTH: 256, HEIGHT: 257, BITS: 258, COMP: 259, PHOTO: 262, STRIP_OFF: 273, SPP: 277, RPS: 278,
  STRIP_CNT: 279, PLANAR: 284, PRED: 317, TILE_W: 322, TILE_H: 323, TILE_OFF: 324, TILE_CNT: 325, FMT: 339,
  SCALE: 33550, TIE: 33922, KEYS: 34735, DOUBLES: 34736, ASCII: 34737, NODATA: 42113 };
const CTOR = { uint8: Uint8Array, int8: Int8Array, uint16: Uint16Array, int16: Int16Array, uint32: Uint32Array,
  int32: Int32Array, float32: Float32Array, float64: Float64Array };
const GET = { uint8: 'getUint8', int8: 'getInt8', uint16: 'getUint16', int16: 'getInt16', uint32: 'getUint32',
  int32: 'getInt32', float32: 'getFloat32', float64: 'getFloat64' };

function readTags(dv, off, le) {
  const n = dv.getUint16(off, le), tags = {};
  for (let i = 0; i < n; i++) {
    const e = off + 2 + 12 * i, tag = dv.getUint16(e, le), type = dv.getUint16(e + 2, le), count = dv.getUint32(e + 4, le);
    const sz = SIZE[type] || 1, at = sz * count <= 4 ? e + 8 : dv.getUint32(e + 8, le);
    if (type === 2) {
      let s = '';
      for (let k = 0; k < count; k++) s += String.fromCharCode(dv.getUint8(at + k));
      tags[tag] = s.replace(/\0+$/, '');
      continue;
    }
    const v = [];
    for (let k = 0; k < count; k++) {
      const p = at + k * sz;
      v.push(type === 3 ? dv.getUint16(p, le) : type === 8 ? dv.getInt16(p, le) : type === 4 || type === 13 ? dv.getUint32(p, le)
        : type === 9 ? dv.getInt32(p, le) : type === 5 ? dv.getUint32(p, le) / dv.getUint32(p + 4, le)
          : type === 10 ? dv.getInt32(p, le) / dv.getInt32(p + 4, le) : type === 11 ? dv.getFloat32(p, le)
            : type === 12 ? dv.getFloat64(p, le) : type === 6 ? dv.getInt8(p) : dv.getUint8(p));
    }
    tags[tag] = v;
  }
  return tags;
}

function dtypeOf(bits, fmt) {
  const name = fmt === 3 ? `float${bits}` : `${fmt === 2 ? 'int' : 'uint'}${bits}`;
  if (!CTOR[name]) throw new Error(`Unsupported sample type (${bits}-bit, format ${fmt})`);
  return name;
}

export async function readGeoTiff(buffer) {
  const dv = new DataView(buffer), u8 = new Uint8Array(buffer);
  const bo = String.fromCharCode(u8[0], u8[1]);
  if (bo !== 'II' && bo !== 'MM') throw new Error('Not a TIFF file');
  const le = bo === 'II', magic = dv.getUint16(2, le);
  if (magic === 43) throw new Error('BigTIFF is not supported in the browser; use the desktop app');
  if (magic !== 42) throw new Error('Not a TIFF file (bad magic number)');
  const t = readTags(dv, dv.getUint32(4, le), le);
  const W = t[T.WIDTH][0], H = t[T.HEIGHT][0], spp = (t[T.SPP] || [1])[0], planar = (t[T.PLANAR] || [1])[0];
  const bits = t[T.BITS][0], fmt = (t[T.FMT] || [1])[0], comp = (t[T.COMP] || [1])[0], pred = (t[T.PRED] || [1])[0];
  if (t[T.BITS].some(b => b !== bits)) throw new Error('Bands with different bit depths are not supported');
  if (pred === 3) throw new Error('Floating-point predictor is not supported in the browser; use the desktop app');
  const dtype = dtypeOf(bits, fmt), bps = bits / 8;
  const tiled = !!t[T.TILE_OFF];
  const bw = tiled ? t[T.TILE_W][0] : W, bh = tiled ? t[T.TILE_H][0] : (t[T.RPS] || [H])[0];
  const offs = tiled ? t[T.TILE_OFF] : t[T.STRIP_OFF], cnts = tiled ? t[T.TILE_CNT] : t[T.STRIP_CNT];
  const across = Math.ceil(W / bw), down = Math.ceil(H / bh), perPlane = across * down;
  const chunkSpp = planar === 2 ? 1 : spp, pxBytes = chunkSpp * bps;
  const bands = Array.from({ length: spp }, () => new CTOR[dtype](W * H)), get = GET[dtype];
  for (let b = 0; b < offs.length; b++) {
    const plane = planar === 2 ? Math.floor(b / perPlane) : 0, bi = b % perPlane;
    const bx = (bi % across) * bw, by = Math.floor(bi / across) * bh;
    const rows = tiled ? bh : Math.min(bh, H - by), rowBytes = bw * pxBytes;
    let data = await decompress(u8.subarray(offs[b], offs[b] + cnts[b]), comp, rows * rowBytes);
    if (pred === 2) data = undoPredictor(data, bw, chunkSpp, bps, le, rows);
    const ddv = new DataView(data.buffer, data.byteOffset, data.byteLength);
    for (let r = 0; r < rows && by + r < H; r++) for (let c = 0; c < bw && bx + c < W; c++) {
      const o = r * rowBytes + c * pxBytes, px = (by + r) * W + bx + c;
      if (o + pxBytes > data.length) continue;
      for (let s = 0; s < chunkSpp; s++) bands[plane + s][px] = ddv[get](o + s * bps, le);
    }
  }
  return { width: W, height: H, bands, dtype, geo: geoInfo(t), nodata: t[T.NODATA] !== undefined ? +t[T.NODATA] : null };
}

function geoInfo(t) {
  if (!t[T.SCALE] && !t[T.TIE] && !t[T.KEYS]) return null;
  const keys = {}, k = t[T.KEYS] || [];
  for (let i = 4; i + 3 < k.length; i += 4) {
    const [id, loc, count, off] = k.slice(i, i + 4);
    keys[id] = loc === 0 ? off : loc === T.DOUBLES ? (t[T.DOUBLES] || []).slice(off, off + count)
      : loc === T.ASCII ? (t[T.ASCII] || '').slice(off, off + count).replace(/\|$/, '') : null;
  }
  const epsg = keys[3072] && keys[3072] !== 32767 ? keys[3072] : keys[2048] && keys[2048] !== 32767 ? keys[2048] : null;
  const geographic = keys[1024] === 2;
  const lin = keys[3076] ?? (keys[1024] === 1 ? 9001 : null);
  const units = geographic ? 'degree' : lin === 9001 ? 'metre' : lin === 9002 ? 'foot' : lin === 9003 ? 'US survey foot' : null;
  const unitsAssumed = !geographic && keys[3076] === undefined && units !== null;  // projected CRS without a unit key
  return { pixelScale: t[T.SCALE] || null, tiepoint: t[T.TIE] || null, keys, epsg, units, unitsAssumed, geographic,
    crsName: keys[3073] || keys[2049] || keys[1026] || null, raw: { keys: t[T.KEYS] || null, doubles: t[T.DOUBLES] || null, ascii: t[T.ASCII] ?? null } };
}

async function decompress(src, comp, want) {
  if (comp === 1) return src;
  if (comp === 32773) return packbits(src, want);
  if (comp === 5) return lzw(src, want);
  if (comp === 8 || comp === 32946) {
    const ds = new Blob([src]).stream().pipeThrough(new DecompressionStream('deflate'));
    return new Uint8Array(await new Response(ds).arrayBuffer());
  }
  throw new Error(`Unsupported TIFF compression (${comp}); save as uncompressed, LZW or Deflate`);
}

function packbits(src, want) {
  const out = new Uint8Array(want);
  let i = 0, o = 0;
  while (i < src.length && o < want) {
    const n = (src[i++] << 24) >> 24;
    if (n >= 0) { for (let k = 0; k <= n && o < want; k++) out[o++] = src[i++]; }
    else if (n !== -128) { const v = src[i++]; for (let k = 0; k < 1 - n && o < want; k++) out[o++] = v; }
  }
  return out;
}

function lzw(src, want) {
  const out = new Uint8Array(want);
  let o = 0, bit = 0, width = 9, prev = null, dict = [];
  const reset = () => { dict = []; for (let i = 0; i < 256; i++) dict.push([i]); dict.push(null, null); width = 9; };
  const read = () => {
    let v = 0;
    for (let k = 0; k < width; k++) {
      const byte = src[(bit + k) >> 3];
      if (byte === undefined) return 257;
      v = (v << 1) | ((byte >> (7 - ((bit + k) & 7))) & 1);
    }
    bit += width;
    return v;
  };
  reset();
  for (;;) {
    const code = read();
    if (code === 257) break;
    if (code === 256) { reset(); prev = null; continue; }
    let seq;
    if (code < dict.length && dict[code]) seq = dict[code];
    else if (prev) seq = prev.concat(prev[0]);
    else throw new Error('Corrupt LZW data');
    for (const b of seq) if (o < want) out[o++] = b;
    if (prev) dict.push(prev.concat(seq[0]));
    prev = seq;
    if (dict.length + 1 >= 1 << width && width < 12) width++;  // TIFF "early change"
  }
  return out;
}

function undoPredictor(data, w, spp, bps, le, rows) {
  const out = new Uint8Array(data), dv = new DataView(out.buffer), rowBytes = w * spp * bps;
  const get = bps === 1 ? 'getUint8' : bps === 2 ? 'getUint16' : 'getUint32', set = get.replace('get', 'set');
  const mod = bps === 4 ? 2 ** 32 : 2 ** (8 * bps);
  for (let r = 0; r < rows && (r + 1) * rowBytes <= out.length; r++) {
    for (let i = spp; i < w * spp; i++) {
      const p = r * rowBytes + i * bps, q = p - spp * bps;
      dv[set](p, (dv[get](p, le) + dv[get](q, le)) % mod, le);
    }
  }
  return out;
}

export function writeGeoTiff(bands, width, height, { dtype = 'uint8', geo = null, nodata = null } = {}) {
  const spp = bands.length, bps = { uint8: 1, uint16: 2, float32: 4 }[dtype];
  if (!bps) throw new Error(`cannot write ${dtype}`);
  const imgBytes = width * height * spp * bps;
  const extras = [];  // [tag, type, count, Uint8Array payload]
  const f64b = a => { const b = new Uint8Array(8 * a.length), d = new DataView(b.buffer); a.forEach((v, i) => d.setFloat64(8 * i, v, true)); return b; };
  const u16b = a => { const b = new Uint8Array(2 * a.length), d = new DataView(b.buffer); a.forEach((v, i) => d.setUint16(2 * i, v, true)); return b; };
  const ascii = s => new TextEncoder().encode(s + '\0');
  if (spp > 1) extras.push([T.BITS, 3, spp, u16b(new Array(spp).fill(8 * bps))], [T.FMT, 3, spp, u16b(new Array(spp).fill(dtype === 'float32' ? 3 : 1))]);
  if (geo?.pixelScale) extras.push([T.SCALE, 12, geo.pixelScale.length, f64b(geo.pixelScale)]);
  if (geo?.tiepoint) extras.push([T.TIE, 12, geo.tiepoint.length, f64b(geo.tiepoint)]);
  if (geo?.raw?.keys) extras.push([T.KEYS, 3, geo.raw.keys.length, u16b(geo.raw.keys)]);
  if (geo?.raw?.doubles) extras.push([T.DOUBLES, 12, geo.raw.doubles.length, f64b(geo.raw.doubles)]);
  if (geo?.raw?.ascii) { const a = ascii(geo.raw.ascii); extras.push([T.ASCII, 2, a.length, a]); }
  if (nodata !== null) { const a = ascii(String(nodata)); extras.push([T.NODATA, 2, a.length, a]); }
  const fmt = dtype === 'float32' ? 3 : 1;
  const base = [[T.WIDTH, 4, 1, width], [T.HEIGHT, 4, 1, height], [T.COMP, 3, 1, 1], [T.PHOTO, 3, 1, 1],
    [T.STRIP_OFF, 4, 1, 0], [T.SPP, 3, 1, spp], [T.RPS, 4, 1, height], [T.STRIP_CNT, 4, 1, imgBytes], [T.PLANAR, 3, 1, 1]];
  if (spp === 1) base.push([T.BITS, 3, 1, 8 * bps], [T.FMT, 3, 1, fmt]);
  if (spp > 1) base.push([338, 3, spp - 1, null]);  // ExtraSamples: unspecified
  const entries = [...base.map(([tag, type, count, v]) => ({ tag, type, count, v })),
    ...extras.map(([tag, type, count, payload]) => ({ tag, type, count, payload }))].sort((a, b) => a.tag - b.tag);
  for (const e of entries) if (e.tag === 338) e.payload = u16b(new Array(spp - 1).fill(0));
  const ifdSize = 2 + 12 * entries.length + 4;
  let pos = 8 + ifdSize;
  for (const e of entries) if (e.payload && e.payload.length > 4) { e.at = pos; pos += e.payload.length + (e.payload.length & 1); }
  const dataAt = pos, buf = new ArrayBuffer(dataAt + imgBytes), dv = new DataView(buf), u8 = new Uint8Array(buf);
  u8[0] = 73; u8[1] = 73; dv.setUint16(2, 42, true); dv.setUint32(4, 8, true);
  dv.setUint16(8, entries.length, true);
  entries.forEach((e, i) => {
    const p = 10 + 12 * i;
    dv.setUint16(p, e.tag, true); dv.setUint16(p + 2, e.type, true); dv.setUint32(p + 4, e.count, true);
    if (e.tag === T.STRIP_OFF) dv.setUint32(p + 8, dataAt, true);
    else if (e.payload) { if (e.payload.length > 4) { dv.setUint32(p + 8, e.at, true); u8.set(e.payload, e.at); } else u8.set(e.payload, p + 8); }
    else if (e.type === 3) dv.setUint16(p + 8, e.v, true);
    else dv.setUint32(p + 8, e.v, true);
  });
  dv.setUint32(10 + 12 * entries.length, 0, true);
  const set = { uint8: 'setUint8', uint16: 'setUint16', float32: 'setFloat32' }[dtype];
  for (let i = 0; i < width * height; i++) for (let s = 0; s < spp; s++) dv[set](dataAt + (i * spp + s) * bps, bands[s][i], true);
  return u8;
}
