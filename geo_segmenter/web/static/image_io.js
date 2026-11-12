// Opening user images in the browser: GeoTIFF / TIFF (geotiff.js) or PNG / JPEG / WebP (decoded by the browser, RGB
// bands, no georeferencing). Images above MAX_PIXELS are reduced by block averaging (factor shown on the page) so the
// feature maps fit in memory; the local app's Python engine then receives the same reduced image.
import { readGeoTiff } from './geotiff.js';

export const MAX_PIXELS = 1_500_000;

export async function openImage(file) {
  const isTiff = /\.tiff?$/i.test(file.name) || file.type === 'image/tiff';
  const img = isTiff ? await readGeoTiff(await file.arrayBuffer()) : await decodeBrowserImage(file);
  return reduce(img);
}

async function decodeBrowserImage(file) {
  const bmp = await createImageBitmap(file);
  const cv = new OffscreenCanvas(bmp.width, bmp.height), g = cv.getContext('2d');
  g.drawImage(bmp, 0, 0);
  const d = g.getImageData(0, 0, bmp.width, bmp.height).data, N = bmp.width * bmp.height;
  const bands = [0, 1, 2].map(c => { const b = new Uint8Array(N); for (let i = 0; i < N; i++) b[i] = d[4 * i + c]; return b; });
  return { width: bmp.width, height: bmp.height, bands, dtype: 'uint8', geo: null, nodata: null };
}

// Block-average by an integer factor when the image is too large.
export function reduce(img, maxPixels = MAX_PIXELS) {
  const k = Math.ceil(Math.sqrt((img.width * img.height) / maxPixels));
  if (k <= 1) return { ...img, factor: 1 };
  const W = Math.floor(img.width / k), H = Math.floor(img.height / k);
  const bands = img.bands.map(b => {
    const out = new Float32Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let s = 0;
      for (let dy = 0; dy < k; dy++) for (let dx = 0; dx < k; dx++) s += b[(y * k + dy) * img.width + x * k + dx];
      out[y * W + x] = s / (k * k);
    }
    return out;
  });
  const geo = img.geo && img.geo.pixelScale ? { ...img.geo, pixelScale: img.geo.pixelScale.map((v, i) => (i < 2 ? v * k : v)) } : img.geo;
  return { ...img, width: W, height: H, bands, dtype: 'float32', geo, factor: k };
}
