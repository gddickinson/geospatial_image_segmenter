// Static-site data layer: answers api() GETs from the JSON exported by scripts/build_static.py. Training,
// classification and GeoTIFF reading run in the browser (state.js + worker.js), so the POST route is never needed.
const files = new Map();
async function file(rel) {
  if (!files.has(rel)) files.set(rel, fetch('api/' + rel + '.json').then(r => { if (!r.ok) throw new Error(`${r.status} ${rel}`); return r.json(); }));
  return files.get(rel);
}

export const NEEDS_SERVER = 'needs the full app running locally (Python server): see the README, "Run the web app"';

export async function staticApi(path, body) {
  const clean = path.split('?')[0].replace(/^\/+/, '');
  if (body) throw new Error(`The Python engine (${clean}) ${NEEDS_SERVER}.`);
  return file(clean);
}
