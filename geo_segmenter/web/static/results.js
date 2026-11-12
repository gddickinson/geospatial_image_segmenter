// Results panel of the Classify page: class statistics (pixels, share, area, mean NDVI and band means), training
// metrics as RandomForestModel.train returns them, feature importances, agreement with the known classes of the demo,
// and exports (classification GeoTIFF, overlay PNG, statistics CSV, training-pixel GeoTIFF).
import { download, esc } from './core.js';
import { pixelArea, rerunDemoForest, S } from './state.js';
import { compose } from './viewer.js';
import { classStats, confusion } from './metrics.js';
import { writeGeoTiff } from './geotiff.js';

const pct = v => (v === null || v === undefined || Number.isNaN(v) ? '–' : `${(100 * v).toFixed(1)}%`);
const num = (v, d = 1) => (v === null || v === undefined ? '–' : (+v).toLocaleString(undefined, { maximumFractionDigits: d }));

function stats() {
  const K = S.classes.length, pa = pixelArea();
  return classStats(S.result.pred, K, S.image.bands, { pixelArea: pa, bandMap: S.bandMap });
}

export function renderResults(host) {
  const r = S.result;
  if (!r) { host.innerHTML = ''; return; }
  const st = stats(), pa = pixelArea(), img = S.image, m = r.metrics || {};
  const truth = img.truth;
  let acc = null, cm = null;
  if (truth) { let ok = 0; for (let i = 0; i < truth.length; i++) ok += truth[i] === r.pred[i]; acc = ok / truth.length; cm = confusion(truth, r.pred, S.classes.length); }
  const imp = Object.entries(m.feature_importances || {}).sort((a, b) => b[1] - a[1]), maxImp = imp[0]?.[1] || 1;
  const rep = m.classification_report || {};
  host.innerHTML = `
  <div class="panel">
    <h3>Classification</h3>
    <div class="tiles">
      <div class="tile"><b>${num(img.W * img.H, 0)}</b><span>pixels classified</span></div>
      <div class="tile"><b>${num(r.n_train, 0)}</b><span>painted training pixels</span></div>
      <div class="tile"><b>${pct(m.training_score)}</b><span>training accuracy</span></div>
      <div class="tile"><b>${pct(m.validation_score)}</b><span>validation accuracy (20% held out)</span></div>
      ${acc !== null ? `<div class="tile"><b>${pct(acc)}</b><span>agreement with the known classes (demo)</span></div>` : ''}
    </div>
    <div class="scroll"><table class="tbl"><thead><tr><th>Class</th><th>Pixels</th><th>Share</th>
      <th>${pa ? 'Area (ha)' : 'Area'}</th><th>Mean NDVI</th>${img.bandNames.map(b => `<th>Mean ${esc(b)}</th>`).join('')}</tr></thead><tbody>
      ${st.map(s => `<tr><td><i class="sw" style="background:${S.classes[s.k].color}"></i>${esc(S.classes[s.k].name)}</td><td>${num(s.pixels, 0)}</td>
        <td>${pct(s.share)}</td><td>${pa ? num(s.area_ha, 2) : '–'}</td><td>${s.ndvi === null ? '–' : s.ndvi.toFixed(3)}</td>
        ${s.means.map(v => `<td>${num(v, 1)}</td>`).join('')}</tr>`).join('')}
    </tbody></table></div>
    <p class="small mute">${pa ? `Area from the pixel size in the file (${num(pa, 3)} m² per pixel${img.factor > 1 ? ', after block averaging' : ''}).` : 'No metric pixel size in the file, so no areas.'}
      Mean NDVI uses the red and NIR band roles${S.bandMap.nir === undefined ? ' (no NIR band set)' : ''}.</p>
    <div class="exports">
      <button id="x-tif">Classification GeoTIFF</button><button id="x-png" class="alt">Overlay PNG</button>
      <button id="x-csv" class="alt">Class statistics CSV</button><button id="x-train" class="alt">Training pixels GeoTIFF</button>
      ${img.source === 'demo' ? '<button id="x-scene" class="alt">Demo scene GeoTIFF</button>' : ''}
    </div>
  </div>
  <div class="two">
    <div class="panel"><h3>Validation report</h3>
      ${Object.keys(rep).length ? `<div class="scroll"><table class="tbl"><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>
      ${Object.entries(rep).filter(([k]) => k !== 'accuracy').map(([k, v]) => `<tr><td>${esc(k)}</td><td>${v.precision.toFixed(3)}</td><td>${v.recall.toFixed(3)}</td><td>${v['f1-score'].toFixed(3)}</td><td>${num(v.support, 0)}</td></tr>`).join('')}
      </tbody></table></div><p class="small mute">On the held-out 20% of the painted pixels (sklearn classification_report).</p>` : '<p class="mute">No validation split.</p>'}
    </div>
    <div class="panel"><h3>Feature importances</h3>
      <div class="bars">${imp.map(([n, v]) => `<span>${esc(n)}</span><span class="bar" style="width:${(100 * v / maxImp).toFixed(1)}%"></span><span>${(100 * v).toFixed(1)}%</span>`).join('')}</div>
      <p class="small mute">Mean decrease in impurity over all trees.</p>
    </div>
  </div>
  ${cm ? `<div class="panel"><h3>Known class (rows) vs classification (columns), demo</h3><div class="scroll"><table class="tbl"><thead><tr><th></th>
    ${S.classes.map(c => `<th>${esc(c.name)}</th>`).join('')}</tr></thead><tbody>
    ${cm.map((row, i) => `<tr><td>${esc(S.classes[i]?.name ?? i)}</td>${row.map((v, j) => `<td class="${i === j ? 'ok' : v ? 'warnc' : 'mute'}">${num(v, 0)}</td>`).join('')}</tr>`).join('')}
    </tbody></table></div>
    ${r.engine === 'python' && S.demoForest ? '<button id="x-rerun" class="alt">Re-run the Python forest in this browser</button><span class="small mute" id="rerun"></span>' : ''}</div>` : ''}`;
  const tag = () => S.image.name.replace(/\.[^.]+$/, '').replace(/[^\w-]+/g, '_');
  host.querySelector('#x-tif').onclick = () => download(`${tag()}_classes.tif`, new Blob([writeGeoTiff([r.pred], img.W, img.H, { dtype: 'uint8', geo: img.geo })], { type: 'image/tiff' }));
  host.querySelector('#x-train').onclick = () => {
    const lab = new Uint8Array(img.W * img.H);  // 0 = not painted, k + 1 = class k
    S.classes.forEach((c, k) => { for (let i = 0; i < lab.length; i++) if (c.mask[i]) lab[i] = k + 1; });
    download(`${tag()}_training.tif`, new Blob([writeGeoTiff([lab], img.W, img.H, { dtype: 'uint8', geo: img.geo, nodata: 0 })], { type: 'image/tiff' }));
  };
  host.querySelector('#x-png').onclick = () => {
    const cv = document.createElement('canvas');
    cv.width = img.W; cv.height = img.H;
    cv.getContext('2d').putImageData(compose({ result: true }), 0, 0);
    cv.toBlob(b => download(`${tag()}_overlay.png`, b), 'image/png');
  };
  host.querySelector('#x-csv').onclick = () => {
    const head = ['class_index', 'class', 'pixels', 'share', 'area_ha', 'mean_ndvi', ...img.bandNames.map(b => `mean_${b}`)];
    const rows = st.map(s => [s.k, `"${S.classes[s.k].name.replace(/"/g, '""')}"`, s.pixels, s.share.toFixed(6), s.area_ha ?? '',
      s.ndvi ?? '', ...s.means.map(v => v ?? '')].join(','));
    download(`${tag()}_class_stats.csv`, [head.join(','), ...rows].join('\n') + '\n', 'text/csv');
  };
  const sc = host.querySelector('#x-scene');
  if (sc) sc.onclick = () => download('synthetic_demo_scene.tif', new Blob([writeGeoTiff(img.bands, img.W, img.H, { dtype: 'uint16', geo: img.geo })], { type: 'image/tiff' }));
  const rr = host.querySelector('#x-rerun');
  if (rr) rr.onclick = async () => {
    rr.disabled = true;
    const p = await rerunDemoForest();
    let same = 0;
    for (let i = 0; i < p.length; i++) same += p[i] === r.pred[i];
    host.querySelector('#rerun').textContent = ` JS features + the exported trees give the Python label on ${same.toLocaleString()} of ${p.length.toLocaleString()} pixels (${pct(same / p.length)}).`;
    rr.disabled = false;
  };
}
