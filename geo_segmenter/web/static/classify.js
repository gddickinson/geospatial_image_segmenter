// Classify page: the desktop workflow in the browser. Open a GeoTIFF / image (or the synthetic demo), set the band
// roles and display, add classes, paint training pixels, train the random forest and classify the whole image.
import { $, $$, app, esc, isStatic, onLeave, route } from './core.js';
import { classify, getInfo, loadDemo, loadFile, painted, PALETTE, S, setBandMap } from './state.js';
import { attachBrush, compose, countPainted, undo } from './viewer.js';
import { renderResults } from './results.js';
import { spectralNames } from './features.js';

let mode = 'paint', zoom = 0;
const show = { result: true, strokes: true, errors: false };
const ROLES = ['red', 'green', 'blue', 'nir'];

route('classify', async () => {
  const info = await getInfo(), d = info.defaults;
  app().innerHTML = `
  <h1>Classify an image</h1>
  <p class="lead">Paint a few pixels of each land-cover class, train the random forest on spectral indices and texture,
    and classify every pixel, as the desktop app's Train / Run Segmentation steps do.</p>
  <div class="bench">
    <aside>
      <div class="panel"><h3>Image</h3>
        <button id="demo" class="alt">Load the synthetic demo</button>
        <label for="file">Or open a GeoTIFF, TIFF, PNG or JPEG</label>
        <input type="file" id="file" accept=".tif,.tiff,image/tiff,image/png,image/jpeg,image/webp">
        <p class="small mute" id="imginfo"></p>
      </div>
      <div class="panel"><h3>Bands</h3>
        <div class="roles" id="roles"></div>
        <p class="small mute" id="featinfo"></p>
      </div>
      <div class="panel"><h3>Classes</h3>
        <ul class="labels" id="labels"></ul>
        <button id="addcls" class="alt">Add class</button>
      </div>
      <div class="panel"><h3>Brush</h3>
        <div class="seg" role="group" aria-label="Tool" id="modes">
          <button data-m="view">Navigate</button><button data-m="paint">Paint</button><button data-m="erase">Erase</button></div>
        <div class="ctl"><label for="brush">Brush size <b id="brushv"></b> px</label>
          <input type="range" id="brush" min="1" max="30" step="1"></div>
        <button id="undo" class="alt">Undo</button><button id="clear" class="alt">Clear all</button>
      </div>
      <div class="panel"><h3>Random forest</h3>
        <label for="engine">Engine</label>
        <select id="engine">
          ${isStatic() ? '' : '<option value="python">Python (RandomForestModel, local server)</option>'}
          <option value="js">Browser (JS port)</option>
        </select>
        ${isStatic() ? '<p class="static-note">The static site trains the JS port in your browser. The local app can also train with the Python code.</p>' : ''}
        <div class="row">
          <div><label for="trees">Trees</label><input type="number" id="trees" min="1" max="300" value="${d.n_estimators}"></div>
          <div><label for="depth">Max depth</label><input type="number" id="depth" min="1" max="30" value="${d.max_depth}"></div>
        </div>
        <label for="seed">Random seed (blank = a new forest each run; the desktop app sets none)</label>
        <input type="number" id="seed" min="0" step="1" placeholder="random">
        <button id="run">Train and classify</button>
        <div class="progress" aria-hidden="true"><i id="bar"></i></div>
        <p class="status" id="status" role="status" aria-live="polite"></p>
      </div>
    </aside>
    <section>
      <div class="toolbar">
        <label class="inl" for="dispbands">Display</label>
        <select id="dispbands" class="narrow"></select>
        <label class="inl" for="stretch">Stretch</label>
        <select id="stretch" class="narrow">${info.stretches.map(s => `<option>${esc(s)}</option>`).join('')}</select>
        <div class="ctl" style="min-width:150px;flex:1"><label for="opacity">Result opacity <b id="opv"></b></label>
          <input type="range" id="opacity" min="0" max="1" step="0.05"></div>
      </div>
      <div class="toolbar">
        <label class="chk"><input type="checkbox" id="sh-result"> classification</label>
        <label class="chk"><input type="checkbox" id="sh-strokes"> training pixels</label>
        <label class="chk" id="errctl"><input type="checkbox" id="sh-errors"> errors vs known classes (demo)</label>
        <div class="seg" role="group" aria-label="Zoom" id="zoom">
          <button data-z="0">Fit</button><button data-z="1">1×</button><button data-z="2">2×</button><button data-z="4">4×</button></div>
      </div>
      <div class="viewwrap" id="wrap"><canvas id="view" aria-label="Image with training pixels and classification overlay"></canvas></div>
      <div class="legend" id="legend"></div>
      <div id="results"></div>
    </section>
  </div>`;
  if (!S.image) { setStatus('loading the demo...', true); await loadDemo(progress); setStatus(''); }
  wire();
  refresh();
  const listener = () => refresh();
  S.listeners.add(listener);
  onLeave(() => S.listeners.delete(listener));
});

function setStatus(msg, busy = false, frac = null) {
  const s = $('#status');
  if (!s) return;
  s.textContent = msg; s.classList.toggle('busy', busy);
  $('#bar').style.width = frac === null ? '0' : `${Math.round(frac * 100)}%`;
}
const progress = p => setStatus(`${p.stage} (${p.done}/${p.total})`, true, p.total ? p.done / p.total : null);

function draw() {
  const cv = $('#view'), { W, H } = S.image;
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  cv.getContext('2d').putImageData(compose(show), 0, 0);
}

function wire() {
  $('#demo').onclick = async () => { await loadDemo(progress); setStatus('demo loaded'); };
  $('#file').onchange = async e => {
    const f = e.target.files[0];
    if (!f) return;
    try { setStatus(`reading ${f.name}...`, true); await loadFile(f, progress); setStatus(`loaded ${f.name}: set the band roles, paint at least two classes, then train`); }
    catch (err) { setStatus(`Could not open ${f.name}: ${err.message}`); }
  };
  $('#addcls').onclick = () => {
    const i = S.classes.length;
    S.classes.push({ name: `Class ${i + 1}`, color: PALETTE[i % PALETTE.length], mask: new Uint8Array(S.image.W * S.image.H) });
    S.active = i; refresh();
  };
  $('#modes').onclick = e => { const m = e.target.dataset.m; if (m) { mode = m; refresh(); } };
  $('#zoom').onclick = e => { const z = e.target.dataset.z; if (z !== undefined) { zoom = +z; refresh(); } };
  $('#brush').oninput = e => { S.brush = +e.target.value; $('#brushv').textContent = S.brush; };
  $('#undo').onclick = () => { if (!undo()) setStatus('nothing to undo'); };
  $('#clear').onclick = () => { S.history.push(S.classes.map(c => c.mask.slice())); S.classes.forEach(c => c.mask.fill(0)); refresh(); };
  $('#opacity').oninput = e => { S.opacity = +e.target.value; $('#opv').textContent = S.opacity.toFixed(2); draw(); };
  $('#dispbands').onchange = e => { S.display.bands = e.target.value.split(',').map(Number); draw(); };
  $('#stretch').onchange = e => { S.display.stretch = e.target.value; draw(); };
  for (const k of Object.keys(show)) { const c = $(`#sh-${k}`); c.checked = show[k]; c.onchange = () => { show[k] = c.checked; draw(); }; }
  attachBrush($('#view'), () => mode, draw);
  $('#run').onclick = run;
}

async function run() {
  const engine = $('#engine').value, seedTxt = $('#seed').value.trim();
  if (painted() < 2) { setStatus('Paint training pixels for at least two classes first'); return; }
  const opts = { engine, n_estimators: Math.max(1, +$('#trees').value || 100), max_depth: Math.max(1, +$('#depth').value || 10),
    seed: seedTxt === '' ? (engine === 'js' ? (Math.random() * 2 ** 31) >>> 0 : null) : +seedTxt };
  $('#run').disabled = true;
  try { await classify(opts, progress); setStatus('classified'); }
  catch (e) { setStatus(`Error: ${e.message}`); }
  finally { $('#run').disabled = false; }
}

function bandOptions() {
  const n = S.image.bands.length, names = S.image.bandNames, opts = [];
  const add = (b, label) => opts.push(`<option value="${b.join(',')}" ${b.join() === S.display.bands.join() ? 'selected' : ''}>${esc(label)}</option>`);
  if (n >= 3) add([1, 2, 3], `${names[0]} / ${names[1]} / ${names[2]}`);
  if (n >= 4) { add([4, 1, 2], `False colour: ${names[3]} / ${names[0]} / ${names[1]}`); add([4, 3, 2], `${names[3]} / ${names[2]} / ${names[1]}`); }
  for (let b = 1; b <= n; b++) add([b, b, b], `${names[b - 1]} (grey)`);
  return opts.join('');
}

function renderRoles() {
  const n = S.image.bands.length, names = S.image.bandNames;
  $('#roles').innerHTML = n < 3
    ? '<p class="small mute">Fewer than three bands: the desktop app uses texture features only.</p>'
    : ROLES.map(r => `<div class="role"><label for="role-${r}">${r === 'nir' ? 'NIR' : r[0].toUpperCase() + r.slice(1)}</label>
      <select id="role-${r}" data-role="${r}">${r === 'nir' ? '<option value="">none</option>' : ''}
      ${names.map((nm, i) => `<option value="${i}" ${S.bandMap[r] === i ? 'selected' : ''}>${esc(nm)}</option>`).join('')}</select></div>`).join('');
  $('#roles').onchange = async () => {
    const m = {};
    for (const s of $$('#roles select')) if (s.value !== '') m[s.dataset.role] = +s.value;
    setStatus('recomputing features...', true);
    await setBandMap(m, progress);
    setStatus('band roles updated');
  };
  const sp = n >= 3 ? spectralNames(S.bandMap, S.info.features.spectral_indices) : [];
  $('#featinfo').innerHTML = `${S.featureNames.length} features: ${sp.length ? `spectral ${esc(sp.join(', '))}; ` : ''}texture (std, range, gradient) at ${S.info.features.texture_windows.join(', ')} px.`
    + ' Texture uses bands 1-3 whatever the roles, as the desktop app does.';
}

function refresh() {
  if (!$('#view') || !S.image) return;
  const img = S.image, g = img.geo;
  $('#imginfo').innerHTML = `<b>${esc(img.name)}</b><br>${img.W}×${img.H} px · ${img.bands.length} band${img.bands.length > 1 ? 's' : ''} · ${esc(img.dtype)}`
    + (img.factor > 1 ? ` · reduced ${img.factor}× by block averaging` : '')
    + (g?.pixelScale ? `<br>pixel ${+g.pixelScale[0].toPrecision(5)} × ${+g.pixelScale[1].toPrecision(5)} ${esc(g.units || 'units?')}${g.unitsAssumed ? ' (assumed)' : ''}` : '<br>no georeferencing')
    + (g ? ` · ${g.epsg ? `EPSG:${g.epsg}` : esc(g.crsName || 'CRS not given')}` : '')
    + (img.source === 'demo' ? '<br><span class="tag b">synthetic</span> generated by utils/synthetic_scene.py, not real imagery' : '');
  $('#labels').innerHTML = S.classes.map((c, i) => `<li class="${i === S.active ? 'on' : ''}">
      <input type="color" value="${c.color}" data-col="${i}" aria-label="Colour of ${esc(c.name)}">
      <button class="pick" data-pick="${i}" aria-pressed="${i === S.active}">${esc(c.name)}</button>
      <span class="n" title="painted pixels">${countPainted(c)}</span>
      <button class="x" data-ren="${i}" aria-label="Rename ${esc(c.name)}" title="Rename">✎</button>
      ${S.classes.length > 2 ? `<button class="x" data-del="${i}" aria-label="Remove ${esc(c.name)}" title="Remove">×</button>` : ''}</li>`).join('');
  $('#labels').onclick = e => {
    const t = e.target.closest('button');
    if (!t) return;
    if (t.dataset.pick) { S.active = +t.dataset.pick; if (mode === 'view') mode = 'paint'; refresh(); }
    if (t.dataset.ren) { const c = S.classes[+t.dataset.ren], n = prompt('Class name', c.name); if (n && n.trim()) { c.name = n.trim(); refresh(); } }
    if (t.dataset.del) { S.classes.splice(+t.dataset.del, 1); S.active = Math.min(S.active, S.classes.length - 1); S.result = null; refresh(); }
  };
  $('#labels').oninput = e => { const i = e.target.dataset.col; if (i !== undefined) { S.classes[+i].color = e.target.value; draw(); renderLegend(); } };
  renderRoles();
  for (const b of $$('#modes button')) b.classList.toggle('on', b.dataset.m === mode);
  for (const b of $$('#zoom button')) b.classList.toggle('on', +b.dataset.z === zoom);
  $('#brush').value = S.brush; $('#brushv').textContent = S.brush;
  $('#opacity').value = S.opacity; $('#opv').textContent = S.opacity.toFixed(2);
  $('#dispbands').innerHTML = bandOptions();
  $('#stretch').value = S.display.stretch;
  $('#errctl').classList.toggle('hide', !img.truth);
  const wrap = $('#wrap');
  wrap.classList.toggle('paint', mode !== 'view');
  wrap.classList.toggle('zoomed', zoom > 0);
  $('#view').style.width = zoom > 0 ? `${img.W * zoom}px` : '';
  draw();
  renderLegend();
  renderResults($('#results'));
}

function renderLegend() {
  const r = S.result;
  $('#legend').innerHTML = S.classes.map(c => `<span><i class="sw" style="background:${c.color}"></i>${esc(c.name)}</span>`).join('')
    + (r ? `<span class="badge ${r.engine === 'python' ? 'py' : 'js'}">${esc(r.note)}</span>` : '<span class="mute">No classification yet: paint and train.</span>');
}
