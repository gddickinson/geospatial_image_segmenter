// Features page: every feature map the random forest sees for the current image (as the desktop Feature dialog
// previews them), with its range and the mean / spread over each class's painted pixels.
import { $, app, esc, route } from './core.js';
import { getInfo, loadDemo, S } from './state.js';
import { call } from './worker_client.js';
import { describe } from './features.js';

// A perceptual blue-green-yellow ramp (viridis key colours).
const RAMP = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
function ramp(t) {
  const x = Math.min(Math.max(t, 0), 1) * (RAMP.length - 1), i = Math.min(Math.floor(x), RAMP.length - 2), f = x - i;
  return RAMP[i].map((c, j) => c + (RAMP[i + 1][j] - c) * f);
}

let current = 0;

route('features', async () => {
  await getInfo();
  if (!S.image) await loadDemo();
  app().innerHTML = `
  <h1>Feature maps</h1>
  <p class="lead">The ${S.featureNames.length} per-pixel features of <b>${esc(S.image.name)}</b> that the random forest is
    trained on: spectral indices from the band roles, and texture of the grey image at three window sizes.</p>
  <div class="featview">
    <section>
      <label for="feat">Feature</label>
      <select id="feat">${S.featureNames.map((n, k) => `<option value="${k}">${esc(n)}</option>`).join('')}</select>
      <p class="small mute" id="fdesc"></p>
      <div class="viewwrap"><canvas id="fmap" aria-label="Feature map"></canvas></div>
      <div class="rampbar"><span id="lo"></span><i></i><span id="hi"></span></div>
    </section>
    <aside class="panel"><h3>Per class (painted pixels)</h3><div id="perclass"></div>
      <p class="small mute">2nd-98th percentile stretch. A feature whose class ranges barely overlap separates those
        classes on its own; the forest combines all of them.</p></aside>
  </div>`;
  $('#feat').value = String(Math.min(current, S.featureNames.length - 1));
  $('#feat').onchange = e => { current = +e.target.value; show(); };
  await show();
});

async function show() {
  const k = +$('#feat').value, { W, H } = S.image, N = W * H, f = await call('feature', { k });
  $('#fdesc').textContent = describe(S.featureNames[k]);
  const sorted = Float64Array.from(f).filter(Number.isFinite).sort();
  const lo = sorted[Math.floor(0.02 * (sorted.length - 1))], hi = sorted[Math.ceil(0.98 * (sorted.length - 1))];
  const cv = $('#fmap');
  cv.width = W; cv.height = H;
  const img = new ImageData(W, H), d = img.data;
  for (let i = 0; i < N; i++) {
    const c = ramp(hi > lo ? (f[i] - lo) / (hi - lo) : 0);
    d[4 * i] = c[0]; d[4 * i + 1] = c[1]; d[4 * i + 2] = c[2]; d[4 * i + 3] = 255;
  }
  cv.getContext('2d').putImageData(img, 0, 0);
  const fmt = v => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toPrecision(3));
  $('#lo').textContent = fmt(lo); $('#hi').textContent = fmt(hi);
  $('#perclass').innerHTML = `<table class="tbl"><thead><tr><th>Class</th><th>n</th><th>Mean</th><th>SD</th></tr></thead><tbody>
    ${S.classes.map(c => {
      let n = 0, s = 0, s2 = 0;
      for (let i = 0; i < N; i++) if (c.mask[i]) { n++; s += f[i]; s2 += f[i] * f[i]; }
      const m = n ? s / n : null, sd = n > 1 ? Math.sqrt(Math.max(s2 / n - m * m, 0)) : null;
      return `<tr><td><i class="sw" style="background:${c.color}"></i>${esc(c.name)}</td><td>${n}</td><td>${m === null ? '–' : fmt(m)}</td><td>${sd === null ? '–' : fmt(sd)}</td></tr>`;
    }).join('')}</tbody></table>`;
}
