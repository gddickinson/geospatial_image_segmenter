// Entry point: lazy page modules, the mode label, a subtle background of drifting field outlines, then the router.
import { $, app, isStatic, lazy, route, start } from './core.js';

const PAGES = { 'home.js': ['home'], 'classify.js': ['classify'], 'features_page.js': ['features'], 'docs.js': ['method', 'about'] };
lazy(Object.fromEntries(Object.entries(PAGES).flatMap(([file, names]) => names.map(n => [n, () => import('./' + file)]))));
route('notfound', () => { app().innerHTML = '<h1>Not found</h1><p><a href="#/">Back to the start</a></p>'; });

$('#mode').textContent = isStatic()
  ? 'static site: classification runs in your browser (JS port tested against the Python code)'
  : 'local app: Python engine on this computer, or the in-browser JS port';

// Background: faint field parcels slowly changing colour, like a classification map. Off under reduced motion.
function background(canvas) {
  if (!canvas || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const g = canvas.getContext('2d'), cols = ['#2f7bff', '#1f9e4a', '#b7e05a', '#c98b4f', '#ff4fa0'];
  const fit = () => { canvas.width = innerWidth; canvas.height = innerHeight; };
  fit(); addEventListener('resize', fit);
  const cells = Array.from({ length: 26 }, () => ({ x: Math.random(), y: Math.random(), w: 40 + Math.random() * 90,
    h: 30 + Math.random() * 70, c: Math.floor(Math.random() * 5), t: Math.random() * 400 }));
  let last = 0;
  const tick = t => {
    requestAnimationFrame(tick);
    if (document.hidden || t - last < 80) return;
    last = t;
    g.clearRect(0, 0, canvas.width, canvas.height);
    for (const c of cells) {
      if (--c.t < 0) { c.c = (c.c + 1) % 5; c.t = 200 + Math.random() * 400; }
      g.strokeStyle = cols[c.c]; g.globalAlpha = 0.5; g.lineWidth = 1.2;
      g.strokeRect(c.x * canvas.width, c.y * canvas.height, c.w, c.h);
    }
  };
  requestAnimationFrame(tick);
}
background($('#bg'));
start();
