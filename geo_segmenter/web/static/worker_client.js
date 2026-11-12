// Promise wrapper around worker.js: call(type, args, onProgress) -> result. One worker for the whole session.
let worker = null, seq = 0;
const pending = new Map();

function get() {
  if (!worker) {
    worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    worker.onmessage = e => {
      const { id, ok, result, error, progress } = e.data, p = pending.get(id);
      if (!p) return;
      if (progress) { p.onProgress(progress); return; }
      pending.delete(id);
      if (ok) p.resolve(result); else p.reject(new Error(error));
    };
    worker.onerror = e => { for (const p of pending.values()) p.reject(new Error(e.message || 'worker error')); pending.clear(); };
  }
  return worker;
}

export function call(type, args = {}, onProgress = () => {}) {
  const id = ++seq;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject, onProgress });
    get().postMessage({ id, type, ...args });
  });
}
