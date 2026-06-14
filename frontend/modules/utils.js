/**
 * Pure utility functions (no side effects, no DOM queries).
 */

export function apiUrl(path) {
  const p = path.startsWith('/') ? path : `/${path}`;
  if (typeof window === 'undefined') {
    return `http://127.0.0.1:8000${p}`;
  }
  if (window.PHOTO_API_BASE_URL != null && String(window.PHOTO_API_BASE_URL).trim() !== '') {
    return `${String(window.PHOTO_API_BASE_URL).replace(/\/$/, '')}${p}`;
  }
  const loc = window.location;
  if (loc.protocol === 'file:') {
    return `http://127.0.0.1:8000${p}`;
  }
  const host = loc.hostname || '127.0.0.1';
  const scheme = loc.protocol === 'https:' ? 'https' : 'http';
  let port = '8000';
  if (window.PHOTO_API_PORT != null && String(window.PHOTO_API_PORT).trim() !== '') {
    port = String(window.PHOTO_API_PORT).trim();
  }
  return `${scheme}://${host}:${port}${p}`;
}

export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function flushPaintFrames() {
  await new Promise((request) => requestAnimationFrame(request));
  await new Promise((request) => requestAnimationFrame(request));
}

export async function dwellMinimumSince(startedAt, minMs) {
  const elapsed = Date.now() - startedAt;
  if (elapsed < minMs) await sleep(minMs - elapsed);
}

export function formatBytes(n) {
  if (n == null || Number.isNaN(n) || n < 0) return '—';
  if (n === 0) return '0 B';
  const u = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  let x = n;
  while (x >= 1024 && i < u.length - 1) {
    x /= 1024;
    i += 1;
  }
  return `${x.toFixed(i > 0 ? 1 : 0)} ${u[i]}`;
}

export function formatElapsedTime(ms) {
  if (ms == null || ms < 0) return '0:00';
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, '0')}`;
}

export function formatFileSize(bytes) {
  if (bytes == null || Number.isNaN(bytes)) return '';
  const n = Number(bytes);
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

export function formatLatLon(lat, lon) {
  if (lat == null || lon == null || Number.isNaN(lat) || Number.isNaN(lon)) return '';
  const ns = lat >= 0 ? 'N' : 'S';
  const ew = lon >= 0 ? 'E' : 'W';
  return `${Math.abs(lat).toFixed(5)}° ${ns}, ${Math.abs(lon).toFixed(5)}° ${ew}`;
}

export function escapeHtml(text) {
  if (text == null) return '';
  const d = document.createElement('div');
  d.textContent = String(text);
  return d.innerHTML;
}

export function mdBoldToHtml(text) {
  if (text == null) return '';
  const esc = escapeHtml(String(text));
  return esc
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');
}

export function localWallDateToIsoString(d) {
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

export function exifAsciiDatetimeToIso(s) {
  const t = String(s).trim();
  const m = t.match(/^(\d{4}):(\d{2}):(\d{2})\s+(\d{2}):(\d{2}):(\d{2})$/);
  if (!m) return '';
  return `${m[1]}-${m[2]}-${m[3]}T${m[4]}:${m[5]}:${m[6]}`;
}

export function getPreferredClientLanguage() {
  if (typeof navigator === 'undefined') return 'en';
  if (Array.isArray(navigator.languages)) {
    const first = navigator.languages.find(
      (value) => typeof value === 'string' && value.trim() !== '',
    );
    if (first) return first.trim();
  }
  if (typeof navigator.language === 'string' && navigator.language.trim() !== '') {
    return navigator.language.trim();
  }
  return 'en';
}

export function appendFormBoolean(form, key, value) {
  form.append(key, value ? 'true' : 'false');
}

/** Convenience to create a DOM element with class and text content. */
export function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

/** Debounce helper for rapid UI events. */
export function debounce(fn, wait) {
  let t;
  return function (...args) {
    clearTimeout(t);
    t = setTimeout(() => fn.apply(this, args), wait);
  };
}
