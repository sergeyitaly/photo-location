/**
 * localStorage cache for UI state + Commons sample list snapshot.
 */
import { apiUrl } from './utils.js';

const CLIENT_CACHE_KEY = 'photo_geo_ui_cache_v1';
const COMMONS_CACHE_TTL_MS = 45 * 60 * 1000;
const STREETCLIP_WARM_MAX_AGE_MS = 30 * 24 * 60 * 60 * 1000;

function getApiOrigin() {
  try {
    return new URL(apiUrl('/')).origin;
  } catch {
    return '';
  }
}

function emptyClientCache() {
  return {
    v: 1,
    apiOrigin: getApiOrigin(),
    streetclipWarmAt: null,
    commons: null,
  };
}

function readClientCache() {
  try {
    const raw = localStorage.getItem(CLIENT_CACHE_KEY);
    if (!raw) return emptyClientCache();
    const o = JSON.parse(raw);
    if (o.v !== 1 || o.apiOrigin !== getApiOrigin()) return emptyClientCache();
    return o;
  } catch {
    return emptyClientCache();
  }
}

function writeClientCache(obj) {
  try {
    obj.apiOrigin = getApiOrigin();
    localStorage.setItem(CLIENT_CACHE_KEY, JSON.stringify(obj));
  } catch (e) {
    console.warn('photo_geo: localStorage unavailable or quota exceeded', e);
  }
}

function mergeClientCache(patch) {
  const cur = readClientCache();
  Object.assign(cur, patch);
  writeClientCache(cur);
}

/** After at least one successful StreetCLIP prediction, progress hints assume server-side weights are on disk. */
export function recordStreetclipWarm() {
  mergeClientCache({ streetclipWarmAt: Date.now() });
}

export function isStreetclipWarmClient() {
  const t = readClientCache().streetclipWarmAt;
  if (t == null) return false;
  return Date.now() - t < STREETCLIP_WARM_MAX_AGE_MS;
}

export function saveCommonsListCache(payload) {
  mergeClientCache({
    commons: { savedAt: Date.now(), payload },
  });
}

export function readCommonsListCache() {
  const { commons } = readClientCache();
  if (!commons?.payload?.samples?.length) return null;
  if (Date.now() - commons.savedAt > COMMONS_CACHE_TTL_MS) return null;
  return commons;
}
