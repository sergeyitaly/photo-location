/**
 * Image selection, preview, EXIF parsing, and metadata panel.
 */
import { formatLatLon, formatFileSize, localWallDateToIsoString, exifAsciiDatetimeToIso, escapeHtml } from './utils.js';
import { state } from './state.js';

const DOM = {
  uploadArea: () => document.getElementById('uploadArea'),
  fileInput: () => document.getElementById('fileInput'),
  previewContainer: () => document.getElementById('previewContainer'),
  preview: () => document.getElementById('preview'),
  fileName: () => document.getElementById('fileName'),
  predictBtn: () => document.getElementById('predictBtn'),
  clearBtn: () => document.getElementById('clearBtn'),
  imageMetaDl: () => document.getElementById('imageMetaDl'),
  imageMetaAside: () => document.getElementById('imageMetaAside'),
};

export function getCheckboxRefs() {
  return {
    includeFeatureAnalysis: document.getElementById('includeFeatureAnalysis'),
    includeGlobeRegionalHints: document.getElementById('includeGlobeRegionalHints'),
    includeSceneGeolocationCues: document.getElementById('includeSceneGeolocationCues'),
    includeCulturalEconomicVisualCues: document.getElementById('includeCulturalEconomicVisualCues'),
    includeExternalValidation: document.getElementById('includeExternalValidation'),
    includeMlImageRecognition: document.getElementById('includeMlImageRecognition'),
    includeInfrastructureEnergyCues: document.getElementById('includeInfrastructureEnergyCues'),
    fastPrediction: document.getElementById('fastPrediction'),
    clearPredictionCache: document.getElementById('clearPredictionCache'),
    includeLlmDetective: document.getElementById('includeLlmDetective'),
  };
}

export function setupImageListeners(handleImageSelectFn) {
  const ua = DOM.uploadArea();
  const fi = DOM.fileInput();
  if (ua) {
    ua.addEventListener('click', () => fi?.click());
    ua.addEventListener('dragover', (e) => {
      e.preventDefault();
      ua.classList.add('dragover');
    });
    ua.addEventListener('dragleave', () => ua.classList.remove('dragover'));
    ua.addEventListener('drop', (e) => {
      e.preventDefault();
      ua.classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) handleImageSelectFn(e.dataTransfer.files[0]);
    });
  }
  if (fi) {
    fi.addEventListener('change', (e) => {
      if (e.target.files.length > 0) handleImageSelectFn(e.target.files[0]);
    });
  }
}

export function handleImageSelect(file) {
  if (!file.type.startsWith('image/')) {
    showError('Please select a valid image file');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showError('Image size must be less than 10MB');
    return;
  }

  state.sampleReference = null;
  state.selectedImage = file;
  const fnEl = DOM.fileName();
  if (fnEl) fnEl.textContent = file.name;

  const reader = new FileReader();
  reader.onload = (e) => {
    const ref = state.sampleReference;
    const p = DOM.preview();
    const pc = DOM.previewContainer();
    const ua = DOM.uploadArea();
    const pb = DOM.predictBtn();
    const cb = DOM.clearBtn();
    if (p) p.src = e.target.result;
    if (pc) pc.style.display = 'block';
    if (ua) ua.style.display = 'none';
    if (pb) pb.disabled = false;
    if (cb) cb.style.display = 'inline-block';
    refreshImageMetadataForCurrentSelection(file, ref);
  };
  reader.readAsDataURL(file);
}

export function showError(message) {
  const errorDiv = document.createElement('div');
  errorDiv.className = 'error-message';
  errorDiv.textContent = message;
  const container = document.querySelector('.main-content');
  if (container) container.insertBefore(errorDiv, container.firstChild);
  setTimeout(() => errorDiv.remove(), 5000);
}

async function updateImageMetadataPanel(file, commonsRef, includePixelDimensions) {
  const dl = DOM.imageMetaDl();
  const aside = DOM.imageMetaAside();
  const previewEl = DOM.preview();
  if (!dl || !aside || !file) return;

  const rows = [];
  const pushRow = (label, value) => {
    if (value == null || value === '') return;
    rows.push(`<dt>${escapeHtml(label)}</dt><dd>${escapeHtml(String(value))}</dd>`);
  };

  pushRow('File', `${file.name} (${formatFileSize(file.size)})`);
  if (file.type) pushRow('Type', file.type);

  let exif = null;
  try {
    if (typeof exifr !== 'undefined' && typeof exifr.parse === 'function') {
      exif = await exifr.parse(file, { gps: true, reviveValues: true, mergeOutput: true });
    }
  } catch (e) {
    console.warn('photo_geo: EXIF parse failed', e);
  }

  if (exif && typeof exif === 'object') {
    const cam = [exif.Make, exif.Model].filter(Boolean).join(' ').trim();
    if (cam) pushRow('Camera', cam);
    if (exif.LensModel) pushRow('Lens', String(exif.LensModel));
    if (exif.FocalLength != null) {
      let fl = String(exif.FocalLength);
      if (exif.FocalLengthIn35mmFormat != null) fl += ` (${exif.FocalLengthIn35mmFormat}mm equiv)`;
      pushRow('Focal length', fl);
    }
    const iso = exif.ISO ?? exif.ISOSpeedRatings;
    if (iso != null) pushRow('ISO', String(iso));
    if (exif.ExposureTime != null) {
      const et = exif.ExposureTime;
      let s;
      if (typeof et === 'number') s = et >= 1 ? `${et.toFixed(1)} s` : `1/${Math.round(1 / et)} s`;
      else s = String(et);
      pushRow('Exposure', s);
    }
    if (exif.FNumber != null) {
      const fn = Number(exif.FNumber);
      if (!Number.isNaN(fn)) pushRow('Aperture', `f/${fn.toFixed(1)}`);
    }
    if (exif.Orientation != null) pushRow('Orientation', String(exif.Orientation));
    const lat = exif.latitude;
    const lon = exif.longitude;
    if (lat != null && lon != null) pushRow('GPS (EXIF)', formatLatLon(Number(lat), Number(lon)));
  }

  if (commonsRef && typeof commonsRef.latitude === 'number' && typeof commonsRef.longitude === 'number') {
    pushRow('Reference (Commons)', formatLatLon(commonsRef.latitude, commonsRef.longitude));
    if (commonsRef.label) pushRow('Commons title', shortCommonsTitle(commonsRef.label));
  }

  if (includePixelDimensions) {
    const w = previewEl?.naturalWidth;
    const h = previewEl?.naturalHeight;
    if (w && h) pushRow('Dimensions', `${w} × ${h} px`);
  }

  dl.innerHTML = rows.join('');
  aside.style.display = rows.length ? 'block' : 'none';
}

export function refreshImageMetadataForCurrentSelection(file, commonsRef) {
  void updateImageMetadataPanel(file, commonsRef, false);
  const previewEl = DOM.preview();
  if (!previewEl) return;
  previewEl.addEventListener('load', () => {
    void updateImageMetadataPanel(file, commonsRef, true);
  }, { once: true });
}

export async function mergeClientExifTimeIfNeeded(prediction) {
  if (prediction.exif_capture_time && prediction.exif_capture_time.iso8601) return;
  if (!state.selectedImage || typeof exifr === 'undefined' || typeof exifr.parse !== 'function') return;
  try {
    const exif = await exifr.parse(state.selectedImage, { gps: true, reviveValues: true, mergeOutput: true });
    if (!exif || typeof exif !== 'object') return;
    const dt = exif.DateTimeOriginal || exif.CreateDate || exif.ModifyDate || exif.DateTime;
    if (!dt) return;
    let iso8601;
    if (dt instanceof Date && !Number.isNaN(dt.getTime())) iso8601 = localWallDateToIsoString(dt);
    else if (typeof dt === 'string') iso8601 = exifAsciiDatetimeToIso(dt);
    else return;
    if (!iso8601) return;
    prediction.exif_capture_time = {
      iso8601,
      source_field: 'DateTimeOriginal (browser)',
      assumed_local_time: true,
    };
  } catch (e) {
    console.warn('photo_geo: client EXIF capture time merge failed', e);
  }
}

function shortCommonsTitle(title) {
  const t = title.replace(/^File:/i, '');
  return t.length > 52 ? `${t.slice(0, 52)}…` : t;
}
