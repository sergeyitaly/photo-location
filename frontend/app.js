/**
 * Photo Geolocation System - Frontend Application
 * Handles image upload, prediction API calls, and result visualization
 */

/**
 * Full URL for API calls (never a bare path — avoids wrong host/port after redirects or dev servers).
 * Default: same hostname as this page, port 8000 (uvicorn).
 * Override full origin: window.PHOTO_API_BASE_URL = 'http://127.0.0.1:8000'
 * Override port only: window.PHOTO_API_PORT = '8080'
 */
function apiUrl(path) {
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

/**
 * localStorage cache for UI state + Commons sample list snapshot.
 * PyTorch / Hugging Face weights are stored on the server (~/.cache/huggingface), not in the browser (~5 MiB limit).
 */
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

/**
 * How long the browser waits for POST /predict before aborting (AbortController).
 * Full vision on CPU routinely exceeds **one hour**; any finite limit causes fake “failures” while the server still runs.
 *
 * Default **0** = no client-side abort (wait for the response).
 * Override: `window.PHOTO_PREDICT_TIMEOUT_MS = 7200000` (e.g. 2 hours), or a positive ms cap for metered networks.
 */
function getPredictFetchTimeoutMs() {
    if (typeof window === 'undefined') {
        return 0;
    }
    if (window.PHOTO_PREDICT_TIMEOUT_MS != null) {
        const n = Number(window.PHOTO_PREDICT_TIMEOUT_MS);
        if (!Number.isFinite(n) || n < 0) {
            return 0;
        }
        return n;
    }
    return 0;
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
function recordStreetclipWarm() {
    mergeClientCache({ streetclipWarmAt: Date.now() });
}

function isStreetclipWarmClient() {
    const t = readClientCache().streetclipWarmAt;
    if (t == null) return false;
    return Date.now() - t < STREETCLIP_WARM_MAX_AGE_MS;
}

function saveCommonsListCache(payload) {
    mergeClientCache({
        commons: { savedAt: Date.now(), payload },
    });
}

function readCommonsListCache() {
    const { commons } = readClientCache();
    if (!commons?.payload?.samples?.length) return null;
    if (Date.now() - commons.savedAt > COMMONS_CACHE_TTL_MS) return null;
    return commons;
}

let selectedImage = null;
let currentPrediction = null;
/** When image was chosen from Wikimedia samples: lat/lon/label for error vs prediction */
let sampleReference = null;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const preview = document.getElementById('preview');
const fileName = document.getElementById('fileName');
const predictBtn = document.getElementById('predictBtn');
const includeFeatureAnalysisCb = document.getElementById('includeFeatureAnalysis');
const includeGlobeRegionalHintsCb = document.getElementById('includeGlobeRegionalHints');
const includeSceneGeolocationCuesCb = document.getElementById('includeSceneGeolocationCues');
const includeCulturalEconomicVisualCuesCb = document.getElementById('includeCulturalEconomicVisualCues');
const includeExternalValidationCb = document.getElementById('includeExternalValidation');
const includeMlImageRecognitionCb = document.getElementById('includeMlImageRecognition');
const includeInfrastructureEnergyCuesCb = document.getElementById('includeInfrastructureEnergyCues');
const fastPredictionCb = document.getElementById('fastPrediction');
const clearPredictionCacheCb = document.getElementById('clearPredictionCache');
const includeLlmDetectiveCb = document.getElementById('includeLlmDetective');
const clearBtn = document.getElementById('clearBtn');
const resultsSection = document.getElementById('resultsSection');

/** Backend `visual_time_of_day.bucket` literals — pixel-only time heuristic. */
const VISUAL_TIME_BUCKET_LABELS = {
    night_or_twilight: 'Night / twilight',
    golden_hour_warm: 'Golden-hour–like',
    midday_bright: 'Bright daylight',
    overcast_diffuse: 'Flat / overcast–like',
    blue_hour_dim: 'Dim / blue-hour–like',
    unclear: 'Unclear',
};

/** Progress overlay (declared early; DOM ready at script load for static UI) */
let progressTimers = [];
let predictStatusPollId = null;
let outputSubStepTimer = null;
let predictStartTime = null;
let predictTimeInterval = null;

/** Client phases for Predict: encode image → GET /config → POST /predict + server work */
const PREDICT_CLIENT_STEPS = 3;

/**
 * Steps 1–2 usually finish before the next paint, so users only saw "Step 3 of 3".
 * Hold each numbered step on screen at least this long unless real work took longer.
 */
const MIN_PREDICT_STEP_DWELL_MS = 340;

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

async function flushPaintFrames() {
    await new Promise((request) => requestAnimationFrame(request));
    await new Promise((request) => requestAnimationFrame(request));
}

async function dwellMinimumSince(startedAt, minMs) {
    const elapsed = Date.now() - startedAt;
    if (elapsed < minMs) await sleep(minMs - elapsed);
}

function clearProgressPhaseStrip() {
    const el = document.getElementById('globalProgressPhase');
    if (el) {
        el.textContent = '';
        el.hidden = true;
    }
}

function applyProgressPhaseStrip(step, total) {
    const el = document.getElementById('globalProgressPhase');
    if (!el) return;
    if (
        typeof step === 'number' &&
        typeof total === 'number' &&
        total > 0 &&
        step >= 1 &&
        step <= total
    ) {
        el.hidden = false;
        el.textContent = `Step ${step} of ${total}`;
    } else {
        clearProgressPhaseStrip();
    }
}

// ============================================================================
// Progress overlay
// ============================================================================

function setProgressBarMode(indeterminate, percent, barId = 'predictProgressBar') {
    const bar = document.getElementById(barId);
    if (!bar) return;
    bar.classList.toggle('predict-progress__bar--indeterminate', indeterminate);
    bar.classList.toggle('predict-progress__bar--determinate', !indeterminate);
    if (!indeterminate && typeof percent === 'number') {
        const p = Math.min(100, Math.max(0, percent));
        bar.style.width = `${p}%`;
        bar.style.animation = 'none';
    } else {
        bar.style.width = '';
        bar.style.animation = '';
    }
}

function clearProgressTimers() {
    progressTimers.forEach((id) => clearTimeout(id));
    progressTimers = [];
}

/**
 * Show blocking progress UI.
 * @param {object} opts
 * @param {string} opts.title
 * @param {string} [opts.detail]
 * @param {string} [opts.hint]
 * @param {boolean} [opts.indeterminate=true]
 * @param {number} [opts.percent] determinate width 0–100
 */
function showProgress(opts) {
    const panel = document.getElementById('predictProgress');
    const titleEl = document.getElementById('predictProgressTitle');
    const messageEl = document.getElementById('predictProgressMessage');
    const metaEl = document.getElementById('predictProgressMeta');
    const percentEl = document.getElementById('predictProgressPercent');
    if (!panel || !titleEl || !messageEl) return;

    clearProgressTimers();
    clearProgressPhaseStrip();
    const pipelineSection = document.getElementById('pipelineSection');
    if (pipelineSection) pipelineSection.style.display = 'none';

    titleEl.textContent = opts.title || 'Finding location';
    messageEl.textContent = opts.detail || opts.message || '';
    if (metaEl) metaEl.textContent = opts.hint || '';
    if (percentEl) percentEl.textContent = '';

    const indeterminate = opts.indeterminate !== false;
    setProgressBarMode(indeterminate, opts.percent);

    panel.style.display = 'block';
    panel.setAttribute('aria-busy', 'true');
    renderProgressLive(opts.live || null);
}

function updateProgress(opts) {
    const titleEl = document.getElementById('predictProgressTitle');
    const messageEl = document.getElementById('predictProgressMessage');
    const metaEl = document.getElementById('predictProgressMeta');
    const percentEl = document.getElementById('predictProgressPercent');
    if (!messageEl) return;
    if (opts.title != null && titleEl) titleEl.textContent = opts.title;
    if (opts.detail != null || opts.message != null) {
        messageEl.textContent = opts.detail || opts.message || '';
    }
    if (opts.hint != null && metaEl) metaEl.textContent = opts.hint;
    if (opts.percent != null) {
        setProgressBarMode(false, opts.percent);
        if (percentEl) percentEl.textContent = `${Math.round(opts.percent)}%`;
    } else if (opts.indeterminate === true) {
        setProgressBarMode(true);
        if (percentEl) percentEl.textContent = '';
    }
    if (opts.live) renderProgressLive(opts.live);
}

function formatBytes(n) {
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

function formatElapsedTime(ms) {
    if (ms == null || ms < 0) return '0:00';
    const totalSeconds = Math.floor(ms / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${String(seconds).padStart(2, '0')}`;
}

function startPredictTimeTracker() {
    stopPredictTimeTracker();
    predictStartTime = Date.now();
    const metaEl = document.getElementById('predictProgressMeta');
    const tick = () => {
        if (!predictStartTime || !metaEl) return;
        const elapsed = formatElapsedTime(Date.now() - predictStartTime);
        const mode = fastPredictionCb?.checked ? 'Fast mode' : 'Full accuracy';
        metaEl.textContent = `Elapsed ${elapsed} · ${mode}`;
    };
    tick();
    predictTimeInterval = setInterval(tick, 1000);
}

function stopPredictTimeTracker() {
    if (predictTimeInterval != null) {
        clearInterval(predictTimeInterval);
        predictTimeInterval = null;
    }
    predictStartTime = null;
}

function stopPredictStatusPoll() {
    if (predictStatusPollId != null) {
        clearInterval(predictStatusPollId);
        predictStatusPollId = null;
    }
}

function renderProgressLive(live, progressStep, progressStatus) {
    const box = document.getElementById('predictProgressLive');
    if (!box) return;
    if (!live || typeof live !== 'object') {
        box.innerHTML = '';
        return;
    }

    const parts = [];
    const note = (live.processing_note || '').trim();
    const region = (live.region_hint || '').trim();
    if (note) {
        parts.push(`<p class="predict-progress__note">${escapeHtml(note)}</p>`);
    }
    if (region && region !== note) {
        parts.push(`<p class="predict-progress__region">${escapeHtml(region)}</p>`);
    }

    const candidates = Array.isArray(live.candidates) ? live.candidates : [];
    if (candidates.length > 0) {
        const rows = candidates
            .map((c) => {
                const place = escapeHtml(formatProgressPlace(c) || c.place || '');
                const src = escapeHtml(c.source || '');
                const lat = Number(c.latitude);
                const lon = Number(c.longitude);
                const coordLabel = formatLatLon(c.latitude, c.longitude);
                const mapsHref =
                    (c.maps_url && String(c.maps_url).trim()) ||
                    (Number.isFinite(lat) && Number.isFinite(lon) ? googleMapsUrl(lat, lon) : '');
                const coord =
                    mapsHref && coordLabel
                        ? `<a class="predict-progress__maps-link" href="${escapeHtml(mapsHref)}" target="_blank" rel="noopener noreferrer" title="Open in Google Maps">${escapeHtml(coordLabel)}</a>`
                        : coordLabel
                          ? `<span class="predict-progress__coord">${escapeHtml(coordLabel)}</span>`
                          : '';
                const conf =
                    c.confidence_pct != null
                        ? `<span class="predict-progress__conf">${escapeHtml(String(c.confidence_pct))}%</span>`
                        : '';
                const reasons = Array.isArray(c.reasons) ? c.reasons : [];
                const reasonsHtml =
                    reasons.length > 0
                        ? `<ul class="predict-progress__reasons">${reasons
                              .map((r) => `<li>${escapeHtml(String(r))}</li>`)
                              .join('')}</ul>`
                        : '';
                const alsoFrom =
                    Array.isArray(c.also_from) && c.also_from.length > 0
                        ? `<span class="predict-progress__also">Also: ${escapeHtml(c.also_from.join(', '))}</span>`
                        : '';
                return `<li class="predict-progress__candidate">
                    <div class="predict-progress__candidate-head">
                        <span class="predict-progress__place">${place}</span>
                        ${conf}
                    </div>
                    ${coord ? `<div class="predict-progress__coord-row">${coord}</div>` : ''}
                    <span class="predict-progress__src">${src}${alsoFrom}</span>
                    ${reasonsHtml}
                </li>`;
            })
            .join('');
        const hasFinalFusion = candidates.some((c) =>
            String(c.source || '').toLowerCase().includes('final fusion'),
        );
        const interimNote =
            hasFinalFusion || progressStatus === 'completed'
                ? '<p class="predict-progress__interim-note">Final fusion ranking is shown in the result card (highest %). Earlier rows may have shown higher interim StreetCLIP scores.</p>'
                : '<p class="predict-progress__interim-note">Interim scores — percentages can change after fusion merges GeoCLIP, StreetCLIP, and grid search.</p>';
        parts.push(
            `<div class="predict-progress__candidates-wrap">
                <p class="predict-progress__candidates-title">Leading guesses</p>
                ${interimNote}
                <ul class="predict-progress__candidates">${rows}</ul>
            </div>`,
        );
    }

    const samples = Array.isArray(live.sample_places) ? live.sample_places : [];
    if (samples.length > 0 && candidates.length === 0) {
        parts.push(
            `<p class="predict-progress__samples"><span class="predict-progress__samples-label">Cities in search area:</span> ${escapeHtml(samples.slice(0, 6).join(' · '))}</p>`,
        );
    }

    const ollamaThoughts = Array.isArray(live.ollama_key_thoughts) ? live.ollama_key_thoughts : [];
    const onOllamaStep = progressStep === 'llm_detective' || live.ollama_status === 'running';
    if (onOllamaStep && ollamaThoughts.length === 0) {
        parts.push(
            `<div class="predict-progress__ollama">
                <p class="predict-progress__ollama-title">Ollama — key thoughts</p>
                <p class="predict-progress__ollama-wait">Waiting for local LLM (usually 30s–3 min on CPU)…</p>
            </div>`,
        );
    } else if (ollamaThoughts.length > 0) {
        const modelLabel = live.ollama_model ? escapeHtml(String(live.ollama_model)) : 'Ollama';
        const statusClass = live.ollama_enabled === false ? ' predict-progress__ollama--muted' : '';
        parts.push(
            `<div class="predict-progress__ollama${statusClass}">
                <p class="predict-progress__ollama-title">Ollama — key thoughts <span class="predict-progress__ollama-model">(${modelLabel})</span></p>
                <ul class="predict-progress__ollama-list">${ollamaThoughts
                    .map((t) => `<li>${escapeHtml(String(t))}</li>`)
                    .join('')}</ul>
            </div>`,
        );
    }

    box.innerHTML = parts.join('');
}

/** Apply GET /predict/progress JSON to the inline main-page panel. */
function applyServerProgress(progress) {
    if (!progress || typeof progress !== 'object') return;

    const live = progress.live && typeof progress.live === 'object' ? progress.live : {};
    const message =
        (live.processing_note || '').trim() ||
        (progress.message || '').trim() ||
        'Working…';
    const title = progress.phase_label || 'Finding location';

    const elapsed =
        progress.elapsed_s != null
            ? formatElapsedTime(Number(progress.elapsed_s) * 1000)
            : progress.elapsed_ms != null
              ? formatElapsedTime(progress.elapsed_ms)
              : '';
    const fastNote = progress.fast_mode ? 'Fast mode' : 'Full accuracy';
    const pct = typeof progress.percent === 'number' ? progress.percent : null;

    let meta = [elapsed && `Elapsed ${elapsed}`, fastNote, pct != null && `${pct}%`]
        .filter(Boolean)
        .join(' · ');
    const step = progress.step || '';
    if (progress.status === 'running' && !progress.fast_mode && step !== 'llm_detective') {
        meta += meta ? ' · ' : '';
        meta += 'Full pipeline on CPU can take 30+ minutes';
    }
    if (progress.status === 'running' && step === 'llm_detective') {
        meta += meta ? ' · ' : '';
        meta += 'Ollama local LLM';
    }

    updateProgress({
        title,
        message,
        hint: meta,
        percent: pct != null && progress.status === 'running' ? pct : undefined,
        indeterminate: pct == null || progress.status !== 'running',
        live,
    });
    renderProgressLive(live, step, progress.status);
}

/**
 * Polls /predict/progress and StreetCLIP load state while POST /predict is in flight.
 */
function startPredictStatusPoll() {
    stopPredictStatusPoll();
    predictStatusPollId = setInterval(async () => {
        let progressRunning = false;
        try {
            const progRes = await fetch(apiUrl('/predict/progress'));
            if (progRes.ok) {
                const progress = await progRes.json();
                if (progress.status === 'running' || progress.status === 'completed') {
                    progressRunning = progress.status === 'running';
                    applyServerProgress(progress);
                    if (progress.status === 'completed') {
                        return;
                    }
                }
            }
        } catch (_) {
            /* ignore */
        }
        if (progressRunning) {
            return;
        }
        try {
            const r = await fetch(apiUrl('/model/streetclip-load-status'));
            if (!r.ok) return;
            const s = await r.json();
            if (s.phase === 'downloading') {
                const pct = typeof s.percent === 'number' ? Math.round(s.percent) : 0;
                const rawFile = s.file != null && String(s.file).trim() ? String(s.file).trim() : '';
                const msg = (s.message || '').trim();
                const aggregate = /^fetching\s/i.test(rawFile) || /^fetching\s/i.test(msg);
                const detail = aggregate
                    ? msg || 'Fetching files from Hugging Face…'
                    : rawFile
                      ? `Downloading ${rawFile}…`
                      : msg || 'Downloading model files from Hugging Face…';
                let hint = `${Math.min(100, pct)}%`;
                if (s.total_bytes > 0) {
                    hint += ` · ${formatBytes(s.current_bytes)} / ${formatBytes(s.total_bytes)}`;
                } else {
                    hint += ' of current file';
                }
                updateProgress({
                    title: 'Loading AI models',
                    detail,
                    hint,
                    percent: Math.min(99, pct),
                    indeterminate: false,
                });
                return;
            }
            if (s.phase === 'loading_ram') {
                updateProgress({
                    title: 'Loading AI models',
                    detail: s.message || 'Loading checkpoint into memory…',
                    hint: 'Reading weights from disk…',
                    percent: 100,
                    indeterminate: false,
                });
                return;
            }
            if (s.ready) {
                updateProgress({
                    title: 'Finding location',
                    detail: 'Models ready — analyzing your photo…',
                    indeterminate: true,
                });
            }
        } catch (_) {
            /* ignore */
        }
    }, 1500);
}

function hideProgress() {
    stopPredictStatusPoll();
    stopPredictTimeTracker();
    clearOutputSubStepTimer();
    clearProgressPhaseStrip();
    renderProgressLive(null);
    const panel = document.getElementById('predictProgress');
    if (panel) {
        panel.style.display = 'none';
        panel.setAttribute('aria-busy', 'false');
    }
    const pipelineSection = document.getElementById('pipelineSection');
    if (pipelineSection) pipelineSection.style.display = '';
    clearProgressTimers();
}

function clearOutputSubStepTimer() {
    if (outputSubStepTimer != null) {
        clearInterval(outputSubStepTimer);
        outputSubStepTimer = null;
    }
}

/**
 * Build ordered list of output-phase sub-steps based on server config and UI options.
 */
function buildOutputSubSteps(cfg, fastOn) {
    const steps = [];
    if (!fastOn) {
        steps.push('Fusing multi-model predictions…');
        if (cfg?.use_cross_reference_database) {
            steps.push('Cross-referencing local gazetteer…');
        }
        steps.push('Validating with open data…');
        if (cfg?.reverse_geocode_enabled) {
            steps.push('Reverse geocoding place names…');
        }
        if (cfg?.globe_regional_torch_ready && includeGlobeRegionalHintsCb?.checked) {
            steps.push('Globe regional CLIP analysis…');
        }
        if (includeSceneGeolocationCuesCb?.checked) {
            steps.push('Scene geolocation cues…');
        }
        if (includeMlImageRecognitionCb?.checked) {
            steps.push('ML image recognition…');
        }
        if (includeInfrastructureEnergyCuesCb?.checked) {
            steps.push('Infrastructure & energy cues…');
        }
        if (cfg?.use_country_elimination || cfg?.use_bayesian_reasoning || cfg?.use_astronomy_solver) {
            steps.push('Geo-reasoning engine…');
        }
        if (cfg?.use_satellite_matching) {
            steps.push('Satellite reverse match…');
        }
        if (cfg?.use_streetview_verification && cfg?.streetview_api_configured) {
            steps.push('Street View verification…');
        }
    }
    if (
        cfg?.use_llm_detective &&
        (includeLlmDetectiveCb == null || includeLlmDetectiveCb.checked)
    ) {
        steps.push(
            cfg?.ollama_available
                ? 'Ollama detective (local LLM)…'
                : 'Ollama detective (start ollama serve)…',
        );
    }
    steps.push('Building final response…');
    return steps;
}

/**
 * Cycle through output sub-steps in the mini pipeline and progress overlay.
 */
function startOutputSubStepCycler(subSteps) {
    clearOutputSubStepTimer();
    if (!subSteps || subSteps.length === 0) return;

    let idx = 0;
    const tick = () => {
        const step = subSteps[idx];
        const statusEl = document.getElementById('mini-status-output');
        if (statusEl) {
            statusEl.textContent = step;
            statusEl.className = 'pipeline-mini__status pipeline-mini__status--active';
        }
        updateProgress({
            detail: `🎯 Output: ${step}`,
            hint: 'Running server-side output pipeline…',
        });
        idx = (idx + 1) % subSteps.length;
    };
    tick();
    outputSubStepTimer = setInterval(tick, 2500);
}

/**
 * While waiting on /predict, escalate messages (first StreetCLIP load can take minutes).
 * If localStorage says this browser has already seen a successful StreetCLIP run for this API
 * origin, we assume server-side HF cache is warm and use gentler copy.
 */
function schedulePredictLongWaitHints() {
    const warm = isStreetclipWarmClient();
    const steps = warm
        ? [
              [
                  7000,
                  {
                      detail: 'Running vision inference…',
                      hint:
                          'Model weights should already be on the server disk from a prior run. Heavy downloads are uncommon unless the cache was cleared.',
                      step: PREDICT_CLIENT_STEPS,
                      stepTotal: PREDICT_CLIENT_STEPS,
                  },
              ],
              [
                  22000,
                  {
                      detail: 'Still processing…',
                      hint: 'Large images or CPU inference can still take a while. Keep this tab open.',
                      step: PREDICT_CLIENT_STEPS,
                      stepTotal: PREDICT_CLIENT_STEPS,
                  },
              ],
          ]
        : [
              [
                  6000,
                  {
                      detail: 'Running inference on the server…',
                      hint:
                          'If the progress bar above is not moving, the server may still be downloading weights (see %). First load can take many minutes.',
                      step: PREDICT_CLIENT_STEPS,
                      stepTotal: PREDICT_CLIENT_STEPS,
                  },
              ],
              [
                  18000,
                  {
                      detail: 'Still processing…',
                      hint: 'Downloads or CPU inference can take several minutes on first load. Keep this tab open.',
                      step: PREDICT_CLIENT_STEPS,
                      stepTotal: PREDICT_CLIENT_STEPS,
                  },
              ],
          ];
    steps.forEach(([delay, payload]) => {
        progressTimers.push(
            setTimeout(async () => {
                try {
                    const r = await fetch(apiUrl('/model/streetclip-load-status'));
                    if (r.ok) {
                        const s = await r.json();
                        if (s.phase === 'downloading' || s.phase === 'loading_ram') return;
                    }
                } catch (_) {
                    /* fall through to payload */
                }
                updateProgress(payload);
            }, delay)
        );
    });
}

/**
 * Milestone hints at 60s, 120s, 180s so users know the system is still alive
 * and can estimate remaining time.
 */
function schedulePredictMilestoneHints() {
    const milestones = [
        [
            60000,
            {
                detail: 'Still working — 1 minute elapsed…',
                hint: 'If this is the first prediction, model downloads are likely in progress. Typical first-run time: 3–5 minutes.',
                step: PREDICT_CLIENT_STEPS,
                stepTotal: PREDICT_CLIENT_STEPS,
            },
        ],
        [
            120000,
            {
                detail: 'Still working — 2 minutes elapsed…',
                hint: 'Downloads may be large (several GB). The server is still active. Do not close this tab.',
                step: PREDICT_CLIENT_STEPS,
                stepTotal: PREDICT_CLIENT_STEPS,
            },
        ],
        [
            180000,
            {
                detail: 'Still working — 3 minutes elapsed…',
                hint: 'If models are nearly downloaded, inference will begin soon. Maximum wait: 5 minutes.',
                step: PREDICT_CLIENT_STEPS,
                stepTotal: PREDICT_CLIENT_STEPS,
            },
        ],
    ];
    milestones.forEach(([delay, payload]) => {
        progressTimers.push(
            setTimeout(async () => {
                try {
                    const r = await fetch(apiUrl('/model/streetclip-load-status'));
                    if (r.ok) {
                        const s = await r.json();
                        if (s.phase === 'downloading' || s.phase === 'loading_ram') return;
                    }
                } catch (_) {
                    /* fall through to payload */
                }
                updateProgress(payload);
            }, delay)
        );
    });
}

// ============================================================================
// Event Listeners (initialized on DOMContentLoaded)
// ============================================================================

function initUploadControls() {
    const area = document.getElementById('uploadArea');
    const input = document.getElementById('fileInput');
    if (!area || !input) {
        console.error('Upload UI missing: #uploadArea or #fileInput not found');
        return;
    }

    input.addEventListener('change', (e) => {
        const files = e.target.files;
        if (files && files.length > 0) {
            handleImageSelect(files[0]);
        }
    });

    area.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        area.classList.add('dragover');
    });

    area.addEventListener('dragleave', (e) => {
        e.preventDefault();
        area.classList.remove('dragover');
    });

    area.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        area.classList.remove('dragover');
        const files = e.dataTransfer?.files;
        if (files && files.length > 0) {
            handleImageSelect(files[0]);
        }
    });

    // Fallback for browsers that ignore overlay input (rare): label click still opens picker.
    area.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            input.click();
        }
    });
}

function initActionButtons() {
    const pb = document.getElementById('predictBtn');
    const cb = document.getElementById('clearBtn');
    if (pb) pb.addEventListener('click', predictLocation);
    if (cb) cb.addEventListener('click', clearUpload);
    document.getElementById('uploadAnotherBtn')?.addEventListener('click', clearUpload);
    document.getElementById('downloadBtn')?.addEventListener('click', downloadResults);
    document.getElementById('loadCommonsSamplesBtn')?.addEventListener('click', loadCommonsSamples);
}

// ============================================================================
// Image Handling
// ============================================================================

function escapeHtml(text) {
    if (text == null) return '';
    const d = document.createElement('div');
    d.textContent = String(text);
    return d.innerHTML;
}

const INFERENCE_MODEL_CATEGORY_LABELS = {
    geolocation: 'Geolocation',
    auxiliary: 'Scene analysis',
    refinement: 'Refinement',
    context: 'Context & data',
};

/**
 * Renders API `inference_models` into grouped HTML, or '' when absent / empty.
 * @param {object} prediction
 * @returns {string}
 */
function buildInferenceModelsHtml(prediction) {
    const rows = prediction.inference_models;
    if (!Array.isArray(rows) || rows.length === 0) return '';
    const order = ['geolocation', 'auxiliary', 'refinement', 'context'];
    /** @type {Record<string, object[]>} */
    const groups = {};
    for (const r of rows) {
        const c = r.category || 'other';
        if (!groups[c]) groups[c] = [];
        groups[c].push(r);
    }
    let html = '<div class="inference-models">';
    for (const c of order) {
        if (!groups[c]) continue;
        const title = INFERENCE_MODEL_CATEGORY_LABELS[c] || c;
        html += `<div class="inference-models__group"><div class="inference-models__group-title">${escapeHtml(title)}</div><ul class="inference-models__list">`;
        for (const r of groups[c]) {
            const idRaw = r.identifier != null ? String(r.identifier).trim() : '';
            const idPart = idRaw
                ? ` <span class="inference-models__id">${escapeHtml(idRaw)}</span>`
                : '';
            html += `<li><span class="inference-models__name">${escapeHtml(r.name)}</span>${idPart}</li>`;
        }
        html += '</ul></div>';
    }
    for (const c of Object.keys(groups)) {
        if (order.includes(c)) continue;
        const title = INFERENCE_MODEL_CATEGORY_LABELS[c] || c;
        html += `<div class="inference-models__group"><div class="inference-models__group-title">${escapeHtml(title)}</div><ul class="inference-models__list">`;
        for (const r of groups[c]) {
            const idRaw = r.identifier != null ? String(r.identifier).trim() : '';
            const idPart = idRaw
                ? ` <span class="inference-models__id">${escapeHtml(idRaw)}</span>`
                : '';
            html += `<li><span class="inference-models__name">${escapeHtml(r.name)}</span>${idPart}</li>`;
        }
        html += '</ul></div>';
    }
    html += '</div>';
    return html;
}

/** Escape then turn **bold** into <strong> (API integrated_estimate strings). */
function mdBoldToHtml(text) {
    if (text == null) return '';
    const esc = escapeHtml(String(text));
    return esc
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');
}

function formatFileSize(bytes) {
    if (bytes == null || Number.isNaN(bytes)) return '';
    const n = Number(bytes);
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

function formatLatLon(lat, lon) {
    if (lat == null || lon == null || Number.isNaN(lat) || Number.isNaN(lon)) return '';
    const ns = lat >= 0 ? 'N' : 'S';
    const ew = lon >= 0 ? 'E' : 'W';
    return `${Math.abs(lat).toFixed(5)}° ${ns}, ${Math.abs(lon).toFixed(5)}° ${ew}`;
}

function googleMapsUrl(lat, lon) {
    if (lat == null || lon == null || Number.isNaN(Number(lat)) || Number.isNaN(Number(lon))) {
        return '';
    }
    return `https://www.google.com/maps?q=${encodeURIComponent(`${Number(lat)},${Number(lon)}`)}`;
}

/** Build local wall-clock ISO fragment (naive) from a parsed Date — for EXIF without timezone. */
function localWallDateToIsoString(d) {
    const pad = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

/** EXIF ASCII "YYYY:MM:DD HH:mm:ss" → ISO-like string for display parsing. */
function exifAsciiDatetimeToIso(s) {
    const t = String(s).trim();
    const m = t.match(/^(\d{4}):(\d{2}):(\d{2})\s+(\d{2}):(\d{2}):(\d{2})$/);
    if (!m) return '';
    return `${m[1]}-${m[2]}-${m[3]}T${m[4]}:${m[5]}:${m[6]}`;
}

/**
 * When the server omitted capture time (or an older build), parse the same file in the browser
 * so "Photo taken" still appears when exifr can read DateTimeOriginal.
 */
async function mergeClientExifTimeIfNeeded(prediction) {
    if (prediction.exif_capture_time && prediction.exif_capture_time.iso8601) return;
    if (!selectedImage || typeof exifr === 'undefined' || typeof exifr.parse !== 'function') return;
    try {
        const exif = await exifr.parse(selectedImage, { gps: true, reviveValues: true, mergeOutput: true });
        if (!exif || typeof exif !== 'object') return;
        const dt = exif.DateTimeOriginal || exif.CreateDate || exif.ModifyDate || exif.DateTime;
        if (!dt) return;
        let iso8601;
        if (dt instanceof Date && !Number.isNaN(dt.getTime())) {
            iso8601 = localWallDateToIsoString(dt);
        } else if (typeof dt === 'string') {
            iso8601 = exifAsciiDatetimeToIso(dt);
        } else {
            return;
        }
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

function formatPhotoTakenDisplayLines(ect) {
    const raw = ect.iso8601 || '';
    const secondary = ect.assumed_local_time
        ? 'Camera clock as stored—many files omit timezone in EXIF.'
        : 'From EXIF UTC offset or GPS timestamp.';
    let primary = raw;
    try {
        const d = new Date(raw);
        if (!Number.isNaN(d.getTime())) {
            const locale = typeof navigator !== 'undefined' ? navigator.language : undefined;
            const opts = { dateStyle: 'full', timeStyle: 'medium' };
            if (!ect.assumed_local_time) {
                opts.timeZone = 'UTC';
            }
            primary = d.toLocaleString(locale, opts);
        }
    } catch (_) {
        /* keep raw */
    }
    return { primary, secondary };
}

function buildPhotoTakenBannerHtml(prediction) {
    const ect = prediction.exif_capture_time;
    if (!ect || !ect.iso8601) return '';
    const { primary, secondary } = formatPhotoTakenDisplayLines(ect);
    const src = ect.source_field
        ? `<div class="results-summary-card__photo-taken-src">${escapeHtml(ect.source_field)}</div>`
        : '';
    return `
        <div class="results-summary-card__photo-taken" role="status" aria-live="polite">
            <div class="results-summary-card__photo-taken-label">Photo taken (EXIF)</div>
            <div class="results-summary-card__photo-taken-main">${escapeHtml(primary)}</div>
            ${src}
            <div class="results-summary-card__photo-taken-sub">${escapeHtml(secondary)}</div>
        </div>`;
}

function buildVisualTimeBannerHtml(prediction) {
    const vtd = prediction.visual_time_of_day;
    if (!vtd || typeof vtd !== 'object') return '';
    const hasBucket = vtd.bucket != null && String(vtd.bucket).length > 0;
    if (!hasBucket && !(String(vtd.summary || '').trim().length > 0)) return '';

    const bn = VISUAL_TIME_BUCKET_LABELS[vtd.bucket] || vtd.bucket || '—';
    const conf =
        vtd.confidence != null && Number.isFinite(Number(vtd.confidence))
            ? `${(Number(vtd.confidence) * 100).toFixed(0)}%`
            : '—';
    const excluded = prediction.embedded_metadata_excluded_from_prediction === true;
    const title = excluded ? 'Time-of-day estimate (from photo only)' : 'Time-of-day estimate (from photo)';
    const sub = excluded
        ? 'Embedded metadata was off—this is not the camera clock. Uses sky strip, CLIP sky/sun cues, and lower-frame edges (coarse daylight bucket, not exact time).'
        : 'No capture timestamp returned from embedded metadata—heuristic uses sky brightness, CLIP cues, and edge orientation (not a wall clock).';

    const ang =
        vtd.gradient_dominant_angle_deg != null &&
        Number.isFinite(Number(vtd.gradient_dominant_angle_deg))
            ? `${Number(vtd.gradient_dominant_angle_deg).toFixed(0)}°`
            : '—';
    const coh =
        vtd.gradient_coherence != null && Number.isFinite(Number(vtd.gradient_coherence))
            ? `${(Number(vtd.gradient_coherence) * 100).toFixed(0)}%`
            : '—';

    const detail = `${bn} · confidence ${conf} · edge emphasis ${ang} (coherence ${coh})`;

    return `
        <div class="results-summary-card__photo-taken results-summary-card__photo-taken--visual" role="status" aria-live="polite">
            <div class="results-summary-card__photo-taken-label">${escapeHtml(title)}</div>
            <div class="results-summary-card__photo-taken-main results-summary-card__photo-taken-main--compact">${escapeHtml(detail)}</div>
            <div class="results-summary-card__photo-taken-sub">${escapeHtml(sub)}</div>
        </div>`;
}

/** EXIF clock when embedded metadata was used; otherwise pixel-only visual estimate when available. */
function buildPhotoTimeBannerHtml(prediction) {
    const exifHtml = buildPhotoTakenBannerHtml(prediction);
    if (exifHtml) return exifHtml;
    return buildVisualTimeBannerHtml(prediction);
}

async function updateImageMetadataPanel(file, commonsRef, includePixelDimensions) {
    const dl = document.getElementById('imageMetaDl');
    const aside = document.getElementById('imageMetaAside');
    const previewEl = document.getElementById('preview');
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
        /* Capture time is shown in the results summary ("Photo taken") after predict — omit here to avoid duplicating the same clock. */

        const cam = [exif.Make, exif.Model].filter(Boolean).join(' ').trim();
        if (cam) pushRow('Camera', cam);

        if (exif.LensModel) pushRow('Lens', String(exif.LensModel));

        if (exif.FocalLength != null) {
            let fl = String(exif.FocalLength);
            if (exif.FocalLengthIn35mmFormat != null) {
                fl += ` (${exif.FocalLengthIn35mmFormat}mm equiv)`;
            }
            pushRow('Focal length', fl);
        }

        const iso = exif.ISO ?? exif.ISOSpeedRatings;
        if (iso != null) pushRow('ISO', String(iso));

        if (exif.ExposureTime != null) {
            const et = exif.ExposureTime;
            let s;
            if (typeof et === 'number') {
                s = et >= 1 ? `${et.toFixed(1)} s` : `1/${Math.round(1 / et)} s`;
            } else {
                s = String(et);
            }
            pushRow('Exposure', s);
        }

        if (exif.FNumber != null) {
            const fn = Number(exif.FNumber);
            if (!Number.isNaN(fn)) pushRow('Aperture', `f/${fn.toFixed(1)}`);
        }

        if (exif.Orientation != null) pushRow('Orientation', String(exif.Orientation));

        const lat = exif.latitude;
        const lon = exif.longitude;
        if (lat != null && lon != null) {
            pushRow('GPS (EXIF)', formatLatLon(Number(lat), Number(lon)));
        }
    }

    if (
        commonsRef &&
        commonsRef.latitude != null &&
        commonsRef.longitude != null &&
        typeof commonsRef.latitude === 'number' &&
        typeof commonsRef.longitude === 'number'
    ) {
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

function refreshImageMetadataForCurrentSelection(file, commonsRef) {
    void updateImageMetadataPanel(file, commonsRef, false);
    const previewEl = document.getElementById('preview');
    if (!previewEl) return;
    previewEl.addEventListener(
        'load',
        () => {
            void updateImageMetadataPanel(file, commonsRef, true);
        },
        { once: true }
    );
}

function handleImageSelect(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file');
        return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showError('Image size must be less than 10MB');
        return;
    }

    sampleReference = null;
    selectedImage = file;
    fileName.textContent = file.name;

    // Display preview
    const reader = new FileReader();
    reader.onload = (e) => {
        const ref = sampleReference;
        preview.src = e.target.result;
        previewContainer.style.display = 'block';
        uploadArea.style.display = 'none';
        predictBtn.disabled = false;
        clearBtn.style.display = 'inline-block';
        refreshImageMetadataForCurrentSelection(file, ref);
    };
    reader.readAsDataURL(file);
}

// ============================================================================
// Pipeline Visualization
// ============================================================================

const PIPELINE_STAGES = [
    { id: 'input', name: 'Input: Photo upload', icon: '📸', type: 'normal' },
    { id: 'parse', name: 'Parse: Request & decode', icon: '📋', type: 'normal' },
    { id: 'exif', name: 'EXIF Check: GPS extraction', icon: '🛰️', type: 'checkpoint' },
    { id: 'filename', name: 'Filename Hint: Keyword match', icon: '📝', type: 'checkpoint' },
    { id: 'features', name: 'Feature Extraction: Visual cues', icon: '🔍', type: 'normal' },
    { id: 'inference', name: 'Vision Inference: Ensemble fusion', icon: '🧠', type: 'major' },
    { id: 'optional', name: 'Optional Modules: Validation', icon: '🧩', type: 'normal' },
    { id: 'analysis', name: 'Additional Analysis: CLIP cues', icon: '🔬', type: 'normal' },
    { id: 'reasoning', name: 'Geo-Reasoning: Re-ranking', icon: '⚙️', type: 'major' },
    { id: 'output', name: 'Output: Location predictions', icon: '🎯', type: 'output' },
];

/** Build the mini pipeline rows inside the progress overlay */
function buildMiniPipeline() {
    const container = document.getElementById('pipelineMini');
    if (!container) return;
    container.innerHTML = PIPELINE_STAGES.map((s) => {
        const typeClass = s.type === 'checkpoint' ? 'pipeline-mini__row--checkpoint' :
                          s.type === 'major' ? 'pipeline-mini__row--major' : '';
        return `
            <div class="pipeline-mini__row ${typeClass}" data-mini-stage="${s.id}" id="mini-stage-${s.id}">
                <span class="pipeline-mini__icon">${s.icon}</span>
                <span class="pipeline-mini__name">${s.name}</span>
                <span class="pipeline-mini__status" id="mini-status-${s.id}"></span>
            </div>
        `;
    }).join('');
}

/** Update a mini pipeline row state */
function updateMiniPipelineStage(stageId, state) {
    const row = document.getElementById(`mini-stage-${stageId}`);
    const status = document.getElementById(`mini-status-${stageId}`);
    if (!row || !status) return;

    row.classList.remove('pipeline-mini__row--active', 'pipeline-mini__row--completed', 'pipeline-mini__row--skipped');
    status.classList.remove('pipeline-mini__status--active', 'pipeline-mini__status--completed', 'pipeline-mini__status--skipped');

    if (state === 'active') {
        row.classList.add('pipeline-mini__row--active');
        status.classList.add('pipeline-mini__status--active');
        status.textContent = 'running…';
        row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } else if (state === 'completed') {
        row.classList.add('pipeline-mini__row--completed');
        status.classList.add('pipeline-mini__status--completed');
        status.textContent = '✓ done';
    } else if (state === 'skipped') {
        row.classList.add('pipeline-mini__row--skipped');
        status.classList.add('pipeline-mini__status--skipped');
        status.textContent = 'skipped';
    } else {
        status.textContent = '';
    }
}

/** Reset all pipeline stages to neutral state */
function resetPipeline() {
    document.querySelectorAll('.pipeline-stage').forEach((stage) => {
        stage.classList.remove('pipeline-stage--active', 'pipeline-stage--completed', 'pipeline-stage--skipped');
    });
    document.querySelectorAll('.pipeline-node__box').forEach((box) => {
        box.classList.remove('pipeline-node__box--active', 'pipeline-node__box--completed');
    });
    const exec = document.getElementById('pipelineExecution');
    const stageEl = document.getElementById('pipelineExecutionStage');
    const prog = document.getElementById('pipelineExecutionProgress');
    if (exec) exec.style.display = 'none';
    if (stageEl) stageEl.textContent = '—';
    if (prog) prog.style.width = '0%';
    // Reset mini pipeline
    PIPELINE_STAGES.forEach((s) => updateMiniPipelineStage(s.id, 'neutral'));
}

/** Activate a specific pipeline stage (and mark prior stages completed) */
function activatePipelineStage(stageId) {
    const stageIndex = PIPELINE_STAGES.findIndex((s) => s.id === stageId);
    if (stageIndex < 0) return;

    // Mark all prior stages as completed
    for (let i = 0; i < stageIndex; i++) {
        const priorId = PIPELINE_STAGES[i].id;
        const priorStage = document.querySelector(`.pipeline-stage[data-stage="${priorId}"]`);
        if (priorStage) {
            priorStage.classList.remove('pipeline-stage--active');
            priorStage.classList.add('pipeline-stage--completed');
        }
        const priorBox = document.querySelector(`.pipeline-stage[data-stage="${priorId}"] .pipeline-node__box`);
        if (priorBox) {
            priorBox.classList.remove('pipeline-node__box--active');
            priorBox.classList.add('pipeline-node__box--completed');
        }
        updateMiniPipelineStage(priorId, 'completed');
    }

    // Activate current stage
    const stage = document.querySelector(`.pipeline-stage[data-stage="${stageId}"]`);
    if (stage) {
        stage.classList.remove('pipeline-stage--completed', 'pipeline-stage--skipped');
        stage.classList.add('pipeline-stage--active');
        // Scroll into view if needed
        stage.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    const boxes = document.querySelectorAll(`.pipeline-stage[data-stage="${stageId}"] .pipeline-node__box`);
    boxes.forEach((box) => {
        box.classList.remove('pipeline-node__box--completed');
        box.classList.add('pipeline-node__box--active');
    });
    updateMiniPipelineStage(stageId, 'active');

    // Update execution indicator
    const exec = document.getElementById('pipelineExecution');
    const stageEl = document.getElementById('pipelineExecutionStage');
    const prog = document.getElementById('pipelineExecutionProgress');
    if (exec) exec.style.display = 'flex';
    if (stageEl) stageEl.textContent = PIPELINE_STAGES[stageIndex].name;
    if (prog) {
        const pct = ((stageIndex + 1) / PIPELINE_STAGES.length) * 100;
        prog.style.width = `${pct}%`;
    }
}

/** Mark a stage as skipped (e.g., EXIF not present, filename not matching) */
function skipPipelineStage(stageId) {
    const stage = document.querySelector(`.pipeline-stage[data-stage="${stageId}"]`);
    if (stage) {
        stage.classList.remove('pipeline-stage--active', 'pipeline-stage--completed');
        stage.classList.add('pipeline-stage--skipped');
    }
    updateMiniPipelineStage(stageId, 'skipped');
}

/** Mark the entire pipeline as complete */
function completePipeline() {
    PIPELINE_STAGES.forEach((s) => {
        const stage = document.querySelector(`.pipeline-stage[data-stage="${s.id}"]`);
        if (stage) {
            stage.classList.remove('pipeline-stage--active', 'pipeline-stage--skipped');
            stage.classList.add('pipeline-stage--completed');
        }
        const boxes = document.querySelectorAll(`.pipeline-stage[data-stage="${s.id}"] .pipeline-node__box`);
        boxes.forEach((box) => {
            box.classList.remove('pipeline-node__box--active');
            box.classList.add('pipeline-node__box--completed');
        });
        updateMiniPipelineStage(s.id, 'completed');
    });
    const exec = document.getElementById('pipelineExecution');
    const stageEl = document.getElementById('pipelineExecutionStage');
    const prog = document.getElementById('pipelineExecutionProgress');
    if (exec) exec.style.display = 'flex';
    if (stageEl) stageEl.textContent = 'Complete ✓';
    if (prog) prog.style.width = '100%';
}

/** Show early-return state (EXIF or filename hit) */
function markEarlyReturn(returnStage) {
    const idx = PIPELINE_STAGES.findIndex((s) => s.id === returnStage);
    if (idx < 0) return;
    for (let i = 0; i <= idx; i++) {
        const s = PIPELINE_STAGES[i];
        const stage = document.querySelector(`.pipeline-stage[data-stage="${s.id}"]`);
        if (stage) {
            stage.classList.remove('pipeline-stage--active', 'pipeline-stage--skipped');
            stage.classList.add('pipeline-stage--completed');
        }
        const boxes = document.querySelectorAll(`.pipeline-stage[data-stage="${s.id}"] .pipeline-node__box`);
        boxes.forEach((box) => {
            box.classList.remove('pipeline-node__box--active');
            box.classList.add('pipeline-node__box--completed');
        });
        updateMiniPipelineStage(s.id, 'completed');
    }
    // Mark remaining stages as skipped
    for (let i = idx + 1; i < PIPELINE_STAGES.length; i++) {
        skipPipelineStage(PIPELINE_STAGES[i].id);
    }
    const exec = document.getElementById('pipelineExecution');
    const stageEl = document.getElementById('pipelineExecutionStage');
    const prog = document.getElementById('pipelineExecutionProgress');
    if (exec) exec.style.display = 'flex';
    if (stageEl) stageEl.textContent = `Early return: ${PIPELINE_STAGES[idx].name}`;
    if (prog) prog.style.width = `${((idx + 1) / PIPELINE_STAGES.length) * 100}%`;
}

function clearUpload() {
    selectedImage = null;
    sampleReference = null;
    const input = document.getElementById('fileInput');
    if (input) input.value = '';
    if (includeFeatureAnalysisCb) includeFeatureAnalysisCb.checked = true;
    if (includeGlobeRegionalHintsCb) includeGlobeRegionalHintsCb.checked = true;
    if (includeSceneGeolocationCuesCb) includeSceneGeolocationCuesCb.checked = true;
    if (includeCulturalEconomicVisualCuesCb) includeCulturalEconomicVisualCuesCb.checked = true;
    if (includeExternalValidationCb) includeExternalValidationCb.checked = true;
    if (includeMlImageRecognitionCb) includeMlImageRecognitionCb.checked = true;
    if (includeInfrastructureEnergyCuesCb) includeInfrastructureEnergyCuesCb.checked = true;
    if (fastPredictionCb) fastPredictionCb.checked = true;
    previewContainer.style.display = 'none';
    const area = document.getElementById('uploadArea');
    if (area) area.style.display = 'block';
    if (predictBtn) predictBtn.disabled = true;
    if (clearBtn) clearBtn.style.display = 'none';
    resultsSection.style.display = 'none';
    const summaryCard = document.getElementById('resultsSummaryCard');
    if (summaryCard) {
        summaryCard.style.display = 'none';
        summaryCard.innerHTML = '';
    }
    const interpCl = document.getElementById('interpretationCluster');
    if (interpCl) interpCl.style.display = 'none';
    const hypGr = document.getElementById('resultsGroupHypotheses');
    if (hypGr) hypGr.style.display = 'none';
    const extraGr = document.getElementById('resultsGroupExtras');
    if (extraGr) extraGr.style.display = 'none';
    const wikiSyn = document.getElementById('wikipediaSynthesisBlock');
    if (wikiSyn) {
        wikiSyn.innerHTML = '';
        wikiSyn.style.display = 'none';
    }
    document.getElementById('alternativesContainer').style.display = 'none';
    const rpr = document.getElementById('resolvedPlaceRow');
    const rpv = document.getElementById('resolvedPlaceValue');
    const rpn = document.getElementById('resolvedPlaceNote');
    if (rpr) rpr.style.display = 'none';
    if (rpv) rpv.textContent = '—';
    if (rpn) {
        rpn.textContent = '';
        rpn.style.display = 'none';
    }
    const gcc = document.getElementById('geoclipRanksContainer');
    const gcl = document.getElementById('geoclipRanksList');
    if (gcc) gcc.style.display = 'none';
    if (gcl) gcl.innerHTML = '';
    document.getElementById('featuresContainer').style.display = 'none';
    const metaDl = document.getElementById('imageMetaDl');
    const metaAside = document.getElementById('imageMetaAside');
    if (metaDl) metaDl.innerHTML = '';
    if (metaAside) metaAside.style.display = 'none';
    resetGoogleReferencePanel();
    const plantBlock = document.getElementById('plantHintsBlock');
    const plantList = document.getElementById('plantHintsList');
    if (plantBlock) plantBlock.style.display = 'none';
    if (plantList) plantList.innerHTML = '';
    const idBlock = document.getElementById('identifiedElementsBlock');
    const idList = document.getElementById('identifiedElementsList');
    const idLoc = document.getElementById('identifiedElementsLocation');
    if (idBlock) idBlock.style.display = 'none';
    if (idList) idList.innerHTML = '';
    if (idLoc) idLoc.textContent = '';
    const archBlock = document.getElementById('architectureHintsBlock');
    const archSecs = document.getElementById('architectureHintsSections');
    if (archBlock) archBlock.style.display = 'none';
    if (archSecs) archSecs.innerHTML = '';
    const globeBlock = document.getElementById('globeRegionalHintsBlock');
    const globeSecs = document.getElementById('globeRegionalHintsSections');
    const globeDetails = document.getElementById('globeRegionalHintsDetails');
    if (globeBlock) globeBlock.style.display = 'none';
    if (globeSecs) globeSecs.innerHTML = '';
    if (globeDetails) globeDetails.open = false;
    const sceneCuesBlock = document.getElementById('sceneGeolocationCuesBlock');
    const sceneCuesContent = document.getElementById('sceneGeolocationCuesContent');
    if (sceneCuesBlock) sceneCuesBlock.style.display = 'none';
    if (sceneCuesContent) sceneCuesContent.innerHTML = '';
    const manualVerifyBlock = document.getElementById('manualVerificationBlock');
    if (manualVerifyBlock) manualVerifyBlock.style.display = 'none';
    const manualScope = document.getElementById('manualVerificationScope');
    const manualList = document.getElementById('manualVerificationChecklist');
    const manualLinks = document.getElementById('manualVerificationLinks');
    if (manualScope) manualScope.textContent = '';
    if (manualList) manualList.innerHTML = '';
    if (manualLinks) manualLinks.innerHTML = '';
    const readingAxesBlock = document.getElementById('readingAxesBlock');
    if (readingAxesBlock) readingAxesBlock.style.display = 'none';
    const rav = document.getElementById('readingAxisView');
    const rab = document.getElementById('readingAxisBuildings');
    const raw = document.getElementById('readingAxisWiki');
    if (rav) rav.textContent = '';
    if (rab) rab.textContent = '';
    if (raw) raw.textContent = '';
    const extValBlock = document.getElementById('externalValidationBlock');
    const extValContent = document.getElementById('externalValidationContent');
    if (extValBlock) extValBlock.style.display = 'none';
    if (extValContent) extValContent.innerHTML = '';
    if (includeExternalValidationCb) includeExternalValidationCb.checked = true;
    const mlRecBlock = document.getElementById('mlRecognitionBlock');
    const mlRecContent = document.getElementById('mlRecognitionContent');
    if (mlRecBlock) mlRecBlock.style.display = 'none';
    if (mlRecContent) mlRecContent.innerHTML = '';
    if (includeMlImageRecognitionCb) includeMlImageRecognitionCb.checked = true;
    const infraBlock = document.getElementById('infrastructureEnergyBlock');
    const infraContent = document.getElementById('infrastructureEnergyContent');
    if (infraBlock) infraBlock.style.display = 'none';
    if (infraContent) infraContent.innerHTML = '';
    if (includeInfrastructureEnergyCuesCb) includeInfrastructureEnergyCuesCb.checked = true;
    const satBlock = document.getElementById('satelliteMatchBlock');
    const satContent = document.getElementById('satelliteMatchContent');
    if (satBlock) satBlock.style.display = 'none';
    if (satContent) satContent.innerHTML = '';
const llmBlock = document.getElementById('llmDetectiveBlock');
    const llmContent = document.getElementById('llmDetectiveContent');
    if (llmBlock) llmBlock.style.display = 'none';
    if (llmContent) llmContent.innerHTML = '';
    const svBlock = document.getElementById('streetviewVerifyBlock');
    const svContent = document.getElementById('streetviewVerifyContent');
    if (svBlock) svBlock.style.display = 'none';
    if (svContent) svContent.innerHTML = '';
    resetPipeline();
    const seasonBlock = document.getElementById('seasonTimeBlock');
    const seasonSecs = document.getElementById('seasonTimeSections');
    const seasonMet = document.getElementById('seasonTimeSkyMetrics');
    const seasonVtd = document.getElementById('seasonTimeVisualEstimate');
    if (seasonBlock) seasonBlock.style.display = 'none';
    if (seasonSecs) seasonSecs.innerHTML = '';
    if (seasonMet) seasonMet.textContent = '';
    if (seasonVtd) seasonVtd.textContent = '';
    const frBlock = document.getElementById('flowerBushRoadBlock');
    const frSecs = document.getElementById('flowerBushRoadSections');
    if (frBlock) frBlock.style.display = 'none';
    if (frSecs) frSecs.innerHTML = '';
    const intBlock = document.getElementById('integratedEstimateBlock');
    if (intBlock) intBlock.style.display = 'none';
    ['integratedHeadline', 'integratedGeo', 'integratedScene', 'integratedRec'].forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.innerHTML = '';
    });
    const agr = document.getElementById('integratedAgreement');
    const ten = document.getElementById('integratedTension');
    const lim = document.getElementById('integratedLimits');
    const aw = document.getElementById('integratedAgreementWrap');
    const tw = document.getElementById('integratedTensionWrap');
    if (agr) agr.innerHTML = '';
    if (ten) ten.innerHTML = '';
    if (lim) lim.innerHTML = '';
    if (aw) aw.style.display = 'none';
    if (tw) tw.style.display = 'none';
    const svRow = document.getElementById('streetviewRefinementRow');
    const svTxt = document.getElementById('streetviewRefinementText');
    if (svRow) svRow.style.display = 'none';
    if (svTxt) svTxt.textContent = '';
}

function resetGoogleReferencePanel() {
    const block = document.getElementById('googleMapsReplicaBlock');
    const sv = document.getElementById('googleStreetViewImg');
    const sm = document.getElementById('googleStaticMapImg');
    const hint = document.getElementById('googleMapsReplicaHint');
    if (sv) {
        sv.removeAttribute('src');
        sv.onerror = null;
        sv.alt = '';
        sv.style.display = '';
        sv.classList.remove('google-maps-replica__img--err');
    }
    if (sm) {
        sm.removeAttribute('src');
        sm.onerror = null;
        sm.alt = '';
        sm.style.display = '';
        sm.classList.remove('google-maps-replica__img--err');
    }
    if (hint) hint.textContent = '';
    if (block) block.style.display = 'none';
}

// ============================================================================
// API Calls
// ============================================================================

async function predictLocation() {
    if (!selectedImage) {
        showError('Please select an image first');
        return;
    }

    try {
        predictBtn.disabled = true;
        resultsSection.style.display = 'none';

        showProgress({
            title: 'Finding location',
            detail: 'Preparing your photo…',
            indeterminate: true,
        });
        startPredictTimeTracker();
        await flushPaintFrames();

        let serverConfig = null;
        let includeStreetviewRefinement = false;
        try {
            const cfgRes = await fetch(apiUrl('/config'));
            if (cfgRes.ok) {
                serverConfig = await cfgRes.json();
                includeStreetviewRefinement = Boolean(serverConfig.use_streetview_refinement && serverConfig.google_maps_configured);
            }
        } catch (_) {
            /* ignore */
        }

        const fastOn = fastPredictionCb != null ? fastPredictionCb.checked : true;
        updateProgress({
            detail: fastOn
                ? 'Running vision models (fast mode)…'
                : 'Running full vision + validation pipeline…',
            indeterminate: true,
        });

        startPredictStatusPoll();

        const pipelineTimers = [];
        const formData = buildPredictFormData({
            includeStreetviewRefinement,
            fastOn,
        });

        const predictTimeoutMs = getPredictFetchTimeoutMs();
        const controller = new AbortController();
        let abortTimeoutId = null;
        if (predictTimeoutMs > 0) {
            abortTimeoutId = setTimeout(() => controller.abort(), predictTimeoutMs);
        }

        const response = await fetch(apiUrl('/predict'), {
            method: 'POST',
            body: formData,
            signal: controller.signal,
        });
        if (abortTimeoutId) {
            clearTimeout(abortTimeoutId);
        }

        pipelineTimers.forEach((id) => clearTimeout(id));

        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            const detail =
                typeof data.detail === 'string'
                    ? data.detail
                    : Array.isArray(data.detail)
                      ? data.detail.map((d) => d.msg || d).join(' ')
                      : data.detail
                        ? JSON.stringify(data.detail)
                        : response.statusText;
            throw new Error(detail || `API Error: ${response.status}`);
        }

        currentPrediction = data;
        await displayResults(data);

    } catch (error) {
        console.error('Prediction error:', error);
        if (error.name === 'AbortError') {
            const capMs = getPredictFetchTimeoutMs();
            const elapsedMs = predictStartTime ? Date.now() - predictStartTime : capMs || 0;
            const elapsedStr = formatElapsedTime(elapsedMs);
            const msg =
                `Prediction was cancelled after ${elapsedStr}` +
                (capMs > 0 ? ` (your PHOTO_PREDICT_TIMEOUT_MS limit ≈ ${Math.round(capMs / 60000)} min)` : '') +
                `. Full vision on CPU often needs 1–2+ hours. Options: leave the default (no client limit), ` +
                `set PHOTO_PREDICT_TIMEOUT_MS = 0 in the console, enable **Fast prediction**, or use a GPU backend.`;
            showError(msg);
            return;
        }
        const detail = await describePredictFailure(error);
        showError(`Failed to predict location: ${detail}`);
    } finally {
        predictBtn.disabled = false;
        hideProgress();
    }
}

function getPreferredClientLanguage() {
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

function appendFormBoolean(form, key, value) {
    form.append(key, value ? 'true' : 'false');
}

function buildPredictFormData({ includeStreetviewRefinement, fastOn }) {
    const form = new FormData();
    form.append('image', selectedImage, selectedImage.name || 'upload');
    form.append('original_filename', selectedImage.name || 'upload');
    form.append('reverse_geocode_accept_language', getPreferredClientLanguage());
    appendFormBoolean(form, 'use_cloud_inference', false);
    appendFormBoolean(form, 'fast_prediction', fastOn);
    appendFormBoolean(
        form,
        'clear_prediction_cache',
        clearPredictionCacheCb != null ? clearPredictionCacheCb.checked : false,
    );
    appendFormBoolean(
        form,
        'include_llm_detective',
        includeLlmDetectiveCb != null ? includeLlmDetectiveCb.checked : true,
    );
    appendFormBoolean(
        form,
        'include_feature_analysis',
        includeFeatureAnalysisCb != null ? includeFeatureAnalysisCb.checked : true,
    );
    appendFormBoolean(
        form,
        'include_globe_regional_hints',
        includeGlobeRegionalHintsCb != null ? includeGlobeRegionalHintsCb.checked : true,
    );
    appendFormBoolean(
        form,
        'include_scene_geolocation_cues',
        includeSceneGeolocationCuesCb != null ? includeSceneGeolocationCuesCb.checked : true,
    );
    appendFormBoolean(
        form,
        'include_cultural_economic_visual_cues',
        includeCulturalEconomicVisualCuesCb != null
            ? includeCulturalEconomicVisualCuesCb.checked
            : true,
    );
    appendFormBoolean(
        form,
        'include_external_validation',
        includeExternalValidationCb != null ? includeExternalValidationCb.checked : true,
    );
    appendFormBoolean(
        form,
        'include_ml_image_recognition',
        includeMlImageRecognitionCb != null ? includeMlImageRecognitionCb.checked : true,
    );
    appendFormBoolean(
        form,
        'include_infrastructure_energy_cues',
        includeInfrastructureEnergyCuesCb != null
            ? includeInfrastructureEnergyCuesCb.checked
            : true,
    );
    appendFormBoolean(form, 'include_streetview_refinement', includeStreetviewRefinement);
    appendFormBoolean(form, 'include_reverse_geocode', true);
    return form;
}

async function describePredictFailure(error) {
    const message =
        error && typeof error.message === 'string' && error.message.trim() !== ''
            ? error.message.trim()
            : 'Unknown error';
    if (!/networkerror|failed to fetch|load failed/i.test(message)) {
        return message;
    }
    try {
        const health = await fetch(apiUrl('/health'));
        if (health.ok) {
            return `The backend is reachable, but ${apiUrl('/predict')} closed before returning a response. Check the backend console for a crash and try the upload again.`;
        }
    } catch (_) {
        return `Could not reach ${apiUrl('/predict')}. Make sure the backend is running and reachable on port 8000.`;
    }
    return message;
}

// ============================================================================
// Results Display
// ============================================================================

/** English Wikipedia payload worth showing (headings, article list, or heuristic fit scores). */
function wikipediaContextHasUsablePayload(wp) {
    if (!wp || typeof wp !== 'object') return false;
    if (String(wp.synthesized_summary || '').trim()) return true;
    if (Array.isArray(wp.articles) && wp.articles.length > 0) return true;
    if (String(wp.physical_setting_summary || '').trim()) return true;
    if (String(wp.alternative_wikipedia_note || '').trim()) return true;
    if (wp.primary_wikipedia_fit_score != null || wp.best_alternative_wikipedia_fit_score != null) {
        return true;
    }
    if (wp.primary_photo_similarity != null) return true;
    if (String(wp.wiki_match_quality || '').trim()) return true;
    return false;
}

/** Use default display for <details> so expand/collapse keeps working (avoid display:block on <details>). */
function setResultsDrillGroupVisible(el, visible) {
    if (!el) return;
    el.style.display = visible ? '' : 'none';
}

function displayWikipediaNearby(prediction) {
    const wrap = document.getElementById('wikipediaPlaceBlock');
    const synBlock = document.getElementById('wikipediaSynthesisBlock');
    const noteEl = document.getElementById('wikipediaPlaceNote');
    const adjEl = document.getElementById('wikipediaPinAdjustment');
    const listEl = document.getElementById('wikipediaArticlesList');
    if (!wrap || !listEl) return;
    const wp = prediction.wikipedia_place_context;

    if (!wikipediaContextHasUsablePayload(wp)) {
        wrap.style.display = 'none';
        listEl.innerHTML = '';
        if (synBlock) {
            synBlock.style.display = 'none';
            synBlock.innerHTML = '';
        }
        if (noteEl) noteEl.textContent = '';
        if (adjEl) {
            adjEl.style.display = 'none';
            adjEl.innerHTML = '';
        }
        return;
    }
    wrap.style.display = 'block';

    if (synBlock) {
        if (String(wp.synthesized_summary || '').trim()) {
            synBlock.innerHTML = mdBoldToHtml(wp.synthesized_summary);
            synBlock.style.display = 'block';
        } else {
            const parts = [];
            if (String(wp.physical_setting_summary || '').trim()) {
                parts.push(
                    `**Physical setting (heuristic):** ${String(wp.physical_setting_summary).trim()}`,
                );
            }
            if (String(wp.alternative_wikipedia_note || '').trim()) {
                parts.push(String(wp.alternative_wikipedia_note).trim());
            }
            if (
                wp.primary_wikipedia_fit_score != null ||
                wp.best_alternative_wikipedia_fit_score != null
            ) {
                let line = 'Wikipedia cue scores (heuristic): ';
                if (wp.primary_wikipedia_fit_score != null) {
                    line += `primary pin **${Number(wp.primary_wikipedia_fit_score).toFixed(1)}**`;
                }
                if (wp.best_alternative_wikipedia_fit_score != null) {
                    line += ` · best alternative **${Number(wp.best_alternative_wikipedia_fit_score).toFixed(1)}**`;
                    if (wp.best_alternative_wikipedia_index != null) {
                        line += ` (candidate #${Number(wp.best_alternative_wikipedia_index) + 1})`;
                    }
                }
                parts.push(line);
            }
            if (String(wp.wiki_match_quality || '').trim()) {
                parts.push(`Wiki match quality: **${String(wp.wiki_match_quality).trim()}**`);
            }
            if (wp.primary_photo_similarity != null) {
                parts.push(
                    `**Wikimedia photo CLIP:** ${(Number(wp.primary_photo_similarity) * 100).toFixed(0)}% similarity to best nearby Commons/Wikipedia image`,
                );
            }
            if (parts.length) {
                synBlock.innerHTML = mdBoldToHtml(parts.join('\n\n'));
                synBlock.style.display = 'block';
            } else {
                synBlock.innerHTML = '';
                synBlock.style.display = 'none';
            }
        }
    }

    if (noteEl && wp.note) noteEl.textContent = wp.note;
    if (adjEl) {
        if (wp.primary_pin_adjusted && wp.pin_adjustment_note) {
            adjEl.style.display = 'block';
            adjEl.innerHTML = mdBoldToHtml(wp.pin_adjustment_note);
        } else {
            adjEl.style.display = 'none';
            adjEl.innerHTML = '';
        }
    }
    const arts = Array.isArray(wp.articles) ? wp.articles : [];
    listEl.innerHTML = arts
        .map((a) => {
            const dist =
                a.distance_m != null && Number.isFinite(a.distance_m)
                    ? `${Math.round(a.distance_m)} m`
                    : 'distance n/a';
            const cues =
                Array.isArray(a.overlap_cues) && a.overlap_cues.length
                    ? ` · cues: ${escapeHtml(a.overlap_cues.join(', '))}`
                    : '';
            const snippet = escapeHtml((a.extract || '').slice(0, 520));
            const title = escapeHtml(a.title || '');
            const url = escapeHtml(a.url || '#');
            const score =
                a.relevance_score != null ? `score ${Number(a.relevance_score).toFixed(1)}` : '';
            const photoPct =
                a.photo_similarity != null && Number.isFinite(Number(a.photo_similarity))
                    ? ` · photo CLIP ${(Number(a.photo_similarity) * 100).toFixed(0)}%`
                    : '';
            const photoUrl = a.photo_match_url ? String(a.photo_match_url).trim() : '';
            const photoLink = photoUrl
                ? ` · <a href="${escapeHtml(photoUrl)}" target="_blank" rel="noopener noreferrer">matched image</a>`
                : '';
            const photoCue =
                Array.isArray(a.overlap_cues) && a.overlap_cues.includes('photo_match')
                    ? ' · visual photo match'
                    : '';
            return `<li class="wiki-nearby__item"><a href="${url}" target="_blank" rel="noopener noreferrer">${title}</a> · ${escapeHtml(dist)} ${score}${photoPct}${photoCue}${photoLink}${cues}<div class="wiki-nearby__snippet">${snippet}</div></li>`;
        })
        .join('');
}

function displayIntegratedEstimate(prediction) {
    const block = document.getElementById('integratedEstimateBlock');
    if (!block) return;
    const ie = prediction.integrated_estimate;
    if (!ie || typeof ie !== 'object') {
        block.style.display = 'none';
        return;
    }
    block.style.display = 'block';
    const headline = document.getElementById('integratedHeadline');
    const geo = document.getElementById('integratedGeo');
    const scene = document.getElementById('integratedScene');
    const rec = document.getElementById('integratedRec');
    const agrUl = document.getElementById('integratedAgreement');
    const tenUl = document.getElementById('integratedTension');
    const limUl = document.getElementById('integratedLimits');
    const aw = document.getElementById('integratedAgreementWrap');
    const tw = document.getElementById('integratedTensionWrap');
    if (headline) headline.innerHTML = mdBoldToHtml(ie.headline);
    if (geo) geo.innerHTML = mdBoldToHtml(ie.geo_narrative);
    if (scene) scene.innerHTML = mdBoldToHtml(ie.scene_narrative);
    if (rec) rec.innerHTML = mdBoldToHtml(ie.recommended_interpretation);
    if (agrUl) {
        const a = Array.isArray(ie.agreement_signals) ? ie.agreement_signals : [];
        agrUl.innerHTML = a.map((s) => `<li>${mdBoldToHtml(s)}</li>`).join('');
    }
    if (aw) aw.style.display = ie.agreement_signals && ie.agreement_signals.length ? 'block' : 'none';
    if (tenUl) {
        const t = Array.isArray(ie.tension_signals) ? ie.tension_signals : [];
        tenUl.innerHTML = t.map((s) => `<li>${mdBoldToHtml(s)}</li>`).join('');
    }
    if (tw) tw.style.display = ie.tension_signals && ie.tension_signals.length ? 'block' : 'none';
    if (limUl) {
        const L = Array.isArray(ie.limitations) ? ie.limitations : [];
        limUl.innerHTML = L.map((s) => `<li>${mdBoldToHtml(s)}</li>`).join('');
    }
}

function displayStreetViewRefinement(prediction) {
    const row = document.getElementById('streetviewRefinementRow');
    const textEl = document.getElementById('streetviewRefinementText');
    if (!row || !textEl) return;
    const r = prediction.streetview_refinement;
    if (!r || !r.attempted) {
        row.style.display = 'none';
        textEl.textContent = '';
        return;
    }
    row.style.display = 'flex';
    const thrPct =
        r.similarity_threshold != null ? (r.similarity_threshold * 100).toFixed(0) : '—';
    const simPct =
        r.best_similarity != null ? (r.best_similarity * 100).toFixed(1) : '—';
    let msg = '';
    if (r.swapped_primary && r.chosen_candidate_index != null) {
        msg = `Promoted geo candidate #${r.chosen_candidate_index + 1}: CLIP vs Street View similarity ${simPct}% (threshold ${thrPct}%).`;
    } else if (r.chosen_candidate_index === 0 && r.best_similarity != null && !r.detail) {
        msg = `Primary geo candidate matched Street View (${simPct}% ≥ ${thrPct}% threshold).`;
    } else if (r.detail) {
        msg = r.detail;
    } else {
        msg = `Evaluated ${r.candidates_evaluated} candidate(s). Best CLIP similarity ${simPct}% (threshold ${thrPct}%).`;
    }
    textEl.textContent = msg;
}

function formatLatLon(lat, lon) {
    const ns = lat >= 0 ? 'N' : 'S';
    const ew = lon >= 0 ? 'E' : 'W';
    return `${Math.abs(lat).toFixed(4)}°${ns}, ${Math.abs(lon).toFixed(4)}°${ew}`;
}

function locationSourceLabel(src) {
    switch (src) {
        case 'exif_gps':
            return 'EXIF GPS (embedded coordinates)';
        case 'filename_hint':
            return 'Filename keyword guess';
        case 'streetclip':
            return 'StreetCLIP place softmax';
        case 'vision_fusion':
            return 'Vision fusion (GeoCLIP + StreetCLIP + CLIP, server pipeline)';
        case 'hybrid_geo':
            return 'CLIP + GeoCLIP hybrid';
        case 'ensemble':
            return 'Legacy ensemble label (deprecated)';
        default:
            return src || '—';
    }
}

/**
 * Prefer legacy `location_source` when present; otherwise map API `coordinate_source`.
 * Never default to `ensemble` — that incorrectly showed the legacy mock banner when the field was omitted.
 */
function deriveLocationSourceForUi(prediction) {
    if (prediction.location_source) {
        return prediction.location_source;
    }
    const cs = prediction.coordinate_source;
    if (cs === 'exif_gps') return 'exif_gps';
    if (cs === 'filename_hint') return 'filename_hint';
    if (cs === 'vision_estimate') return 'vision_fusion';
    return 'vision_fusion';
}

function wireSummaryCopyCoords(lat, lon) {
    const btn = document.getElementById('summaryCopyCoordsBtn');
    if (!btn) return;
    btn.addEventListener('click', () => {
        const t = `${Number(lat).toFixed(5)}, ${Number(lon).toFixed(5)}`;
        const done = () => {
            const prev = btn.textContent;
            btn.textContent = 'Copied';
            setTimeout(() => {
                btn.textContent = prev;
            }, 1600);
        };
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(t).then(done).catch(() => {});
        }
    });
}

async function populateResultsSummary(prediction) {
    await mergeClientExifTimeIfNeeded(prediction);

    const card = document.getElementById('resultsSummaryCard');
    if (!card || !prediction.primary_prediction) return;
    const primary = prediction.primary_prediction;
    const lat = primary.latitude;
    const lon = primary.longitude;
    const locationSource = deriveLocationSourceForUi(prediction);

    const badges = [];
    if (prediction.has_exif_gps) {
        badges.push('<span class="result-badge result-badge--muted">EXIF GPS → primary pin</span>');
    }
    if (
        prediction.embedded_metadata_excluded_from_prediction &&
        prediction.exif_gps_present_in_image
    ) {
        badges.push(
            '<span class="result-badge result-badge--warning">EXIF GPS in file (ignored)</span>',
        );
    } else if (prediction.embedded_metadata_excluded_from_prediction) {
        badges.push(
            '<span class="result-badge result-badge--muted">Vision-only (embedded metadata off)</span>',
        );
    }
    if (locationSource === 'hybrid_geo') {
        badges.push('<span class="result-badge">CLIP + GeoCLIP</span>');
    }
    const wp = prediction.wikipedia_place_context;
    if (wp && wp.primary_pin_adjusted) {
        badges.push('<span class="result-badge result-badge--wiki">Wikipedia-assisted pin</span>');
    }
    const terr = prediction.terrain_elevation_context;
    if (terr && terr.summary && terr.summary.primary_swapped_ocean_mismatch) {
        badges.push('<span class="result-badge result-badge--wiki">Terrain consistency edit</span>');
    }
    if (terr && terr.summary && terr.summary.primary_local_relief_m != null) {
        const r = Number(terr.summary.primary_local_relief_m);
        if (Number.isFinite(r)) {
            badges.push(
                `<span class="result-badge result-badge--muted">Local relief ≈ ${r.toFixed(0)} m</span>`,
            );
        }
    }
    if (prediction.wikipedia_enabled_in_request === false) {
        badges.push(
            '<span class="result-badge result-badge--warning">Wikipedia skipped (this run)</span>',
        );
    }
    if (prediction.from_cache) {
        badges.push(
            '<span class="result-badge result-badge--warning">Cached result — enable “Clear prediction cache” for a fresh run</span>',
        );
    }
    if (prediction.fast_prediction_applied) {
        badges.push(
            '<span class="result-badge result-badge--muted">Fast prediction path</span>',
        );
    }
    if (prediction.nationality_cues_enabled_in_request === false) {
        badges.push(
            '<span class="result-badge result-badge--muted">Nationality/civic CLIP prompts off</span>',
        );
    }

    const osmFmt = formatPrimaryPlaceResolution(primary);
    const osmErr = primary.place_resolution && primary.place_resolution.error;
    const placeLine =
        (osmFmt && osmFmt.title && !osmErr ? osmFmt.title : null) ||
        [primary.city, primary.country].filter(Boolean).join(', ') ||
        'Location';
    const refKm = prediction.reference_error_km;
    const refBlock =
        refKm != null && refKm >= 0
            ? `<div><dt>Commons reference</dt><dd>${refKm.toFixed(1)} km away</dd></div>`
            : '';

    const vm = prediction.vision_models_used;
    const invPresent =
        Array.isArray(prediction.inference_models) && prediction.inference_models.length > 0;
    const vmBlock =
        !invPresent && Array.isArray(vm) && vm.length > 0
            ? `<div><dt>Vision checkpoints</dt><dd>${escapeHtml(vm.join(', '))}</dd></div>`
            : '';

    const sv = prediction.streetview_refinement;
    const svBlock =
        sv && sv.attempted
            ? `<div><dt>Street View check</dt><dd>${
                  sv.swapped_primary
                      ? 'Promoted another geo candidate'
                      : sv.best_similarity != null
                        ? `Best similarity ${(sv.best_similarity * 100).toFixed(1)}%`
                        : `Evaluated ${sv.candidates_evaluated || 0} candidate(s)`
              }</dd></div>`
            : '';

    const hf = prediction.hybrid_fusion;
    let hybridFusionMetaBlock = '';
    if (hf && typeof hf === 'object') {
        let fuseLine = `${hf.streetclip_vs_geoclip_centroid_km ?? '—'} km apart · kept place names: ${
            hf.kept_streetclip_place_labels ? 'yes' : 'no'
        } (≤${hf.label_match_max_km ?? '—'} km band) · scene tags below do not drive this math.`;
        if (hf.alt_geoclip_reconcile_note && String(hf.alt_geoclip_reconcile_note).trim()) {
            fuseLine += `\n${String(hf.alt_geoclip_reconcile_note).trim()}`;
        }
        hybridFusionMetaBlock = `<div><dt>StreetCLIP ↔ GeoCLIP fuse</dt><dd>${escapeHtml(fuseLine).replace(
            /\n/g,
            '<br/>',
        )}</dd></div>`;
    }

    const photoTimeBanner = buildPhotoTimeBannerHtml(prediction);

    const gmapsUrl = `https://www.google.com/maps?q=${encodeURIComponent(`${Number(lat)},${Number(lon)}`)}`;
    card.innerHTML = `
        <div class="results-summary-card__top">
            <div>
                <div class="results-summary-card__title">${escapeHtml(placeLine)}</div>
                <div class="results-summary-badges">${badges.join('')}</div>
                ${photoTimeBanner}
            </div>
        </div>
        <div class="results-summary-card__coords-row">
            <span>${escapeHtml(formatLatLon(lat, lon))}</span>
            <div class="results-summary-card__coord-actions">
                <a class="results-summary-card__gmap" id="summaryGoogleMapsLink" href="${escapeHtml(
                    gmapsUrl,
                )}" target="_blank" rel="noopener noreferrer">Open in Google Maps</a>
                <button type="button" class="results-summary-card__copy" id="summaryCopyCoordsBtn">Copy coords</button>
            </div>
        </div>
        <dl class="results-summary-card__meta">
            <div><dt>Pin source</dt><dd>${escapeHtml(locationSourceLabel(locationSource))}</dd></div>
            ${hybridFusionMetaBlock}
            ${refBlock}
            ${vmBlock}
            ${svBlock}
        </dl>
    `;
    card.style.display = 'block';
    wireSummaryCopyCoords(lat, lon);
}

function updateResultsGroupVisibility(prediction) {
    const ic = document.getElementById('interpretationCluster');
    if (ic) {
        const hasIE =
            prediction.integrated_estimate && typeof prediction.integrated_estimate === 'object';
        const hasWiki = wikipediaContextHasUsablePayload(prediction.wikipedia_place_context);
        setResultsDrillGroupVisible(ic, hasIE || hasWiki);
    }
    const hyp = document.getElementById('resultsGroupHypotheses');
    if (hyp) {
        const hasGeo =
            Array.isArray(prediction.geoclip_ranked_predictions) &&
            prediction.geoclip_ranked_predictions.length > 0;
        const hasAlt =
            prediction.alternative_predictions && prediction.alternative_predictions.length > 0;
        setResultsDrillGroupVisible(hyp, hasGeo || hasAlt);
    }
    const ex = document.getElementById('resultsGroupExtras');
    const feat = document.getElementById('featuresContainer');
    if (ex && feat) {
        setResultsDrillGroupVisible(ex, feat.style.display !== 'none');
    }
}

function escapeHtml(s) {
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

/** OpenStreetMap reverse-geocode line for primary pin (any city/town/village). */
function formatPrimaryPlaceResolution(primary) {
    const pr = primary && primary.place_resolution;
    if (!pr || typeof pr !== 'object') return null;
    if (pr.error) return { title: null, detail: `Lookup failed: ${String(pr.error)}`, attribution: null };
    const parts = [];
    if (pr.locality) {
        parts.push(pr.locality_kind ? `${pr.locality} (${pr.locality_kind})` : pr.locality);
    }
    if (pr.administrative_area) parts.push(pr.administrative_area);
    if (pr.country) parts.push(pr.country);
    let title = parts.length ? parts.join(' · ') : pr.display_name || null;
    if (title && pr.country_code && !String(title).includes(pr.country_code)) {
        title = `${title} · ${pr.country_code}`;
    }
    return {
        title,
        detail: pr.display_name && parts.length ? pr.display_name : null,
        attribution: pr.attribution || null,
    };
}

async function displayResults(prediction) {
    const primary = prediction && prediction.primary_prediction;
    if (!primary || typeof primary.latitude !== 'number' || typeof primary.longitude !== 'number') {
        showError('API returned no usable coordinates (primary_prediction missing). Check the JSON response in DevTools → Network.');
        return;
    }
    const locationSource = deriveLocationSourceForUi(prediction);

    if (
        locationSource === 'streetclip' ||
        locationSource === 'vision_fusion' ||
        locationSource === 'hybrid_geo'
    ) {
        recordStreetclipWarm();
    }

    const notice = document.getElementById('sourceNotice');
    if (notice) {
        notice.style.display = 'block';
        if (locationSource === 'filename_hint') {
            notice.className = 'source-notice source-notice--warning';
            notice.textContent =
                'This pin comes from your filename (keyword match). The image itself was not geolocated with a trained vision model.';
        } else if (locationSource === 'exif_gps') {
            notice.className = 'source-notice source-notice--info';
            notice.textContent =
                'This pin comes from GPS embedded in the photo (EXIF), not from analyzing the scene with AI.';
        } else if (
            locationSource === 'streetclip' ||
            locationSource === 'vision_fusion' ||
            locationSource === 'hybrid_geo'
        ) {
            notice.className = 'source-notice source-notice--success';
            let fusionExtra = '';
            const vm = prediction.vision_models_used;
            if (locationSource === 'vision_fusion' && Array.isArray(vm) && vm.length > 1) {
                fusionExtra =
                    ` Multiple Hugging Face CLIP checkpoints were fused (${vm.length}): ${vm.join(', ')}.`;
            }
            if (locationSource === 'hybrid_geo') {
                fusionExtra +=
                    ' Coordinates may blend GeoCLIP (continuous GPS model) with CLIP place labels when they agree within ~1500 km (server GEOCLIP_LABEL_MATCH_MAX_KM).';
            }
            notice.textContent =
                'This pin comes from the server vision stack (GeoCLIP neural GPS, StreetCLIP over a gazetteer, CLIP zero-shot cues, fused)—not embedded GPS in the file. Confidence is model-estimate only; similar-looking scenes anywhere can be confused. Region-scale cues are often directionally useful; exact village or street labels are not verified until you check satellite and street imagery.' +
                fusionExtra;
        } else {
            notice.className = 'source-notice source-notice--neutral';
            let msg =
                'Pin source could not be classified for display (unexpected coordinate_source). Check API response and server logs.';
            if (prediction.uninformative_filename) {
                msg +=
                    ' Your filename looks generic (e.g. Untitled, IMG_…) so no place was inferred from the name—only from placeholder pixel statistics.';
            }
            notice.textContent = msg;
        }
        if (prediction.embedded_metadata_excluded_from_prediction) {
            notice.textContent +=
                '\n\nEmbedded metadata was excluded for this run: EXIF GPS, filename hints, and the EXIF capture clock were not used by the server.' +
                ' Any time-of-day line in the summary uses pixel-based cues only—not the embedded camera time.';
            if (prediction.exif_gps_present_in_image) {
                notice.textContent +=
                    ' GPS coordinates were still present in the file but ignored for the prediction.';
            }
        }
        if (prediction.wikipedia_enabled_in_request === false) {
            notice.textContent +=
                '\n\nEnglish Wikipedia was skipped for this run (unchecked in the UI): no article geosearch, thematic matches, or Wikipedia-assisted pin blend. Coordinates usually match having Wikipedia enabled unless a rare landmark blend would have shifted the pin.';
        }
        if (prediction.nationality_cues_enabled_in_request === false) {
            notice.textContent +=
                '\n\nNationality & civic CLIP prompts were off for this run: scene-element and architecture softmax lists excluded the optional flag/postal/civic/regional-house prompts. Core geolocation is unchanged; re-enable the checkbox on upload to include those tags.';
        }
    }

    await populateResultsSummary(prediction);

    displayIntegratedEstimate(prediction);
    displayWikipediaNearby(prediction);

    const refRow = document.getElementById('referenceCompareRow');
    const refKmEl = document.getElementById('referenceErrorKm');
    if (refRow && refKmEl) {
        if (prediction.reference_error_km != null && prediction.reference_error_km >= 0) {
            refRow.style.display = 'flex';
            refKmEl.textContent = `${prediction.reference_error_km.toFixed(1)} km`;
        } else {
            refRow.style.display = 'none';
        }
    }

    const headline = resolvePrimaryLocationHeadline(primary, prediction);
    document.getElementById('primaryCountry').textContent = headline.country || '—';
    document.getElementById('primaryCity').textContent = headline.city || '—';
    const resolvedRow = document.getElementById('resolvedPlaceRow');
    const resolvedVal = document.getElementById('resolvedPlaceValue');
    const resolvedNote = document.getElementById('resolvedPlaceNote');
    const osmFmt = formatPrimaryPlaceResolution(primary);
    if (resolvedRow && resolvedVal) {
        if (osmFmt && (osmFmt.title || osmFmt.detail)) {
            resolvedRow.style.display = 'flex';
            resolvedVal.textContent = osmFmt.title || osmFmt.detail || '—';
            if (resolvedNote) {
                const bits = [];
                if (osmFmt.detail && osmFmt.title && osmFmt.detail !== osmFmt.title) {
                    bits.push(osmFmt.detail);
                }
                if (osmFmt.attribution) bits.push(osmFmt.attribution);
                if (bits.length) {
                    resolvedNote.textContent = bits.join(' — ');
                    resolvedNote.style.display = 'block';
                } else {
                    resolvedNote.textContent = '';
                    resolvedNote.style.display = 'none';
                }
            }
        } else {
            resolvedRow.style.display = 'none';
            resolvedVal.textContent = '—';
            if (resolvedNote) {
                resolvedNote.textContent = '';
                resolvedNote.style.display = 'none';
            }
        }
    }
    document.getElementById('primaryCoords').textContent = formatLatLon(primary.latitude, primary.longitude);
    
    const confidencePercent = (primary.confidence * 100).toFixed(1);
    document.getElementById('primaryConfidence').style.width = `${primary.confidence * 100}%`;
    document.getElementById('primaryConfidenceText').textContent = `${confidencePercent}%`;
    
    document.getElementById('modelUsed').textContent = prediction.model_used;
    const pinWrap = document.getElementById('inferenceModelsPinWrap');
    const pinModelsEl = document.getElementById('inferenceModelsPin');
    if (pinWrap && pinModelsEl) {
        const invHtml = buildInferenceModelsHtml(prediction);
        if (invHtml) {
            pinModelsEl.innerHTML = invHtml;
            pinWrap.style.display = 'flex';
        } else {
            pinModelsEl.innerHTML = '';
            pinWrap.style.display = 'none';
        }
    }
    const processingTimeEl = document.getElementById('processingTime');
    if (processingTimeEl) {
        processingTimeEl.textContent = `${prediction.processing_time_ms.toFixed(1)}ms`;
    }

    const coordSrcEl = document.getElementById('coordinateSource');
    if (coordSrcEl && prediction.coordinate_source) {
        const srcLabels = {
            exif_gps: 'GPS in EXIF (highest trust in coordinates when present)',
            filename_hint: 'Filename keyword (demo only)',
            vision_estimate: 'Vision / model estimate',
        };
        coordSrcEl.textContent =
            srcLabels[prediction.coordinate_source] || prediction.coordinate_source;
    }

    const accuracyNoteEl = document.getElementById('geopositionAccuracyNote');
    if (accuracyNoteEl) {
        const note = prediction.geoposition_accuracy_note;
        if (note && String(note).trim()) {
            accuracyNoteEl.textContent = note;
            accuracyNoteEl.style.display = 'block';
        } else {
            accuracyNoteEl.textContent = '';
            accuracyNoteEl.style.display = 'none';
        }
    }

    displayReadingAxes(prediction);
    displayManualVerificationGuide(prediction, locationSource);

    displayExternalValidation(prediction);
    displayMlImageRecognition(prediction);
    displayInfrastructureEnergyCues(prediction);
    displaySatelliteMatch(prediction);
    displayLlmDetective(prediction);
    displayStreetviewVerify(prediction);

    // Display map
    displayMap(
        primary.latitude,
        primary.longitude,
        (osmFmt && osmFmt.title) || primary.city || primary.country,
    );
    void updateGoogleReferencePanel(primary.latitude, primary.longitude);
    displayIdentifiedElements(prediction, primary);
    displayArchitectureHints(prediction);
    displayGlobeRegionalHints(prediction);
    displaySceneGeolocationCues(prediction);
    displaySeasonTimeHints(prediction);
    displayFlowerBushRoadHints(prediction);
    displayPlantHints(prediction);

    displayGeoClipRanks(prediction);

    const altEl = document.getElementById('alternativesContainer');
    if (prediction.alternative_predictions && prediction.alternative_predictions.length > 0) {
        displayAlternatives(prediction.alternative_predictions);
        altEl.style.display = 'block';
    } else {
        altEl.style.display = 'none';
    }

    const featEl = document.getElementById('featuresContainer');
    if (prediction.feature_analysis) {
        displayFeatureAnalysis(prediction.feature_analysis);
        featEl.style.display = 'block';
    } else {
        featEl.style.display = 'none';
    }

    displayStreetViewRefinement(prediction);

    updateResultsGroupVisibility(prediction);

    document.querySelectorAll('details.results-drill').forEach((d) => {
        d.open = false;
    });

    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function displayIdentifiedElements(prediction, primary) {
    const block = document.getElementById('identifiedElementsBlock');
    const list = document.getElementById('identifiedElementsList');
    const locEl = document.getElementById('identifiedElementsLocation');
    if (!block || !list) return;
    const els = prediction.identified_elements;
    const lat = primary.latitude;
    const lon = primary.longitude;
    if (locEl) {
        const place = [primary.city, primary.country].filter(Boolean).join(', ') || '—';
        locEl.innerHTML =
            `<span class="identified-elements__pin-label">Primary pin shown on map</span> · ` +
            `<span class="identified-elements__pin-coords">${escapeHtml(formatLatLon(lat, lon))}</span> · ` +
            `<span class="identified-elements__pin-place">${escapeHtml(place)}</span>` +
            `<span class="identified-elements__pin-note"> — independent of the tag list below (different model output).</span>`;
    }
    if (!Array.isArray(els) || els.length === 0) {
        block.style.display = 'none';
        list.innerHTML = '';
        return;
    }
    block.style.display = 'block';
    list.innerHTML = els
        .map(
            (e) =>
                `<li class="identified-elements__item"><span class="identified-elements__item-label">${escapeHtml(
                    e.label || '—',
                )}</span> <span class="identified-elements__conf">${(
                    (e.confidence != null ? e.confidence : 0) * 100
                ).toFixed(1)}%</span></li>`,
        )
        .join('');
}

function displayArchitectureHints(prediction) {
    const block = document.getElementById('architectureHintsBlock');
    const container = document.getElementById('architectureHintsSections');
    if (!block || !container) return;
    const arch = prediction.architecture_hints;
    if (!arch || typeof arch !== 'object') {
        block.style.display = 'none';
        container.innerHTML = '';
        return;
    }
    const sections = [
        { key: 'structural_edges', title: 'Structural edges / silhouette' },
        { key: 'color_palette', title: 'Color palette' },
        {
            key: 'construction_scale',
            title: 'Built finish / “spend” cues (visual only)',
        },
        { key: 'building_density', title: 'Building density' },
    ];
    const parts = [];
    let any = false;
    for (const { key, title } of sections) {
        const items = arch[key];
        if (!Array.isArray(items) || items.length === 0) continue;
        any = true;
        const lis = items
            .map(
                (e) =>
                    `<li><span class="architecture-hints__item-label">${escapeHtml(
                        e.label || '—',
                    )}</span> <span class="architecture-hints__conf">${(
                        (e.confidence != null ? e.confidence : 0) * 100
                    ).toFixed(1)}%</span></li>`,
            )
            .join('');
        parts.push(
            `<div class="architecture-hints__group"><h5 class="architecture-hints__group-title">${escapeHtml(
                title,
            )}</h5><ul class="architecture-hints__list">${lis}</ul></div>`,
        );
    }
    if (!any) {
        block.style.display = 'none';
        container.innerHTML = '';
        return;
    }
    container.innerHTML = parts.join('');
    block.style.display = 'block';
}

function displayGlobeRegionalHints(prediction) {
    const block = document.getElementById('globeRegionalHintsBlock');
    const container = document.getElementById('globeRegionalHintsSections');
    if (!block || !container) return;
    const globe = prediction.globe_regional_hints;
    if (!globe || typeof globe !== 'object') {
        block.style.display = 'none';
        container.innerHTML = '';
        return;
    }
    const regions = globe.regions;
    if (!Array.isArray(regions) || regions.length === 0) {
        block.style.display = 'none';
        container.innerHTML = '';
        return;
    }

    const parts = [];
    let any = false;

    for (const reg of regions) {
        const rTitle = reg.title || reg.region_id || 'Region';
        const cats = reg.categories;
        if (!Array.isArray(cats) || cats.length === 0) continue;

        const catParts = [];
        for (const cat of cats) {
            const items = cat.items;
            if (!Array.isArray(items) || items.length === 0) continue;
            any = true;
            const catTitle = cat.title || cat.category_id || 'Category';
            const lis = items
                .map(
                    (e) =>
                        `<li><span class="architecture-hints__item-label">${escapeHtml(
                            e.label || '—',
                        )}</span> <span class="architecture-hints__conf">${(
                            (e.confidence != null ? e.confidence : 0) * 100
                        ).toFixed(1)}%</span></li>`,
                )
                .join('');
            catParts.push(
                `<div class="architecture-hints__group"><h5 class="architecture-hints__group-title">${escapeHtml(
                    catTitle,
                )}</h5><ul class="architecture-hints__list">${lis}</ul></div>`,
            );
        }

        if (catParts.length === 0) continue;
        parts.push(
            `<div class="globe-regional-region"><h4 class="architecture-hints__group-title globe-regional-region__heading">${escapeHtml(
                rTitle,
            )}</h4>${catParts.join('')}</div>`,
        );
    }

    if (!any) {
        if (globe.note || globe.clip_available === false) {
            const st = !globe.clip_available
                ? 'CLIP regional cues are off (install torch and transformers on the server, or see the message below).'
                : '';
            const n = globe.note ? String(globe.note) : '';
            container.innerHTML = `<p class="globe-regional-fallback">${escapeHtml(st + (st && n ? ' ' : '') + n)}</p>`;
            block.style.display = 'block';
        } else {
            block.style.display = 'none';
            container.innerHTML = '';
        }
        return;
    }
    container.innerHTML = parts.join('');
    block.style.display = 'block';
}

function renderCueList(title, cues) {
    if (!Array.isArray(cues) || cues.length === 0) return '';
    const lis = cues
        .map(
            (c) =>
                `<li><span class="scene-cues__score">${((c.score != null ? c.score : 0) * 100).toFixed(
                    0,
                )}%</span> <span class="scene-cues__label">${escapeHtml(c.label || '—')}</span> ` +
                `<span class="scene-cues__src">${escapeHtml(c.source || '')}</span></li>`,
        )
        .join('');
    return `<div class="scene-cues__group"><h5 class="scene-cues__group-title">${escapeHtml(title)}</h5><ul class="scene-cues__list">${lis}</ul></div>`;
}

/**
 * True when other result panels already show overlapping CLIP / softmax / flora / light cues.
 * In that case we only show compact pixel metrics + summary to avoid duplicating lists.
 */
function otherVisualCuePanelsAlreadyPopulated(prediction) {
    const arch = prediction.architecture_hints;
    if (arch && typeof arch === 'object') {
        for (const k of ['structural_edges', 'color_palette', 'construction_scale', 'building_density']) {
            if (Array.isArray(arch[k]) && arch[k].length > 0) return true;
        }
    }
    if (Array.isArray(prediction.plant_geo_hints) && prediction.plant_geo_hints.length > 0) return true;

    const globe = prediction.globe_regional_hints;
    if (globe && typeof globe === 'object') {
        const regs = globe.regions;
        if (Array.isArray(regs)) {
            for (const reg of regs) {
                const cats = reg.categories;
                if (!Array.isArray(cats)) continue;
                for (const cat of cats) {
                    if (Array.isArray(cat.items) && cat.items.length > 0) return true;
                }
            }
        }
    }

    const st = prediction.season_time_hints;
    if (st && typeof st === 'object') {
        if (Array.isArray(st.month_band_scores) && st.month_band_scores.length > 0) return true;
        if (Array.isArray(st.month_scores) && st.month_scores.length > 0) return true;
        if (String(st.summary || '').trim()) return true;
    }
    const sk = prediction.sky_image_metrics;
    if (sk && typeof sk === 'object' && sk.mean_rgb_upper != null) return true;
    const vtd = prediction.visual_time_of_day;
    if (vtd && typeof vtd === 'object' && (vtd.bucket || String(vtd.summary || '').trim())) return true;

    const fb = prediction.flower_bush_road_hints;
    if (fb && typeof fb === 'object') {
        for (const k of ['flowers_bushes', 'road_surface']) {
            if (Array.isArray(fb[k]) && fb[k].length > 0) return true;
        }
    }

    const ie = prediction.identified_elements;
    if (Array.isArray(ie) && ie.length > 0) return true;

    return false;
}

function displaySceneGeolocationCues(prediction) {
    const block = document.getElementById('sceneGeolocationCuesBlock');
    const container = document.getElementById('sceneGeolocationCuesContent');
    if (!block || !container) return;
    const sc = prediction.scene_geolocation_cues;
    if (!sc || typeof sc !== 'object') {
        block.style.display = 'none';
        container.innerHTML = '';
        return;
    }

    const stats = sc.pixel_stats && typeof sc.pixel_stats === 'object' ? sc.pixel_stats : {};
    const statRows = Object.keys(stats)
        .map((k) => `<tr><td>${escapeHtml(k)}</td><td>${escapeHtml(String(stats[k]))}</td></tr>`)
        .join('');
    const statsHtml = statRows
        ? `<table class="scene-cues__stats"><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>${statRows}</tbody></table>`
        : '';

    const compact = otherVisualCuePanelsAlreadyPopulated(prediction);

    let html = '';
    if (compact) {
        html += `<p class="scene-cues__compact-note">Other panels above already show architecture, plant/region, globe-regional CLIP, season/light, or roadside cues. This section shows only <strong>extra normalized pixel metrics</strong> + a short summary — not repeated softmax lists.</p>`;
    } else {
        html += `<p class="scene-cues__methodology">${escapeHtml(sc.methodology || '')}</p>`;
    }

    if (sc.interpretive_summary) {
        html += `<p class="scene-cues__summary"><strong>Summary:</strong> ${escapeHtml(sc.interpretive_summary)}</p>`;
    }
    html += statsHtml;

    if (!compact) {
        html += renderCueList('Vegetation & landscape', sc.vegetation);
        html += renderCueList('Built environment', sc.built_environment);
        html += renderCueList('Palette & finish', sc.palette_and_finish);
        html += renderCueList('Climate & light', sc.climate_and_light);
        html += renderCueList('Design / upkeep (speculative)', sc.design_and_upkeep_proxy);

        if (Array.isArray(sc.clip_banks_detail) && sc.clip_banks_detail.length > 0) {
            html += `<details class="scene-cues__clip-details"><summary>CLIP cue banks (detail)</summary>`;
            for (const bank of sc.clip_banks_detail) {
                const cats = bank.categories || [];
                for (const cat of cats) {
                    const items = cat.items || [];
                    if (items.length === 0) continue;
                    html += `<div class="scene-cues__clip-bank"><strong>${escapeHtml(bank.title || '')}</strong><ul>`;
                    for (const it of items) {
                        html += `<li>${((it.confidence != null ? it.confidence : 0) * 100).toFixed(1)}% — ${escapeHtml(
                            it.label || '',
                        )}</li>`;
                    }
                    html += `</ul></div>`;
                }
            }
            html += `</details>`;
        }

        const ce = sc.cultural_economic_visual;
        if (ce && typeof ce === 'object') {
            html += `<div class="cultural-economic-visual">`;
            html += `<h4 class="cultural-economic-visual__title">Built form & street commerce (CLIP — not GDP or culture)</h4>`;
            html += `<p class="cultural-economic-visual__warn" role="alert">${escapeHtml(ce.disclaimer || '')}</p>`;
            html += `<p class="cultural-economic-visual__meth">${escapeHtml(ce.methodology || '')}</p>`;
            if (ce.clip_available === false) {
                html += `<p class="scene-cues__clip-off">These phrase banks need the same CLIP runtime as other cues.</p>`;
            }
            const cebanks = ce.clip_banks_detail || [];
            if (Array.isArray(cebanks) && cebanks.length > 0) {
                html += `<details class="scene-cues__clip-details cultural-economic-visual__banks"><summary>Phrase softmax banks (built environment / commerce / façades)</summary>`;
                for (const bank of cebanks) {
                    const cats = bank.categories || [];
                    for (const cat of cats) {
                        const items = cat.items || [];
                        if (items.length === 0) continue;
                        html += `<div class="scene-cues__clip-bank"><strong>${escapeHtml(bank.title || '')}</strong><ul>`;
                        for (const it of items) {
                            html += `<li>${((it.confidence != null ? it.confidence : 0) * 100).toFixed(1)}% — ${escapeHtml(
                                it.label || '',
                            )}</li>`;
                        }
                        html += `</ul></div>`;
                    }
                }
                html += `</details>`;
            }
            html += `</div>`;
        }
    } else {
        html += `<p class="scene-cues__json-hint">Full breakdown: <code>scene_geolocation_cues</code> in downloaded JSON (includes <code>cultural_economic_visual</code> CLIP banks when enabled — not measures of economy or society).</p>`;
    }

    let clipNote = '';
    if (!compact) {
        clipNote =
            sc.clip_available === false
                ? '<p class="scene-cues__clip-off">CLIP softmax cue banks unavailable — pixel metrics above still apply. Install <code>torch</code> + <code>transformers</code> on the server for phrase rankings.</p>'
                : `<p class="scene-cues__clip-on">CLIP softmax cue banks active — ${escapeHtml(sc.clip_model_id || 'openai/clip-vit-base-patch32')}</p>`;
    }

    container.innerHTML = clipNote + html;

    const hasVisibleBody =
        String(html).trim().length > 0 ||
        String(sc.interpretive_summary || '').trim().length > 0 ||
        statRows.length > 0;
    if (compact && !hasVisibleBody) {
        block.style.display = 'none';
        container.innerHTML = '';
        return;
    }

    block.style.display = 'block';
}

function displaySeasonTimeHints(prediction) {
    const block = document.getElementById('seasonTimeBlock');
    const container = document.getElementById('seasonTimeSections');
    const metricsEl = document.getElementById('seasonTimeSkyMetrics');
    const visualEl = document.getElementById('seasonTimeVisualEstimate');
    if (!block || !container) return;
    const hints = prediction.season_time_hints;
    const metrics = prediction.sky_image_metrics;
    const vtd = prediction.visual_time_of_day;

    if (visualEl) {
        if (vtd && typeof vtd === 'object' && (vtd.summary || vtd.bucket)) {
            const bn = VISUAL_TIME_BUCKET_LABELS[vtd.bucket] || vtd.bucket || '—';
            const ang =
                vtd.gradient_dominant_angle_deg != null &&
                Number.isFinite(Number(vtd.gradient_dominant_angle_deg))
                    ? `${Number(vtd.gradient_dominant_angle_deg).toFixed(0)}°`
                    : '—';
            const coh =
                vtd.gradient_coherence != null && Number.isFinite(Number(vtd.gradient_coherence))
                    ? `${(Number(vtd.gradient_coherence) * 100).toFixed(0)}%`
                    : '—';
            visualEl.innerHTML =
                `<strong>Visual time-of-day (heuristic):</strong> ${escapeHtml(bn)} · confidence ${(
                    (vtd.confidence != null ? vtd.confidence : 0) * 100
                ).toFixed(0)}% · lower-frame edge emphasis ${escapeHtml(ang)} (coherence ${escapeHtml(coh)}).`;
        } else {
            visualEl.textContent = '';
        }
    }

    if (metricsEl) {
        if (metrics && typeof metrics === 'object' && metrics.mean_rgb_upper != null) {
            const rgb = metrics.mean_rgb_upper;
            const rgbStr = Array.isArray(rgb)
                ? rgb.map((x) => Math.round(Number(x))).join(', ')
                : '—';
            metricsEl.innerHTML =
                `<strong>Upper-band RGB (heuristic sky strip):</strong> (${escapeHtml(rgbStr)}) · brightness ${(
                    (metrics.mean_brightness_upper != null ? metrics.mean_brightness_upper : 0) * 100
                ).toFixed(0)}% · hue bucket: ${escapeHtml(metrics.hue_bucket || '—')}`;
        } else {
            metricsEl.textContent = '';
        }
    }

    if (!hints || typeof hints !== 'object') {
        const hasMetrics = metrics && metrics.mean_rgb_upper != null;
        const hasVtd = vtd && typeof vtd === 'object' && (vtd.summary || vtd.bucket);
        container.innerHTML = '';
        block.style.display = hasMetrics || hasVtd ? 'block' : 'none';
        if (!hasMetrics && !hasVtd && metricsEl) metricsEl.textContent = '';
        return;
    }
    const sections = [
        { key: 'sky_color_light', title: 'Sky color & light' },
        { key: 'sun_position_shape', title: 'Sun position / shape (CLIP cues)' },
        { key: 'trees_wind_vegetation', title: 'Trees / leaves / wind (visual proxies)' },
        {
            key: 'month_season_estimate',
            title: 'Month stereotypes (NH temperate wording)',
        },
    ];
    const parts = [];
    let any = false;
    for (const { key, title } of sections) {
        const items = hints[key];
        if (!Array.isArray(items) || items.length === 0) continue;
        any = true;
        const lis = items
            .map(
                (e) =>
                    `<li><span class="season-time-hints__item-label">${escapeHtml(
                        e.label || '—',
                    )}</span> <span class="season-time-hints__conf">${(
                        (e.confidence != null ? e.confidence : 0) * 100
                    ).toFixed(1)}%</span></li>`,
            )
            .join('');
        parts.push(
            `<div class="season-time-hints__group"><h5 class="season-time-hints__group-title">${escapeHtml(
                title,
            )}</h5><ul class="season-time-hints__list">${lis}</ul></div>`,
        );
    }
    const showBlock =
        any ||
        (metrics && metrics.mean_rgb_upper != null) ||
        (vtd && typeof vtd === 'object' && (vtd.summary || vtd.bucket));
    if (!any) container.innerHTML = '';
    else container.innerHTML = parts.join('');
    block.style.display = showBlock ? 'block' : 'none';
}

function displayFlowerBushRoadHints(prediction) {
    const block = document.getElementById('flowerBushRoadBlock');
    const container = document.getElementById('flowerBushRoadSections');
    if (!block || !container) return;
    const hints = prediction.flower_bush_road_hints;
    if (!hints || typeof hints !== 'object') {
        block.style.display = 'none';
        container.innerHTML = '';
        return;
    }
    const sections = [
        { key: 'flowers_bushes', title: 'Flowers & bushes' },
        { key: 'road_surface', title: 'Road / path surface' },
    ];
    const parts = [];
    let any = false;
    for (const { key, title } of sections) {
        const items = hints[key];
        if (!Array.isArray(items) || items.length === 0) continue;
        any = true;
        const lis = items
            .map(
                (e) =>
                    `<li><span class="flower-bush-road-hints__item-label">${escapeHtml(
                        e.label || '—',
                    )}</span> <span class="flower-bush-road-hints__conf">${(
                        (e.confidence != null ? e.confidence : 0) * 100
                    ).toFixed(1)}%</span></li>`,
            )
            .join('');
        parts.push(
            `<div class="flower-bush-road-hints__group"><h5 class="flower-bush-road-hints__group-title">${escapeHtml(
                title,
            )}</h5><ul class="flower-bush-road-hints__list">${lis}</ul></div>`,
        );
    }
    if (!any) {
        block.style.display = 'none';
        container.innerHTML = '';
        return;
    }
    container.innerHTML = parts.join('');
    block.style.display = 'block';
}

function displayPlantHints(prediction) {
    const block = document.getElementById('plantHintsBlock');
    const list = document.getElementById('plantHintsList');
    if (!block || !list) return;
    const hints = prediction.plant_geo_hints;
    if (!Array.isArray(hints) || hints.length === 0) {
        block.style.display = 'none';
        list.innerHTML = '';
        return;
    }
    block.style.display = 'block';
    list.innerHTML = hints
        .map(
            (h) =>
                `<li class="plant-hints__item"><span class="plant-hints__prompt">${escapeHtml(
                    h.plant_prompt || '—',
                )}</span> <span class="plant-hints__region">→ ${escapeHtml(
                    h.native_region || '—',
                )}</span> <span class="plant-hints__conf">${(
                    (h.confidence != null ? h.confidence : 0) * 100
                ).toFixed(1)}%</span> <span class="plant-hints__ll">${formatLatLon(
                    h.latitude,
                    h.longitude,
                )}</span></li>`,
        )
        .join('');
}

function googleMapsSatelliteUrl(lat, lon) {
    return `https://www.google.com/maps/@${Number(lat)},${Number(lon)},18z/data=!3m1!1e3`;
}

function googleMapsStreetViewBrowseUrl(lat, lon) {
    return `https://www.google.com/maps/@?api=1&map_action=pano&viewpoint=${Number(lat)},${Number(lon)}`;
}

const MANUAL_VERIFICATION_CHECKLIST = [
    'Open the predicted coordinates in satellite imagery (roof layout, fields, tree lines, relief).',
    'Open the same point in street-level imagery where coverage exists (facades, poles, curbs).',
    'Yellow outdoor gas pipes crossing roads or along façades (common in parts of Eastern Europe / former USSR).',
    'Hillside orientation — which slope is built up vs open sky in the photo.',
    'Road curvature and intersection geometry vs the map.',
    'Utility pole spacing, crossarms, and wire density along the street.',
    'House placement — setbacks, gates, garden plots, and spacing between roofs.',
];

function displayManualVerificationGuide(prediction, locationSource) {
    const block = document.getElementById('manualVerificationBlock');
    const scopeEl = document.getElementById('manualVerificationScope');
    const listEl = document.getElementById('manualVerificationChecklist');
    const linksEl = document.getElementById('manualVerificationLinks');
    if (!block || !scopeEl || !listEl || !linksEl) return;

    const primary = prediction && prediction.primary_prediction;
    const lat = primary && Number(primary.latitude);
    const lon = primary && Number(primary.longitude);
    const hasCoords = Number.isFinite(lat) && Number.isFinite(lon);

    if (locationSource === 'filename_hint') {
        block.style.display = 'none';
        return;
    }

    if (locationSource === 'exif_gps') {
        scopeEl.textContent =
            'GPS in the file is the strongest coordinate signal here. The resolved place name from reverse geocoding can still be wrong when accuracy is coarse — confirm the label in imagery if the name matters.';
    } else {
        scopeEl.textContent =
            'Region-level prediction from pixels is often useful (climate, relief, broad built form). Exact village or street identification is not verified — treat named places as hypotheses until imagery agrees.';
    }

    listEl.innerHTML = MANUAL_VERIFICATION_CHECKLIST.map((item) => `<li>${escapeHtml(item)}</li>`).join('');

    if (hasCoords) {
        const sat = googleMapsSatelliteUrl(lat, lon);
        const sv = googleMapsStreetViewBrowseUrl(lat, lon);
        const maps = `https://www.google.com/maps?q=${encodeURIComponent(`${lat},${lon}`)}`;
        linksEl.innerHTML = [
            `<a href="${escapeHtml(sat)}" target="_blank" rel="noopener noreferrer">Open satellite view</a>`,
            `<a href="${escapeHtml(sv)}" target="_blank" rel="noopener noreferrer">Open Street View</a>`,
            `<a href="${escapeHtml(maps)}" target="_blank" rel="noopener noreferrer">Open in Google Maps</a>`,
        ].join('');
    } else {
        linksEl.textContent = 'Coordinates unavailable — cannot open map links.';
    }

    block.style.display = 'block';
}

function displayReadingAxes(prediction) {
    const block = document.getElementById('readingAxesBlock');
    const elV = document.getElementById('readingAxisView');
    const elB = document.getElementById('readingAxisBuildings');
    const elW = document.getElementById('readingAxisWiki');
    if (!block || !elV || !elB || !elW) return;
    const axes = prediction.geolocation_reading_axes;
    if (!axes || typeof axes !== 'object') {
        block.style.display = 'none';
        elV.textContent = '';
        elB.textContent = '';
        elW.textContent = '';
        return;
    }
    elV.textContent = axes.perspective_of_view || '—';
    elB.textContent = axes.building_proportions || '—';
    elW.textContent = axes.estimated_wikipedia || '—';
    block.style.display = 'block';
}

function displayExternalValidation(prediction) {
    const block = document.getElementById('externalValidationBlock');
    const content = document.getElementById('externalValidationContent');
    if (!block || !content) return;
    const ev = prediction.external_validation;
    if (!ev || typeof ev !== 'object') {
        block.style.display = 'none';
        content.innerHTML = '';
        return;
    }
    if (!ev.enabled) {
        block.style.display = 'block';
        let reasonText = ev.skipped_reason || 'skipped';
        if (ev.skipped_reason === 'disabled_in_request') {
            reasonText = 'Cross-check was turned off for this run.';
        } else if (ev.skipped_reason === 'not_ensemble_source') {
            reasonText =
                'Cross-check runs only for vision ensemble predictions (not EXIF GPS or filename hints).';
        }
        const note = ev.summary_note ? `<p class="external-validation__note">${escapeHtml(ev.summary_note)}</p>` : '';
        content.innerHTML = `<p class="external-validation__muted">${escapeHtml(reasonText)}</p>${note}`;
        return;
    }

    const idx = ev.selected_candidate_index ?? 0;
    const semRows = ev.wikipedia_semantic_checks || [];
    const semActive = semRows.some(
        (r) =>
            r &&
            typeof r.detail === 'string' &&
            !r.detail.includes('semantic gate disabled') &&
            !r.detail.includes('no image array'),
    );
    const adjustedNote = ev.pin_adjusted
        ? semActive
            ? 'The map pin may have been promoted from an alternative candidate after Wikipedia, relief, and optional CLIP-vs-Wikipedia-lead checks.'
            : 'The map pin may have been promoted from an alternative candidate after Wikipedia and relief checks passed.'
        : semActive
          ? 'The primary candidate passed Wikipedia, relief, and semantic checks first (when enabled), or no alternative scored better.'
          : 'The primary candidate passed both checks first, or no alternative scored better under the same rules.';

    const wikiRows = (ev.wikipedia_checks || [])
        .map((row) => {
            const ok = row.proven ? '✓' : '✗';
            const cls = row.proven ? 'external-validation__ok' : 'external-validation__bad';
            const dist =
                row.nearest_distance_m != null
                    ? `${escapeHtml(String(Math.round(Number(row.nearest_distance_m))))} m`
                    : '—';
            return `<tr><td>#${escapeHtml(String(row.candidate_index ?? ''))}</td><td class="${cls}">${ok}</td><td>${escapeHtml(String(row.articles_found ?? '—'))}</td><td>${escapeHtml(row.nearest_title || '—')}</td><td>${dist}</td><td>${escapeHtml(row.detail || '')}</td></tr>`;
        })
        .join('');

    const reliefRows = (ev.relief_checks || [])
        .map((row) => {
            const ok = row.proven ? '✓' : '✗';
            const cls = row.proven ? 'external-validation__ok' : 'external-validation__bad';
            const elev =
                row.center_elevation_m != null
                    ? `${escapeHtml(String(Math.round(Number(row.center_elevation_m))))} m`
                    : '—';
            const rel =
                row.local_relief_m != null
                    ? `${escapeHtml(String(Math.round(Number(row.local_relief_m))))} m`
                    : '—';
            return `<tr><td>#${escapeHtml(String(row.candidate_index ?? ''))}</td><td class="${cls}">${ok}</td><td>${elev}</td><td>${rel}</td><td>${escapeHtml(row.detail || '')}</td></tr>`;
        })
        .join('');

    const semanticRows = semRows
        .map((row) => {
            const ok = row.proven ? '✓' : '✗';
            const cls = row.proven ? 'external-validation__ok' : 'external-validation__bad';
            const sim =
                row.similarity != null && !Number.isNaN(Number(row.similarity))
                    ? Number(row.similarity).toFixed(3)
                    : '—';
            const thr =
                row.threshold != null && !Number.isNaN(Number(row.threshold))
                    ? Number(row.threshold).toFixed(3)
                    : '—';
            const bestArt = row.best_semantic_title || row.nearest_title || '—';
            const scanned =
                row.titles_scanned != null || row.titles_cap != null
                    ? `${escapeHtml(String(row.titles_scanned ?? '—'))}/${escapeHtml(String(row.titles_cap ?? '—'))}`
                    : '—';
            return `<tr><td>#${escapeHtml(String(row.candidate_index ?? ''))}</td><td class="${cls}">${ok}</td><td>${sim}</td><td>${thr}</td><td>${escapeHtml(String(bestArt))}</td><td>${scanned}</td><td>${escapeHtml(row.detail || '')}</td></tr>`;
        })
        .join('');

    const proofBanner =
        ev.proof_satisfied === false
            ? `<p class="external-validation__note" style="border-left:3px solid #c45c26;padding-left:10px;margin-bottom:12px;" role="alert"><strong>Incomplete proof:</strong> no fusion candidate passed Wikipedia + relief + semantic checks together. The pin stays on the model’s primary guess — treat locations cautiously.</p>`
            : '';

    content.innerHTML = `
        ${proofBanner}
        <p class="external-validation__lead">${escapeHtml(ev.summary_note || '')}</p>
        <p class="external-validation__meta">Selected candidate index <strong>${escapeHtml(String(idx))}</strong>. ${escapeHtml(adjustedNote)}</p>
        <div class="external-validation__tables">
            <div class="external-validation__table-wrap">
                <h5 class="external-validation__sub">Wikipedia (English geosearch)</h5>
                <table class="external-validation__table">
                    <thead><tr><th>Cand.</th><th>OK</th><th>Articles</th><th>Nearest title</th><th>Dist.</th><th>Detail</th></tr></thead>
                    <tbody>${wikiRows || '<tr><td colspan="6">No rows</td></tr>'}</tbody>
                </table>
            </div>
            <div class="external-validation__table-wrap">
                <h5 class="external-validation__sub">OpenTopoData (SRTM grid relief)</h5>
                <table class="external-validation__table">
                    <thead><tr><th>Cand.</th><th>OK</th><th>Elevation</th><th>Local relief</th><th>Detail</th></tr></thead>
                    <tbody>${reliefRows || '<tr><td colspan="5">No rows</td></tr>'}</tbody>
                </table>
            </div>
            <div class="external-validation__table-wrap">
                <h5 class="external-validation__sub">CLIP vs Wikipedia leads (best of N nearest articles)</h5>
                <table class="external-validation__table">
                    <thead><tr><th>Cand.</th><th>OK</th><th>Best CLIP</th><th>Threshold</th><th>Best article</th><th>Titles scanned</th><th>Detail</th></tr></thead>
                    <tbody>${semanticRows || '<tr><td colspan="7">No rows</td></tr>'}</tbody>
                </table>
            </div>
        </div>
        <p class="external-validation__fineprint">“OK” here means consistency with open geodata near the coordinates, not proof the image was captured at that pin.</p>
    `;
    block.style.display = 'block';
}

function displayMlImageRecognition(prediction) {
    const block = document.getElementById('mlRecognitionBlock');
    const content = document.getElementById('mlRecognitionContent');
    if (!block || !content) return;
    const ml = prediction.ml_image_recognition;
    if (!ml || typeof ml !== 'object') {
        block.style.display = 'none';
        content.innerHTML = '';
        return;
    }
    const mid = ml.model_id ? escapeHtml(ml.model_id) : '—';
    const methodology = ml.methodology ? `<p class="ml-recognition__method">${escapeHtml(ml.methodology)}</p>` : '';
    const labels = Array.isArray(ml.scene_and_object_labels) ? ml.scene_and_object_labels : [];
    if (!labels.length && !ml.clip_available) {
        content.innerHTML = `
            ${methodology}
            <p class="ml-recognition__note">CLIP is not available on this server.</p>
        `;
        block.style.display = 'block';
        return;
    }
    const rows = labels
        .map((row) => {
            const pct = row.score != null ? (Number(row.score) * 100).toFixed(2) : '—';
            return `<li class="ml-recognition__item"><span class="ml-recognition__label">${escapeHtml(row.label || '—')}</span> <span class="ml-recognition__score">${pct}%</span></li>`;
        })
        .join('');
    content.innerHTML = `
        <p class="ml-recognition__meta">Model: <code class="ml-recognition__code">${mid}</code></p>
        ${methodology}
        <ol class="ml-recognition__list">${rows}</ol>
        <p class="ml-recognition__fineprint">Scores are softmax probabilities over the fixed prompt list only — not a full object detector.</p>
    `;
    block.style.display = 'block';
}

function displayInfrastructureEnergyCues(prediction) {
    const block = document.getElementById('infrastructureEnergyBlock');
    const content = document.getElementById('infrastructureEnergyContent');
    if (!block || !content) return;
    const ie = prediction.infrastructure_energy_cues;
    if (!ie || typeof ie !== 'object') {
        block.style.display = 'none';
        content.innerHTML = '';
        return;
    }

    const mid = ie.clip_model_id ? escapeHtml(String(ie.clip_model_id)) : '—';
    const summary = ie.interpretive_summary
        ? `<p class="infra-energy__summary">${escapeHtml(ie.interpretive_summary)}</p>`
        : '';

    if (ie.enabled === false || ie.skipped_reason === 'disabled_in_settings') {
        content.innerHTML = `
            <p class="infra-energy__muted">Infrastructure-energy CLIP bundle is turned off on the server (<code>USE_INFRASTRUCTURE_ENERGY_CLIP=false</code>).</p>
            ${ie.methodology ? `<p class="infra-energy__method">${escapeHtml(ie.methodology)}</p>` : ''}
        `;
        block.style.display = 'block';
        return;
    }

    const meth = ie.methodology ? `<p class="infra-energy__method">${escapeHtml(ie.methodology)}</p>` : '';
    const disc = ie.disclaimer ? `<p class="infra-energy__disc">${escapeHtml(ie.disclaimer)}</p>` : '';

    if (!ie.clip_available && ie.skipped_reason === 'torch_transformers_missing') {
        content.innerHTML = `
            <p class="infra-energy__meta">CLIP model id: <code class="infra-energy__code">${mid}</code></p>
            ${meth}
            ${disc}
            <p class="infra-energy__note">PyTorch / transformers not available on this server.</p>
        `;
        block.style.display = 'block';
        return;
    }

    const banks = Array.isArray(ie.clip_banks_detail) ? ie.clip_banks_detail : [];
    const bankHtml = banks
        .map((bank) => {
            const title = escapeHtml(bank.title || bank.bank_id || 'Bank');
            const cats = bank.categories || [];
            const items = (cats[0] && cats[0].items) || [];
            const lis = items
                .slice(0, 10)
                .map((it) => {
                    const pct =
                        it.confidence != null ? (Number(it.confidence) * 100).toFixed(1) : '—';
                    return `<li><span class="infra-energy__lab">${escapeHtml(it.label || '—')}</span> <span class="infra-energy__pct">${pct}%</span></li>`;
                })
                .join('');
            return `<div class="infra-energy__bank"><h5 class="infra-energy__bank-title">${title}</h5><ol class="infra-energy__ol">${lis}</ol></div>`;
        })
        .join('');

    content.innerHTML = `
        <p class="infra-energy__meta">CLIP model: <code class="infra-energy__code">${mid}</code> · top-N per bank from server <code>INFRASTRUCTURE_ENERGY_CLIP_TOP_N</code></p>
        ${meth}
        ${disc}
        ${summary}
        ${bankHtml || '<p class="infra-energy__muted">No bank data returned.</p>'}
        <p class="infra-energy__fineprint">Scores are softmax-normalized within each bank only (gas vs solar banks are not comparable). Does not locate buried pipes.</p>
    `;
    block.style.display = 'block';
}

function displaySatelliteMatch(prediction) {
    const block = document.getElementById('satelliteMatchBlock');
    const content = document.getElementById('satelliteMatchContent');
    if (!block || !content) return;
    const sm = prediction.inference_debug?.satellite_match;
    if (!sm || typeof sm !== 'object') {
        block.style.display = 'none';
        content.innerHTML = '';
        return;
    }
    if (!sm.enabled) {
        block.style.display = 'block';
        content.innerHTML = `
            <p class="satellite-match__muted">${escapeHtml(sm.summary || 'Satellite matching skipped.')}</p>
        `;
        return;
    }
    const cmp = sm.comparison || {};
    const score = cmp.overall_match_score ?? '—';
    const interp = cmp.interpretation || 'unknown';
    const scoreColor = interp === 'strong_mismatch' ? 'var(--pipeline-danger)' : interp === 'weak_mismatch' ? 'var(--pipeline-warning)' : 'var(--pipeline-success)';
    content.innerHTML = `
        <p class="satellite-match__meta">Source: <code>${escapeHtml(sm.source || '—')}</code> · zoom ${sm.zoom || '—'}</p>
        <p class="satellite-match__score" style="color:${scoreColor}">
            Match score: <strong>${score}</strong> (${escapeHtml(interp)})
        </p>
        <p class="satellite-match__summary">${escapeHtml(sm.summary || '')}</p>
        <p class="satellite-match__fineprint">Automated histogram/vegetation match at the pin — not proof of capture location. Use the verification checklist above for human confirmation.</p>
        ${cmp.histogram_similarity != null ? `
        <div class="satellite-match__metrics">
            <div>Histogram: ${cmp.histogram_similarity}</div>
            <div>Vegetation: ${cmp.vegetation_match}</div>
            <div>Edge density: ${cmp.edge_density_match}</div>
            <div>Brightness: ${cmp.brightness_match}</div>
        </div>` : ''}
    `;
    block.style.display = 'block';
}

function isGeoclipModelLabel(city, country) {
    const c = String(city || '').toLowerCase();
    const co = String(country || '').toLowerCase();
    return (
        co.includes('geoclip') ||
        co.includes('vision gps estimate') ||
        c.includes('geoclip rank')
    );
}

function formatProgressPlace(candidate) {
    if (!candidate || typeof candidate !== 'object') return '';
    const shown = candidate.display_place || candidate.place;
    if (shown && !/geoclip\s+rank/i.test(String(shown))) return String(shown);
    if (shown) return String(shown);
    return '';
}

function resolvePrimaryLocationHeadline(primary, prediction) {
    const osm = formatPrimaryPlaceResolution(primary);
    if (osm && osm.title) {
        return {
            city: osm.title,
            country: (primary.place_resolution && primary.place_resolution.country) || primary.country || '—',
        };
    }
    if (!isGeoclipModelLabel(primary.city, primary.country)) {
        const city = (primary.city || '').trim();
        const country = (primary.country || '').trim();
        if (city && country && city.toLowerCase() !== country.toLowerCase()) {
            return { city, country };
        }
    }
    const alts = prediction && prediction.alternative_predictions;
    if (Array.isArray(alts)) {
        const ranked = [...alts]
            .filter((a) => a && !isGeoclipModelLabel(a.city, a.country))
            .filter((a) => {
                const c = (a.city || '').trim();
                const co = (a.country || '').trim();
                return c && co && c.toLowerCase() !== co.toLowerCase();
            })
            .sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
        const named = ranked[0];
        if (named && (named.confidence || 0) >= (primary.confidence || 0)) {
            return { city: named.city || '—', country: named.country || '—' };
        }
    }
    const lat = Number(primary.latitude);
    const lon = Number(primary.longitude);
    if (Number.isFinite(lat) && Number.isFinite(lon)) {
        return {
            city: formatLatLon(lat, lon) || '—',
            country: 'GPS estimate',
        };
    }
    return { city: '—', country: '—' };
}

function isPlaceholderOllamaThought(text) {
    const t = String(text || '').trim().toLowerCase();
    if (!t || t.length < 3) return true;
    if (/^(clue|contradiction|location|evidence)\s*#?\s*\d+$/.test(t)) return true;
    if (t === 'string explaining confidence') return true;
    if (/\b(clue|evidence|contradiction|location)\s+\d+\b/.test(t) && t.length < 48) return true;
    if (/^need:\s*one sentence/i.test(t)) return true;
    if (/geoclip\s+rank\s*\d/i.test(t) && t.length < 80) return true;
    if (/^fits:\s*visual clue:/i.test(t) && t.length < 90) return true;
    if (/^infrastructure type:\s*urban$/i.test(t) && t.length < 40) return true;
    return false;
}

function collectOllamaKeyThoughts(ld) {
    if (!ld || typeof ld !== 'object') return [];
    if (Array.isArray(ld.key_thoughts) && ld.key_thoughts.length > 0) {
        return ld.key_thoughts
            .map((t) => String(t))
            .filter((t) => !isPlaceholderOllamaThought(t));
    }
    const out = [];
    if (ld.detective_summary) out.push(String(ld.detective_summary));
    (ld.strongest_clues || []).forEach((c) => out.push(String(c)));
    (ld.contradictions || []).forEach((c) => out.push(`⚠ ${c}`));
    (ld.most_consistent_locations || []).forEach((c) => {
        const t = String(c);
        if (isPlaceholderOllamaThought(t) || isGeoclipModelLabel(t, '')) return;
        out.push(`Fits: ${t}`);
    });
    if (ld.confidence_assessment) out.push(String(ld.confidence_assessment));
    if (!out.length && ld.summary) out.push(String(ld.summary));
    return out;
}

function displayLlmDetective(prediction) {
    const block = document.getElementById('llmDetectiveBlock');
    const content = document.getElementById('llmDetectiveContent');
    if (!block || !content) return;
    const ld = prediction.inference_debug?.llm_detective;
    if (!ld || typeof ld !== 'object') {
        block.style.display = 'none';
        content.innerHTML = '';
        return;
    }
    const keyThoughts = collectOllamaKeyThoughts(ld);
    const model = escapeHtml(ld.model || '—');
    const synthNote =
        ld.llm_enhanced === false || ld.synthesized
            ? '<p class="llm-detective__hint">Bullets below are from vision cues and fusion ranks — not verified Ollama reasoning. Install a larger model or fix Ollama errors for real LLM text.</p>'
            : '';
    const keyList =
        keyThoughts.length > 0
            ? `<ul class="llm-detective__key-thoughts">${keyThoughts
                  .map((t) => `<li>${escapeHtml(t)}</li>`)
                  .join('')}</ul>`
            : `<p class="llm-detective__muted">${escapeHtml(ld.summary || 'No thoughts returned.')}</p>`;

    if (!ld.enabled) {
        block.style.display = 'block';
        content.innerHTML = `
            <p class="llm-detective__model">Ollama: <code>${model}</code></p>
            <div class="llm-detective__key-wrap">
                <h5 class="llm-detective__key-heading">Key thoughts</h5>
                ${keyList}
            </div>
            ${ld.skipped_reason === 'ollama_query_failed' && /500/i.test(ld.summary || '')
                ? `<p class="llm-detective__hint">HTTP 500 on CPU often means qwen2.5:7b ran out of RAM. Set <code>OLLAMA_MODEL=tinyllama:1.1b</code> in <code>backend/.env</code>.</p>`
                : `<p class="llm-detective__hint">Run <code>ollama serve</code> and <code>ollama pull qwen2.5:7b</code></p>`}
        `;
        block.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        return;
    }
    const clues = Array.isArray(ld.strongest_clues) ? ld.strongest_clues : [];
    const contradictions = Array.isArray(ld.contradictions) ? ld.contradictions : [];
    const consistent = Array.isArray(ld.most_consistent_locations) ? ld.most_consistent_locations : [];
    const evidence = Array.isArray(ld.additional_evidence_needed) ? ld.additional_evidence_needed : [];
    content.innerHTML = `
        <p class="llm-detective__model">Ollama model: <code>${model}</code></p>
        ${synthNote}
        <div class="llm-detective__key-wrap">
            <h5 class="llm-detective__key-heading">Key thoughts</h5>
            ${keyList}
        </div>
        ${clues.length ? `<div class="llm-detective__section"><h5>Strongest clues</h5><ul>${clues.map(c => `<li>${escapeHtml(c)}</li>`).join('')}</ul></div>` : ''}
        ${contradictions.length ? `<div class="llm-detective__section llm-detective__section--warning"><h5>⚠️ Contradictions</h5><ul>${contradictions.map(c => `<li>${escapeHtml(c)}</li>`).join('')}</ul></div>` : ''}
        ${consistent.length ? `<div class="llm-detective__section"><h5>Most consistent locations</h5><ul>${consistent.map(c => `<li>${escapeHtml(c)}</li>`).join('')}</ul></div>` : ''}
        ${ld.confidence_assessment ? `<div class="llm-detective__section"><h5>Confidence assessment</h5><p>${escapeHtml(ld.confidence_assessment)}</p></div>` : ''}
        ${evidence.length ? `<div class="llm-detective__section"><h5>Additional evidence needed</h5><ul>${evidence.map(c => `<li>${escapeHtml(c)}</li>`).join('')}</ul></div>` : ''}
    `;
    block.style.display = 'block';
}

function displayStreetviewVerify(prediction) {
    const block = document.getElementById('streetviewVerifyBlock');
    const content = document.getElementById('streetviewVerifyContent');
    if (!block || !content) return;
    const sv = prediction.inference_debug?.streetview_verification;
    if (!sv || typeof sv !== 'object') {
        block.style.display = 'none';
        content.innerHTML = '';
        return;
    }
    if (!sv.enabled) {
        block.style.display = 'block';
        content.innerHTML = `
            <p class="streetview-verify__muted">${escapeHtml(sv.summary || 'Street View verification skipped.')}</p>
            <p class="streetview-verify__hint">Set <code>GOOGLE_MAPS_API_KEY</code> with Street View Static API enabled, or use the manual verification links above (satellite + Street View browse).</p>
        `;
        return;
    }
    const swapped = sv.swapped_primary;
    const bestSim = sv.best_similarity != null ? sv.best_similarity : '—';
    const threshold = sv.similarity_threshold || 0.72;
    const statusColor = swapped ? 'var(--pipeline-success)' : (bestSim >= threshold ? 'var(--pipeline-success)' : 'var(--pipeline-warning)');

    const perCand = Array.isArray(sv.per_candidate) ? sv.per_candidate : [];
    const candHtml = perCand.map((c) => {
        const isBest = c.index === sv.best_candidate_index;
        const badge = isBest ? '<span class="streetview-verify__best">★ BEST</span>' : '';
        const sim = c.best_similarity != null ? c.best_similarity : 'N/A';
        return `<div class="streetview-verify__cand ${isBest ? 'streetview-verify__cand--best' : ''}">
            <span class="streetview-verify__cand-label">Candidate #${c.index} ${badge}</span>
            <span class="streetview-verify__cand-sim">CLIP sim: ${sim}</span>
            <span class="streetview-verify__cand-head">Heading: ${c.best_heading != null ? c.best_heading + '°' : 'default'}</span>
        </div>`;
    }).join('');

    content.innerHTML = `
        <p class="streetview-verify__status" style="color:${statusColor}">
            <strong>${swapped ? '✓ Promoted alternative' : '✓ Primary confirmed'}</strong>
            · Best similarity: ${bestSim} (threshold ${threshold})
        </p>
        <p class="streetview-verify__detail">${escapeHtml(sv.detail || '')}</p>
        <div class="streetview-verify__candidates">${candHtml}</div>
        <p class="streetview-verify__fineprint">Automated CLIP vs nearest panorama — not a substitute for checking road layout, poles, and pipes in the checklist above.</p>
    `;
    block.style.display = 'block';
}

async function updateGoogleReferencePanel(lat, lon) {
    const block = document.getElementById('googleMapsReplicaBlock');
    const sv = document.getElementById('googleStreetViewImg');
    const sm = document.getElementById('googleStaticMapImg');
    const hint = document.getElementById('googleMapsReplicaHint');
    if (!block || !sv || !sm || !hint) return;

    let configured = false;
    try {
        const r = await fetch(apiUrl('/config'));
        if (r.ok) {
            const c = await r.json();
            configured = Boolean(c.google_maps_configured);
        }
    } catch {
        configured = false;
    }

    if (!configured) {
        block.style.display = 'none';
        hint.textContent = '';
        sv.removeAttribute('src');
        sm.removeAttribute('src');
        return;
    }

    block.style.display = 'block';
    hint.textContent =
        'Nearest Street View panorama and map at the predicted coordinates (not a pixel match to your upload). Requires Google Maps Static API + Street View Static API on your key.';
    const ts = Date.now();
    const base = apiUrl('/maps');
    sv.alt = `Street View near ${formatLatLon(lat, lon)}`;
    sm.alt = `Map at ${formatLatLon(lat, lon)}`;
    sv.onerror = null;
    sm.onerror = null;
    sv.style.display = '';
    sm.style.display = '';
    sv.classList.remove('google-maps-replica__img--err');
    sm.classList.remove('google-maps-replica__img--err');

    sv.src = `${base}/streetview?latitude=${encodeURIComponent(lat)}&longitude=${encodeURIComponent(lon)}&_=${ts}`;
    sm.src = `${base}/staticmap?latitude=${encodeURIComponent(lat)}&longitude=${encodeURIComponent(lon)}&_=${ts}`;

    const onErr = (el, label) => {
        el.onerror = () => {
            el.classList.add('google-maps-replica__img--err');
            el.alt = `${label} could not be loaded. Check server key, API enablement, and billing.`;
        };
    };
    onErr(sv, 'Street View');
    onErr(sm, 'Map');
}

function displayMap(lat, lon, title) {
    const mapContainer = document.getElementById('mapContainer');
    
    // Simple map visualization using a service
    const zoomLevel = 10;
    const mapImage = `https://api.mapbox.com/styles/v1/mapbox/light-v11/static/${lon},${lat},${zoomLevel},0/800x400@2x?access_token=placeholder`;
    
    // Alternative: Create a simple SVG map
    mapContainer.innerHTML = `
        <div style="display: flex; align-items: center; justify-content: center; height: 100%; background: linear-gradient(135deg, #e0e7ff 0%, #f3e8ff 100%);">
            <div style="text-align: center; padding: 6px;">
                <div style="font-size: 1.25rem; margin-bottom: 4px;">📍</div>
                <p style="font-size: 0.9rem; font-weight: 600; margin-bottom: 2px;">${title}</p>
                <p style="color: #64748b; font-family: ui-monospace, monospace; font-size: 0.75rem;">${formatLatLon(lat, lon)}</p>
                <p style="color: #94a3b8; font-size: 0.72rem; margin-top: 6px;">
                    <a href="https://maps.google.com/?q=${lat},${lon}" target="_blank" rel="noopener" style="color: #2563eb; text-decoration: none;">
                        Google Maps →
                    </a>
                </p>
            </div>
        </div>
    `;
}

function displayGeoClipRanks(prediction) {
    const wrap = document.getElementById('geoclipRanksContainer');
    const container = document.getElementById('geoclipRanksList');
    if (!wrap || !container) return;
    const ranks = prediction.geoclip_ranked_predictions;
    if (!Array.isArray(ranks) || ranks.length === 0) {
        wrap.style.display = 'none';
        container.innerHTML = '';
        return;
    }
    wrap.style.display = 'block';
    container.innerHTML = ranks.map((row, index) => {
        const pct = row.confidence != null ? (row.confidence * 100).toFixed(1) : '—';
        const label = escapeHtml(row.city || `Rank ${index + 1}`);
        const lat = Number(row.latitude);
        const lon = Number(row.longitude);
        const mapsLink =
            Number.isFinite(lat) && Number.isFinite(lon)
                ? `<br><a class="prediction-maps-link" href="https://www.google.com/maps?q=${encodeURIComponent(
                      `${lat},${lon}`,
                  )}" target="_blank" rel="noopener noreferrer">Open in Google Maps</a>`
                : '';
        return `
        <div class="alternative-item">
            <div>
                <div class="label">${index === 0 ? 'Rank 1 (top)' : `Rank ${index + 1}`}</div>
                <div class="value">${label}</div>
            </div>
            <div>
                <div class="label">Coordinates</div>
                <div class="value" style="font-size: 0.75rem;">
                    ${formatLatLon(row.latitude, row.longitude)}
                    ${mapsLink}
                </div>
            </div>
            <div>
                <div class="confidence-bar" style="width: 64px;">
                    <div class="confidence-fill" style="width: ${Math.min(100, (row.confidence || 0) * 100)}%"></div>
                </div>
                <div class="value" style="font-size: 0.72rem; margin-top: 2px;">${pct}%</div>
            </div>
        </div>`;
    }).join('');
}

function formatAlternativeHeadline(alt) {
    if (!alt) return '—';
    const co = String(alt.country || '');
    const rankFromCountry = co.match(/geoclip\s+rank\s*(\d+)/i);
    if (rankFromCountry) {
        const coord = formatLatLon(alt.latitude, alt.longitude);
        return coord
            ? `Vision GPS #${rankFromCountry[1]} (${coord})`
            : `Vision GPS #${rankFromCountry[1]}`;
    }
    if (isGeoclipModelLabel(alt.city, alt.country)) {
        const rank = String(alt.city || '').replace(/geoclip\s*rank\s*/i, '').trim() || '?';
        return `Vision GPS estimate #${rank}`;
    }
    const c = (alt.city || '').trim();
    const country = co.trim();
    if (c && country && c.toLowerCase() !== country.toLowerCase()) {
        return `${c}, ${country}`;
    }
    if (alt.place_resolution && !alt.place_resolution.error && alt.place_resolution.locality) {
        return alt.place_resolution.locality;
    }
    return c || country || '—';
}

function displayAlternatives(alternatives) {
    const container = document.getElementById('alternativesList');
    
    if (alternatives.length === 0) {
        container.innerHTML = '<p>No alternative predictions available</p>';
        return;
    }

    const sorted = [...alternatives].sort((a, b) => (b.confidence || 0) - (a.confidence || 0));

    container.innerHTML = sorted.map((alt, index) => {
        const lat = Number(alt.latitude);
        const lon = Number(alt.longitude);
        const mapsLink =
            Number.isFinite(lat) && Number.isFinite(lon)
                ? `<br><a class="prediction-maps-link" href="https://www.google.com/maps?q=${encodeURIComponent(
                      `${lat},${lon}`,
                  )}" target="_blank" rel="noopener noreferrer">Open in Google Maps</a>`
                : '';
        const altOsm =
            alt.place_resolution && !alt.place_resolution.error && alt.place_resolution.display_name
                ? `<div class="value" style="font-size:0.72rem;margin-top:4px;opacity:0.9">${escapeHtml(
                      alt.place_resolution.locality || alt.place_resolution.display_name.split(',')[0],
                  )} <span style="opacity:0.75">(OSM)</span></div>`
                : '';
        return `
        <div class="alternative-item">
            <div>
                <div class="label">Location ${index + 1}</div>
                <div class="value">${escapeHtml(formatAlternativeHeadline(alt))}</div>
                ${altOsm}
            </div>
            <div>
                <div class="label">Country</div>
                <div class="value">${alt.country}</div>
            </div>
            <div>
                <div class="label">Coordinates</div>
                <div class="value" style="font-size: 0.75rem;">
                    ${formatLatLon(alt.latitude, alt.longitude)}
                    ${mapsLink}
                </div>
            </div>
            <div>
                <div class="confidence-bar" style="width: 64px;">
                    <div class="confidence-fill" style="width: ${alt.confidence * 100}%"></div>
                </div>
                <div class="value" style="font-size: 0.75rem; margin-top: 2px;">
                    ${(alt.confidence * 100).toFixed(1)}%
                </div>
            </div>
        </div>`;
    }).join('');
}

function displayFeatureAnalysis(features) {
    const container = document.getElementById('featuresList');
    const items = [];

    if (features.landmarks && features.landmarks.length > 0) {
        const landmarkMin = 0.50;
        const shown = features.landmarks.filter(
            (l) => l && (l.confidence == null || Number(l.confidence) >= landmarkMin),
        );
        if (shown.length > 0) {
            items.push(`
            <div class="feature-item">
                <strong>🏛️ Landmark hints (CLIP, unverified)</strong>
                <p>${shown
                    .map((l) => `${escapeHtml(l.name)} (${(Number(l.confidence || 0) * 100).toFixed(0)}%)`)
                    .join(', ')}</p>
                <p class="accuracy-note" style="margin-top:4px;">Softmax over a fixed list of famous monuments — not object detection. Weak matches are hidden.</p>
            </div>
        `);
        }
    }

    if (features.vegetation_types) {
        items.push(`
            <div class="feature-item">
                <strong>🌿 Vegetation (pixel heuristic)</strong>
                <p>${features.vegetation_types.join(', ')}</p>
            </div>
        `);
    }

    if (features.architecture_style) {
        items.push(`
            <div class="feature-item">
                <strong>🏗️ Architecture</strong>
                <p>${features.architecture_style}</p>
            </div>
        `);
    }

    if (features.weather_condition) {
        items.push(`
            <div class="feature-item">
                <strong>🌤️ Weather</strong>
                <p>${features.weather_condition}</p>
            </div>
        `);
    }

    if (features.time_of_day) {
        items.push(`
            <div class="feature-item">
                <strong>⏰ Time of Day</strong>
                <p>${features.time_of_day}</p>
            </div>
        `);
    }

    if (features.detected_text) {
        items.push(`
            <div class="feature-item">
                <strong>📝 Text Detected</strong>
                <p>${features.detected_text.join(', ')}</p>
            </div>
        `);
    }

    container.innerHTML = items.length > 0 ? items.join('') : '<p>No features detected</p>';
}

// ============================================================================
// Wikimedia Commons samples
// ============================================================================

function shortCommonsTitle(title) {
    const t = title.replace(/^File:/i, '');
    return t.length > 52 ? `${t.slice(0, 52)}…` : t;
}

/**
 * @param {object} data API JSON: { samples, warning?, source? }
 * @param {{ fromCache?: boolean, savedAt?: number }} meta
 */
function renderCommonsGridFromPayload(data, meta = {}) {
    const grid = document.getElementById('commonsSamplesGrid');
    const status = document.getElementById('commonsSamplesStatus');
    if (!grid || !status) return;

    const samples = data.samples || [];
    if (samples.length === 0) return;

    grid.innerHTML = '';

    samples.forEach((sample) => {
        const card = document.createElement('button');
        card.type = 'button';
        card.className = 'commons-sample-card';

        const img = document.createElement('img');
        img.src = sample.thumb_url;
        img.alt = '';
        img.loading = 'lazy';

        const titleEl = document.createElement('span');
        titleEl.className = 'commons-sample-title';
        titleEl.textContent = shortCommonsTitle(sample.title);

        const coordsEl = document.createElement('span');
        coordsEl.className = 'commons-sample-coords';
        coordsEl.textContent = formatLatLon(sample.latitude, sample.longitude);

        card.appendChild(img);
        card.appendChild(titleEl);
        card.appendChild(coordsEl);
        card.addEventListener('click', () => selectCommonsSample(sample));
        grid.appendChild(card);
    });

    grid.style.display = 'grid';

    let base = `${samples.length} geotagged photos (coordinates from Commons geosearch). Click one to run StreetCLIP.${
        data.warning ? ` (${data.warning})` : ''
    }`;
    if (meta.fromCache && meta.savedAt) {
        const ageMin = Math.max(1, Math.round((Date.now() - meta.savedAt) / 60000));
        base += ` Restored from browser cache (~${ageMin} min old). Click “Load sample images” to refresh from Wikimedia.`;
    }
    status.textContent = base;
}

function restoreCommonsGridFromCache() {
    const entry = readCommonsListCache();
    if (!entry) return;
    renderCommonsGridFromPayload(entry.payload, { fromCache: true, savedAt: entry.savedAt });
}

async function loadCommonsSamples() {
    const grid = document.getElementById('commonsSamplesGrid');
    const status = document.getElementById('commonsSamplesStatus');
    const btn = document.getElementById('loadCommonsSamplesBtn');
    if (!grid || !status || !btn) return;

    btn.disabled = true;
    status.textContent = '';
    grid.style.display = 'none';
    grid.innerHTML = '';

    showProgress({
        title: 'Loading Commons samples',
        detail: 'Fetching geotagged files from Wikimedia API…',
        hint: 'Calling /samples/wikimedia (geosearch + thumbnails).',
        indeterminate: false,
        percent: 12,
    });

    try {
        const listUrl = apiUrl('/samples/wikimedia');
        updateProgress({
            detail: 'Contacting Wikimedia Commons…',
            percent: 28,
        });
        const r = await fetch(listUrl);
        if (!r.ok) {
            throw new Error(
                `HTTP ${r.status} for ${listUrl}. Open the UI from http://127.0.0.1:8000/static/index.html and restart uvicorn after updating the app.`
            );
        }
        const data = await r.json();
        const samples = data.samples || [];
        if (data.warning && samples.length === 0) {
            status.textContent = data.warning;
            return;
        }
        if (samples.length === 0) {
            status.textContent = data.warning || 'No samples returned. Try again later.';
            return;
        }

        updateProgress({
            detail: `Building gallery (${samples.length} samples)…`,
            percent: 72,
            hint: 'Rendering thumbnails in the grid below.',
        });

        saveCommonsListCache(data);
        renderCommonsGridFromPayload(data, {});

        updateProgress({ percent: 100 });
    } catch (e) {
        console.error(e);
        status.textContent = `Could not load Commons samples: ${e.message}`;
    } finally {
        btn.disabled = false;
        hideProgress();
    }
}

async function selectCommonsSample(sample) {
    try {
        showProgress({
            title: 'Loading sample image',
            detail: `Fetching thumbnail via API…`,
            hint: shortCommonsTitle(sample.title),
            indeterminate: true,
        });
        const url = `${apiUrl('/samples/image')}?title=${encodeURIComponent(sample.title)}`;
        const r = await fetch(url);
        if (!r.ok) throw new Error(`Could not proxy image (${r.status})`);
        const blob = await r.blob();
        const file = new File([blob], 'commons_sample.jpg', {
            type: blob.type || 'image/jpeg',
        });

        sampleReference = {
            latitude: sample.latitude,
            longitude: sample.longitude,
            label: sample.title,
        };

        selectedImage = file;
        fileName.textContent = shortCommonsTitle(sample.title);

        const reader = new FileReader();
        reader.onload = (e) => {
            const ref = sampleReference;
            preview.src = e.target.result;
            previewContainer.style.display = 'block';
            uploadArea.style.display = 'none';
            predictBtn.disabled = false;
            clearBtn.style.display = 'inline-block';
            hideProgress();
            refreshImageMetadataForCurrentSelection(file, ref);
        };
        reader.onerror = () => {
            hideProgress();
            showError('Could not read sample image');
        };
        reader.readAsDataURL(file);
    } catch (e) {
        console.error(e);
        hideProgress();
        showError(e.message || 'Failed to load sample image');
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    
    const container = document.querySelector('.main-content');
    container.insertBefore(errorDiv, container.firstChild);
    
    setTimeout(() => errorDiv.remove(), 5000);
}

function downloadResults() {
    if (!currentPrediction) {
        showError('No results to download');
        return;
    }

    const data = JSON.stringify(currentPrediction, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `geolocation-result-${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
}

// ============================================================================
// StreetCLIP gazetteer — server autoloads GeoNames JSON at startup; poll status here
// ============================================================================

function pollGazetteerAutoloadBanner() {
    const banner = document.getElementById('gazetteerStartupBanner');
    const textEl = document.getElementById('gazetteerStartupBannerText');
    if (!banner || !textEl) return;

    async function tick() {
        try {
            const r = await fetch(apiUrl('/config'));
            if (!r.ok) return;
            const c = await r.json();
            const ga = c.gazetteer_autoload;
            if (!ga || !ga.enabled) {
                banner.style.display = 'none';
                return;
            }
            const phase = ga.phase || 'idle';
            const terminalOk = phase === 'ready' || phase === 'skipped';
            if (terminalOk) {
                banner.style.display = 'none';
                banner.classList.remove('gazetteer-startup-banner--error');
                return;
            }
            if (phase === 'error') {
                banner.style.display = 'flex';
                banner.classList.add('gazetteer-startup-banner--error');
                textEl.textContent = ga.message || ga.error || 'Gazetteer build failed.';
                return;
            }
            banner.style.display = 'flex';
            banner.classList.remove('gazetteer-startup-banner--error');
            textEl.textContent =
                ga.message ||
                'Loading StreetCLIP gazetteer (GeoNames worldwide — first run may take several minutes)…';
            setTimeout(tick, 900);
        } catch (_) {
            setTimeout(tick, 2000);
        }
    }

    tick();
}

// ============================================================================
// Initialization
// ============================================================================

async function refreshApiStatus() {
    try {
        const r = await fetch(apiUrl('/config'));
        if (!r.ok) return;
        const c = await r.json();
        const el = document.getElementById('apiStatusBanner');
        if (!el) return;
        if (!c.streetclip_installed && !c.allow_legacy_mock_ensemble) {
            el.style.display = 'block';
            const ver = c.python_version ? ` (server Python ${c.python_version})` : '';
            const extra = c.vision_ml_note ? ` ${c.vision_ml_note}` : '';
            el.textContent =
                `Vision geolocation is off: no PyTorch/StreetCLIP in this environment${ver}.` +
                ` /predict returns an error when the image has no EXIF GPS.` +
                ` For local dev only: set ALLOW_LEGACY_MOCK_ENSEMBLE=true in backend/.env (demo classifier, not for real use).` +
                extra;
        } else {
            el.style.display = 'none';
        }
    } catch (_) {
        /* ignore */
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    console.log('Photo Geolocation System loaded');

    initUploadControls();
    initActionButtons();

    restoreCommonsGridFromCache();
    pollGazetteerAutoloadBanner();

    try {
        const response = await fetch(apiUrl('/health'));
        if (response.ok) {
            const data = await response.json();
            console.log(`API Health: ${data.status} - ${data.app} v${data.version}`);
        }
        await refreshApiStatus();
    } catch (error) {
        console.warn('API not available:', error.message);
        showError('API server not available. Please ensure the backend is running on port 8000.');
    }
});
