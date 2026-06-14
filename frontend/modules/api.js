/**
 * API calls and prediction orchestration.
 */
import { apiUrl, flushPaintFrames, dwellMinimumSince, getPreferredClientLanguage, appendFormBoolean } from './utils.js';
import { state } from './state.js';
import { PREDICT_CLIENT_STEPS, MIN_PREDICT_STEP_DWELL_MS } from './state.js';
import { isStreetclipWarmClient } from './cache.js';
import { showProgress, updateProgress, hideProgress, startPredictStatusPoll, schedulePredictLongWaitHints, buildOutputSubSteps, startOutputSubStepCycler } from './progress.js';
import { buildMiniPipeline, activatePipelineStage, markEarlyReturn, completePipeline, resetPipeline } from './pipeline.js';
import { getCheckboxRefs, showError } from './image.js';
import { displayResults } from './results.js';

export async function predictLocation() {
  if (!state.selectedImage) {
    showError('Please select an image first');
    return;
  }

  const predictBtn = document.getElementById('predictBtn');
  const resultsSection = document.getElementById('resultsSection');

  try {
    if (predictBtn) predictBtn.disabled = true;
    if (resultsSection) resultsSection.style.display = 'none';
    resetPipeline();
    buildMiniPipeline();
    activatePipelineStage('input');

    showProgress({
      title: 'Predicting location',
      detail: 'Preparing upload…',
      hint: `Next: ${PREDICT_CLIENT_STEPS - 1} HTTP round-trips (quick /config, then /predict).`,
      indeterminate: true,
      step: 1,
      stepTotal: PREDICT_CLIENT_STEPS,
    });
    const step1ShownAt = Date.now();
    await flushPaintFrames();
    await dwellMinimumSince(step1ShownAt, MIN_PREDICT_STEP_DWELL_MS);

    activatePipelineStage('parse');
    const step2ShownAt = Date.now();
    updateProgress({
      detail: 'Checking server…',
      hint: 'GET /config · 1 HTTP round-trip left after this (POST /predict with your photo).',
      step: 2,
      stepTotal: PREDICT_CLIENT_STEPS,
      indeterminate: true,
    });
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

    await dwellMinimumSince(step2ShownAt, MIN_PREDICT_STEP_DWELL_MS);

    activatePipelineStage('exif');
    updateProgress({
      detail: 'Sending image and waiting for prediction…',
      hint: 'Awaiting server (first run may download weights).',
      step: 3,
      stepTotal: PREDICT_CLIENT_STEPS,
      indeterminate: true,
    });

    schedulePredictLongWaitHints(isStreetclipWarmClient(), updateProgress, PREDICT_CLIENT_STEPS);
    startPredictStatusPoll(apiUrl);

    const pipelineTimers = [];
    const stageDelays = [
      { stage: 'filename', delay: 800 },
      { stage: 'features', delay: 1600 },
      { stage: 'inference', delay: 2800 },
      { stage: 'optional', delay: 5000 },
      { stage: 'analysis', delay: 6500 },
      { stage: 'reasoning', delay: 8000 },
      { stage: 'output', delay: 9500 },
    ];
    const cbs = getCheckboxRefs();
    const fastOn = cbs.fastPrediction != null ? cbs.fastPrediction.checked : true;
    stageDelays.forEach(({ stage, delay }) => {
      pipelineTimers.push(setTimeout(() => {
        activatePipelineStage(stage);
        if (stage === 'output') {
          startOutputSubStepCycler(buildOutputSubSteps(serverConfig, fastOn, cbs), updateProgress);
        }
      }, delay));
    });

    const formData = buildPredictFormData({
      includeStreetviewRefinement,
      fastOn,
    });

    const response = await fetch(apiUrl('/predict'), {
      method: 'POST',
      body: formData,
    });

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

    const coordSource = data.coordinate_source || 'vision_estimate';
    if (coordSource === 'exif_gps') {
      markEarlyReturn('exif');
    } else if (coordSource === 'filename_hint') {
      markEarlyReturn('filename');
    } else {
      completePipeline();
    }

    state.currentPrediction = data;
    await displayResults(data);

  } catch (error) {
    console.error('Prediction error:', error);
    const detail = await describePredictFailure(error);
    showError(`Failed to predict location: ${detail}`);
    resetPipeline();
  } finally {
    if (predictBtn) predictBtn.disabled = false;
    hideProgress();
  }
}

function buildPredictFormData({ includeStreetviewRefinement, fastOn }) {
  const form = new FormData();
  form.append('image', state.selectedImage, state.selectedImage.name || 'upload');
  form.append('original_filename', state.selectedImage.name || 'upload');
  form.append('reverse_geocode_accept_language', getPreferredClientLanguage());
  appendFormBoolean(form, 'use_cloud_inference', false);
  appendFormBoolean(form, 'fast_prediction', fastOn);

  const cbs = getCheckboxRefs();
  appendFormBoolean(
    form,
    'clear_prediction_cache',
    cbs.clearPredictionCache?.checked ?? false,
  );
  appendFormBoolean(form, 'include_llm_detective', cbs.includeLlmDetective?.checked ?? true);
  appendFormBoolean(form, 'include_feature_analysis', cbs.includeFeatureAnalysis?.checked ?? true);
  appendFormBoolean(form, 'include_globe_regional_hints', cbs.includeGlobeRegionalHints?.checked ?? true);
  appendFormBoolean(form, 'include_scene_geolocation_cues', cbs.includeSceneGeolocationCues?.checked ?? true);
  appendFormBoolean(form, 'include_cultural_economic_visual_cues', cbs.includeCulturalEconomicVisualCues?.checked ?? true);
  appendFormBoolean(form, 'include_external_validation', cbs.includeExternalValidation?.checked ?? true);
  appendFormBoolean(form, 'include_ml_image_recognition', cbs.includeMlImageRecognition?.checked ?? true);
  appendFormBoolean(form, 'include_infrastructure_energy_cues', cbs.includeInfrastructureEnergyCues?.checked ?? true);
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

export async function refreshApiStatus() {
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

export async function pollGazetteerAutoloadBanner() {
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
