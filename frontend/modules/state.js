/**
 * Shared mutable state and constants for the Photo Geolocation frontend.
 */

export const state = {
  selectedImage: null,
  currentPrediction: null,
  sampleReference: null,
};

export const VISUAL_TIME_BUCKET_LABELS = {
  night_or_twilight: 'Night / twilight',
  golden_hour_warm: 'Golden-hour–like',
  midday_bright: 'Bright daylight',
  overcast_diffuse: 'Flat / overcast–like',
  blue_hour_dim: 'Dim / blue-hour–like',
  unclear: 'Unclear',
};

/** Client phases for Predict: encode image → GET /config → POST /predict + server work */
export const PREDICT_CLIENT_STEPS = 3;

/**
 * Steps 1–2 usually finish before the next paint, so users only saw "Step 3 of 3".
 * Hold each numbered step on screen at least this long unless real work took longer.
 */
export const MIN_PREDICT_STEP_DWELL_MS = 340;
