"""API endpoints for photo geolocation"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
import base64
import time
from io import BytesIO
from typing import Any, Dict, Optional

import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from PIL import Image

from app.models.schemas import (
    ImageUploadRequest,
    PredictionResponse,
    LocationPrediction,
    GlobeRegionalHintsResult,
    SceneGeolocationCues,
    CoordinateSource,
    ExternalValidationSummary,
    WikipediaPlaceContext,
    CrossReferenceDatabaseSummary,
    MLImageRecognition,
    InfrastructureEnergyVisualCues,
    CountryEliminationResult,
    GeoReasoningResult,
    AstronomyConstraints,
)
from app.features.extractor import FeatureExtractor
from app.inference.ensemble import EnsembleInference
from app.models.database import GeoLocationResult, results_store
from app.utils.helpers import extract_exif_gps, convert_to_numpy
from app.utils.filename_hints import location_from_filename
from app.config import settings
from app.data.gazetteer_loader import streetclip_gazetteer_json_resolved
from app.services.gazetteer_autoload import get_gazetteer_autoload_status
from app.inference.globe_regional_clip_hint import compute_globe_regional_hints
from app.features.scene_geolocation_cues import compute_scene_geolocation_cues
from app.features.ml_image_recognition import compute_ml_image_recognition
from app.features.reading_axes import build_geolocation_reading_axes
from app.features.infrastructure_energy_cues import compute_infrastructure_energy_cues
from app.features.specialist_detectors import (
    extract_all_specialist_cues,
    get_astronomy_constraints,
)
from app.reasoning.country_elimination import (
    CountryEliminationEngine,
    DetectedCue,
    cues_from_detected_text,
    cues_from_vegetation,
    cues_from_infrastructure,
)
from app.reasoning.bayesian_geo_reasoner import BayesianGeoReasoner
from app.inference.location_fusion import apply_reasoning_to_predictions
from app.services.cross_reference_database import cross_reference_candidates_with_local_database
from app.services.external_validation import validate_candidates_with_open_data
from app.services.reverse_geocode import (
    enrich_predictions_with_reverse_geocode,
    effective_accept_language_for_nominatim,
)
from app.services.llm_detective import run_llm_detective
from app.services.satellite_matcher import satellite_reverse_match
from app.services.streetview_matcher import streetview_verify_predictions
from app.services.prediction_cache import (
    delete_cached_prediction,
    get_cached_prediction,
    set_cached_prediction,
)
from app.services import global_pipeline
from app.inference.streetclip_inference import get_streetclip_load_status
from app.services.pipeline_progress import get_progress_tracker
from app.services.pipeline_live import (
    candidates_from_predictions,
    format_coord_short,
    format_place,
)
from app.services.place_display import (
    display_place_label,
    is_geoclip_placeholder,
    enrich_predictions_for_display,
    enrich_response_payload_for_display,
    promote_named_primary_if_available,
    sort_predictions_by_confidence,
)
from app.services.wikipedia_place_context import build_wikipedia_place_context

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize ML components
feature_extractor = FeatureExtractor()
ensemble_model = EnsembleInference()


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000.0, 3)


def _is_vision_fusion_result(model_used: str) -> bool:
    mu = (model_used or "").strip()
    return (
        mu == "ensemble"
        or mu == "fusion"
        or mu.startswith("fusion[")
        or mu.startswith("emergency[")
        or mu == "clip_zs_fallback"
    )


@router.get("/config")
async def get_public_config():
    """Flags and model ids for the web client."""

    from app.inference.globe_regional_clip_hint import is_globe_clip_runtime_available
    from app.inference.geoclip_inference import is_geoclip_available
    from app.inference.streetclip_inference import is_transformers_streetclip_available
    from app.services.llm_detective import _is_ollama_available, DEFAULT_OLLAMA_URL, DEFAULT_MODEL

    ollama_url = getattr(settings, "ollama_base_url", DEFAULT_OLLAMA_URL)
    ollama_ready = await _is_ollama_available(ollama_url)

    return {
        "globe_clip_model_id": settings.globe_clip_model_id,
        "globe_regional_torch_ready": is_globe_clip_runtime_available(),
        "geoclip_available": is_geoclip_available(),
        "streetclip_transformers_ready": is_transformers_streetclip_available(),
        "use_geoclip": settings.use_geoclip,
        "use_streetclip": settings.use_streetclip,
        "streetclip_model_id": settings.streetclip_model_id,
        "streetclip_gazetteer_source": (
            "file" if streetclip_gazetteer_json_resolved(settings) is not None else "embedded"
        ),
        "fusion_weights": {
            "geoclip": settings.fusion_weight_geoclip,
            "streetclip": settings.fusion_weight_streetclip,
            "clip_zs": settings.fusion_weight_clip_zs,
            "grid_search": settings.fusion_weight_grid_search,
        },
        "use_multi_resolution_grid_search": settings.use_multi_resolution_grid_search,
        "use_cross_reference_database": settings.use_cross_reference_database,
        "fusion_dedupe_decimals": settings.fusion_dedupe_decimals,
        "geoclip_merge_max_ranks": settings.geoclip_merge_max_ranks,
        "top_k_predictions": settings.top_k_predictions,
        "use_infrastructure_energy_clip": settings.use_infrastructure_energy_clip,
        "infrastructure_energy_clip_top_n": settings.infrastructure_energy_clip_top_n,
        "use_country_elimination": settings.use_country_elimination,
        "use_bayesian_reasoning": settings.use_bayesian_reasoning,
        "use_astronomy_solver": settings.use_astronomy_solver,
        "use_specialist_detectors": settings.use_specialist_detectors,
        "reasoning_fusion_boost_weight": settings.reasoning_fusion_boost_weight,
        "reasoning_latitude_penalty_weight": settings.reasoning_latitude_penalty_weight,
        "preload_torch_models_at_startup": settings.preload_torch_models_at_startup,
        "hybrid_streetclip_alt_geoclip_reconcile": settings.hybrid_streetclip_alt_geoclip_reconcile,
        "reverse_geocode_enabled": settings.reverse_geocode_enabled,
        "nominatim_base_url": settings.nominatim_base_url,
        "global_pipeline_version": global_pipeline.PIPELINE_VERSION,
        "capabilities": {
            "coverage": "worldwide_coordinates",
            "vision_fusion": True,
            "sparse_streetclip_gazetteer": settings.use_streetclip,
            "geoclip_gallery_retrieval": settings.use_geoclip,
            "open_data_pin_validation": True,
            "reverse_geocode_place_names": settings.reverse_geocode_enabled,
            "place_naming_source": "openstreetmap_nominatim_odbl",
            "production_note": "High traffic: self-host Nominatim or use a commercial geocoder; set NOMINATIM_HTTP_USER_AGENT.",
        },
        "streetclip_gazetteer_autoload_dump": getattr(settings, "streetclip_gazetteer_autoload_dump", "cities1000"),
        "streetclip_gazetteer_autoload_enabled": getattr(
            settings, "streetclip_gazetteer_autoload_at_startup", False
        ),
        "gazetteer_autoload": get_gazetteer_autoload_status(),
        "ollama_available": ollama_ready,
        "ollama_url": ollama_url,
        "ollama_model": getattr(settings, "ollama_model", DEFAULT_MODEL),
        "use_llm_detective": getattr(settings, "use_llm_detective", True),
        "use_satellite_matching": getattr(settings, "use_satellite_matching", True),
        "streetview_api_configured": bool(
            getattr(settings, "google_maps_api_key", None)
            and str(getattr(settings, "google_maps_api_key", "")).strip().lower()
            not in ("", "none", "null")
        ),
        "use_streetview_verification": getattr(settings, "use_streetview_verification", True),
        "use_prediction_cache": getattr(settings, "use_prediction_cache", True),
        "prediction_cache_ttl_seconds": getattr(settings, "prediction_cache_ttl_seconds", 86400),
        "prediction_cache_max_entries": getattr(settings, "prediction_cache_max_entries", 1000),
        # Legacy keys (older clients); same data as /health-style introspection
        "app_name": settings.app_name,
        "version": settings.app_version,
        "debug": settings.debug,
        "model_device": settings.model_device,
        "confidence_threshold": settings.confidence_threshold,
        "enable_landmark_detection": settings.enable_landmark_detection,
        "enable_vegetation_analysis": settings.enable_vegetation_analysis,
        "enable_architecture_analysis": settings.enable_architecture_analysis,
        "enable_text_ocr": settings.enable_text_ocr,
    }


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "global_pipeline_version": global_pipeline.PIPELINE_VERSION,
        "globe_ready": True,
    }


@router.post("/cache/clear")
async def clear_prediction_cache_endpoint():
    """Clear all disk-cached prediction results."""
    from app.services.prediction_cache import clear_prediction_cache
    count = clear_prediction_cache(settings)
    return {"cleared": count, "ok": True}


@router.get("/model/streetclip-load-status")
async def streetclip_load_status():
    """Pollable StreetCLIP load phase for the web UI during long first-time downloads."""
    return get_streetclip_load_status(settings)


@router.get("/predict/progress")
async def get_prediction_progress():
    """
    Poll while POST /predict is running.

    Returns a short human-oriented status: ``message``, ``phase_label``, ``percent``,
    and a ``checklist`` of coarse stages (prepare → vision → enrich → finish).
    """
    tracker = get_progress_tracker()
    return tracker.get_current()


@router.get("/feature-catalog/meta")
async def feature_catalog_meta():
    """
    Count and version of the static geolocation feature-spec catalog
    (analysis dimensions; not per-image values).
    """
    from app.data.geolocation_feature_catalog import (
        geolocation_feature_catalog_count,
        geolocation_feature_catalog_version,
    )

    return {
        "version": geolocation_feature_catalog_version(),
        "feature_spec_count": geolocation_feature_catalog_count(),
        "note": "Cue templates for extractors — does not imply implemented detectors for every row.",
    }


@router.post("/predict", response_model=PredictionResponse)
async def predict_location(req: Request):
    """
    Global inference pipeline: multi-model GPS hypothesis → optional open-data validation → OSM place naming.

    Accepts either JSON (``base64_image`` / ``image_url``) or multipart form-data with an ``image`` file.

    Place names for arbitrary settlements use Nominatim at the predicted pin (worldwide OSM coverage).
Locale: set ``reverse_geocode_accept_language`` or standard ``Accept-Language`` header.
    """
    try:
        start_time = time.time()
        perf_start = time.perf_counter()
        timings_ms: dict[str, float] = {}
        progress_tracker = get_progress_tracker()
        
        # Initialize image_id early for progress tracking
        image_id_early = f"img_{uuid.uuid4().hex[:8]}"
        progress_tracker.start_prediction(image_id_early)
        
        t0 = time.perf_counter()
        payload, image_bytes = await _read_predict_payload(req)
        timings_ms["request_parse"] = _elapsed_ms(t0)
        fast_prediction = bool(getattr(payload, "fast_prediction", False))
        clear_prediction_cache = bool(getattr(payload, "clear_prediction_cache", False))
        progress_tracker.set_options(
            fast_mode=fast_prediction,
            include_features=bool(getattr(payload, "include_feature_analysis", True)),
        )
        progress_tracker.update_step("request_parse", timings=timings_ms)
        logger.info(
            "Predict request received: fast=%s clear_cache=%s filename=%s",
            fast_prediction,
            clear_prediction_cache,
            payload.original_filename,
        )

        if image_bytes and clear_prediction_cache:
            if delete_cached_prediction(image_bytes, settings):
                logger.info(
                    "Cleared prediction cache entry for %s",
                    payload.original_filename,
                )

        # Disk cache check (identical image bytes → instant response)
        skip_prediction_cache = fast_prediction or clear_prediction_cache
        if image_bytes and settings.use_prediction_cache and not skip_prediction_cache:
            cached = get_cached_prediction(image_bytes, settings)
            if cached:
                cached_result = cached.get("result")
                if cached_result and isinstance(cached_result, dict):
                    cached_result = enrich_response_payload_for_display(cached_result)
                    cached_result["from_cache"] = True
                    cached_result["processing_time_ms"] = _elapsed_ms(start_time)
                    if "timings_ms" in cached_result and isinstance(cached_result["timings_ms"], dict):
                        cached_result["timings_ms"]["total"] = round(cached_result["processing_time_ms"], 3)
                    logger.info("Returning cached prediction for %s", payload.original_filename)
                    progress_tracker.complete({"total": cached_result.get("processing_time_ms", 0)})
                    return PredictionResponse.model_validate(cached_result)

        # Parse image input
        t0 = time.perf_counter()
        if image_bytes is not None:
            image_array = _bytes_to_numpy_rgb(image_bytes)
        elif payload.image_url:
            raise HTTPException(status_code=400, detail="URL image loading not yet implemented")
        else:
            raise HTTPException(status_code=400, detail="Must provide either base64_image or image_url")
        timings_ms["image_decode"] = _elapsed_ms(t0)
        progress_tracker.update_step("image_decode", timings=timings_ms)

        if image_array is None or image_array.size == 0:
            raise HTTPException(status_code=400, detail="Invalid or empty image")

        # Generate image ID
        image_id = f"img_{uuid.uuid4().hex[:8]}"

        # Extract features if requested
        feature_analysis = None
        if payload.include_feature_analysis:
            t0 = time.perf_counter()
            progress_tracker.update_step("feature_analysis")
            feature_analysis = await asyncio.to_thread(
                feature_extractor.extract_all_features,
                image_array,
            )
            timings_ms["feature_analysis"] = _elapsed_ms(t0)
            progress_tracker.update_step("feature_analysis", timings=timings_ms)

        has_exif_gps = False
        exif_gps = extract_exif_gps(image_bytes) if image_bytes else None
        inference_debug: dict[str, Any] = {}
        llm_detective: Dict[str, Any] | None = None

        filename_prediction = location_from_filename(payload.original_filename)

        coordinate_source: CoordinateSource = "vision_estimate"
        geoposition_accuracy_note = settings.geoposition_note_vision

        if exif_gps:
            has_exif_gps = True
            coordinate_source = "exif_gps"
            geoposition_accuracy_note = settings.geoposition_note_exif
            primary_prediction = LocationPrediction(
                latitude=exif_gps["latitude"],
                longitude=exif_gps["longitude"],
                country="From EXIF",
                confidence=0.999,
                distance_confidence_km=0.05,
            )
            alternative_predictions = []
            model_used = "EXIF GPS"
        elif filename_prediction:
            coordinate_source = "filename_hint"
            geoposition_accuracy_note = settings.geoposition_note_filename
            primary_prediction = filename_prediction
            alternative_predictions = []
            model_used = "filename_hint"
        else:
            clip_id = settings.globe_clip_model_id if settings.ensemble_use_clip_zero_shot else None
            progress_tracker.update_step("vision_inference")
            t0 = time.perf_counter()
            inference_results = await asyncio.to_thread(
                ensemble_model.predict,
                image_array,
                True,
                settings.top_k_predictions,
                clip_id,
                fast_prediction,
            )
            timings_ms["vision_inference_total"] = _elapsed_ms(t0)
            progress_tracker.update_step("vision_inference", timings=timings_ms)

            primary_prediction = inference_results.get("primary_prediction")
            alternative_predictions = inference_results.get("alternative_predictions", [])
            model_used = inference_results.get("model_used", "ensemble")
            inference_debug = {
                "fusion_sources": inference_results.get("fusion_sources", []),
                "source_counts": inference_results.get("source_counts", {}),
                "source_predictions": inference_results.get("source_predictions", {}),
                "inference_timings_ms": inference_results.get("timings_ms", {}),
                "grid_search_debug": inference_results.get("grid_search_debug", {}),
            }
            
            if not primary_prediction:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Vision fusion produced no geographic hypothesis. "
                        "Install pip packages: torch, torchvision, transformers, geoclip. "
                        "Set GLOBE_CLIP_MODEL_ID, USE_GEOCLIP=True, USE_STREETCLIP=True "
                        "(StreetCLIP downloads geolocal/StreetCLIP; GeoCLIP downloads encoder weights). "
                        "Ensure ENSEMBLE_USE_CLIP_ZERO_SHOT=True for CLIP country/landmark softmax."
                    ),
                )

            t0_llm = time.perf_counter()
            llm_detective = await _run_llm_detective_step(
                progress_tracker,
                feature_analysis=feature_analysis,
                primary_prediction=primary_prediction,
                alternative_predictions=alternative_predictions or [],
                model_used=model_used,
                include_llm_detective=bool(getattr(payload, "include_llm_detective", True)),
                settings=settings,
            )
            if llm_detective is not None:
                timings_ms["llm_detective"] = _elapsed_ms(t0_llm)

            primary_prediction, alternative_predictions = sort_predictions_by_confidence(
                primary_prediction,
                list(alternative_predictions or []),
            )
            if getattr(settings, "place_promote_named_primary", False):
                primary_prediction, alternative_predictions = promote_named_primary_if_available(
                    primary_prediction,
                    alternative_predictions,
                    max_distance_km=float(
                        getattr(settings, "place_promote_max_distance_km", 85.0)
                    ),
                    min_confidence_ratio=float(
                        getattr(settings, "place_promote_min_confidence_ratio", 1.0)
                    ),
                )
            accept_lang_early = effective_accept_language_for_nominatim(
                req.headers.get("accept-language"),
                getattr(payload, "reverse_geocode_accept_language", None),
                settings.reverse_geocode_default_accept_language,
            )
            need_osm_name = is_geoclip_placeholder(
                primary_prediction.city, primary_prediction.country
            )
            if (
                settings.reverse_geocode_enabled
                and getattr(payload, "include_reverse_geocode", True)
                and need_osm_name
                and (fast_prediction or not getattr(primary_prediction, "place_resolution", None))
            ):
                progress_tracker.set_live(
                    processing_note=(
                        f"OpenStreetMap: place name at "
                        f"{format_coord_short(primary_prediction.latitude, primary_prediction.longitude)}"
                    ),
                )
                t0 = time.perf_counter()
                try:
                    primary_prediction, alternative_predictions = await asyncio.wait_for(
                        enrich_predictions_with_reverse_geocode(
                            primary_prediction,
                            alternative_predictions,
                            settings=settings,
                            accept_language=accept_lang_early,
                        ),
                        timeout=settings.reverse_geocode_batch_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    logger.warning("Fast primary reverse geocode timed out")
                else:
                    timings_ms["reverse_geocode_primary"] = _elapsed_ms(t0)

        cross_reference_database: CrossReferenceDatabaseSummary | None = None
        enable_cross_reference_database = (
            _is_vision_fusion_result(model_used)
            and getattr(settings, "use_cross_reference_database", True)
            and not fast_prediction
        )
        if enable_cross_reference_database:
            progress_tracker.update_step("cross_reference")
            t0 = time.perf_counter()
            primary_prediction, alternative_predictions, xref_dict = cross_reference_candidates_with_local_database(
                primary_prediction,
                alternative_predictions,
                settings=settings,
                original_filename=payload.original_filename,
            )
            timings_ms["cross_reference_database"] = _elapsed_ms(t0)
            cross_reference_database = CrossReferenceDatabaseSummary.model_validate(xref_dict)
        elif _is_vision_fusion_result(model_used):
            cross_reference_database = CrossReferenceDatabaseSummary(
                enabled=False,
                skipped_reason="fast_prediction" if fast_prediction else "disabled_in_settings",
                summary_note=(
                    "Local gazetteer cross-reference skipped for fast prediction."
                    if fast_prediction
                    else "Local gazetteer cross-reference disabled in settings."
                ),
            )

        external_validation: ExternalValidationSummary | None = None
        enable_external_validation = (
            _is_vision_fusion_result(model_used)
            and payload.include_external_validation
            and not fast_prediction
        )
        if enable_external_validation:
            progress_tracker.update_step("external_validation")
            progress_tracker.set_live(
                processing_note=(
                    f"Checking “{display_place_label(primary_prediction)}” "
                    "against Wikipedia and terrain data"
                ),
                candidates=candidates_from_predictions(
                    [primary_prediction] + list(alternative_predictions or [])[:3],
                    source="Validating",
                    limit=4,
                ),
            )
            t0 = time.perf_counter()
            try:
                primary_prediction, alternative_predictions, ev_dict = await asyncio.wait_for(
                    validate_candidates_with_open_data(
                        primary_prediction,
                        alternative_predictions,
                        settings,
                        image_rgb=image_array,
                    ),
                    timeout=settings.external_validation_timeout_seconds,
                )
                timings_ms["external_validation"] = _elapsed_ms(t0)
                external_validation = ExternalValidationSummary.model_validate(ev_dict)
            except asyncio.TimeoutError:
                logger.warning(
                    "External validation timed out after %.0fs",
                    settings.external_validation_timeout_seconds,
                )
                timings_ms["external_validation"] = _elapsed_ms(t0)
                external_validation = ExternalValidationSummary(
                    enabled=False,
                    skipped_reason="timeout",
                    summary_note=(
                        f"External validation timed out after {settings.external_validation_timeout_seconds:.0f}s."
                    ),
                )
        elif _is_vision_fusion_result(model_used):
            external_validation = ExternalValidationSummary(
                enabled=False,
                skipped_reason="fast_prediction" if fast_prediction else "disabled_in_request",
                summary_note=(
                    "Wikipedia + OpenTopoData validation skipped for fast prediction."
                    if fast_prediction
                    else "Wikipedia + OpenTopoData validation skipped for this request."
                ),
            )
        else:
            external_validation = ExternalValidationSummary(
                enabled=False,
                skipped_reason="not_ensemble_source",
                summary_note="External validation applies only to vision ensemble candidates (not EXIF/filename pins).",
            )

        accept_lang = effective_accept_language_for_nominatim(
            req.headers.get("accept-language"),
            getattr(payload, "reverse_geocode_accept_language", None),
            settings.reverse_geocode_default_accept_language,
        )
        enable_reverse_geocode = (
            settings.reverse_geocode_enabled
            and getattr(payload, "include_reverse_geocode", True)
            and not fast_prediction
        )
        if enable_reverse_geocode:
            progress_tracker.update_step("reverse_geocode")
            progress_tracker.set_live(
                processing_note=(
                    f"OpenStreetMap: resolving place name at "
                    f"{format_coord_short(primary_prediction.latitude, primary_prediction.longitude)}"
                ),
                lead_place=display_place_label(primary_prediction),
                candidates=candidates_from_predictions(
                    [primary_prediction] + list(alternative_predictions or [])[:4],
                    source="Candidate pin",
                    limit=5,
                ),
            )
            t0 = time.perf_counter()
            try:
                primary_prediction, alternative_predictions = await asyncio.wait_for(
                    enrich_predictions_with_reverse_geocode(
                        primary_prediction,
                        alternative_predictions,
                        settings=settings,
                        accept_language=accept_lang,
                    ),
                    timeout=settings.reverse_geocode_batch_timeout_seconds,
                )
                timings_ms["reverse_geocode"] = _elapsed_ms(t0)
            except asyncio.TimeoutError:
                logger.warning(
                    "Reverse geocode timed out after %.0fs",
                    settings.reverse_geocode_batch_timeout_seconds,
                )
                timings_ms["reverse_geocode"] = _elapsed_ms(t0)

        globe_regional_hints: GlobeRegionalHintsResult | None = None
        if payload.include_globe_regional_hints and not fast_prediction:
            progress_tracker.update_step("analysis_panels")
            t0 = time.perf_counter()
            raw_hints = compute_globe_regional_hints(
                image_array,
                model_id=settings.globe_clip_model_id,
            )
            timings_ms["globe_regional_hints"] = _elapsed_ms(t0)
            globe_regional_hints = GlobeRegionalHintsResult.model_validate(raw_hints)

        scene_geolocation_cues: SceneGeolocationCues | None = None
        if payload.include_scene_geolocation_cues and not fast_prediction:
            t0 = time.perf_counter()
            raw_scene = compute_scene_geolocation_cues(
                image_array,
                model_id=settings.globe_clip_model_id,
                include_cultural_economic_visual=payload.include_cultural_economic_visual_cues,
            )
            timings_ms["scene_geolocation_cues"] = _elapsed_ms(t0)
            scene_geolocation_cues = SceneGeolocationCues.model_validate(raw_scene)

        ml_image_recognition: MLImageRecognition | None = None
        if payload.include_ml_image_recognition and not fast_prediction:
            t0 = time.perf_counter()
            raw_ml = compute_ml_image_recognition(
                image_array,
                settings=settings,
                top_n=settings.ml_recognition_top_n,
            )
            timings_ms["ml_image_recognition"] = _elapsed_ms(t0)
            ml_image_recognition = MLImageRecognition.model_validate(raw_ml)

        infrastructure_energy_cues: InfrastructureEnergyVisualCues | None = None
        if payload.include_infrastructure_energy_cues and not fast_prediction:
            t0 = time.perf_counter()
            raw_ie = compute_infrastructure_energy_cues(image_array, settings=settings)
            timings_ms["infrastructure_energy_cues"] = _elapsed_ms(t0)
            infrastructure_energy_cues = InfrastructureEnergyVisualCues.model_validate(raw_ie)

        # ------------------------------------------------------------------
        # Geo-Reasoning Engine (country elimination + Bayesian + astronomy)
        # ------------------------------------------------------------------
        country_elimination: CountryEliminationResult | None = None
        geo_reasoning: GeoReasoningResult | None = None
        astronomy_constraints: AstronomyConstraints | None = None

        enable_reasoning = _is_vision_fusion_result(model_used) and not fast_prediction
        if enable_reasoning:
            progress_tracker.update_step("reasoning")
            t0 = time.perf_counter()

            # 1. Collect all detected cues
            all_cues = []

            # From specialist pixel detectors
            if settings.use_specialist_detectors:
                specialist_cues = extract_all_specialist_cues(image_array, scene_cues=raw_scene if scene_geolocation_cues else None)
                all_cues.extend(specialist_cues)

            # From detected text (OCR)
            if feature_analysis and feature_analysis.detected_text:
                text_cues = cues_from_detected_text(feature_analysis.detected_text, confidence=0.7)
                all_cues.extend(text_cues)

            # From vegetation
            if feature_analysis and feature_analysis.vegetation_types:
                veg_cues = cues_from_vegetation(feature_analysis.vegetation_types, confidence=0.6)
                all_cues.extend(veg_cues)

            # From infrastructure type
            if feature_analysis and feature_analysis.infrastructure_type:
                infra_cues = cues_from_infrastructure(feature_analysis.infrastructure_type, confidence=0.5)
                all_cues.extend(infra_cues)

            # From scene CLIP cues (convert top cues to elimination cues)
            if scene_geolocation_cues and scene_geolocation_cues.vegetation:
                for veg in scene_geolocation_cues.vegetation[:3]:
                    label = (veg.label or "").lower()
                    if "palm" in label or "tropical" in label:
                        all_cues.append(DetectedCue(cue_type="latitude_band", value="tropical", confidence=veg.score * 0.7, source="clip_softmax"))
                    elif "pine" in label or "conifer" in label:
                        all_cues.append(DetectedCue(cue_type="latitude_band", value="temperate", confidence=veg.score * 0.6, source="clip_softmax"))

            # From infrastructure energy cues
            if infrastructure_energy_cues and infrastructure_energy_cues.clip_banks_detail:
                for bank in infrastructure_energy_cues.clip_banks_detail:
                    for cat in bank.get("categories", []):
                        for item in cat.get("items", [])[:2]:
                            lab = (item.get("label") or "").lower()
                            if "pole" in lab or "utility" in lab:
                                all_cues.append(DetectedCue(cue_type="pole_type", value="wooden_crossarm_us_style", confidence=item.get("confidence", 0.5) * 0.6, source="clip_softmax"))
                            elif "solar" in lab or "wind" in lab:
                                pass  # not strong country signals alone

            logger.info("Geo-Reasoning: collected %d cues for analysis", len(all_cues))

            # 2. Country Elimination
            if settings.use_country_elimination and all_cues:
                elim_engine = CountryEliminationEngine()
                elim_result = elim_engine.eliminate(all_cues)
                country_elimination = CountryEliminationResult(
                    remaining_countries=sorted(elim_result.remaining_countries)[:50],
                    eliminated_countries=sorted(elim_result.eliminated_countries)[:50],
                    country_scores=dict(sorted(elim_result.country_scores.items(), key=lambda x: x[1], reverse=True)[:30]),
                    applied_cue_count=len(elim_result.applied_cues),
                    contradiction_penalties=elim_result.contradiction_penalties,
                    summary=elim_result.summary,
                    num_remaining=elim_result.num_remaining,
                    num_eliminated=elim_result.num_eliminated,
                )
                inference_debug["country_elimination"] = {
                    "num_remaining": elim_result.num_remaining,
                    "num_eliminated": elim_result.num_eliminated,
                    "top_remaining": sorted(elim_result.remaining_countries)[:10],
                }

            # 3. Bayesian Reasoning
            if settings.use_bayesian_reasoning and all_cues:
                reasoner = BayesianGeoReasoner()
                reasoning_result = reasoner.reason(all_cues)
                geo_reasoning = GeoReasoningResult(
                    country_posteriors=dict(sorted(reasoning_result.country_posteriors.items(), key=lambda x: x[1], reverse=True)[:30]),
                    top_country=reasoning_result.top_country,
                    top_confidence=reasoning_result.top_confidence,
                    evidence_breakdown=reasoning_result.evidence_breakdown,
                    contradiction_penalties_applied=reasoning_result.contradiction_penalties_applied,
                    summary=reasoning_result.summary,
                )
                inference_debug["geo_reasoning"] = {
                    "top_country": reasoning_result.top_country,
                    "top_confidence": reasoning_result.top_confidence,
                    "num_posteriors": len(reasoning_result.country_posteriors),
                }

            # 4. Astronomy Solver
            if settings.use_astronomy_solver:
                astro = get_astronomy_constraints(
                    image_array,
                    scene_cues=raw_scene if scene_geolocation_cues else None,
                )
                astronomy_constraints = AstronomyConstraints(
                    latitude_min=astro.latitude_min,
                    latitude_max=astro.latitude_max,
                    latitude_confidence=astro.latitude_confidence,
                    hemisphere_hint=astro.hemisphere_hint,
                    season_hint=astro.season_hint,
                    solar_elevation_deg=astro.solar_elevation_deg,
                    shadow_direction_deg=astro.shadow_direction_deg,
                    time_of_day_hint=astro.time_of_day_hint,
                    summary=astro.summary,
                )
                inference_debug["astronomy_constraints"] = {
                    "latitude_min": astro.latitude_min,
                    "latitude_max": astro.latitude_max,
                    "confidence": astro.latitude_confidence,
                    "hemisphere": astro.hemisphere_hint,
                }

            # 5. Apply reasoning to re-rank predictions
            if (country_elimination or geo_reasoning or astronomy_constraints) and primary_prediction:
                primary_prediction, alternative_predictions = apply_reasoning_to_predictions(
                    primary_prediction,
                    alternative_predictions,
                    country_elimination=country_elimination,
                    geo_reasoning=geo_reasoning,
                    astronomy_constraints=astronomy_constraints,
                    settings=settings,
                )

            timings_ms["geo_reasoning"] = _elapsed_ms(t0)


        # ------------------------------------------------------------------
        satellite_match: Dict[str, Any] | None = None
        if _is_vision_fusion_result(model_used) and not fast_prediction and primary_prediction:
            progress_tracker.update_step("satellite_match")
            t0 = time.perf_counter()
            try:
                satellite_match = await asyncio.wait_for(
                    satellite_reverse_match(
                        image_array,
                        primary_prediction.latitude,
                        primary_prediction.longitude,
                        settings=settings,
                    ),
                    timeout=settings.satellite_match_timeout_seconds,
                )
                timings_ms["satellite_match"] = _elapsed_ms(t0)
            except asyncio.TimeoutError:
                logger.warning(
                    "Satellite match timed out after %.0fs",
                    settings.satellite_match_timeout_seconds,
                )
                satellite_match = {
                    "enabled": False,
                    "skipped_reason": "timeout",
                    "summary": (
                        f"Satellite match timed out after {settings.satellite_match_timeout_seconds:.0f}s."
                    ),
                }
                timings_ms["satellite_match"] = _elapsed_ms(t0)
            except Exception as e:
                logger.warning("Satellite match failed: %s", e)
                satellite_match = {"enabled": False, "skipped_reason": "exception", "summary": str(e)}
                timings_ms["satellite_match"] = _elapsed_ms(t0)

        # ------------------------------------------------------------------
        # NEW: Street View visual verification (highest-accuracy signal)
        # ------------------------------------------------------------------
        streetview_verification: Dict[str, Any] | None = None
        if _is_vision_fusion_result(model_used) and not fast_prediction and primary_prediction:
            progress_tracker.update_step("streetview_verify")
            t0 = time.perf_counter()
            try:
                sv_thresh = float(getattr(settings, "streetview_similarity_threshold", 0.72))
                streetview_verification = await asyncio.wait_for(
                    streetview_verify_predictions(
                        image_array,
                        primary_prediction,
                        alternative_predictions or [],
                        settings=settings,
                        similarity_threshold=sv_thresh,
                    ),
                    timeout=settings.streetview_verify_timeout_seconds,
                )
                timings_ms["streetview_verification"] = _elapsed_ms(t0)
                # Potentially swap primary based on Street View match
                if streetview_verification.get("swapped_primary"):
                    chosen_idx = streetview_verification.get("chosen_candidate_index", 0)
                    if chosen_idx == 0:
                        pass  # primary confirmed
                    elif chosen_idx == 1 and alternative_predictions:
                        # Promote first alternative to primary
                        old_primary = primary_prediction
                        primary_prediction = alternative_predictions[0]
                        alternative_predictions = [old_primary] + alternative_predictions[1:]
                        logger.info("Street View promoted alternative #1 to primary")
                    elif chosen_idx > 1 and len(alternative_predictions) >= chosen_idx:
                        old_primary = primary_prediction
                        primary_prediction = alternative_predictions[chosen_idx - 1]
                        alternative_predictions[chosen_idx - 1] = old_primary
                        logger.info("Street View promoted alternative #%d to primary", chosen_idx)
            except asyncio.TimeoutError:
                logger.warning(
                    "Street View verification timed out after %.0fs",
                    settings.streetview_verify_timeout_seconds,
                )
                streetview_verification = {
                    "enabled": False,
                    "api_configured": True,
                    "skipped_reason": "timeout",
                    "summary": (
                        f"Street View verification timed out after {settings.streetview_verify_timeout_seconds:.0f}s."
                    ),
                }
                timings_ms["streetview_verification"] = _elapsed_ms(t0)
            except Exception as e:
                logger.warning("Street View verification failed: %s", e)
                streetview_verification = {"enabled": False, "api_configured": False, "summary": str(e)}
                timings_ms["streetview_verification"] = _elapsed_ms(t0)

        wikipedia_place_context: WikipediaPlaceContext | None = None
        if _is_vision_fusion_result(model_used):
            wikipedia_place_context = build_wikipedia_place_context(
                external_validation,
                primary_prediction,
                alternative_predictions or [],
                enabled_in_request=bool(
                    payload.include_external_validation and not fast_prediction
                ),
            )

        # Re-rank after validation / reasoning / Street View so the UI primary always
        # has the highest fusion confidence (interim progress may show higher StreetCLIP %).
        if _is_vision_fusion_result(model_used) and primary_prediction:
            primary_prediction, alternative_predictions = sort_predictions_by_confidence(
                primary_prediction,
                list(alternative_predictions or []),
            )
            if getattr(settings, "place_promote_named_primary", False):
                primary_prediction, alternative_predictions = promote_named_primary_if_available(
                    primary_prediction,
                    alternative_predictions,
                    max_distance_km=float(
                        getattr(settings, "place_promote_max_distance_km", 85.0)
                    ),
                    min_confidence_ratio=float(
                        getattr(settings, "place_promote_min_confidence_ratio", 1.0)
                    ),
                )
            final_conf = float(primary_prediction.confidence or 0.0)
            progress_tracker.set_live(
                lead_place=display_place_label(primary_prediction),
                candidates=candidates_from_predictions(
                    [primary_prediction] + list(alternative_predictions or [])[:5],
                    source="Final fusion",
                    limit=6,
                ),
                processing_note=(
                    f"Final pin: {display_place_label(primary_prediction)} "
                    f"({final_conf * 100:.1f}% fusion score — highest among returned hypotheses)"
                ),
            )

        progress_tracker.update_step("finish")
        t0 = time.perf_counter()
        geolocation_reading_axes = build_geolocation_reading_axes(
            image_array,
            scene_geolocation_cues,
            external_validation,
            coordinate_source=coordinate_source,
        )
        timings_ms["reading_axes"] = _elapsed_ms(t0)

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        timings_ms["total"] = round(processing_time_ms, 3)
        inference_debug["coordinate_source"] = coordinate_source
        inference_debug["model_used"] = model_used
        inference_debug["fast_prediction"] = fast_prediction
        if clear_prediction_cache:
            inference_debug["clear_prediction_cache"] = True
        if satellite_match:
            inference_debug["satellite_match"] = satellite_match
        if streetview_verification:
            inference_debug["streetview_verification"] = streetview_verification
        if llm_detective:
            inference_debug["llm_detective"] = llm_detective

        primary_prediction, alternative_predictions = enrich_predictions_for_display(
            primary_prediction,
            list(alternative_predictions or []),
        )

        # Create response
        response = PredictionResponse(
            status="success",
            image_id=image_id,
            primary_prediction=primary_prediction,
            alternative_predictions=alternative_predictions,
            feature_analysis=feature_analysis,
            processing_time_ms=processing_time_ms,
            model_used=model_used,
            has_exif_gps=has_exif_gps,
            globe_regional_hints=globe_regional_hints,
            coordinate_source=coordinate_source,
            geoposition_accuracy_note=geoposition_accuracy_note,
            scene_geolocation_cues=scene_geolocation_cues,
            geolocation_reading_axes=geolocation_reading_axes,
            external_validation=external_validation,
            cross_reference_database=cross_reference_database,
            ml_image_recognition=ml_image_recognition,
            infrastructure_energy_cues=infrastructure_energy_cues,
            country_elimination=country_elimination,
            geo_reasoning=geo_reasoning,
            astronomy_constraints=astronomy_constraints,
            fast_prediction_applied=fast_prediction,
            wikipedia_enabled_in_request=bool(
                payload.include_external_validation and not fast_prediction
            ),
            wikipedia_place_context=wikipedia_place_context,
            timings_ms=timings_ms,
            inference_debug=inference_debug,
        )

        # Store result
        result = GeoLocationResult(
            image_id=image_id,
            latitude=primary_prediction.latitude,
            longitude=primary_prediction.longitude,
            country=primary_prediction.country,
            city=primary_prediction.city,
            confidence=primary_prediction.confidence,
            processing_time_ms=processing_time_ms,
            model_used=model_used,
            has_exif_gps=has_exif_gps,
        )
        results_store[image_id] = result

        # Store result in disk cache for instant re-uploads
        if image_bytes and settings.use_prediction_cache and not fast_prediction and not response.from_cache:
            try:
                set_cached_prediction(image_bytes, response.model_dump(mode="json"), settings)
            except Exception as e:
                logger.warning("Failed to store prediction cache: %s", e)

        logger.info(
            "Processed image %s -> %s, %s in %.0f ms (fast=%s, timings=%s)",
            image_id,
            primary_prediction.city,
            primary_prediction.country,
            processing_time_ms,
            fast_prediction,
            timings_ms,
        )
        progress_tracker.complete(timings_ms)
        return response

    except HTTPException:
        progress_tracker.error("request rejected")
        raise
    except Exception as e:
        progress_tracker.error(str(e))
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@router.post("/predict_batch")
async def predict_batch(images: list[UploadFile] = File(...)):
    """
    Batch predict locations for multiple images.
    
    Returns list of predictions with processing times.
    """
    try:
        results = []
        
        for image in images:
            try:
                contents = await image.read()
                image_array = convert_to_numpy(contents)
                
                if image_array is None or image_array.size == 0:
                    results.append({
                        "filename": image.filename,
                        "status": "error",
                        "error": "Invalid image format"
                    })
                    continue
                
                # Run inference
                start_time = time.time()
                clip_id = settings.globe_clip_model_id if settings.ensemble_use_clip_zero_shot else None
                inference_results = ensemble_model.predict(
                    image_array,
                    include_retrieval=True,
                    top_k=3,
                    clip_model_id=clip_id,
                )
                
                primary_pred = inference_results.get("primary_prediction")
                processing_time = (time.time() - start_time) * 1000
                
                if not primary_pred:
                    results.append({
                        "filename": image.filename,
                        "status": "error",
                        "error": (
                            "No vision hypothesis (CLIP stack required — same as POST /predict 503)."
                        ),
                        "processing_time_ms": processing_time,
                    })
                    continue

                if settings.reverse_geocode_enabled:
                    primary_pred, _ = await enrich_predictions_with_reverse_geocode(
                        primary_pred,
                        [],
                        settings=settings,
                        accept_language=settings.reverse_geocode_default_accept_language,
                    )

                pred_payload = {
                    "latitude": primary_pred.latitude,
                    "longitude": primary_pred.longitude,
                    "country": primary_pred.country,
                    "city": primary_pred.city,
                    "confidence": primary_pred.confidence,
                }
                if primary_pred.place_resolution:
                    pred_payload["place_resolution"] = primary_pred.place_resolution.model_dump()

                results.append({
                    "filename": image.filename,
                    "status": "success",
                    "model_used": inference_results.get("model_used", "ensemble"),
                    "prediction": pred_payload,
                    "processing_time_ms": processing_time,
                })
                
            except Exception as e:
                results.append({
                    "filename": image.filename,
                    "status": "error",
                    "error": str(e)
                })
        
        return {"results": results, "total": len(results)}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")


@router.get("/results/{image_id}")
async def get_result(image_id: str):
    """Retrieve previously computed result by image ID"""
    if image_id not in results_store:
        raise HTTPException(status_code=404, detail=f"Result not found: {image_id}")
    
    return results_store[image_id].to_dict()


@router.get("/results")
async def list_results(limit: int = 100, skip: int = 0):
    """List all stored results"""
    all_results = list(results_store.values())
    paginated = all_results[skip:skip + limit]
    
    return {
        "total": len(all_results),
        "skip": skip,
        "limit": limit,
        "results": [r.to_dict() for r in paginated]
    }


@router.get("/models/info")
async def get_model_info():
    """Get information about loaded models"""
    return ensemble_model.get_model_info()


@router.get("/capabilities/executive-summary")
async def executive_summary_capabilities():
    """JSON checklist: executive design doc vs this repository."""
    from app.services.executive_design_map import executive_design_capabilities

    return executive_design_capabilities()


# Helper functions

async def _run_llm_detective_step(
    progress_tracker: Any,
    *,
    feature_analysis: Any,
    primary_prediction: LocationPrediction,
    alternative_predictions: list,
    model_used: str,
    include_llm_detective: bool,
    settings: Any,
) -> Optional[Dict[str, Any]]:
    """Ollama detective right after vision so thoughts appear before slow enrich steps."""
    if not include_llm_detective or not getattr(settings, "use_llm_detective", True):
        result = {
            "enabled": False,
            "skipped_reason": "disabled_in_request",
            "summary": "Ollama detective disabled in prediction options.",
            "key_thoughts": ["Enable “Ollama detective” in prediction options to run local LLM reasoning."],
        }
        _publish_ollama_live(progress_tracker, result)
        return result
    if not _is_vision_fusion_result(model_used) or not primary_prediction:
        return None
    if not feature_analysis:
        result = {
            "enabled": False,
            "skipped_reason": "no_feature_analysis",
            "summary": "Ollama needs feature analysis cues — enable Feature analysis in options.",
            "key_thoughts": ["Enable Feature analysis, then run again for Ollama key thoughts."],
        }
        _publish_ollama_live(progress_tracker, result)
        return result

    progress_tracker.update_step("llm_detective")
    progress_tracker.set_live(
        processing_note="Querying Ollama (local LLM) — key thoughts on this scene…",
        ollama_status="running",
        ollama_key_thoughts=[
            "Waiting for Ollama response (typical 30s–3min on CPU after vision finishes)…",
        ],
        ollama_enabled=None,
    )
    t0 = time.perf_counter()
    pred_list = [
        {
            "country": primary_prediction.country,
            "city": primary_prediction.city,
            "confidence": primary_prediction.confidence,
        }
    ]
    for alt in (alternative_predictions or [])[:2]:
        pred_list.append(
            {
                "country": alt.country,
                "city": alt.city,
                "confidence": alt.confidence,
            }
        )
    async def _heartbeat() -> None:
        t0 = time.perf_counter()
        while True:
            await asyncio.sleep(4.0)
            elapsed = int(time.perf_counter() - t0)
            progress_tracker.set_live(
                processing_note=f"Ollama still thinking… ({elapsed}s on CPU)",
                ollama_status="running",
                ollama_key_thoughts=[
                    f"Waiting for local LLM… {elapsed}s elapsed",
                    "If this fails with HTTP 500, try OLLAMA_MODEL=tinyllama:1.1b",
                ],
                ollama_enabled=None,
            )

    hb = asyncio.create_task(_heartbeat())
    try:
        result = await asyncio.wait_for(
            run_llm_detective(
                feature_analysis.model_dump(mode="json"),
                pred_list,
                settings=settings,
            ),
            timeout=settings.llm_detective_timeout_seconds,
        )
        _publish_ollama_live(progress_tracker, result)
        return result
    except asyncio.TimeoutError:
        logger.warning(
            "LLM detective timed out after %.0fs",
            settings.llm_detective_timeout_seconds,
        )
        result = {
            "enabled": False,
            "skipped_reason": "timeout",
            "summary": (
                f"Ollama timed out after {settings.llm_detective_timeout_seconds:.0f}s — "
                "increase LLM_DETECTIVE_TIMEOUT_SECONDS or use a smaller model."
            ),
            "key_thoughts": [
                f"Timed out after {int(settings.llm_detective_timeout_seconds)}s",
                "Try: ollama pull llama3.2:3b",
            ],
        }
        _publish_ollama_live(progress_tracker, result)
        return result
    except Exception as e:
        logger.warning("LLM detective failed: %s", e)
        result = {
            "enabled": False,
            "skipped_reason": "exception",
            "summary": str(e),
            "key_thoughts": [str(e)],
        }
        _publish_ollama_live(progress_tracker, result)
        return result
    finally:
        hb.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await hb


def _publish_ollama_live(progress_tracker: Any, llm_detective: Optional[Dict[str, Any]]) -> None:
    """Push Ollama detective key thoughts into GET /predict/progress live payload."""
    if not llm_detective or not isinstance(llm_detective, dict):
        return
    from app.services.llm_detective import build_key_thoughts

    thoughts = list(llm_detective.get("key_thoughts") or [])
    if not thoughts:
        thoughts = build_key_thoughts(llm_detective)
    model = llm_detective.get("model")
    if llm_detective.get("enabled") and thoughts:
        lead = thoughts[0][:160] + ("…" if len(thoughts[0]) > 160 else "")
        progress_tracker.set_live(
            processing_note=f"Ollama ({model or 'local LLM'}): {lead}",
            ollama_model=model,
            ollama_key_thoughts=thoughts,
            ollama_enabled=True,
        )
    else:
        summary = str(llm_detective.get("summary") or "Ollama detective skipped.").strip()
        if not thoughts and summary:
            thoughts = [summary]
        progress_tracker.set_live(
            processing_note=(thoughts[0] if thoughts else summary)[:220],
            ollama_model=model,
            ollama_key_thoughts=thoughts[:10],
            ollama_enabled=False,
            ollama_status="done",
        )


def _decode_base64_to_bytes(base64_str: str) -> Optional[bytes]:
    """Decode base64 payload (optional data-URL prefix) to raw image bytes."""
    try:
        if base64_str.startswith("data:image"):
            base64_str = base64_str.split(",", 1)[1]
        return base64.b64decode(base64_str)
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        return None


def _bytes_to_numpy_rgb(image_data: bytes) -> Optional[np.ndarray]:
    """Load image bytes as RGB numpy array (H, W, 3)."""
    try:
        image = Image.open(BytesIO(image_data)).convert("RGB")
        return np.array(image, dtype=np.uint8)
    except Exception as e:
        logger.error(f"Error loading image from bytes: {e}")
        return None


def _coerce_form_bool(value: Any, default: bool) -> bool:
    """HTML form helpers send booleans as strings; normalize them for Pydantic input."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _coerce_form_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


async def _read_predict_payload(req: Request) -> tuple[ImageUploadRequest, Optional[bytes]]:
    """Support both JSON API clients and browser multipart uploads."""
    content_type = (req.headers.get("content-type") or "").split(";", 1)[0].strip().lower()

    if content_type in {"", "application/json"} or content_type.endswith("+json"):
        try:
            raw = await req.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid JSON request body") from e
        if not isinstance(raw, dict):
            raise HTTPException(status_code=400, detail="JSON request body must be an object")
        payload = ImageUploadRequest.model_validate(raw)
        image_bytes = _decode_base64_to_bytes(payload.base64_image) if payload.base64_image else None
        return payload, image_bytes

    if content_type in {"multipart/form-data", "application/x-www-form-urlencoded"}:
        try:
            form = await req.form()
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid multipart form body") from e

        upload = form.get("image")
        image_bytes: Optional[bytes] = None
        upload_filename: Optional[str] = None
        if upload is not None and hasattr(upload, "read"):
            upload_filename = _coerce_form_text(getattr(upload, "filename", None))
            image_bytes = await upload.read()
            if hasattr(upload, "close"):
                await upload.close()

        payload = ImageUploadRequest.model_validate(
            {
                "image_url": _coerce_form_text(form.get("image_url")),
                "base64_image": _coerce_form_text(form.get("base64_image")),
                "original_filename": _coerce_form_text(form.get("original_filename")) or upload_filename,
                "use_cloud_inference": _coerce_form_bool(form.get("use_cloud_inference"), False),
                "fast_prediction": _coerce_form_bool(form.get("fast_prediction"), False),
                "clear_prediction_cache": _coerce_form_bool(
                    form.get("clear_prediction_cache"), False
                ),
                "include_llm_detective": _coerce_form_bool(form.get("include_llm_detective"), True),
                "include_feature_analysis": _coerce_form_bool(form.get("include_feature_analysis"), True),
                "include_globe_regional_hints": _coerce_form_bool(
                    form.get("include_globe_regional_hints"), True
                ),
                "include_scene_geolocation_cues": _coerce_form_bool(
                    form.get("include_scene_geolocation_cues"), True
                ),
                "include_cultural_economic_visual_cues": _coerce_form_bool(
                    form.get("include_cultural_economic_visual_cues"), True
                ),
                "include_external_validation": _coerce_form_bool(
                    form.get("include_external_validation"), True
                ),
                "include_ml_image_recognition": _coerce_form_bool(
                    form.get("include_ml_image_recognition"), True
                ),
                "include_infrastructure_energy_cues": _coerce_form_bool(
                    form.get("include_infrastructure_energy_cues"), True
                ),
                "include_reverse_geocode": _coerce_form_bool(form.get("include_reverse_geocode"), True),
                "reverse_geocode_accept_language": _coerce_form_text(
                    form.get("reverse_geocode_accept_language")
                ),
            }
        )
        if image_bytes is None and payload.base64_image:
            image_bytes = _decode_base64_to_bytes(payload.base64_image)
        return payload, image_bytes

    raise HTTPException(
        status_code=415,
        detail="Unsupported Content-Type for /predict. Use application/json or multipart/form-data.",
    )
