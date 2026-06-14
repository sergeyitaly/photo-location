"""
Maps the Photo Geolocation executive design document to this repository’s implementation.

This backend is FastAPI + browser UI — not the Android app described in the PDF.
"""

from __future__ import annotations

from typing import Any, Dict, List


def executive_design_capabilities() -> Dict[str, Any]:
    """
    Structured checklist for GET /capabilities/executive-summary.

    Values:
      - implemented: shipped in this repo now
      - partial: simplified heuristics, CLIP softmax proxies, or optional wiring
      - not_applicable: Android / Google Play / on-device-only
      - planned: intentional stub / extension point only
    """
    items: List[Dict[str, Any]] = [
        # Platform scope
        {
            "id": "android_camera_photos",
            "layer": "mobile_client",
            "status": "not_applicable",
            "detail": "CameraX, MediaStore, Google Photos Picker — build a native Android app against this HTTP API.",
        },
        {
            "id": "cloud_fastapi_backend",
            "layer": "server",
            "status": "implemented",
            "detail": "FastAPI + uvicorn; /predict multimodal fusion pipeline.",
        },
        {
            "id": "exif_gps_priority",
            "layer": "server",
            "status": "implemented",
            "detail": "EXIF GPS wins when present; ACCESS_MEDIA_LOCATION is client-side.",
        },
        {
            "id": "on_device_tf_lite",
            "layer": "mobile_client",
            "status": "not_applicable",
            "detail": "TensorFlow Lite / ONNX Runtime — optional companion app; server uses PyTorch.",
        },
        # Models
        {
            "id": "ensemble_geoclip_streetclip_clipzs",
            "layer": "ml",
            "status": "implemented",
            "detail": "GeoCLIP + StreetCLIP gazetteer + CLIP zero-shot countries/landmarks + weighted fusion.",
        },
        {
            "id": "planet_style_cell_classifier",
            "layer": "ml",
            "status": "planned",
            "detail": "No PlaNet-style million-cell classifier; CLIP country softmax + GeoCLIP continuous GPS.",
        },
        {
            "id": "faiss_geotagged_retrieval",
            "layer": "ml",
            "status": "partial",
            "detail": (
                "Wired in ImageRetrieval (merge with landmark softmax) when FAISS_GEOTAG_INDEX_PATH + "
                "FAISS_GEOTAG_COORDS_NPY_PATH are set and faiss-cpu is installed; no prebuilt index in repo."
            ),
        },
        {
            "id": "lvlm_chain_of_thought",
            "layer": "ml",
            "status": "planned",
            "detail": "No GPT-4V/Gemini integration in-repo.",
        },
        # Features / cues
        {
            "id": "feature_catalog_dimensions",
            "layer": "features",
            "status": "implemented",
            "detail": "geolocation_feature_catalog.py — versioned semantic catalog (500+ cue templates).",
        },
        {
            "id": "scene_clip_softmax_cues",
            "layer": "features",
            "status": "implemented",
            "detail": "Scene, globe regional, infrastructure/energy, cultural-economic CLIP banks.",
        },
        {
            "id": "extractor_fifty_plus_specialists",
            "layer": "features",
            "status": "partial",
            "detail": "FeatureExtractor uses heuristics (vegetation green ratio, etc.); not 50 separate CNN specialists.",
        },
        {
            "id": "ocr_script_speed_limits",
            "layer": "features",
            "status": "partial",
            "detail": "Generic OCR toggle in pipeline where wired; no dedicated script/lane-marking detectors.",
        },
        {
            "id": "postprocess_reverse_geocode",
            "layer": "server",
            "status": "implemented",
            "detail": "OpenStreetMap Nominatim reverse geocode (not Google Maps Geocoding by default).",
        },
        {
            "id": "wikipedia_dem_validation",
            "layer": "server",
            "status": "implemented",
            "detail": "Wikipedia geosearch + OpenTopoData relief + optional CLIP vs article lead.",
        },
        {
            "id": "geonames_streetclip_gazetteer",
            "layer": "server",
            "status": "implemented",
            "detail": "GeoNames-derived JSON; startup autoload + bbox filter around GeoCLIP prior.",
        },
        {
            "id": "privacy_face_plate_blur",
            "layer": "server",
            "status": "planned",
            "detail": "Not implemented server-side; suitable for on-device ML Kit before upload.",
        },
        {
            "id": "multi_image_album_lstm",
            "layer": "ml",
            "status": "planned",
            "detail": "Single-image /predict only; no temporal album fusion.",
        },
        {
            "id": "benchmarks_im2gps_yfcc",
            "layer": "eval",
            "status": "planned",
            "detail": "No bundled benchmark harness; models are pretrained packages + zero-shot.",
        },
    ]
    counts = {"implemented": 0, "partial": 0, "planned": 0, "not_applicable": 0}
    for it in items:
        counts[it["status"]] = counts.get(it["status"], 0) + 1

    return {
        "repository_scope": (
            "This repo is a Python FastAPI geolocation server and static web UI. "
            "It implements the cloud half of the hybrid architecture; Android-specific "
            "components must be a separate client."
        ),
        "reference": "Photo Geolocation System Design — Executive Summary (internal)",
        "items": items,
        "status_counts": counts,
    }
