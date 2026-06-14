"""StreetCLIP (geolocal/StreetCLIP) zero-shot over gazetteer cities — chunked logits merge."""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from app.config import Settings
from app.data.gazetteer_loader import (
    filter_gazetteer_for_streetclip,
    haversine_km,
    load_gazetteer_rows_from_disk,
)
from app.inference.streetclip_chunked_search import (
    ProgressCallback,
    _predictions_from_heap,
    score_gazetteer_chunked_early_stop,
)
from app.models.schemas import LocationPrediction

logger = logging.getLogger(__name__)

_streetclip_status_lock = threading.Lock()
# Updated while `_load_clip_pair` runs (download + RAM); cached loads skip the body.
_streetclip_public: Dict[str, Any] = {
    "phase": "idle",
    "ready": False,
    "message": "",
}


def get_streetclip_load_status(settings: Settings) -> Dict[str, Any]:
    """
    JSON for GET /model/streetclip-load-status (frontend polls during /predict).

    Fine-grained Hugging Face download % is not tracked; ``loading_ram`` covers first-time fetch + load.
    """
    base: Dict[str, Any] = {
        "model_id": settings.streetclip_model_id,
        "use_streetclip": settings.use_streetclip,
        "phase": "idle",
        "ready": False,
        "message": "",
        "percent": None,
        "current_bytes": 0,
        "total_bytes": 0,
        "file": None,
    }
    if not settings.use_streetclip:
        base["phase"] = "disabled"
        base["message"] = "StreetCLIP disabled in server configuration."
        return base
    mid = (settings.streetclip_model_id or "").strip()
    if not mid:
        base["phase"] = "disabled"
        base["message"] = "streetclip_model_id is empty."
        return base
    if not is_transformers_streetclip_available():
        base["phase"] = "disabled"
        base["message"] = "torch/transformers not available on this server."
        return base
    with _streetclip_status_lock:
        merged = {**base, **_streetclip_public.copy()}
    return merged


def is_transformers_streetclip_available() -> bool:
    try:
        import torch  # noqa: F401
        from transformers import CLIPModel  # noqa: F401

        return True
    except Exception:
        return False


def warmup_streetclip_model(model_id: str) -> None:
    """Eager-load StreetCLIP (HF) weights; may download on first boot."""
    mid = (model_id or "").strip()
    if not mid:
        return
    _load_clip_pair(mid)


@lru_cache(maxsize=2)
def _load_clip_pair(model_id: str):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    with _streetclip_status_lock:
        _streetclip_public.update(
            phase="loading_ram",
            ready=False,
            message="Loading StreetCLIP (first run may download large files from Hugging Face)…",
        )
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    )
    try:
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id).to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    except Exception:
        with _streetclip_status_lock:
            _streetclip_public.update(
                phase="error",
                ready=False,
                message="StreetCLIP load failed (see server logs).",
            )
        raise
    with _streetclip_status_lock:
        _streetclip_public.update(
            phase="idle",
            ready=True,
            message="StreetCLIP weights are in memory.",
        )
    return model, processor, device


def _softmax_np(x: np.ndarray) -> np.ndarray:
    import numpy as np

    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


def score_labels_with_streetclip(
    image_rgb: np.ndarray,
    labels: List[str],
    *,
    settings: Settings,
) -> Optional[np.ndarray]:
    """Raw image-vs-text logits for arbitrary labels using the configured StreetCLIP model."""
    if not settings.use_streetclip or not is_transformers_streetclip_available():
        return None
    if image_rgb is None or image_rgb.size == 0 or not labels:
        return None

    mid = settings.streetclip_model_id.strip()
    if not mid:
        return None

    pil = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
    chunk = max(8, min(settings.streetclip_gazetteer_chunk_size, 96))

    try:
        model, processor, device = _load_clip_pair(mid)
    except Exception as e:
        logger.warning("StreetCLIP load failed (%s): %s", mid, e)
        return None

    try:
        import torch

        scored_parts: List[np.ndarray] = []
        for start in range(0, len(labels), chunk):
            batch_labels = labels[start : start + chunk]
            inputs = processor(text=batch_labels, images=pil, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits_per_image[0].float().cpu().numpy()
            scored_parts.append(np.asarray(logits, dtype=np.float64))
        if not scored_parts:
            return None
        return np.concatenate(scored_parts, axis=0)
    except Exception as e:
        logger.warning("StreetCLIP forward failed: %s", e)
        return None


def predict_locations_streetclip_gazetteer(
    image_rgb: np.ndarray,
    *,
    settings: Settings,
    top_k: int = 5,
    geo_prior: Optional[Tuple[float, float]] = None,
    country_allowlist: Optional[List[str]] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> List[LocationPrediction]:
    """
    Chunked forward: compare raw logits across gazetteer labels (large JSON + GeoCLIP bbox filter);
    top-k global by logit, then softmax over top `min(16, len)` logits for calibration.

    When early-stop is enabled, search runs country-by-country and stops if later batches peak
    below earlier ones (wrong direction); best-so-far cities are kept as anchors.
    """
    if not settings.use_streetclip or not is_transformers_streetclip_available():
        return []
    if image_rgb is None or image_rgb.size == 0:
        return []

    rows = load_gazetteer_rows_from_disk(settings)
    rows = filter_gazetteer_for_streetclip(
        rows,
        settings=settings,
        geo_prior=geo_prior,
        country_allowlist=country_allowlist,
    )
    if not rows:
        return []

    chunk_size = max(8, min(int(settings.streetclip_gazetteer_chunk_size), 96))
    use_early = bool(getattr(settings, "streetclip_early_stop_enabled", True))
    if use_early or progress_callback is not None or len(rows) > chunk_size:
        ranked, _meta = score_gazetteer_chunked_early_stop(
            image_rgb,
            rows,
            settings=settings,
            geo_prior=geo_prior,
            progress_callback=progress_callback,
        )
        if not ranked:
            return []
        return _predictions_from_heap(
            ranked, rows, top_k=top_k, geo_prior=geo_prior, settings=settings
        )

    labels_all: List[str] = [f"{c['city']}, {c['country']}" for c in rows]
    logits = score_labels_with_streetclip(image_rgb, labels_all, settings=settings)
    if logits is None:
        return []
    scored: List[Tuple[float, int]] = [(float(logit), idx) for idx, logit in enumerate(logits)]
    if not scored:
        return []
    scored.sort(key=lambda t: -t[0])
    top_n = min(16, len(scored))
    top_slice = scored[:top_n]
    logits_vec = np.array([t[0] for t in top_slice], dtype=np.float64)
    probs = _softmax_np(logits_vec)

    bbox_km = float(getattr(settings, "streetclip_gazetteer_bbox_lat_deg", 2.0)) * 111.0

    out: List[LocationPrediction] = []
    for rank, ((logit, gidx), pr) in enumerate(zip(top_slice, probs)):
        if rank >= top_k:
            break
        row = rows[gidx]
        rlat, rlon = float(row["lat"]), float(row["lon"])
        if geo_prior:
            dist_km = haversine_km(geo_prior[0], geo_prior[1], rlat, rlon)
            dist_conf = min(50.0, max(10.0, dist_km * 0.35 + 8.0))
        else:
            dist_conf = min(90.0, max(25.0, bbox_km * 0.55))
        out.append(
            LocationPrediction(
                latitude=rlat,
                longitude=rlon,
                country=str(row["country"]),
                city=str(row["city"]),
                confidence=float(pr),
                distance_confidence_km=dist_conf,
            )
        )
    return out
