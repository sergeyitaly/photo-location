"""
Street View image verification service.

Fetches Google Street View panoramas at predicted coordinates,
encodes them with CLIP, and compares visual similarity to the
uploaded photo. Can promote alternative predictions when a
better Street View match is found.

This is one of the highest-accuracy verification signals available:
- Ground-level perspective match
- Building facade verification
- Road geometry confirmation
"""

from __future__ import annotations

import asyncio
import logging
import io
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Limit concurrent Street View API requests to avoid rate limits and long waits
_STREETVIEW_SEMAPHORE = asyncio.Semaphore(3)


def _has_streetview_api_key(settings) -> bool:
    """Check if Google Maps API key is configured."""
    key = getattr(settings, "google_maps_api_key", None)
    return bool(key and str(key).strip() and str(key).strip().lower() not in ("", "none", "null"))


def _build_streetview_url(
    lat: float,
    lon: float,
    api_key: str,
    size: str = "640x640",
    fov: int = 90,
    pitch: int = 0,
    heading: Optional[int] = None,
) -> str:
    """Build Google Street View Static API URL."""
    base = "https://maps.googleapis.com/maps/api/streetview"
    h = f"&heading={heading}" if heading is not None else ""
    return (
        f"{base}?size={size}&location={lat},{lon}"
        f"&fov={fov}&pitch={pitch}{h}&key={api_key}"
    )


async def _fetch_streetview_image(
    lat: float,
    lon: float,
    api_key: str,
    heading: Optional[int] = None,
    timeout: float = 8.0,
) -> Optional[np.ndarray]:
    """Fetch Street View image as RGB numpy array."""
    try:
        import httpx

        url = _build_streetview_url(lat, lon, api_key, heading=heading)
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url)
            if r.status_code != 200:
                logger.warning("Street View HTTP %s at %.4f,%.4f", r.status_code, lat, lon)
                return None
            # Check for Google's error image (no panorama available)
            content_type = r.headers.get("content-type", "")
            if "image" not in content_type:
                logger.warning("Street View non-image response: %s", content_type)
                return None
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            return np.array(img, dtype=np.uint8)
    except Exception as e:
        logger.warning("Street View fetch failed at %.4f,%.4f: %s", lat, lon, e)
        return None


def _clip_similarity(
    image_a: np.ndarray,
    image_b: np.ndarray,
    settings,
) -> Optional[float]:
    """Compute CLIP cosine similarity between two images."""
    try:
        from app.inference.clip_common import encode_image_embedding

        emb_a = encode_image_embedding(image_a, model_id=settings.globe_clip_model_id)
        emb_b = encode_image_embedding(image_b, model_id=settings.globe_clip_model_id)
        if emb_a is None or emb_b is None:
            return None
        # Cosine similarity
        dot = float(np.dot(emb_a, emb_b))
        norm_a = float(np.linalg.norm(emb_a))
        norm_b = float(np.linalg.norm(emb_b))
        if norm_a == 0 or norm_b == 0:
            return None
        return dot / (norm_a * norm_b)
    except Exception as e:
        logger.warning("CLIP similarity failed: %s", e)
        return None


def _heading_variations() -> List[Optional[int]]:
    """Headings to try for Street View (None = default facing)."""
    return [None, 0, 90, 180, 270]


async def _evaluate_candidate(
    idx: int,
    cand: Any,
    query_image: np.ndarray,
    api_key: str,
    settings: Any,
) -> Dict[str, Any]:
    """Evaluate one candidate location against all Street View headings in parallel."""
    lat, lon = float(cand.latitude), float(cand.longitude)
    headings = _heading_variations()

    async def _fetch_and_score(heading: Optional[int]) -> Optional[Dict[str, Any]]:
        async with _STREETVIEW_SEMAPHORE:
            sv_img = await _fetch_streetview_image(lat, lon, api_key, heading=heading, timeout=8.0)
            if sv_img is None:
                return None
            sim = _clip_similarity(query_image, sv_img, settings)
            if sim is None:
                return None
            return {"heading": heading, "similarity": round(sim, 4)}

    # Fetch all headings in parallel (limited by semaphore)
    results = await asyncio.gather(*[_fetch_and_score(h) for h in headings], return_exceptions=True)

    all_heads: List[Dict[str, Any]] = []
    best_sim = -1.0
    best_head: Optional[int] = None

    for res in results:
        if isinstance(res, Exception):
            continue
        if res is None:
            continue
        all_heads.append(res)
        if res["similarity"] > best_sim:
            best_sim = res["similarity"]
            best_head = res["heading"]

    return {
        "index": idx,
        "lat": lat,
        "lon": lon,
        "best_heading": best_head,
        "best_similarity": round(best_sim, 4) if best_sim >= 0 else None,
        "all_headings": all_heads,
    }


async def _streetview_verify_impl(
    query_image: np.ndarray,
    primary: Any,
    alternatives: List[Any],
    settings: Any,
    similarity_threshold: float = 0.72,
) -> Dict[str, Any]:
    """Implementation of Street View verification."""
    api_key = str(settings.google_maps_api_key).strip()
    # Limit to primary + top 2 alternatives to reduce API calls and latency
    candidates = [(0, primary)] + [(i + 1, alt) for i, alt in enumerate(alternatives[:2])]

    per_candidate: List[Dict[str, Any]] = []
    best_overall_sim = -1.0
    best_overall_idx: Optional[int] = None
    best_overall_heading: Optional[int] = None

    for idx, cand in candidates:
        result = await _evaluate_candidate(idx, cand, query_image, api_key, settings)
        per_candidate.append(result)

        best_sim = result["best_similarity"] if result["best_similarity"] is not None else -1.0
        if best_sim > best_overall_sim:
            best_overall_sim = best_sim
            best_overall_idx = idx
            best_overall_heading = result["best_heading"]

    swapped = False
    chosen_idx = 0
    detail = "No Street View panoramas found for any candidate."

    if best_overall_idx is not None and best_overall_sim >= similarity_threshold:
        chosen_idx = best_overall_idx
        swapped = chosen_idx != 0
        if swapped:
            detail = (
                f"Promoted candidate #{chosen_idx} based on Street View CLIP similarity "
                f"{best_overall_sim:.3f} (threshold {similarity_threshold})."
            )
        else:
            detail = (
                f"Primary candidate confirmed by Street View CLIP similarity "
                f"{best_overall_sim:.3f} (threshold {similarity_threshold})."
            )
    elif best_overall_idx is not None:
        detail = (
            f"Best Street View similarity {best_overall_sim:.3f} below threshold "
            f"{similarity_threshold}. Keeping primary."
        )

    return {
        "enabled": True,
        "api_configured": True,
        "candidates_evaluated": len(candidates),
        "best_similarity": round(best_overall_sim, 4) if best_overall_sim >= 0 else None,
        "best_candidate_index": best_overall_idx,
        "best_heading": best_overall_heading,
        "swapped_primary": swapped,
        "chosen_candidate_index": chosen_idx,
        "similarity_threshold": similarity_threshold,
        "detail": detail,
        "per_candidate": per_candidate,
    }


async def streetview_verify_predictions(
    query_image: np.ndarray,
    primary: Any,
    alternatives: List[Any],
    settings: Any,
    similarity_threshold: float = 0.72,
) -> Dict[str, Any]:
    """
    Verify predictions by comparing query image against Street View
    panoramas at each candidate location.

    Returns:
        {
            "enabled": bool,
            "api_configured": bool,
            "candidates_evaluated": int,
            "best_similarity": float | None,
            "best_candidate_index": int | None,
            "best_heading": int | None,
            "swapped_primary": bool,
            "chosen_candidate_index": int | None,
            "detail": str,
            "per_candidate": [
                {
                    "index": int,
                    "lat": float,
                    "lon": float,
                    "best_heading": int | None,
                    "best_similarity": float,
                    "all_headings": [{"heading": int, "similarity": float}],
                }
            ],
        }
    """
    if not _has_streetview_api_key(settings):
        return {
            "enabled": False,
            "api_configured": False,
            "summary": "Google Maps API key not configured. Set GOOGLE_MAPS_API_KEY.",
        }

    try:
        # Overall timeout: 30 seconds max for entire Street View verification
        return await asyncio.wait_for(
            _streetview_verify_impl(
                query_image,
                primary,
                alternatives,
                settings,
                similarity_threshold=similarity_threshold,
            ),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        logger.warning("Street View verification timed out after 30s")
        return {
            "enabled": False,
            "api_configured": True,
            "skipped_reason": "timeout",
            "summary": "Street View verification timed out (30s limit exceeded). Keeping primary prediction.",
        }
