"""Lightweight CLIP hints for feature analysis (landmarks + architecture), no extra API."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from app.config import settings as app_settings

logger = logging.getLogger(__name__)

_ARCHITECTURE_PROMPTS: List[tuple[str, str]] = [
    ("european_historic", "historic European stone buildings and ornate facades"),
    ("north_american_suburban", "North American suburban houses and wooden siding"),
    ("east_asian_urban", "dense East Asian city apartment blocks and signage"),
    ("mediterranean", "Mediterranean white stucco walls and terracotta roof tiles"),
    ("tropical_open", "tropical open-air structures and palm-lined streets"),
    ("soviet_panel", "Soviet-era concrete panel apartment blocks"),
    ("modern_glass", "modern glass skyscrapers and steel office towers"),
    ("rural_vernacular", "rural vernacular farm buildings and unpaved roads"),
]


def clip_architecture_hint(image_array: np.ndarray, *, clip_model_id: Optional[str] = None) -> Optional[str]:
    """Best-matching architecture style id from CLIP softmax over fixed prompts."""
    model_id = (clip_model_id or app_settings.globe_clip_model_id or "").strip()
    if not model_id:
        return None
    try:
        from app.inference.clip_common import clip_softmax_probs_ordered, is_clip_runtime_available

        if not is_clip_runtime_available():
            return None
        prompts = [p for _id, p in _ARCHITECTURE_PROMPTS]
        probs = clip_softmax_probs_ordered(image_array, prompts, model_id)
        if probs is None or len(probs) != len(_ARCHITECTURE_PROMPTS):
            return None
        best_i = int(max(range(len(probs)), key=lambda i: float(probs[i])))
        if float(probs[best_i]) < 0.09:
            return None
        return _ARCHITECTURE_PROMPTS[best_i][0]
    except Exception as exc:
        logger.debug("CLIP architecture hint skipped: %s", exc)
        return None


def clip_landmark_hints(
    image_array: np.ndarray,
    *,
    clip_model_id: Optional[str] = None,
    top_k: int = 2,
    min_prob: float = 0.50,
) -> List[Dict[str, Any]]:
    """Landmark detections as {name, country, confidence} dicts for FeatureAnalysis.landmarks."""
    model_id = (clip_model_id or app_settings.globe_clip_model_id or "").strip()
    if not model_id:
        return []
    try:
        from app.inference.zero_shot_geo import clip_landmark_predictions

        preds = clip_landmark_predictions(
            image_array,
            model_id,
            top_k=top_k,
            min_prob=min_prob,
        )
        out: List[Dict[str, Any]] = []
        for p in preds:
            out.append(
                {
                    "name": p.city or p.country,
                    "country": p.country,
                    "confidence": float(p.confidence),
                }
            )
        return out
    except Exception as exc:
        logger.debug("CLIP landmark hints skipped: %s", exc)
        return []
