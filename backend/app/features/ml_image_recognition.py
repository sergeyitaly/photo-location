"""High-level ML / image recognition summary using shared CLIP model."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from app.config import Settings
from app.data.ml_recognition_prompts import ML_RECOGNITION_PROMPTS
from app.inference.clip_common import clip_softmax_for_prompts, is_clip_runtime_available


def compute_ml_image_recognition(
    image_rgb: np.ndarray,
    *,
    settings: Settings,
    top_n: int = 15,
) -> Dict[str, Any]:
    """
    Scene/object-style softmax readout — same contrastive machinery as cue panels,
    fixed prompt list for a compact “recognition” summary.
    """
    mid = settings.globe_clip_model_id
    if not is_clip_runtime_available():
        return {
            "clip_available": False,
            "model_id": mid,
            "methodology": (
                "CLIP image–text softmax is disabled: install torch + transformers "
                "and restart the backend to enable neural image recognition."
            ),
            "scene_and_object_labels": [],
            "note": None,
        }

    pairs = clip_softmax_for_prompts(image_rgb, ML_RECOGNITION_PROMPTS, model_id=mid)
    labels = [{"label": lab, "score": round(float(sc), 6)} for lab, sc in pairs[: max(1, top_n)]]
    return {
        "clip_available": True,
        "model_id": mid,
        "methodology": (
            "Open-vocabulary contrastive classification: softmax probabilities over "
            "the curated prompt list only (not global object detection)."
        ),
        "scene_and_object_labels": labels,
        "note": None,
    }
