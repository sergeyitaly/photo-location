"""
CLIP softmax readouts per macro-region category. Does not affect coordinates.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.data.globe_regional_labels import GLOBE_REGION_DEFINITIONS
from app.inference.clip_common import (
    clip_softmax_for_prompts,
    is_clip_runtime_available,
)

logger = logging.getLogger(__name__)


def is_globe_clip_runtime_available() -> bool:
    """True if torch+transformers can be imported (CLIP can be loaded on first use)."""

    return is_clip_runtime_available()


def compute_globe_regional_hints(
    image_rgb,
    *,
    model_id: str = "openai/clip-vit-base-patch32",
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    For every region/category, softmax over prompts in that category vs the image.

    Returns:
        regions: nested structure with per-prompt probabilities (sum to 1 within category).
        clip_available: bool
        note: optional install hint
    """
    import numpy as np

    if image_rgb is None or np.asarray(image_rgb).size == 0:
        return {
            "clip_available": False,
            "note": "empty image",
            "regions": [],
        }

    if not is_clip_runtime_available():
        return {
            "clip_available": False,
            "note": "Install torch and transformers to enable CLIP softmax globe cues.",
            "regions": _empty_regions_skeleton(),
        }

    try:
        regions_out: List[Dict[str, Any]] = []

        for region in GLOBE_REGION_DEFINITIONS:
            cat_payloads: List[Dict[str, Any]] = []
            for cat in region["categories"]:
                prompts = cat["prompts"]
                if not prompts:
                    continue
                pairs = clip_softmax_for_prompts(image_rgb, prompts, model_id=model_id)
                rows = pairs[:top_n]
                cat_payloads.append(
                    {
                        "category_id": cat["id"],
                        "title": cat["title"],
                        "items": [{"label": r[0], "confidence": r[1]} for r in rows],
                    }
                )

            regions_out.append(
                {
                    "region_id": region["id"],
                    "title": region["title"],
                    "categories": cat_payloads,
                }
            )

        return {
            "clip_available": True,
            "note": None,
            "model_id": model_id,
            "regions": regions_out,
        }

    except Exception as e:
        logger.warning("Globe regional CLIP hints failed: %s", e, exc_info=True)
        return {
            "clip_available": False,
            "note": f"CLIP inference error: {e}",
            "regions": _empty_regions_skeleton(),
        }


def _empty_regions_skeleton() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for region in GLOBE_REGION_DEFINITIONS:
        cats = [
            {"category_id": c["id"], "title": c["title"], "items": []} for c in region["categories"]
        ]
        out.append({"region_id": region["id"], "title": region["title"], "categories": cats})
    return out
