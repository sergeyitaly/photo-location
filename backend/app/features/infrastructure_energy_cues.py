"""CLIP readouts for infrastructure, energy hardware, and visual economic-activity proxies."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.config import Settings
from app.data.infrastructure_energy_clip_prompts import (
    DISCLAIMER_INFRASTRUCTURE_ENERGY,
    INFRASTRUCTURE_ENERGY_CLIP_BANKS,
    METHODOLOGY_INFRASTRUCTURE_ENERGY,
)
from app.inference.clip_common import clip_softmax_for_prompts, is_clip_runtime_available

logger = logging.getLogger(__name__)


def compute_infrastructure_energy_cues(
    image_rgb: np.ndarray,
    *,
    settings: Settings,
) -> Dict[str, Any]:
    """Returns serialized cue bundle for API (may be skipped when disabled or CLIP unavailable)."""
    top_n = max(3, min(20, settings.infrastructure_energy_clip_top_n))
    mid = settings.globe_clip_model_id

    if image_rgb is None or image_rgb.size == 0:
        return _empty(mid, "Empty image.")

    if not settings.use_infrastructure_energy_clip:
        return {
            "enabled": False,
            "skipped_reason": "disabled_in_settings",
            "methodology": METHODOLOGY_INFRASTRUCTURE_ENERGY,
            "disclaimer": DISCLAIMER_INFRASTRUCTURE_ENERGY,
            "clip_banks_detail": [],
            "clip_available": False,
            "clip_model_id": mid,
            "interpretive_summary": "Infrastructure-energy CLIP bundle disabled (USE_INFRASTRUCTURE_ENERGY_CLIP).",
        }

    if not is_clip_runtime_available():
        return {
            "enabled": True,
            "skipped_reason": "torch_transformers_missing",
            "methodology": METHODOLOGY_INFRASTRUCTURE_ENERGY,
            "disclaimer": DISCLAIMER_INFRASTRUCTURE_ENERGY,
            "clip_banks_detail": [],
            "clip_available": False,
            "clip_model_id": mid,
            "interpretive_summary": "Install torch + transformers on the server to run infrastructure-energy CLIP banks.",
        }

    banks_out: List[Dict[str, Any]] = []
    clip_ok = False
    try:
        for bank in INFRASTRUCTURE_ENERGY_CLIP_BANKS:
            pairs = clip_softmax_for_prompts(image_rgb, bank["prompts"], model_id=mid)
            rows = pairs[:top_n]
            banks_out.append(
                {
                    "bank_id": bank["id"],
                    "title": bank["title"],
                    "categories": [
                        {
                            "category_id": bank["id"],
                            "title": bank["title"],
                            "items": [
                                {"label": x[0], "confidence": x[1], "source": "clip_softmax"} for x in rows
                            ],
                        }
                    ],
                }
            )
        clip_ok = True
    except Exception as e:
        logger.warning("Infrastructure-energy CLIP failed: %s", e, exc_info=True)

    summary = _summary_from_banks(banks_out)

    return {
        "enabled": True,
        "skipped_reason": None,
        "methodology": METHODOLOGY_INFRASTRUCTURE_ENERGY,
        "disclaimer": DISCLAIMER_INFRASTRUCTURE_ENERGY,
        "clip_banks_detail": banks_out,
        "clip_available": clip_ok,
        "clip_model_id": mid if clip_ok else None,
        "interpretive_summary": summary,
    }


def _empty(model_id: str, note: str) -> Dict[str, Any]:
    return {
        "enabled": True,
        "skipped_reason": None,
        "methodology": METHODOLOGY_INFRASTRUCTURE_ENERGY,
        "disclaimer": DISCLAIMER_INFRASTRUCTURE_ENERGY,
        "clip_banks_detail": [],
        "clip_available": False,
        "clip_model_id": model_id,
        "interpretive_summary": note,
    }


def _summary_from_banks(banks_out: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for bank in banks_out[:4]:
        cats = bank.get("categories") or []
        if not cats:
            continue
        items = cats[0].get("items") or []
        if items:
            lab = str(items[0].get("label", ""))[:72]
            conf = items[0].get("confidence")
            try:
                pct = float(conf) * 100.0
            except (TypeError, ValueError):
                pct = 0.0
            parts.append(f"{bank.get('title', 'Bank')}: “{lab}…” (~{pct:.1f}% within bank)")
    if not parts:
        return "No dominant infrastructure cue in softmax readout; scene may lack clear energy/grid hardware."
    return " ".join(parts)
