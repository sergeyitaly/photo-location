"""
Preload Hugging Face CLIP, StreetCLIP, and GeoCLIP weights at process start when configured.

Avoids multi-minute delays on the first HTTP /predict after server boot (downloads still need network once).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from app.config import Settings
from app.inference.clip_common import clip_softmax_for_prompts, is_clip_runtime_available, load_clip

logger = logging.getLogger(__name__)


def warmup_torch_models(settings: Settings) -> Dict[str, Any]:
    """
    Load all enabled PyTorch stacks into memory. Safe to call even if CUDA/MPS unavailable (CPU fallback).

    Returns a report dict for logging; failures are logged but do not raise (non-fatal startup).
    """
    report: Dict[str, Any] = {
        "preload_requested": settings.preload_torch_models_at_startup,
        "clip_globe_model_id": settings.globe_clip_model_id,
        "clip_base_loaded": False,
        "clip_smoke_ok": False,
        "geoclip_loaded": False,
        "streetclip_loaded": False,
        "warnings": [],
    }

    if not settings.preload_torch_models_at_startup:
        report["skipped"] = "preload_torch_models_at_startup=false"
        logger.info("Torch warmup skipped (PRELOAD_TORCH_MODELS_AT_STARTUP=false)")
        return report

    if not is_clip_runtime_available():
        report["skipped"] = "torch_or_transformers_import_failed"
        logger.warning("Torch warmup skipped: torch and/or transformers not importable")
        return report

    mid = settings.globe_clip_model_id.strip()
    if not mid:
        report["warnings"].append("globe_clip_model_id empty")
        return report

    try:
        load_clip(mid)
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        clip_softmax_for_prompts(dummy, ["outdoor daylight scene", "indoor scene"], model_id=mid)
        report["clip_base_loaded"] = True
        report["clip_smoke_ok"] = True
        logger.info("Warmup: base CLIP loaded and smoke-forward OK (%s)", mid)
    except Exception as e:
        report["warnings"].append(f"clip_base: {e}")
        logger.warning("Warmup: base CLIP failed (non-fatal): %s", e, exc_info=True)

    if settings.use_geoclip:
        try:
            from app.inference.geoclip_inference import warmup_geoclip_model

            warmup_geoclip_model()
            report["geoclip_loaded"] = True
            logger.info("Warmup: GeoCLIP loaded")
        except Exception as e:
            report["warnings"].append(f"geoclip: {e}")
            logger.warning("Warmup: GeoCLIP failed (non-fatal): %s", e, exc_info=True)

    if settings.use_streetclip:
        sid = settings.streetclip_model_id.strip()
        if sid:
            try:
                from app.inference.streetclip_inference import warmup_streetclip_model

                warmup_streetclip_model(sid)
                report["streetclip_loaded"] = True
                logger.info("Warmup: StreetCLIP loaded (%s)", sid)
            except Exception as e:
                report["warnings"].append(f"streetclip: {e}")
                logger.warning("Warmup: StreetCLIP failed (non-fatal): %s", e, exc_info=True)

    if settings.use_streetclip or settings.use_multi_resolution_grid_search:
        try:
            from app.data.gazetteer_loader import (
                load_gazetteer_rows_from_disk,
                streetclip_gazetteer_json_resolved,
            )

            gaz_path = streetclip_gazetteer_json_resolved(settings)
            if gaz_path is not None:
                rows = load_gazetteer_rows_from_disk(settings)
                report["gazetteer_rows"] = len(rows)
                report["gazetteer_path"] = str(gaz_path)
                logger.info("Warmup: StreetCLIP gazetteer preloaded (%s rows)", len(rows))
        except Exception as e:
            report["warnings"].append(f"gazetteer: {e}")
            logger.warning("Warmup: gazetteer preload failed (non-fatal): %s", e, exc_info=True)

    return report
