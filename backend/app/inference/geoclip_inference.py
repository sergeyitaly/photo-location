"""GeoCLIP neural GPS retrieval over a pretrained gallery (pip package `geoclip`)."""

from __future__ import annotations

import logging
import os
import tempfile
from functools import lru_cache
from typing import List

import numpy as np
from PIL import Image

from app.models.schemas import LocationPrediction

logger = logging.getLogger(__name__)


def _pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def _load_geoclip():
    from geoclip import GeoCLIP

    model = GeoCLIP(from_pretrained=True)
    model.to(_pick_device())
    model.eval()
    return model


def warmup_geoclip_model() -> None:
    """Eager-load GeoCLIP at application startup when USE_GEOCLIP is enabled."""
    _load_geoclip()


def is_geoclip_available() -> bool:
    try:
        import geoclip  # noqa: F401
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def predict_locations_geoclip(
    image_rgb: np.ndarray,
    *,
    top_k: int = 5,
) -> List[LocationPrediction]:
    """
    Top-k GPS hypotheses from GeoCLIP's contrastive gallery (downloads weights on first use).
    """
    if not is_geoclip_available():
        return []
    if image_rgb is None or image_rgb.size == 0:
        return []

    pil = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        pil.save(path, quality=92, format="JPEG")
        model = _load_geoclip()
        top_pred_gps, top_pred_prob = model.predict(path, top_k=top_k)
    except Exception as e:
        logger.warning("GeoCLIP predict failed: %s", e)
        return []
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    out: List[LocationPrediction] = []
    try:
        import torch

        if isinstance(top_pred_gps, torch.Tensor):
            gps_np = top_pred_gps.detach().cpu().numpy()
        else:
            gps_np = np.asarray(top_pred_gps)
        if isinstance(top_pred_prob, torch.Tensor):
            prob_np = top_pred_prob.detach().cpu().numpy().flatten()
        else:
            prob_np = np.asarray(top_pred_prob).flatten()
    except Exception as e:
        logger.warning("GeoCLIP output parse failed: %s", e)
        return []

    for i in range(min(len(gps_np), len(prob_np), top_k)):
        lat, lon = float(gps_np[i][0]), float(gps_np[i][1])
        p = float(prob_np[i])
        p = max(0.0, min(1.0, p))
        out.append(
            LocationPrediction(
                latitude=lat,
                longitude=lon,
                country="GeoCLIP gallery",
                city=f"GeoCLIP rank {i + 1}",
                confidence=p,
                distance_confidence_km=None,
            )
        )
    return out
