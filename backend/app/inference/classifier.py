"""Coarse country prior from CLIP zero-shot only (no synthetic coordinate fallbacks)."""
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from app.models.schemas import LocationPrediction
from app.config import settings as app_settings

logger = logging.getLogger(__name__)


class CountryClassifier:
    """
    CLIP text–image softmax over “a photograph taken in {country}” prompts.
    If torch/transformers are missing or the run fails, returns an empty list
    (callers must not invent lat/lon from colour statistics).
    """

    def __init__(self) -> None:
        from app.inference.zero_shot_geo import COUNTRY_ENTRIES

        self._num_countries = len(COUNTRY_ENTRIES)
        logger.info("CountryClassifier: CLIP zero-shot over %d country priors", self._num_countries)

    def predict_from_image(
        self,
        image_array: np.ndarray,
        *,
        clip_model_id: Optional[str],
        confidence_threshold: float = 0.008,
        top_k: int = 8,
    ) -> List[LocationPrediction]:
        effective_id = (clip_model_id or "").strip() or (app_settings.globe_clip_model_id or "").strip()
        if not effective_id:
            logger.info("CountryClassifier: no CLIP model id (set GLOBE_CLIP_MODEL_ID) — skipping country softmax.")
            return []

        try:
            from app.inference.zero_shot_geo import clip_country_predictions

            preds = clip_country_predictions(
                image_array,
                effective_id,
                top_k=top_k,
                min_prob=confidence_threshold,
            )
            if preds:
                return preds
            logger.warning(
                "CountryClassifier: CLIP returned no country above threshold %.4f "
                "(install torch+transformers or lower threshold).",
                confidence_threshold,
            )
        except Exception as e:
            logger.warning("CountryClassifier: CLIP country softmax failed: %s", e)

        return []

    def get_confidence_distribution(self) -> Dict[str, Any]:
        return {
            "model_type": "CLIP zero-shot country softmax (no histogram fallback)",
            "num_country_prompts": self._num_countries,
            "approach": "Contrastive text prompts × image; softmax over fixed country list",
        }
