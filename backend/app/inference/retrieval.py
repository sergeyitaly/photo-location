"""Landmark pins from CLIP softmax + optional FAISS neighbours (offline-built index)."""
import logging
import numpy as np
from typing import List, Dict, Any, Optional

from app.config import settings as app_settings
from app.models.schemas import LocationPrediction

logger = logging.getLogger(__name__)


def _score_to_confidence(score: float) -> float:
    """Map Faiss inner-product cosine score (typically [-1, 1]) to [0.05, 0.98]."""
    if score <= 1.0 and score >= -1.0:
        return float(np.clip((score + 1.0) / 2.0, 0.05, 0.98))
    return float(np.clip(score, 0.05, 0.98))


class ImageRetrieval:
    """
    CLIP similarity over curated landmark prompts with known coordinates.
    Optional Faiss nearest-neighbour search when ``faiss_geotag_*`` paths are configured
    (offline index built with the same CLIP image encoder).
    """

    def __init__(self) -> None:
        logger.info(
            "ImageRetrieval: CLIP landmark softmax + optional Faiss geotag index."
        )

    def retrieve_from_image(
        self,
        image_array: np.ndarray,
        *,
        clip_model_id: Optional[str],
        k: int = 5,
        confidence_threshold: float = 0.05,
    ) -> List[LocationPrediction]:
        preds: List[LocationPrediction] = []
        cid = (clip_model_id or "").strip() or getattr(app_settings, "globe_clip_model_id", "") or ""

        if cid:
            try:
                from app.inference.zero_shot_geo import clip_landmark_predictions

                preds = clip_landmark_predictions(
                    image_array,
                    cid,
                    top_k=k,
                    min_prob=confidence_threshold,
                )
                if not preds:
                    logger.warning(
                        "ImageRetrieval: CLIP returned no landmark above threshold %.4f.",
                        confidence_threshold,
                    )
            except Exception as e:
                logger.warning("ImageRetrieval: CLIP landmark softmax failed: %s", e)

        nn_preds: List[LocationPrediction] = []
        if cid:
            try:
                from app.inference.clip_common import encode_image_embedding
                from app.inference.faiss_georetrieval import load_faiss_bundle, search_neighbors

                if load_faiss_bundle(app_settings) is None:
                    pass
                else:
                    emb = encode_image_embedding(image_array, cid)
                    if emb is not None and len(emb) > 0:
                        for lat, lon, score in search_neighbors(
                            app_settings,
                            emb.astype(np.float32),
                            k=min(k, 16),
                        ):
                            conf = _score_to_confidence(score)
                            if conf < confidence_threshold:
                                continue
                            nn_preds.append(
                                LocationPrediction(
                                    latitude=lat,
                                    longitude=lon,
                                    country="Indexed retrieval",
                                    city=f"ANN cosine={score:.3f}",
                                    confidence=conf,
                                    distance_confidence_km=max(15.0, 250.0 * (1.05 - conf)),
                                )
                            )
            except Exception as e:
                logger.debug("ImageRetrieval: Faiss merge skipped: %s", e)

        merged = preds + nn_preds
        merged.sort(key=lambda p: p.confidence, reverse=True)
        if merged:
            return merged[: max(k, 1)]
        return []

    def retrieve_similar_locations(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        confidence_threshold: float = 0.3,
    ) -> List[LocationPrediction]:
        """Nearest indexed (lat, lon) from Faiss when bundle is configured."""
        if query_embedding is None or len(query_embedding) == 0:
            return []
        try:
            from app.inference.faiss_georetrieval import search_neighbors

            rows = search_neighbors(app_settings, np.asarray(query_embedding, dtype=np.float32), k=k)
            out: List[LocationPrediction] = []
            for lat, lon, score in rows:
                conf = _score_to_confidence(score)
                if conf < confidence_threshold:
                    continue
                out.append(
                    LocationPrediction(
                        latitude=lat,
                        longitude=lon,
                        country="Indexed retrieval",
                        city=f"ANN cosine={score:.3f}",
                        confidence=conf,
                        distance_confidence_km=max(15.0, 250.0 * (1.05 - conf)),
                    )
                )
            return out
        except Exception as e:
            logger.debug("retrieve_similar_locations: %s", e)
            return []

    def encode_image(self, image_array: np.ndarray, clip_model_id: Optional[str] = None) -> np.ndarray:
        """CLIP image embedding when available; otherwise empty (no histogram padding)."""
        if not clip_model_id:
            return np.zeros(0, dtype=np.float64)
        try:
            from app.inference.clip_common import encode_image_embedding

            emb = encode_image_embedding(image_array, clip_model_id)
            if emb is not None and len(emb) > 0:
                out = emb.astype(np.float64)
                if len(out) < 512:
                    out = np.pad(out, (0, 512 - len(out)))
                return out[:512]
        except Exception as e:
            logger.warning("CLIP image encode failed: %s", e)
        return np.zeros(0, dtype=np.float64)

    def get_vector_db_stats(self) -> Dict[str, Any]:
        from app.inference.faiss_georetrieval import faiss_bundle_stats

        fs = faiss_bundle_stats(app_settings)
        n = int(fs.get("vectors") or 0)
        return {
            "model_type": "CLIP landmark softmax + optional Faiss IP index",
            "indexed_geotagged_images": n,
            "approach": (
                "Configure FAISS_GEOTAG_INDEX_PATH + FAISS_GEOTAG_COORDS_NPY_PATH (same CLIP dim as GLOBE_CLIP_MODEL_ID)."
            ),
            "landmark_prompts": "see zero_shot_geo.LANDMARK_ENTRIES",
            "faiss": fs,
        }
