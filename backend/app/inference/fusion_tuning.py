"""
Dynamic fusion weights: trust StreetCLIP when its margin is strong; damp GeoCLIP when scattered.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from app.config import Settings
from app.data.gazetteer_loader import haversine_km
from app.models.schemas import LocationPrediction


def geoclip_top_spread_km(geo_preds: List[LocationPrediction], scan: int = 6) -> float:
    if len(geo_preds) < 2:
        return 0.0
    lead = geo_preds[0]
    lat0, lon0 = float(lead.latitude), float(lead.longitude)
    max_km = 0.0
    for p in geo_preds[:scan]:
        max_km = max(
            max_km,
            haversine_km(lat0, lon0, float(p.latitude), float(p.longitude)),
        )
    return max_km


def streetclip_confidence_margin(sc_preds: List[LocationPrediction]) -> float:
    if len(sc_preds) < 2:
        return 1.0 if sc_preds else 0.0
    return float(sc_preds[0].confidence) - float(sc_preds[1].confidence)


def tune_fusion_source_weights(
    sources: List[Tuple[str, float, List[LocationPrediction]]],
    *,
    geo_preds: List[LocationPrediction],
    sc_preds: List[LocationPrediction],
    settings: Settings,
    image_rgb=None,
) -> List[Tuple[str, float, List[LocationPrediction]]]:
    """
    Return sources with adjusted weights (same structure). Other sources unchanged.
    """
    if not sources:
        return sources

    margin_thr = float(getattr(settings, "streetclip_confident_margin_threshold", 0.10))
    geo_down = float(getattr(settings, "geoclip_downweight_when_streetclip_confident", 0.55))
    sc_boost = float(getattr(settings, "streetclip_boost_when_confident", 1.12))
    spread_thr = float(getattr(settings, "geoclip_scatter_spread_km_threshold", 350.0))
    scatter_down = float(getattr(settings, "geoclip_downweight_when_scattered", 0.72))

    sc_margin = streetclip_confidence_margin(sc_preds)
    geo_spread = geoclip_top_spread_km(geo_preds)

    open_water_boost = 1.0
    if image_rgb is not None and getattr(settings, "fusion_open_water_streetclip_boost", True):
        try:
            from app.features.water_pixels import water_fraction_central

            if water_fraction_central(image_rgb) >= float(
                getattr(settings, "fusion_open_water_fraction_threshold", 0.08)
            ):
                open_water_boost = float(
                    getattr(settings, "fusion_open_water_streetclip_weight_multiplier", 1.06)
                )
        except Exception:
            open_water_boost = 1.0

    tuned: List[Tuple[str, float, List[LocationPrediction]]] = []
    for name, weight, preds in sources:
        w = float(weight)
        if name == "geoclip" and sc_margin >= margin_thr and sc_preds:
            w *= geo_down
        if name == "streetclip" and sc_margin >= margin_thr and sc_preds:
            w *= sc_boost
        if name == "streetclip" and open_water_boost > 1.0 and sc_preds:
            w *= open_water_boost
        if name == "geoclip" and geo_spread >= spread_thr and len(geo_preds) >= 2:
            w *= scatter_down
        tuned.append((name, w, preds))
    return tuned


def should_run_fast_confidence_grid(
    *,
    fast: bool,
    geo_preds: List[LocationPrediction],
    country_predictions: List[LocationPrediction],
    settings: Settings,
) -> bool:
    """Run a reduced grid search in fast mode when GeoCLIP or CLIP country signal is weak."""
    if not fast or not getattr(settings, "fast_mode_confidence_gated_grid", True):
        return False
    if not getattr(settings, "use_multi_resolution_grid_search", True):
        return False
    if float(getattr(settings, "fusion_weight_grid_search", 0)) <= 0:
        return False

    max_conf = float(getattr(settings, "fast_grid_geoclip_max_confidence", 0.14))
    spread_km = float(getattr(settings, "fast_grid_geoclip_spread_km", 100.0))

    if geo_preds:
        if float(geo_preds[0].confidence) <= max_conf:
            return True
        if geoclip_top_spread_km(geo_preds) >= spread_km:
            return True

    if country_predictions:
        top = max(country_predictions, key=lambda p: float(p.confidence))
        country_floor = float(getattr(settings, "fast_grid_clip_country_max_confidence", 0.10))
        if float(top.confidence) <= country_floor:
            return True

    return False
