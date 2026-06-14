"""
When StreetCLIP's top gazetteer city differs from GeoCLIP rank-1 GPS, promote the GeoCLIP hypothesis
whose coordinates are closest to StreetCLIP's city centroid (same country, capital vs regional).
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

from app.config import Settings
from app.models.schemas import LocationPrediction

logger = logging.getLogger(__name__)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def reconcile_fusion_with_geoclip_streetclip(
    merged: List[LocationPrediction],
    geo_preds: List[LocationPrediction],
    sc_preds: List[LocationPrediction],
    *,
    settings: Settings,
) -> List[LocationPrediction]:
    """
    If enabled: among GeoCLIP ranks 0..scan_top-1, pick the pin closest to StreetCLIP top-1's
    gazetteer coordinates; if that rank beats rank-0 by ``min_improvement_km`` and separation rules,
    move that fused hypothesis to primary (fixes Kyiv-vs-Zhytomyr-style dominance when GeoCLIP scatters).
    """
    if not merged:
        return merged
    if not getattr(settings, "hybrid_streetclip_alt_geoclip_reconcile", False):
        return merged
    if not geo_preds or not sc_preds:
        return merged

    scan = max(1, min(int(settings.hybrid_alt_geoclip_scan_top), len(geo_preds)))
    sc0 = sc_preds[0]
    # StreetCLIP lat/lon match gazetteer centroids — use as anchor for “named city”
    t_lat, t_lon = float(sc0.latitude), float(sc0.longitude)

    dists: List[float] = []
    for i in range(scan):
        g = geo_preds[i]
        dists.append(_haversine_km(t_lat, t_lon, float(g.latitude), float(g.longitude)))

    best_i = min(range(scan), key=lambda i: dists[i])
    if best_i == 0:
        return merged

    d0 = dists[0]
    db = dists[best_i]
    improvement_km = d0 - db
    sep_km = _haversine_km(
        float(geo_preds[0].latitude),
        float(geo_preds[0].longitude),
        float(geo_preds[best_i].latitude),
        float(geo_preds[best_i].longitude),
    )

    if improvement_km < float(settings.hybrid_alt_geoclip_min_improvement_km):
        return merged
    if sep_km < float(settings.hybrid_alt_geoclip_min_rank_sep_km):
        return merged
    if float(sc0.confidence) < float(settings.hybrid_alt_geoclip_min_softmax_alt_conf):
        return merged
    if float(geo_preds[best_i].confidence) < float(settings.hybrid_alt_geoclip_min_geoclip_rank_conf):
        return merged

    # Confidence: prefer fused row at this GPS if present
    dedupe = int(getattr(settings, "fusion_dedupe_decimals", 3))
    key_b = (
        round(geo_preds[best_i].latitude, dedupe),
        round(geo_preds[best_i].longitude, dedupe),
    )
    fused_here: Optional[LocationPrediction] = None
    for p in merged:
        k = (round(p.latitude, dedupe), round(p.longitude, dedupe))
        if k == key_b:
            fused_here = p
            break

    base_conf = float(fused_here.confidence) if fused_here else float(geo_preds[best_i].confidence)
    promoted = LocationPrediction(
        latitude=float(geo_preds[best_i].latitude),
        longitude=float(geo_preds[best_i].longitude),
        country=str(sc0.country),
        city=str(sc0.city),
        confidence=min(1.0, max(base_conf, float(sc0.confidence) * 0.85)),
        distance_confidence_km=geo_preds[best_i].distance_confidence_km,
    )

    logger.info(
        "Hybrid reconcile: promoted GeoCLIP rank %s over rank 1 (Δ≈%.1f km vs StreetCLIP %s, %s); sep=%.1f km",
        best_i + 1,
        improvement_km,
        sc0.city,
        sc0.country,
        sep_km,
    )

    others: List[LocationPrediction] = []
    seen = {(round(promoted.latitude, 2), round(promoted.longitude, 2))}
    for p in merged:
        k = (round(p.latitude, 2), round(p.longitude, 2))
        if k in seen:
            continue
        others.append(p)
        seen.add(k)

    # Keep old primary as an alternative unless duplicate
    old = merged[0]
    old_k = (round(old.latitude, 2), round(old.longitude, 2))
    if old_k not in seen:
        others.insert(0, old)

    return [promoted] + others
