"""Fuse LocationPrediction lists from multiple neural sources with per-source normalization."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from app.models.schemas import (
    LocationPrediction,
    CountryEliminationResult,
    GeoReasoningResult,
    AstronomyConstraints,
)
from app.config import Settings


def fuse_weighted_predictions(
    sources: List[Tuple[str, float, List[LocationPrediction]]],
    *,
    dedupe_decimals: int = 2,
) -> List[LocationPrediction]:
    """
    Each source is (name, weight, predictions). Confidences are normalized within the source
    by the max confidence, then multiplied by `weight`. Lists are merged, deduped by rounded
    lat/lon (keeping the highest fused confidence), and sorted descending.
    """
    merged: dict[tuple[float, float], LocationPrediction] = {}

    for _name, weight, preds in sources:
        if weight <= 0 or not preds:
            continue
        mx = max((p.confidence for p in preds), default=0.0) or 1e-9
        for p in preds:
            conf = min(1.0, max(0.0, (p.confidence / mx) * weight))
            key = (round(p.latitude, dedupe_decimals), round(p.longitude, dedupe_decimals))
            if key not in merged or merged[key].confidence < conf:
                merged[key] = LocationPrediction(
                    latitude=p.latitude,
                    longitude=p.longitude,
                    country=p.country,
                    city=p.city,
                    confidence=conf,
                    distance_confidence_km=p.distance_confidence_km,
                )

    result = sorted(merged.values(), key=lambda x: x.confidence, reverse=True)
    return result


def apply_reasoning_to_predictions(
    primary: LocationPrediction,
    alternatives: List[LocationPrediction],
    *,
    country_elimination: Optional[CountryEliminationResult] = None,
    geo_reasoning: Optional[GeoReasoningResult] = None,
    astronomy_constraints: Optional[AstronomyConstraints] = None,
    settings: Settings,
) -> Tuple[LocationPrediction, List[LocationPrediction]]:
    """
    Re-rank predictions using geo-reasoning outputs.

    - Boost candidates from countries with high Bayesian posterior
    - Penalize candidates outside astronomy-derived latitude bounds
    - Filter out candidates eliminated by rule engine (if strict)
    """
    candidates = [primary] + list(alternatives)
    if not candidates:
        return primary, alternatives

    scored: List[Tuple[float, LocationPrediction]] = []

    for cand in candidates:
        score = float(cand.confidence)

        # 1. Bayesian posterior boost
        if geo_reasoning and geo_reasoning.country_posteriors and settings.reasoning_fusion_boost_weight > 0:
            posterior = geo_reasoning.country_posteriors.get(cand.country, 0.0)
            # Scale boost: max 25% confidence increase for top posterior country
            boost = posterior * settings.reasoning_fusion_boost_weight
            score *= (1.0 + boost)

        # 2. Country elimination penalty
        if country_elimination and country_elimination.remaining_countries:
            if cand.country not in country_elimination.remaining_countries:
                # Country was eliminated — heavy penalty but not zero
                score *= 0.3
            else:
                # Country survived — small bonus proportional to its score
                elim_score = country_elimination.country_scores.get(cand.country, 0.0)
                score *= (1.0 + elim_score * 0.1)

        # 3. Astronomy latitude penalty
        if astronomy_constraints and settings.reasoning_latitude_penalty_weight > 0:
            lat = cand.latitude
            lat_min = astronomy_constraints.latitude_min
            lat_max = astronomy_constraints.latitude_max
            lat_conf = astronomy_constraints.latitude_confidence

            if lat_conf > 0.2:
                if lat < lat_min or lat > lat_max:
                    # Outside bounds — penalty proportional to confidence
                    penalty = lat_conf * settings.reasoning_latitude_penalty_weight
                    score *= (1.0 - penalty)
                else:
                    # Inside bounds — small bonus
                    score *= (1.0 + lat_conf * settings.reasoning_latitude_penalty_weight * 0.3)

            # Hemisphere consistency
            hem = astronomy_constraints.hemisphere_hint
            if hem == "northern" and lat < -10:
                score *= 0.5
            elif hem == "southern" and lat > 10:
                score *= 0.5

        scored.append((score, cand))

    # Sort by new scores
    scored.sort(key=lambda x: x[0], reverse=True)

    new_primary = scored[0][1]
    new_primary.confidence = min(1.0, scored[0][0])
    new_alternatives = []
    for s, c in scored[1:]:
        c.confidence = min(1.0, s)
        new_alternatives.append(c)

    return new_primary, new_alternatives
