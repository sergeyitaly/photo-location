"""
Human-readable place labels for UI and progress (avoid raw "GeoCLIP rank 1, GeoCLIP gallery").
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.data.gazetteer_loader import haversine_km
from app.models.schemas import LocationPrediction, PlaceResolution

logger = logging.getLogger(__name__)


def _format_coord_short(lat: float, lon: float) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{abs(lat):.2f}°{ns}, {abs(lon):.2f}°{ew}"


def _format_place(city: Optional[str], country: Optional[str]) -> str:
    c = (city or "").strip()
    co = (country or "").strip()
    if c and co:
        return f"{c}, {co}"
    return c or co or "Unknown"


def is_geoclip_placeholder(city: Optional[str], country: Optional[str]) -> bool:
    c = (city or "").strip().lower()
    co = (country or "").strip().lower()
    if "geoclip" in co or co == "geoclip gallery":
        return True
    if "geoclip rank" in c or c.startswith("geoclip"):
        return True
    return False


def is_named_gazetteer_place(city: Optional[str], country: Optional[str]) -> bool:
    if not (city or "").strip() or not (country or "").strip():
        return False
    if (city or "").strip().lower() == (country or "").strip().lower():
        # Country-only softmax centroid (e.g. city=Italy, country=Italy) — not a city pin.
        if len((city or "").split()) <= 2:
            return False
    return not is_geoclip_placeholder(city, country)


def label_from_place_resolution(pr: Optional[PlaceResolution]) -> Optional[str]:
    if not pr or pr.error:
        return None
    parts: List[str] = []
    if pr.locality:
        parts.append(str(pr.locality).strip())
    if pr.administrative_area:
        parts.append(str(pr.administrative_area).strip())
    elif pr.country:
        parts.append(str(pr.country).strip())
    if parts:
        return " · ".join(parts)
    if pr.display_name:
        return str(pr.display_name).strip()
    return None


def display_place_label(pred: LocationPrediction) -> str:
    """User-facing place string for maps, progress, and validation messages."""
    pr = getattr(pred, "place_resolution", None)
    resolved = label_from_place_resolution(pr)
    if resolved:
        return resolved
    if is_named_gazetteer_place(pred.city, pred.country):
        return _format_place(pred.city, pred.country)
    lat, lon = float(pred.latitude), float(pred.longitude)
    return f"Near {_format_coord_short(lat, lon)}"


def sort_predictions_by_confidence(
    primary: LocationPrediction,
    alternatives: List[LocationPrediction],
) -> Tuple[LocationPrediction, List[LocationPrediction]]:
    """Ensure primary is the highest-confidence hypothesis (fusion order preserved on ties)."""
    all_preds = [primary] + list(alternatives)
    ranked = sorted(
        all_preds,
        key=lambda p: float(p.confidence or 0.0),
        reverse=True,
    )
    return ranked[0], ranked[1:]


def promote_named_primary_if_available(
    primary: LocationPrediction,
    alternatives: List[LocationPrediction],
    *,
    max_distance_km: float = 85.0,
    min_confidence_ratio: float = 1.0,
) -> Tuple[LocationPrediction, List[LocationPrediction]]:
    """
    Optional: swap a GeoCLIP-labelled primary for a **higher-confidence** named city nearby.

    Never picks the nearest named place if that would **lower** the displayed confidence
    versus the current primary or versus another named alternative with higher fusion score.
    Default ``min_confidence_ratio=1.0`` means the named candidate must beat the primary.
    """
    primary, alternatives = sort_predictions_by_confidence(primary, alternatives)

    if is_named_gazetteer_place(primary.city, primary.country):
        return primary, alternatives

    alts = list(alternatives)
    plat, plon = float(primary.latitude), float(primary.longitude)
    primary_conf = float(primary.confidence or 0.0)

    best_idx: Optional[int] = None
    best_conf = primary_conf

    for i, alt in enumerate(alts):
        if not is_named_gazetteer_place(alt.city, alt.country):
            continue
        d = haversine_km(plat, plon, float(alt.latitude), float(alt.longitude))
        if d > max_distance_km:
            continue
        alt_conf = float(alt.confidence or 0.0)
        if alt_conf >= best_conf * min_confidence_ratio and alt_conf > best_conf + 1e-9:
            best_conf = alt_conf
            best_idx = i

    if best_idx is None:
        return primary, alternatives

    named = alts[best_idx]
    logger.info(
        "Promoted named primary %s, %s (%.1f%%) over GeoCLIP-labelled pin (%.1f%%)",
        named.city,
        named.country,
        best_conf * 100.0,
        primary_conf * 100.0,
    )
    promoted = named.model_copy(update={"confidence": best_conf})
    new_alts = [primary] + [a for j, a in enumerate(alts) if j != best_idx]
    new_alts = sorted(new_alts, key=lambda p: float(p.confidence or 0.0), reverse=True)
    return promoted, new_alts


def _geoclip_rank_from_city(city: Optional[str]) -> Optional[str]:
    m = re.search(r"rank\s*(\d+)", str(city or ""), re.I)
    return m.group(1) if m else None


def enrich_prediction_display_labels(pred: LocationPrediction) -> LocationPrediction:
    """
    Replace internal GeoCLIP gallery labels with coordinate + rank text for the UI.
    Does not change lat/lon or fusion confidence — only city/country strings shown to users.
    """
    if not is_geoclip_placeholder(pred.city, pred.country):
        return pred
    rank = _geoclip_rank_from_city(pred.city) or "?"
    lat, lon = float(pred.latitude), float(pred.longitude)
    return pred.model_copy(
        update={
            "city": _format_coord_short(lat, lon),
            "country": f"Vision GPS estimate (GeoCLIP rank {rank})",
        }
    )


def enrich_predictions_for_display(
    primary: LocationPrediction,
    alternatives: List[LocationPrediction],
) -> Tuple[LocationPrediction, List[LocationPrediction]]:
    return enrich_prediction_display_labels(primary), [
        enrich_prediction_display_labels(a) for a in alternatives
    ]


def enrich_response_payload_for_display(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Apply display-label enrichment to a serialized PredictionResponse dict."""
    out = dict(payload)
    primary_raw = out.get("primary_prediction")
    if isinstance(primary_raw, dict):
        primary = LocationPrediction.model_validate(primary_raw)
        alts_raw = out.get("alternative_predictions") or []
        alts = [LocationPrediction.model_validate(a) for a in alts_raw if isinstance(a, dict)]
        primary, alts = enrich_predictions_for_display(primary, alts)
        out["primary_prediction"] = primary.model_dump(mode="json")
        out["alternative_predictions"] = [a.model_dump(mode="json") for a in alts]
    return out
