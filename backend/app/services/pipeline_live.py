"""Format live location hints for GET /predict/progress (city, country, coordinates)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.models.schemas import LocationPrediction
from app.services.place_display import display_place_label, is_geoclip_placeholder


def format_coord_short(lat: float, lon: float) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{abs(lat):.2f}°{ns}, {abs(lon):.2f}°{ew}"


def format_place(city: Optional[str], country: Optional[str]) -> str:
    c = (city or "").strip()
    co = (country or "").strip()
    if c and co:
        return f"{c}, {co}"
    return c or co or "Unknown"


def google_maps_url(lat: float, lon: float) -> str:
    return f"https://www.google.com/maps?q={lat},{lon}"


def _confidence_note(conf: Optional[float]) -> Optional[str]:
    if conf is None:
        return None
    c = float(conf)
    if c <= 1.0:
        return f"Model confidence {c * 100:.1f}%"
    return f"Model score {c:.3f}"


def reasons_for_candidate(pred: LocationPrediction, source: str) -> List[str]:
    """Short, user-facing bullets explaining why this hypothesis appeared."""
    city = (pred.city or "").strip()
    country = (pred.country or "").strip()
    place = format_place(city, country)
    coord = format_coord_short(float(pred.latitude), float(pred.longitude))
    conf_note = _confidence_note(pred.confidence)
    reasons: List[str] = []

    src = (source or "").strip()
    if src == "CLIP country":
        if country and country not in ("Unknown", "GeoCLIP gallery"):
            reasons.append(
                f"Scene best matches CLIP prompt “a photograph taken in {country}”"
            )
        if city and city != country and "rank" not in city.lower():
            reasons.append(f"Landmark / place label in softmax: “{city}”")
    elif src == "CLIP similar places":
        reasons.append("Image embedding is closest to a reference location in the CLIP index")
        if place and place != "Unknown":
            reasons.append(f"Nearest indexed place: {place}")
    elif src == "GeoCLIP GPS":
        reasons.append(
            "GeoCLIP contrastive model matched your photo to GPS-tagged images in its gallery"
        )
        if city and "rank" in city.lower():
            reasons.append(f"Gallery match: {city} (not a city name — a ranked GPS hypothesis)")
        reasons.append(f"Estimated pin at {coord}")
    elif src == "Map grid":
        reasons.append(
            "Strongest city match inside the winning coarse→fine map grid cell"
        )
        reasons.append(f"Cell centred near {coord}")
    elif src == "StreetCLIP city":
        reasons.append(
            f"StreetCLIP compared your photo to the text “{place}” among regional city labels"
        )
        if pred.distance_confidence_km is not None:
            reasons.append(
                f"Gazetteer pins are approximate (typical ±{float(pred.distance_confidence_km):.0f} km)"
            )
    elif src == "Fused estimate":
        reasons.append(
            "Weighted fusion of GeoCLIP GPS, StreetCLIP city text, and CLIP country/landmark scores"
        )
        if place and place != "Unknown":
            reasons.append(f"Top fused hypothesis: {place}")
    elif src == "Candidate pin":
        reasons.append("Current best coordinate before place-name lookup")
        if place and place != "Unknown":
            reasons.append(f"Label from vision models: {place}")
    elif src == "Validating":
        reasons.append("Cross-checking against Wikipedia articles, Wikimedia photos, and terrain")
        if place and place != "Unknown":
            reasons.append(f"Candidate: {place}")
    else:
        if src:
            reasons.append(f"Signal from {src}")

    if conf_note and conf_note not in reasons:
        reasons.append(conf_note)

    if not reasons:
        reasons.append(f"Location hypothesis at {coord}")

    return reasons[:4]


def candidate_from_prediction(
    pred: LocationPrediction,
    *,
    source: str,
    rank: Optional[int] = None,
) -> Dict[str, Any]:
    conf = float(pred.confidence) if pred.confidence is not None else None
    lat = float(pred.latitude)
    lon = float(pred.longitude)
    model_place = format_place(pred.city, pred.country)
    shown = display_place_label(pred)
    out: Dict[str, Any] = {
        "place": shown,
        "display_place": shown,
        "model_place": model_place if is_geoclip_placeholder(pred.city, pred.country) else None,
        "city": pred.city,
        "country": pred.country,
        "latitude": lat,
        "longitude": lon,
        "source": source,
        "maps_url": google_maps_url(lat, lon),
        "reasons": reasons_for_candidate(pred, source),
    }
    if conf is not None:
        out["confidence"] = round(conf, 4)
        out["confidence_pct"] = round(conf * 100, 1) if conf <= 1.0 else None
    if rank is not None:
        out["rank"] = rank
    if pred.distance_confidence_km is not None:
        out["distance_confidence_km"] = float(pred.distance_confidence_km)
    return out


def candidates_from_predictions(
    preds: Sequence[LocationPrediction],
    *,
    source: str,
    limit: int = 6,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, pred in enumerate(preds[: max(1, limit)]):
        out.append(candidate_from_prediction(pred, source=source, rank=i + 1))
    return out


def merge_candidate_lists(
    *lists: Sequence[Dict[str, Any]],
    limit: int = 8,
) -> List[Dict[str, Any]]:
    """De-dupe by place string; merge reasons when the same place appears from another model."""
    by_key: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for lst in lists:
        for item in lst:
            key = (item.get("place") or "").strip().lower()
            if not key:
                continue
            if key not in by_key:
                by_key[key] = dict(item)
                order.append(key)
            else:
                existing = by_key[key]
                extra_src = item.get("source")
                if extra_src and extra_src != existing.get("source"):
                    reasons = list(existing.get("reasons") or [])
                    for r in item.get("reasons") or []:
                        tag = f"[{extra_src}] {r}"
                        if tag not in reasons:
                            reasons.append(tag)
                    existing["reasons"] = reasons[:5]
                    existing["also_from"] = list(
                        dict.fromkeys(
                            [*(existing.get("also_from") or []), extra_src],
                        )
                    )
            if len(order) >= limit:
                break
        if len(order) >= limit:
            break
    merged = [by_key[k] for k in order[:limit]]
    def _sort_key(item: Dict[str, Any]) -> float:
        conf = float(item.get("confidence") or 0.0)
        place = (item.get("place") or "").lower()
        if "geoclip rank" in place or "geoclip gallery" in place:
            return conf - 0.35
        return conf

    merged.sort(key=_sort_key, reverse=True)
    return merged


def merge_candidates_monotonic(
    previous: Sequence[Dict[str, Any]],
    new: Sequence[Dict[str, Any]],
    *,
    limit: int = 8,
) -> List[Dict[str, Any]]:
    """
    Merge candidate lists but do not let the leading guess confidence drop in the UI
    when a later search batch scores weaker cities.
    """
    if not new:
        return list(previous)[:limit]
    if not previous:
        return merge_candidate_lists(new, limit=limit)

    prev_top = float(previous[0].get("confidence") or 0.0)
    new_top = float(new[0].get("confidence") or 0.0)
    if new_top + 1e-6 < prev_top:
        return list(previous)[:limit]
    return merge_candidate_lists(new, previous, limit=limit)


def _margin_deg_str(deg: float) -> str:
    return f"{deg:.0f}" if deg >= 1.0 else f"{deg:.1f}"


def region_hint_from_prior(
    geo_prior: Optional[Tuple[float, float]],
    *,
    lat_margin_deg: float = 2.0,
    lon_margin_deg: float = 2.5,
    settings: Any = None,
) -> str:
    if settings is not None:
        lat_margin_deg = float(getattr(settings, "streetclip_gazetteer_bbox_lat_deg", lat_margin_deg))
        lon_margin_deg = float(getattr(settings, "streetclip_gazetteer_bbox_lon_deg", lon_margin_deg))
    if not geo_prior:
        return "Worldwide search"
    lat, lon = geo_prior
    return (
        f"Near {format_coord_short(lat, lon)} "
        f"(±{_margin_deg_str(lat_margin_deg)}° lat, ±{_margin_deg_str(lon_margin_deg)}° lon)"
    )


def sample_places_from_gazetteer_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    limit: int = 8,
) -> List[str]:
    """Populous / diverse city names in the active gazetteer slice."""
    if not rows:
        return []
    ranked = sorted(
        rows,
        key=lambda r: (
            -int(r.get("pop") or 0),
            str(r.get("country") or ""),
            str(r.get("city") or ""),
        ),
    )
    places: List[str] = []
    seen: set[str] = set()
    for row in ranked:
        label = format_place(row.get("city"), row.get("country"))
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        places.append(label)
        if len(places) >= limit:
            break
    return places


def streetclip_scope_note(
    rows: Sequence[Dict[str, Any]],
    *,
    geo_prior: Optional[Tuple[float, float]],
    settings: Any,
) -> str:
    n = len(rows)
    region = region_hint_from_prior(
        geo_prior,
        lat_margin_deg=float(getattr(settings, "streetclip_gazetteer_bbox_lat_deg", 10.0)),
        lon_margin_deg=float(getattr(settings, "streetclip_gazetteer_bbox_lon_deg", 14.0)),
    )
    if n == 0:
        return "No cities in search area"
    return f"Comparing photo to {n:,} cities — {region}"
