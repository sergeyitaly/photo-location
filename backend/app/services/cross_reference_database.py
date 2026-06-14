"""Cross-reference fused location candidates against the local gazetteer database."""

from __future__ import annotations

import heapq
import math
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from app.config import Settings
from app.data.gazetteer_loader import load_gazetteer_rows_from_disk
from app.models.schemas import LocationPrediction


def _normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    value = unicodedata.normalize("NFKD", str(text))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    value = re.sub(r"[_\-/]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _candidate_country_is_specific(country: Optional[str]) -> bool:
    text = _normalize_text(country)
    return bool(text) and "gallery" not in text and "indexed retrieval" not in text


def _nearby_rows(
    rows: List[Dict[str, Any]],
    *,
    lat: float,
    lon: float,
    radius_km: float,
    nearest_k: int,
) -> List[Tuple[float, Dict[str, Any]]]:
    lat_band = radius_km / 111.0
    lon_band = radius_km / (111.0 * max(0.2, math.cos(math.radians(lat))))
    shortlist = [
        row
        for row in rows
        if abs(float(row["lat"]) - lat) <= lat_band and abs(float(row["lon"]) - lon) <= lon_band
    ]
    if not shortlist:
        shortlist = rows

    return heapq.nsmallest(
        max(1, nearest_k),
        ((_haversine_km(lat, lon, float(row["lat"]), float(row["lon"])), row) for row in shortlist),
        key=lambda item: item[0],
    )


def _support_score(
    cand: LocationPrediction,
    row: Dict[str, Any],
    *,
    distance_km: float,
    radius_km: float,
    filename_norm: str,
) -> Tuple[float, Dict[str, Any]]:
    cand_city = _normalize_text(cand.city)
    cand_country = _normalize_text(cand.country)
    row_city = _normalize_text(row.get("city"))
    row_country = _normalize_text(row.get("country"))

    city_name_match = bool(cand_city) and cand_city == row_city
    country_match = _candidate_country_is_specific(cand.country) and cand_country == row_country
    filename_city_match = bool(filename_norm) and bool(row_city) and row_city in filename_norm

    proximity = max(0.0, 1.0 - min(distance_km, radius_km) / max(radius_km, 1e-6))
    pop = int(row.get("pop") or 0)
    pop_bonus = min(0.15, math.log10(max(10, pop)) / 40.0)

    score = proximity * 0.65
    if city_name_match:
        score += 0.75
    if filename_city_match:
        score += 0.55
    if country_match:
        score += 0.20
    score += pop_bonus

    return score, {
        "matched_city_name": city_name_match,
        "matched_country_name": country_match,
        "filename_city_match": filename_city_match,
        "distance_km": round(distance_km, 3),
        "support_score": round(score, 6),
        "population_hint": pop if pop > 0 else None,
    }


def cross_reference_candidates_with_local_database(
    primary: LocationPrediction,
    alternatives: List[LocationPrediction],
    *,
    settings: Settings,
    original_filename: Optional[str] = None,
) -> Tuple[LocationPrediction, List[LocationPrediction], Dict[str, Any]]:
    """
    Rerank fused candidates using the local gazetteer as a cross-reference database.

    This stage is intentionally local/offline and does not require network access.
    """
    rows = load_gazetteer_rows_from_disk(settings)
    if not rows:
        return primary, alternatives, {
            "enabled": False,
            "skipped_reason": "gazetteer_unavailable",
            "selected_candidate_index": 0,
            "pin_adjusted": False,
            "matched_place_name": None,
            "matched_country": None,
            "candidate_checks": [],
            "summary_note": "Local gazetteer unavailable; cross-reference skipped.",
        }

    radius_km = float(getattr(settings, "cross_reference_search_radius_km", 80.0))
    nearest_k = int(getattr(settings, "cross_reference_nearest_k", 5))
    promote_delta = float(getattr(settings, "cross_reference_promote_min_score_delta", 0.12))
    filename_norm = _normalize_text(original_filename)

    candidates = [primary] + list(alternatives)
    checks: List[Dict[str, Any]] = []
    best_idx = 0
    best_score = float("-inf")

    for idx, cand in enumerate(candidates):
        nearby = _nearby_rows(
            rows,
            lat=float(cand.latitude),
            lon=float(cand.longitude),
            radius_km=radius_km,
            nearest_k=nearest_k,
        )
        if not nearby:
            checks.append(
                {
                    "candidate_index": idx,
                    "candidate_city": cand.city,
                    "candidate_country": cand.country,
                    "nearest_place": None,
                    "nearest_country": None,
                    "distance_km": None,
                    "support_score": 0.0,
                    "matched_city_name": False,
                    "matched_country_name": False,
                    "filename_city_match": False,
                }
            )
            continue

        nearest_distance, nearest_row = nearby[0]
        score, detail = _support_score(
            cand,
            nearest_row,
            distance_km=nearest_distance,
            radius_km=radius_km,
            filename_norm=filename_norm,
        )
        checks.append(
            {
                "candidate_index": idx,
                "candidate_city": cand.city,
                "candidate_country": cand.country,
                "nearest_place": nearest_row.get("city"),
                "nearest_country": nearest_row.get("country"),
                **detail,
            }
        )
        if score > best_score:
            best_score = score
            best_idx = idx

    base_score = float(checks[0].get("support_score") or 0.0) if checks else 0.0
    pin_adjusted = bool(best_idx != 0 and best_score >= base_score + promote_delta)
    selected_idx = best_idx if pin_adjusted else 0

    chosen = candidates[selected_idx]
    new_alternatives = [cand for idx, cand in enumerate(candidates) if idx != selected_idx]
    selected_row = next((row for row in checks if int(row.get("candidate_index") or -1) == selected_idx), None)

    if pin_adjusted:
        summary_note = (
            f"Promoted candidate #{selected_idx} after local gazetteer cross-reference "
            f"(support {best_score:.3f} vs primary {base_score:.3f})."
        )
    else:
        summary_note = (
            "Primary candidate kept after local gazetteer cross-reference "
            f"(best support {best_score:.3f}, primary {base_score:.3f})."
        )

    return chosen, new_alternatives, {
        "enabled": True,
        "skipped_reason": None,
        "selected_candidate_index": selected_idx,
        "pin_adjusted": pin_adjusted,
        "matched_place_name": selected_row.get("nearest_place") if selected_row else None,
        "matched_country": selected_row.get("nearest_country") if selected_row else None,
        "candidate_checks": checks,
        "summary_note": summary_note,
    }
