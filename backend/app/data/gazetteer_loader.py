"""
Load StreetCLIP gazetteer rows from a generated JSON file (GeoNames-scale) or embedded fallback.

StreetCLIP scores every label in the active set — for large DBs we **filter by a bbox around GeoCLIP
rank-1** (and cap count) so inference stays feasible.
"""

from __future__ import annotations

import heapq
import json
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import Settings

logger = logging.getLogger(__name__)

# Minimal embedded list when no file / empty file (not country-specific “mocks”).
from app.data.world_city_gazetteer import WORLD_CITIES_EMBEDDED_FALLBACK


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _lon_diff_deg(lon_a: float, lon_b: float) -> float:
    d = ((lon_a - lon_b + 180.0) % 360.0) - 180.0
    return abs(d)


def _in_bbox(
    lat: float,
    lon: float,
    lat0: float,
    lon0: float,
    dlat: float,
    dlon: float,
) -> bool:
    if abs(lat - lat0) > dlat:
        return False
    # Scale lon tolerance by latitude (cos lat) for similar km band
    cos_lat = max(0.2, math.cos(math.radians(lat0)))
    eff_dlon = dlon / cos_lat
    return _lon_diff_deg(lon, lon0) <= eff_dlon


@lru_cache(maxsize=2)
def _load_json_file_cached(path: str) -> Tuple[Tuple[Dict[str, Any], ...], int]:
    """Returns (tuple of row dicts, byte_size). Tuple for cache hashability."""
    p = Path(path)
    if not p.is_file():
        return ((), 0)
    raw = p.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, list):
        logger.warning("Gazetteer JSON must be a list; got %s", type(data))
        return ((), len(raw))
    rows: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        city = item.get("city") or item.get("name")
        country = item.get("country")
        lat, lon = item.get("lat"), item.get("lon")
        if city is None or country is None or lat is None or lon is None:
            continue
        row = {
            "city": str(city).strip(),
            "country": str(country).strip(),
            "lat": float(lat),
            "lon": float(lon),
        }
        if "pop" in item and item["pop"] is not None:
            try:
                row["pop"] = int(item["pop"])
            except (TypeError, ValueError):
                pass
        rows.append(row)
    return (tuple(rows), len(raw))


@lru_cache(maxsize=2)
def _load_json_file_cached_as_list(path: str) -> Tuple[List[Dict[str, Any]], int]:
    """
    Materialize the cached tuple rows as one reusable list per path.

    This avoids rebuilding ~100k+ row lists on every prediction while still letting
    ``clear_gazetteer_json_cache`` invalidate both layers after a rebuild.
    """
    rows_tup, nbytes = _load_json_file_cached(path)
    rows = [dict(row) for row in rows_tup]
    logger.info("Loaded StreetCLIP gazetteer into memory: %s rows (~%s KB)", len(rows), nbytes // 1024)
    return rows, nbytes


def clear_gazetteer_json_cache() -> None:
    """Call after rebuilding gazetteer JSON on disk so the next inference reloads."""
    _load_json_file_cached.cache_clear()
    _load_json_file_cached_as_list.cache_clear()


def streetclip_gazetteer_json_resolved(settings: Settings) -> Optional[Path]:
    """
    Return path to gazetteer JSON if the file exists (explicit STREETCLIP_GAZETTEER_PATH,
    else autoload output ``streetclip_gazetteer_{dump}_world.json``).
    """
    explicit = (getattr(settings, "streetclip_gazetteer_path", None) or "").strip()
    if explicit:
        p = _resolve_gazetteer_path(explicit)
        return p if p.is_file() else None
    dump = (getattr(settings, "streetclip_gazetteer_autoload_dump", "cities1000") or "cities1000").strip().lower()
    rel = f"app/data/generated/streetclip_gazetteer_{dump}_world.json"
    p = _resolve_gazetteer_path(rel)
    return p if p.is_file() else None


def _resolve_gazetteer_path(raw: str) -> Path:
    """Absolute path, cwd-relative, or under backend/app/data/."""
    p = Path(raw).expanduser()
    if p.is_file():
        return p.resolve()
    backend_root = Path(__file__).resolve().parents[2]
    for cand in (backend_root / raw, backend_root / "app" / "data" / raw):
        if cand.is_file():
            return cand.resolve()
    return p


def load_gazetteer_rows_from_disk(settings: Settings) -> List[Dict[str, Any]]:
    """Load full gazetteer from configured path, autoload JSON if present, or embedded fallback."""
    p = streetclip_gazetteer_json_resolved(settings)
    if p is None:
        explicit = (getattr(settings, "streetclip_gazetteer_path", None) or "").strip()
        if explicit:
            logger.warning("streetclip_gazetteer_path not found (%s); using embedded fallback", explicit)
        return list(WORLD_CITIES_EMBEDDED_FALLBACK)
    rows, _nbytes = _load_json_file_cached_as_list(str(p))
    if not rows:
        logger.warning("Gazetteer file empty or invalid (%s); using embedded fallback", p)
        return list(WORLD_CITIES_EMBEDDED_FALLBACK)
    return rows


def geoclip_prior_bbox_half_degrees(
    geo_preds: List[Any],
    settings: Settings,
) -> Tuple[float, float]:
    """
    Half-extents (lat, lon degrees) for the gazetteer filter box around GeoCLIP rank-1.
    Expands when top GeoCLIP hypotheses disagree (wide spread → wider search).
    """
    base_lat = float(getattr(settings, "streetclip_gazetteer_bbox_lat_deg", 2.0))
    base_lon = float(getattr(settings, "streetclip_gazetteer_bbox_lon_deg", 2.5))
    if not geo_preds:
        return base_lat, base_lon

    lead = geo_preds[0]
    lat0 = float(getattr(lead, "latitude", lead.get("lat") if isinstance(lead, dict) else 0.0))
    lon0 = float(getattr(lead, "longitude", lead.get("lon") if isinstance(lead, dict) else 0.0))
    scan = min(8, len(geo_preds))
    max_km = 0.0
    for p in geo_preds[:scan]:
        plat = float(getattr(p, "latitude", p.get("lat") if isinstance(p, dict) else lat0))
        plon = float(getattr(p, "longitude", p.get("lon") if isinstance(p, dict) else lon0))
        max_km = max(max_km, haversine_km(lat0, lon0, plat, plon))

    mult = float(getattr(settings, "geoclip_bbox_spread_multiplier", 1.4))
    pad_km = float(getattr(settings, "geoclip_bbox_spread_pad_km", 40.0))
    effective_km = max(pad_km, max_km * mult)

    km_per_deg_lat = 111.0
    cos_lat = max(0.2, math.cos(math.radians(lat0)))
    lat_deg = max(base_lat, effective_km / km_per_deg_lat)
    lon_deg = max(base_lon, effective_km / (km_per_deg_lat * cos_lat))

    cap_lat = float(getattr(settings, "streetclip_gazetteer_bbox_lat_max_deg", 6.0))
    cap_lon = float(getattr(settings, "streetclip_gazetteer_bbox_lon_max_deg", 8.0))
    return min(lat_deg, cap_lat), min(lon_deg, cap_lon)


def _bbox_filter_rows(
    rows: List[Dict[str, Any]],
    lat0: float,
    lon0: float,
    dlat: float,
    dlon: float,
) -> List[Dict[str, Any]]:
    return [
        r
        for r in rows
        if _in_bbox(float(r["lat"]), float(r["lon"]), lat0, lon0, dlat, dlon)
    ]


def filter_gazetteer_for_streetclip(
    rows: List[Dict[str, Any]],
    *,
    settings: Settings,
    geo_prior: Optional[Tuple[float, float]],
    country_allowlist: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Reduce rows before chunked StreetCLIP. Uses bbox around GeoCLIP prior when enabled;
    optional CLIP country allowlist; then caps to ``streetclip_gazetteer_max_labels``.
    """
    max_labels = int(getattr(settings, "streetclip_gazetteer_max_labels", 6000))
    use_filter = getattr(settings, "streetclip_gazetteer_geo_filter", True) and geo_prior is not None

    working = rows
    if country_allowlist:
        from app.inference.country_gazetteer import row_country_in_allowlist

        min_keep = int(getattr(settings, "streetclip_country_filter_min_rows", 40))
        by_country = [
            r
            for r in working
            if row_country_in_allowlist(str(r.get("country") or ""), country_allowlist)
        ]
        if len(by_country) >= min_keep:
            working = by_country
            logger.info(
                "StreetCLIP country filter: %s → %s rows (CLIP countries: %s)",
                len(rows),
                len(working),
                ", ".join(country_allowlist[:4]),
            )
        else:
            logger.warning(
                "Country filter left only %s rows (need %s); keeping bbox-trimmed set",
                len(by_country),
                min_keep,
            )
    if use_filter and geo_prior:
        lat0, lon0 = geo_prior
        dlat = float(getattr(settings, "streetclip_gazetteer_bbox_lat_deg", 2.0))
        dlon = float(getattr(settings, "streetclip_gazetteer_bbox_lon_deg", 2.5))
        min_rows = int(getattr(settings, "gazetteer_bbox_min_rows_after_filter", 60))
        adaptive = bool(getattr(settings, "gazetteer_bbox_adaptive_widen", True))
        widen_steps = (1.0, 1.5, 2.0, 3.0, 4.0) if adaptive else (1.0,)

        filt: List[Dict[str, Any]] = []
        used_scale = 1.0
        for scale in widen_steps:
            used_scale = scale
            filt = _bbox_filter_rows(rows, lat0, lon0, dlat * scale, dlon * scale)
            if len(filt) >= min_rows or scale == widen_steps[-1]:
                break

        if filt:
            working = filt
            logger.debug(
                "StreetCLIP gazetteer geo-filter: %s → %s rows (prior %.4f, %.4f, box ±%.2f°/±%.2f° ×%.1f)",
                len(rows),
                len(working),
                lat0,
                lon0,
                dlat,
                dlon,
                used_scale,
            )
        else:
            logger.warning(
                "Geo-filter returned 0 rows after widen; using full gazetteer (prior %.4f, %.4f)",
                lat0,
                lon0,
            )
            working = rows

    if len(working) <= max_labels:
        return working

    min_pop = int(getattr(settings, "streetclip_gazetteer_min_population", 0))
    if min_pop > 0:
        working = [r for r in working if int(r.get("pop") or 0) >= min_pop]

    if len(working) <= max_labels:
        return working

    prioritize_distance = bool(
        getattr(settings, "streetclip_gazetteer_prioritize_distance", True)
    )
    if geo_prior:
        lat0, lon0 = geo_prior

        if prioritize_distance:

            def sort_key(r: Dict[str, Any]) -> Tuple[float, int]:
                dist = haversine_km(lat0, lon0, float(r["lat"]), float(r["lon"]))
                pop = int(r.get("pop") or 0)
                return (dist, -pop)

        else:

            def sort_key(r: Dict[str, Any]) -> Tuple[int, float]:
                pop = int(r.get("pop") or 0)
                dist = haversine_km(lat0, lon0, float(r["lat"]), float(r["lon"]))
                return (-pop, dist)

        working = heapq.nsmallest(max_labels, working, key=sort_key)
    elif any(r.get("pop") for r in working):
        working = heapq.nlargest(max_labels, working, key=lambda r: int(r.get("pop") or 0))
    else:
        working = working[:max_labels]

    logger.debug(
        "StreetCLIP gazetteer trimmed to %s labels (cap=%s)",
        len(working),
        max_labels,
    )
    return working
