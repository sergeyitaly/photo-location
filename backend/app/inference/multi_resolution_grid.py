"""Coarse-to-fine StreetCLIP grid search over the gazetteer, then city-level refinement."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.config import Settings
from app.data.gazetteer_loader import filter_gazetteer_for_streetclip, load_gazetteer_rows_from_disk
from app.inference.streetclip_inference import score_labels_with_streetclip
from app.models.schemas import LocationPrediction

_GRID_GROUP_CACHE: dict[tuple[int, float, float], Dict[tuple[int, int], List[Dict[str, Any]]]] = {}


def _cell_key(lat: float, lon: float, lat_step: float, lon_step: float) -> tuple[int, int]:
    lat_idx = int(math.floor((lat + 90.0) / lat_step))
    lon_idx = int(math.floor((lon + 180.0) / lon_step))
    return lat_idx, lon_idx


def _group_rows_cached(
    rows: List[Dict[str, Any]],
    *,
    lat_step: float,
    lon_step: float,
) -> Dict[tuple[int, int], List[Dict[str, Any]]]:
    cache_key = (id(rows), float(lat_step), float(lon_step))
    cached = _GRID_GROUP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    grouped: Dict[tuple[int, int], List[Dict[str, Any]]] = {}
    for row in rows:
        key = _cell_key(float(row["lat"]), float(row["lon"]), lat_step, lon_step)
        grouped.setdefault(key, []).append(row)
    _GRID_GROUP_CACHE[cache_key] = grouped
    return grouped


def _sorted_rows_by_priority(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            -int(r.get("pop") or 0),
            str(r.get("country") or ""),
            str(r.get("city") or ""),
        ),
    )


def _pick_representatives(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    picked: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in _sorted_rows_by_priority(rows):
        key = (str(row.get("city") or ""), str(row.get("country") or ""))
        if key in seen:
            continue
        picked.append(row)
        seen.add(key)
        if len(picked) >= max(1, limit):
            break
    return picked


def _score_cells(
    image_rgb: np.ndarray,
    groups: Dict[tuple[int, int], List[Dict[str, Any]]],
    *,
    settings: Settings,
    reps_per_cell: int,
    limit_cells: int,
) -> List[Dict[str, Any]]:
    labels: List[str] = []
    label_to_cell: List[tuple[int, int]] = []
    cell_meta: Dict[tuple[int, int], Dict[str, Any]] = {}

    for cell_key, cell_rows in groups.items():
        reps = _pick_representatives(cell_rows, reps_per_cell)
        if not reps:
            continue
        cell_meta[cell_key] = {
            "rows": cell_rows,
            "centroid_lat": float(np.mean([float(r["lat"]) for r in cell_rows])),
            "centroid_lon": float(np.mean([float(r["lon"]) for r in cell_rows])),
            "sample_places": [f"{r['city']}, {r['country']}" for r in reps],
        }
        for row in reps:
            labels.append(f"{row['city']}, {row['country']}")
            label_to_cell.append(cell_key)

    logits = score_labels_with_streetclip(image_rgb, labels, settings=settings)
    if logits is None or len(logits) != len(label_to_cell):
        return []

    cell_scores: Dict[tuple[int, int], List[float]] = {}
    for cell_key, logit in zip(label_to_cell, logits):
        cell_scores.setdefault(cell_key, []).append(float(logit))

    ranked: List[Dict[str, Any]] = []
    for cell_key, scores in cell_scores.items():
        meta = cell_meta[cell_key]
        ranked.append(
            {
                "cell_key": cell_key,
                "score": max(scores),
                "centroid_lat": meta["centroid_lat"],
                "centroid_lon": meta["centroid_lon"],
                "rows": meta["rows"],
                "sample_places": meta["sample_places"],
            }
        )
    ranked.sort(key=lambda row: float(row["score"]), reverse=True)
    return ranked[: max(1, limit_cells)]


def _refine_city_predictions(
    image_rgb: np.ndarray,
    candidate_rows: List[Dict[str, Any]],
    *,
    settings: Settings,
    top_k: int,
) -> List[LocationPrediction]:
    labels = [f"{row['city']}, {row['country']}" for row in candidate_rows]
    logits = score_labels_with_streetclip(image_rgb, labels, settings=settings)
    if logits is None or len(logits) != len(candidate_rows):
        return []

    scored = sorted(
        ((float(logit), idx) for idx, logit in enumerate(logits)),
        key=lambda item: item[0],
        reverse=True,
    )
    if not scored:
        return []

    top_slice = scored[: min(16, len(scored))]
    logits_vec = np.asarray([score for score, _idx in top_slice], dtype=np.float64)
    logits_vec = logits_vec - np.max(logits_vec)
    probs = np.exp(logits_vec)
    probs = probs / (np.sum(probs) + 1e-12)

    out: List[LocationPrediction] = []
    for rank, ((_, idx), prob) in enumerate(zip(top_slice, probs)):
        if rank >= top_k:
            break
        row = candidate_rows[idx]
        out.append(
            LocationPrediction(
                latitude=float(row["lat"]),
                longitude=float(row["lon"]),
                country=str(row["country"]),
                city=str(row["city"]),
                confidence=float(prob),
                distance_confidence_km=min(
                    55.0,
                    max(
                        8.0,
                        float(getattr(settings, "grid_search_prior_fine_lat_deg", 0.35)) * 111.0 * 0.75,
                    ),
                ),
            )
        )
    return out


def predict_locations_multi_resolution_grid(
    image_rgb: np.ndarray,
    *,
    settings: Settings,
    top_k: int = 5,
    geo_prior: Optional[Tuple[float, float]] = None,
    country_allowlist: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Search the gazetteer in stages:
    1. score coarse grid cells using representative city labels
    2. refine inside the top coarse cells using finer grid cells
    3. score actual city labels only inside the winning fine cells
    """
    report: Dict[str, Any] = {
        "predictions": [],
        "timings_ms": {},
        "coarse_cell_count": 0,
        "fine_cell_count": 0,
        "top_coarse_cells": [],
        "top_fine_cells": [],
    }
    if not getattr(settings, "use_multi_resolution_grid_search", True):
        return report
    if image_rgb is None or image_rgb.size == 0:
        return report

    t0 = time.perf_counter()
    rows = load_gazetteer_rows_from_disk(settings)
    rows = filter_gazetteer_for_streetclip(
        rows,
        settings=settings,
        geo_prior=geo_prior,
        country_allowlist=country_allowlist,
    )
    report["timings_ms"]["gazetteer_load"] = round((time.perf_counter() - t0) * 1000.0, 3)
    if not rows:
        return report

    if geo_prior is not None:
        coarse_lat = float(getattr(settings, "grid_search_prior_coarse_lat_deg", 1.5))
        coarse_lon = float(getattr(settings, "grid_search_prior_coarse_lon_deg", 1.5))
        fine_lat = float(getattr(settings, "grid_search_prior_fine_lat_deg", 0.35))
        fine_lon = float(getattr(settings, "grid_search_prior_fine_lon_deg", 0.35))
    else:
        coarse_lat = float(getattr(settings, "grid_search_coarse_lat_deg", 12.0))
        coarse_lon = float(getattr(settings, "grid_search_coarse_lon_deg", 12.0))
        fine_lat = float(getattr(settings, "grid_search_fine_lat_deg", 2.0))
        fine_lon = float(getattr(settings, "grid_search_fine_lon_deg", 2.0))
    reps_per_cell = int(getattr(settings, "grid_search_representatives_per_cell", 3))
    top_coarse = int(getattr(settings, "grid_search_top_coarse_cells", 6))
    top_fine = int(getattr(settings, "grid_search_top_fine_cells", 8))
    city_cap = int(getattr(settings, "grid_search_city_limit_per_fine_cell", 24))

    t0 = time.perf_counter()
    coarse_groups = _group_rows_cached(rows, lat_step=coarse_lat, lon_step=coarse_lon)
    report["coarse_cell_count"] = len(coarse_groups)
    coarse_ranked = _score_cells(
        image_rgb,
        coarse_groups,
        settings=settings,
        reps_per_cell=reps_per_cell,
        limit_cells=top_coarse,
    )
    report["timings_ms"]["coarse_grid"] = round((time.perf_counter() - t0) * 1000.0, 3)
    report["top_coarse_cells"] = [
        {
            "score": float(cell["score"]),
            "centroid_lat": float(cell["centroid_lat"]),
            "centroid_lon": float(cell["centroid_lon"]),
            "sample_places": list(cell["sample_places"]),
            "row_count": len(cell["rows"]),
        }
        for cell in coarse_ranked
    ]
    if not coarse_ranked:
        return report

    selected_rows: List[Dict[str, Any]] = []
    for cell in coarse_ranked:
        selected_rows.extend(cell["rows"])

    t0 = time.perf_counter()
    fine_groups = _group_rows_cached(selected_rows, lat_step=fine_lat, lon_step=fine_lon)
    report["fine_cell_count"] = len(fine_groups)
    fine_ranked = _score_cells(
        image_rgb,
        fine_groups,
        settings=settings,
        reps_per_cell=max(2, reps_per_cell),
        limit_cells=top_fine,
    )
    report["timings_ms"]["fine_grid"] = round((time.perf_counter() - t0) * 1000.0, 3)
    report["top_fine_cells"] = [
        {
            "score": float(cell["score"]),
            "centroid_lat": float(cell["centroid_lat"]),
            "centroid_lon": float(cell["centroid_lon"]),
            "sample_places": list(cell["sample_places"]),
            "row_count": len(cell["rows"]),
        }
        for cell in fine_ranked
    ]
    if not fine_ranked:
        return report

    city_rows: List[Dict[str, Any]] = []
    seen_city_keys: set[tuple[str, str, float, float]] = set()
    for cell in fine_ranked:
        for row in _pick_representatives(cell["rows"], city_cap):
            key = (
                str(row["city"]),
                str(row["country"]),
                float(row["lat"]),
                float(row["lon"]),
            )
            if key in seen_city_keys:
                continue
            seen_city_keys.add(key)
            city_rows.append(row)

    t0 = time.perf_counter()
    report["predictions"] = _refine_city_predictions(
        image_rgb,
        city_rows,
        settings=settings,
        top_k=top_k,
    )
    report["timings_ms"]["city_refine"] = round((time.perf_counter() - t0) * 1000.0, 3)
    return report
