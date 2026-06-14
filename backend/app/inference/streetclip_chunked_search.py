"""
Country-ordered, chunked StreetCLIP gazetteer search with early stop on confidence decline.

When a later batch peaks below earlier batches (raw logits), remaining labels are skipped and
the best-so-far hypotheses are kept as anchors for the final softmax ranking.
"""

from __future__ import annotations

import heapq
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.config import Settings
from app.data.gazetteer_loader import haversine_km
from app.models.schemas import LocationPrediction

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[List[LocationPrediction], Dict[str, Any]], None]


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


def _country_bands(
    rows: Sequence[Dict[str, Any]],
    geo_prior: Optional[Tuple[float, float]],
) -> List[Tuple[str, List[int]]]:
    """Group row indices by country; order bands by nearest city to prior (or population)."""
    by_country: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        country = str(row.get("country") or "").strip() or "Unknown"
        by_country[country].append(idx)

    def band_key(country: str) -> Tuple[float, str]:
        indices = by_country[country]
        if geo_prior:
            lat0, lon0 = geo_prior
            nearest = min(
                haversine_km(lat0, lon0, float(rows[i]["lat"]), float(rows[i]["lon"]))
                for i in indices
            )
            return (nearest, country)
        pop_peak = max(int(rows[i].get("pop") or 0) for i in indices)
        return (-float(pop_peak), country)

    ordered = sorted(by_country.keys(), key=band_key)
    return [(country, by_country[country]) for country in ordered]


def _sort_indices_nearest_first(
    rows: Sequence[Dict[str, Any]],
    indices: List[int],
    geo_prior: Optional[Tuple[float, float]],
) -> List[int]:
    if not geo_prior:
        return indices
    lat0, lon0 = geo_prior

    def key(i: int) -> Tuple[float, int]:
        return (
            haversine_km(lat0, lon0, float(rows[i]["lat"]), float(rows[i]["lon"])),
            -int(rows[i].get("pop") or 0),
        )

    return sorted(indices, key=key)


def _heap_push(
    heap: List[Tuple[float, int]],
    logit: float,
    idx: int,
    *,
    cap: int,
) -> None:
    if len(heap) < cap:
        heapq.heappush(heap, (logit, idx))
        return
    if logit > heap[0][0]:
        heapq.heapreplace(heap, (logit, idx))


def _predictions_from_heap(
    heap: List[Tuple[float, int]],
    rows: Sequence[Dict[str, Any]],
    *,
    top_k: int,
    geo_prior: Optional[Tuple[float, float]],
    settings: Settings,
) -> List[LocationPrediction]:
    if not heap:
        return []

    ranked = sorted(heap, key=lambda t: -t[0])
    top_n = min(16, len(ranked))
    top_slice = ranked[:top_n]
    logits_vec = np.array([t[0] for t in top_slice], dtype=np.float64)
    probs = _softmax_np(logits_vec)
    bbox_km = float(getattr(settings, "streetclip_gazetteer_bbox_lat_deg", 2.0)) * 111.0

    out: List[LocationPrediction] = []
    for rank, ((logit, gidx), pr) in enumerate(zip(top_slice, probs)):
        if rank >= top_k:
            break
        row = rows[gidx]
        rlat, rlon = float(row["lat"]), float(row["lon"])
        if geo_prior:
            dist_km = haversine_km(geo_prior[0], geo_prior[1], rlat, rlon)
            dist_conf = min(50.0, max(10.0, dist_km * 0.35 + 8.0))
        else:
            dist_conf = min(90.0, max(25.0, bbox_km * 0.55))
        out.append(
            LocationPrediction(
                latitude=rlat,
                longitude=rlon,
                country=str(row["country"]),
                city=str(row["city"]),
                confidence=float(pr),
                distance_confidence_km=dist_conf,
            )
        )
    return out


def score_gazetteer_chunked_early_stop(
    image_rgb: np.ndarray,
    rows: Sequence[Dict[str, Any]],
    *,
    settings: Settings,
    geo_prior: Optional[Tuple[float, float]] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[List[Tuple[float, int]], Dict[str, Any]]:
    """
    Score all gazetteer rows in chunks; stop when peaks decline vs the running global best.

    Returns global top (logit, row_index) pairs and a debug dict.
    """
    meta: Dict[str, Any] = {
        "early_stopped": False,
        "chunks_scored": 0,
        "chunks_skipped": 0,
        "countries_scored": 0,
        "countries_skipped": 0,
        "labels_scored": 0,
        "labels_total": len(rows),
        "global_peak_logit": None,
    }
    if not rows:
        return [], meta

    chunk_size = max(8, min(int(settings.streetclip_gazetteer_chunk_size), 96))
    heap_cap = int(getattr(settings, "streetclip_search_top_heap", 32))
    early_stop = bool(getattr(settings, "streetclip_early_stop_enabled", True))
    logit_margin = float(getattr(settings, "streetclip_early_stop_logit_margin", 0.35))
    weak_limit = int(getattr(settings, "streetclip_early_stop_weak_chunks", 2))
    country_order = bool(getattr(settings, "streetclip_country_ordered_search", True))
    skip_country = bool(getattr(settings, "streetclip_skip_country_on_decline", True))

    global_peak = float("-inf")
    weak_streak = 0
    heap: List[Tuple[float, int]] = []

    if country_order:
        bands = _country_bands(rows, geo_prior)
    else:
        bands = [("", list(range(len(rows))))]

    total_chunks_est = sum(
        max(1, (len(indices) + chunk_size - 1) // chunk_size) for _, indices in bands
    )
    chunk_serial = 0

    for country, indices in bands:
        indices = _sort_indices_nearest_first(rows, indices, geo_prior)
        country_peak = float("-inf")
        country_scored = False

        for start in range(0, len(indices), chunk_size):
            chunk_serial += 1
            batch_idx = indices[start : start + chunk_size]
            labels = [f"{rows[i]['city']}, {rows[i]['country']}" for i in batch_idx]
            from app.inference.streetclip_inference import score_labels_with_streetclip

            logits = score_labels_with_streetclip(image_rgb, labels, settings=settings)
            if logits is None or len(logits) != len(batch_idx):
                meta["chunks_skipped"] += 1
                continue

            meta["chunks_scored"] += 1
            meta["labels_scored"] += len(batch_idx)
            country_scored = True

            chunk_peak = float(np.max(logits))
            country_peak = max(country_peak, chunk_peak)

            for local_i, logit in enumerate(logits):
                _heap_push(heap, float(logit), batch_idx[local_i], cap=heap_cap)

            if chunk_peak > global_peak:
                global_peak = chunk_peak
                meta["global_peak_logit"] = global_peak
                weak_streak = 0
            elif early_stop and global_peak > float("-inf"):
                if chunk_peak < global_peak - logit_margin:
                    weak_streak += 1
                else:
                    weak_streak = 0

            if progress_callback is not None:
                preds = _predictions_from_heap(
                    heap, rows, top_k=5, geo_prior=geo_prior, settings=settings
                )
                progress_callback(
                    preds,
                    {
                        "country": country or None,
                        "chunk": chunk_serial,
                        "chunks_total": total_chunks_est,
                        "chunk_peak_logit": chunk_peak,
                        "global_peak_logit": global_peak,
                        "weak_streak": weak_streak,
                        "labels_scored": meta["labels_scored"],
                        "labels_total": meta["labels_total"],
                    },
                )

            if early_stop and weak_streak >= weak_limit:
                meta["early_stopped"] = True
                meta["chunks_skipped"] = max(0, total_chunks_est - chunk_serial)
                logger.info(
                    "StreetCLIP early stop after chunk %s/%s (peak %.3f, last chunk %.3f, margin %.2f)",
                    chunk_serial,
                    total_chunks_est,
                    global_peak,
                    chunk_peak,
                    logit_margin,
                )
                break

        if country_scored:
            meta["countries_scored"] += 1
        elif country:
            meta["countries_skipped"] += 1

        if meta["early_stopped"]:
            remaining_countries = len(bands) - meta["countries_scored"] - meta["countries_skipped"]
            meta["countries_skipped"] += max(0, remaining_countries)
            break

        if (
            early_stop
            and skip_country
            and country_scored
            and global_peak > float("-inf")
            and country_peak < global_peak - logit_margin
        ):
            meta["countries_skipped"] += len(bands) - meta["countries_scored"] - meta["countries_skipped"]
            logger.info(
                "StreetCLIP skipping %s remaining countries after weak band %r (country peak %.3f < global %.3f)",
                meta["countries_skipped"],
                country,
                country_peak,
                global_peak,
            )
            break

    ranked = sorted(heap, key=lambda t: -t[0])
    return ranked, meta
