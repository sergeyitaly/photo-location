"""
Cross-check vision candidates with English Wikipedia geosearch + OpenTopoData (SRTM) relief.

Optionally (when enabled in settings), after geosearch and relief pass:
- CLIP image vs English Wikipedia **lead text** (nearest geo-titles).
- CLIP image vs **Wikimedia Commons / Wikipedia lead photos** near the pin.

Scores all validated candidates and only moves the pin away from fusion primary (#0) when another
candidate clearly wins on combined CLIP proof — not merely because it was the first alternate
to pass the optional Wikimedia photo gate.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

from app.models.schemas import LocationPrediction
from app.config import Settings
from app.inference.clip_common import clip_image_text_cosine_similarity
from app.services.external_validation_selection import (
    CandidateProof,
    select_validation_candidate_index,
)
from app.services.throttled_http import OutboundHttpPolicy
from app.services.wikipedia_photo_match import score_wikipedia_photo_match

logger = logging.getLogger(__name__)

WIKI_API = "https://en.wikipedia.org/w/api.php"
OPENTOPO_BASE = "https://api.opentopodata.org/v1"


def _coord_cache_key(lat: float, lon: float, settings: Settings) -> Tuple[float, float]:
    d = int(getattr(settings, "external_validation_coord_cache_decimals", 3))
    return (round(lat, d), round(lon, d))


async def _wikipedia_geosearch(
    lat: float,
    lon: float,
    *,
    radius_m: int,
    limit: int,
    client: httpx.AsyncClient,
    policy: OutboundHttpPolicy,
) -> Dict[str, Any]:
    params = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "list": "geosearch",
        "gscoord": f"{lat}|{lon}",
        "gsradius": radius_m,
        "gsnamespace": 0,
        "gslimit": limit,
    }
    try:
        data, err = await policy.get_json(client, WIKI_API, params=params)
        if err or data is None:
            raise RuntimeError(err or "empty response")
        pages = (data.get("query") or {}).get("geosearch") or []
        best = None
        if pages:
            best = min(pages, key=lambda p: float(p.get("dist", 1e12)))
        ranked = sorted(
            pages,
            key=lambda p: float(p.get("dist", 1e12)),
        )
        ranked_articles: List[Dict[str, Any]] = []
        for p in ranked:
            tit = p.get("title")
            if not tit:
                continue
            dm = p.get("dist")
            ranked_articles.append(
                {
                    "title": tit,
                    "distance_m": float(dm) if dm is not None else None,
                }
            )
        return {
            "ok": True,
            "count": len(pages),
            "nearest_title": best.get("title") if best else None,
            "nearest_distance_m": float(best["dist"]) if best and best.get("dist") is not None else None,
            "titles": [p.get("title") for p in pages if p.get("title")],
            "ranked_articles": ranked_articles,
        }
    except Exception as e:
        logger.warning("Wikipedia geosearch failed: %s", e)
        return {
            "ok": False,
            "count": 0,
            "nearest_title": None,
            "nearest_distance_m": None,
            "titles": [],
            "ranked_articles": [],
            "error": str(e),
        }


def _title_hints_city(city: Optional[str], titles: List[str]) -> bool:
    if not city or not titles:
        return True
    token = city.strip().lower().split()[0]
    if len(token) < 3:
        return True
    for t in titles:
        if token in t.lower():
            return True
    return False


async def _opentopodata_elevations(
    lat: float,
    lon: float,
    *,
    step_deg: float,
    dataset: str,
    client: httpx.AsyncClient,
    policy: OutboundHttpPolicy,
) -> Dict[str, Any]:
    """3×3 grid around (lat,lon); local relief = max − min elevation (m)."""
    locs: List[str] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            locs.append(f"{lat + dy * step_deg},{lon + dx * step_deg}")
    url = f"{OPENTOPO_BASE}/{dataset}"
    params = {"locations": "|".join(locs)}
    try:
        data, err = await policy.get_json(client, url, params=params, timeout=20.0)
        if err or data is None:
            raise RuntimeError(err or "empty response")
        results = data.get("results") or []
        elevs: List[float] = []
        for row in results:
            e = row.get("elevation")
            if e is not None and isinstance(e, (int, float)):
                elevs.append(float(e))
        center_elev = None
        if len(results) >= 5:
            center_elev = results[4].get("elevation")
            if center_elev is not None:
                center_elev = float(center_elev)
        relief = None
        if len(elevs) >= 2:
            relief = max(elevs) - min(elevs)
        return {
            "ok": True,
            "center_elevation_m": center_elev,
            "local_relief_m": relief,
            "samples_ok": len(elevs),
        }
    except Exception as e:
        logger.warning("OpenTopoData lookup failed: %s", e)
        return {
            "ok": False,
            "center_elevation_m": None,
            "local_relief_m": None,
            "samples_ok": 0,
            "error": str(e),
        }


async def _wikipedia_extract(
    title: str,
    client: httpx.AsyncClient,
    policy: OutboundHttpPolicy,
) -> Tuple[str, Optional[str]]:
    """Lead-section plain text via MediaWiki extracts API. Returns (text, error_message)."""
    params = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "prop": "extracts",
        "explaintext": True,
        "exintro": True,
        "titles": title,
        "redirects": 1,
    }
    try:
        data, err = await policy.get_json(client, WIKI_API, params=params, use_cache=True)
        if err or data is None:
            return "", err
        pages = (data.get("query") or {}).get("pages") or []
        if not pages:
            return "", None
        page = pages[0]
        if page.get("missing"):
            return "", None
        extract = page.get("extract") or ""
        return str(extract).strip(), None
    except Exception as e:
        logger.warning("Wikipedia extract failed: %s", e)
        return "", str(e)


async def validate_candidates_with_open_data(
    primary: LocationPrediction,
    alternatives: List[LocationPrediction],
    settings: Settings,
    image_rgb: Optional[np.ndarray] = None,
) -> Tuple[LocationPrediction, List[LocationPrediction], Dict[str, Any]]:
    """
    Returns possibly reordered primary, alternatives, and a serializable summary dict.
    """
    candidates: List[LocationPrediction] = [primary] + list(alternatives)
    max_validate = int(getattr(settings, "external_validation_max_candidates", 4))
    max_dist = settings.wikipedia_validation_max_distance_m
    radius = settings.wikipedia_geosearch_radius_m

    wikipedia_checks: List[Dict[str, Any]] = []
    relief_checks: List[Dict[str, Any]] = []
    wikipedia_semantic_checks: List[Dict[str, Any]] = []
    wikipedia_photo_checks: List[Dict[str, Any]] = []
    candidate_proofs: List[CandidateProof] = []
    rate_limited_hosts: set[str] = set()

    wiki_by_coord: Dict[Tuple[float, float], Dict[str, Any]] = {}
    relief_by_coord: Dict[Tuple[float, float], Dict[str, Any]] = {}

    async with httpx.AsyncClient(
        timeout=25.0,
        headers=settings.outbound_http_headers(),
    ) as client:
        policy = OutboundHttpPolicy(settings)

        def _append_skipped_proof() -> None:
            candidate_proofs.append(
                CandidateProof(
                    index=idx,
                    wiki_proven=False,
                    relief_proven=False,
                    semantic_proven=False,
                    photo_proven=False,
                )
            )

        for idx, cand in enumerate(candidates):
            if idx >= max_validate:
                wikipedia_checks.append(
                    {
                        "candidate_index": idx,
                        "articles_found": 0,
                        "nearest_title": None,
                        "nearest_distance_m": None,
                        "proven": False,
                        "detail": f"skipped (validation limited to top {max_validate} candidates)",
                    }
                )
                relief_checks.append(
                    {
                        "candidate_index": idx,
                        "center_elevation_m": None,
                        "local_relief_m": None,
                        "proven": False,
                        "detail": f"skipped (validation limited to top {max_validate} candidates)",
                    }
                )
                wikipedia_semantic_checks.append(
                    {
                        "candidate_index": idx,
                        "similarity": None,
                        "threshold": float(settings.wikipedia_semantic_min_similarity),
                        "nearest_title": None,
                        "best_semantic_title": None,
                        "titles_scanned": 0,
                        "titles_cap": int(settings.wikipedia_semantic_eval_top_titles),
                        "proven": True,
                        "detail": "skipped (candidate cap)",
                    }
                )
                wikipedia_photo_checks.append(
                    {
                        "candidate_index": idx,
                        "images_found": 0,
                        "images_scored": 0,
                        "best_similarity": None,
                        "threshold": float(settings.wikipedia_photo_min_similarity),
                        "proven": True,
                        "detail": "skipped (candidate cap)",
                        "top_matches": [],
                    }
                )
                _append_skipped_proof()
                continue

            if rate_limited_hosts:
                skip_detail = (
                    "skipped — outbound API rate limit (Wikipedia/OpenTopo); "
                    "retry this prediction in ~2 minutes"
                )
                wikipedia_checks.append(
                    {
                        "candidate_index": idx,
                        "articles_found": 0,
                        "nearest_title": None,
                        "nearest_distance_m": None,
                        "proven": False,
                        "detail": skip_detail,
                    }
                )
                relief_checks.append(
                    {
                        "candidate_index": idx,
                        "center_elevation_m": None,
                        "local_relief_m": None,
                        "proven": False,
                        "detail": skip_detail,
                    }
                )
                wikipedia_semantic_checks.append(
                    {
                        "candidate_index": idx,
                        "similarity": None,
                        "threshold": float(settings.wikipedia_semantic_min_similarity),
                        "nearest_title": None,
                        "best_semantic_title": None,
                        "titles_scanned": 0,
                        "titles_cap": int(settings.wikipedia_semantic_eval_top_titles),
                        "proven": True,
                        "detail": "skipped (rate limit)",
                    }
                )
                wikipedia_photo_checks.append(
                    {
                        "candidate_index": idx,
                        "images_found": 0,
                        "images_scored": 0,
                        "best_similarity": None,
                        "threshold": float(settings.wikipedia_photo_min_similarity),
                        "proven": True,
                        "detail": "skipped (rate limit)",
                        "top_matches": [],
                    }
                )
                _append_skipped_proof()
                continue

            lat, lon = cand.latitude, cand.longitude
            ck = _coord_cache_key(lat, lon, settings)

            if ck in wiki_by_coord:
                wiki = wiki_by_coord[ck]
            else:
                wiki = await _wikipedia_geosearch(
                    lat, lon, radius_m=radius, limit=12, client=client, policy=policy
                )
                wiki_by_coord[ck] = wiki
            wiki_ok_api = wiki.get("ok", False)
            count = int(wiki.get("count") or 0)
            near_m = wiki.get("nearest_distance_m")
            titles = wiki.get("titles") or []

            wiki_proven = wiki_ok_api and count >= settings.wikipedia_min_articles_for_proof
            if wiki_proven and near_m is not None:
                wiki_proven = near_m <= max_dist
            if wiki_proven and settings.wikipedia_require_title_city_match and cand.city and titles:
                wiki_proven = wiki_proven and _title_hints_city(cand.city, titles[:8])

            if ck in relief_by_coord:
                relief = relief_by_coord[ck]
            else:
                relief = await _opentopodata_elevations(
                    lat,
                    lon,
                    step_deg=settings.opentopodata_grid_step_deg,
                    dataset=settings.opentopodata_dataset,
                    client=client,
                    policy=policy,
                )
                relief_by_coord[ck] = relief

            wiki_err = str(wiki.get("error") or "")
            relief_err = str(relief.get("error") or "")
            if not wiki.get("ok") and ("Rate limit" in wiki_err or "429" in wiki_err):
                rate_limited_hosts.add("en.wikipedia.org")
            if not relief.get("ok") and ("Rate limit" in relief_err or "429" in relief_err):
                rate_limited_hosts.add("api.opentopodata.org")
            relief_proven = bool(relief.get("ok")) and int(relief.get("samples_ok") or 0) >= settings.opentopodata_min_samples

            wikipedia_checks.append(
                {
                    "candidate_index": idx,
                    "articles_found": count,
                    "nearest_title": wiki.get("nearest_title"),
                    "nearest_distance_m": near_m,
                    "proven": wiki_proven,
                    "detail": "geosearch hit within radius/distance" if wiki_proven else (wiki.get("error") or "no sufficient Wikipedia hit"),
                }
            )
            relief_checks.append(
                {
                    "candidate_index": idx,
                    "center_elevation_m": relief.get("center_elevation_m"),
                    "local_relief_m": relief.get("local_relief_m"),
                    "proven": relief_proven,
                    "detail": "OpenTopoData elevation grid ok" if relief_proven else (relief.get("error") or "insufficient elevation samples"),
                }
            )

            thr = float(settings.wikipedia_semantic_min_similarity)
            max_titles = max(1, int(getattr(settings, "wikipedia_semantic_eval_top_titles", 5)))
            semantic_row: Dict[str, Any] = {
                "candidate_index": idx,
                "similarity": None,
                "threshold": thr,
                "nearest_title": wiki.get("nearest_title"),
                "best_semantic_title": None,
                "titles_scanned": 0,
                "titles_cap": max_titles,
                "proven": True,
                "detail": "",
            }
            semantic_proven = True

            if not settings.wikipedia_semantic_gate_enabled:
                semantic_row["detail"] = "semantic gate disabled"
            elif image_rgb is None:
                semantic_row["detail"] = "no image array; semantic gate skipped"
                semantic_proven = True
            elif not (wiki_proven and relief_proven):
                semantic_row["detail"] = "skipped until Wikipedia + relief both pass"
                semantic_proven = True
            else:
                ranked_a = wiki.get("ranked_articles") or []
                to_scan = ranked_a[:max_titles]
                if not to_scan:
                    semantic_proven = False
                    semantic_row["proven"] = False
                    semantic_row["detail"] = "no geosearch article titles to compare"
                else:
                    best_sim: Optional[float] = None
                    best_tit: Optional[str] = None
                    scanned = 0
                    clip_any = False
                    for art in to_scan:
                        tit = art.get("title")
                        if not tit:
                            continue
                        extract, ext_err = await _wikipedia_extract(
                            str(tit), client, policy
                        )
                        scanned += 1
                        if ext_err or not (extract or "").strip():
                            continue
                        sim = clip_image_text_cosine_similarity(
                            image_rgb, extract, settings.globe_clip_model_id
                        )
                        if sim is not None:
                            clip_any = True
                            if best_sim is None or sim > best_sim:
                                best_sim = float(sim)
                                best_tit = str(tit)
                    semantic_row["titles_scanned"] = scanned
                    semantic_row["best_semantic_title"] = best_tit
                    semantic_row["similarity"] = best_sim
                    if not clip_any:
                        semantic_row["detail"] = "CLIP unavailable; semantic gate skipped"
                        semantic_proven = True
                    elif best_sim is None:
                        semantic_proven = False
                        semantic_row["proven"] = False
                        semantic_row["detail"] = (
                            f"no usable extracts among {scanned} nearest article(s)"
                        )
                    else:
                        semantic_proven = best_sim >= thr
                        semantic_row["proven"] = semantic_proven
                        if semantic_proven:
                            semantic_row["detail"] = (
                                f"best CLIP {best_sim:.3f} vs lead ({best_tit}) ≥ {thr:.3f}; "
                                f"scanned {scanned} title(s)"
                            )
                        else:
                            semantic_row["detail"] = (
                                f"best CLIP {best_sim:.3f} ({best_tit}) < {thr:.3f}; "
                                f"scanned {scanned} title(s)"
                            )

            wikipedia_semantic_checks.append(semantic_row)

            photo_row: Dict[str, Any] = {
                "candidate_index": idx,
                "images_found": 0,
                "images_scored": 0,
                "best_similarity": None,
                "threshold": float(getattr(settings, "wikipedia_photo_min_similarity", 0.68)),
                "proven": True,
                "detail": "",
                "top_matches": [],
            }
            photo_proven = True

            if not settings.wikipedia_photo_gate_enabled:
                photo_row["detail"] = "Wikipedia photo gate disabled"
            elif image_rgb is None:
                photo_row["detail"] = "no image array; photo gate skipped"
            elif not (wiki_proven and relief_proven):
                photo_row["detail"] = "skipped until geosearch + relief pass"
            else:
                photo_report = await score_wikipedia_photo_match(
                    image_rgb,
                    lat,
                    lon,
                    settings=settings,
                    ranked_articles=wiki.get("ranked_articles") or [],
                    client=client,
                    policy=policy,
                )
                photo_row = {
                    "candidate_index": idx,
                    **photo_report,
                }
                photo_proven = bool(photo_report.get("proven"))

            wikipedia_photo_checks.append(photo_row)

            photo_sim_val: Optional[float] = None
            if photo_row.get("best_similarity") is not None:
                try:
                    photo_sim_val = float(photo_row["best_similarity"])
                except (TypeError, ValueError):
                    photo_sim_val = None

            candidate_proofs.append(
                CandidateProof(
                    index=idx,
                    wiki_proven=wiki_proven,
                    relief_proven=relief_proven,
                    semantic_proven=semantic_proven,
                    photo_proven=photo_proven,
                    semantic_similarity=(
                        float(semantic_row["similarity"])
                        if semantic_row.get("similarity") is not None
                        else None
                    ),
                    photo_similarity=photo_sim_val,
                )
            )

    promote_delta = float(
        getattr(settings, "cross_reference_promote_min_score_delta", 0.12)
    )
    selected_idx, pin_adjusted, proof_satisfied = select_validation_candidate_index(
        candidate_proofs,
        promote_min_score_delta=promote_delta,
    )

    chosen = candidates[selected_idx]
    new_alternatives: List[LocationPrediction] = []
    for i, c in enumerate(candidates):
        if i == selected_idx:
            continue
        new_alternatives.append(c)

    sem_note = ""
    if settings.wikipedia_semantic_gate_enabled and image_rgb is not None:
        sem_note = " + CLIP vs article text"
    photo_note = ""
    if settings.wikipedia_photo_gate_enabled and image_rgb is not None:
        photo_note = " + CLIP vs Wikimedia photos"

    strict_gates = image_rgb is not None and (
        settings.wikipedia_semantic_gate_enabled or settings.wikipedia_photo_gate_enabled
    )
    if strict_gates and not proof_satisfied:
        summary_note = (
            "No candidate passed Wikipedia + relief + CLIP text/photo checks together; "
            "pin remains fusion primary (#0) — treat as best-effort."
        )
    elif pin_adjusted:
        summary_note = (
            f"Promoted candidate #{selected_idx} after Wikipedia + relief{sem_note}{photo_note} "
            f"(combined CLIP proof beat primary by ≥ {promote_delta:.2f})."
        )
    elif proof_satisfied:
        summary_note = (
            f"Fusion primary (#0) kept — passed Wikipedia + relief{sem_note}; "
            "photo match alone does not override a stronger primary text match."
        )
    else:
        summary_note = (
            f"No candidate passed all open-data gates; pin remains fusion primary (#0)."
        )

    if rate_limited_hosts:
        summary_note += (
            " Wikipedia/OpenTopo returned rate limits — remaining candidates were skipped; "
            "wait ~2 minutes before another full validation run."
        )

    summary: Dict[str, Any] = {
        "enabled": True,
        "skipped_reason": None,
        "selected_candidate_index": selected_idx,
        "pin_adjusted": pin_adjusted,
        "proof_satisfied": proof_satisfied,
        "wikipedia_checks": wikipedia_checks,
        "relief_checks": relief_checks,
        "wikipedia_semantic_checks": wikipedia_semantic_checks,
        "wikipedia_photo_checks": wikipedia_photo_checks,
        "summary_note": summary_note,
        "rate_limited_hosts": sorted(rate_limited_hosts),
    }

    return chosen, new_alternatives, summary
