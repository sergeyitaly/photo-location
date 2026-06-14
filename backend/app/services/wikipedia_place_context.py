"""Build wikipedia_place_context for the API / UI from external validation results."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.models.schemas import ExternalValidationSummary, LocationPrediction, WikipediaPlaceContext
from app.services.wikipedia_photo_match import wikipedia_article_url


def build_wikipedia_place_context(
    external_validation: Optional[ExternalValidationSummary],
    primary: LocationPrediction,
    alternatives: List[LocationPrediction],
    *,
    enabled_in_request: bool,
) -> Optional[WikipediaPlaceContext]:
    if not enabled_in_request:
        return WikipediaPlaceContext(
            enabled=False,
            note="Wikipedia validation was off for this request (fast mode or checkbox).",
        )
    if external_validation is None or not external_validation.enabled:
        reason = (external_validation.skipped_reason if external_validation else None) or "not run"
        return WikipediaPlaceContext(
            enabled=False,
            note=f"Wikipedia cross-check skipped ({reason}).",
        )

    ev = external_validation
    idx = int(ev.selected_candidate_index or 0)

    wiki_checks = ev.wikipedia_checks or []
    photo_checks = getattr(ev, "wikipedia_photo_checks", None) or []
    sem_checks = ev.wikipedia_semantic_checks or []

    primary_wiki = next((w for w in wiki_checks if int(w.get("candidate_index", -1)) == 0), {})
    primary_photo = next((p for p in photo_checks if int(p.get("candidate_index", -1)) == 0), {})
    primary_sem = next((s for s in sem_checks if int(s.get("candidate_index", -1)) == 0), {})

    def _fit_score(wiki_row: Dict[str, Any], photo_row: Dict[str, Any], sem_row: Dict[str, Any]) -> float:
        score = 0.0
        if wiki_row.get("proven"):
            score += 40.0
        near = wiki_row.get("nearest_distance_m")
        if near is not None:
            score += max(0.0, 20.0 - float(near) / 500.0)
        sim = sem_row.get("similarity")
        if sim is not None:
            score += float(sim) * 25.0
        psim = photo_row.get("best_similarity")
        if psim is not None:
            score += max(0.0, float(psim)) * 35.0
        return round(score, 1)

    primary_fit = _fit_score(primary_wiki, primary_photo, primary_sem)

    best_alt_fit: Optional[float] = None
    best_alt_idx: Optional[int] = None
    for i in range(1, 1 + len(alternatives)):
        wr = next((w for w in wiki_checks if int(w.get("candidate_index", -1)) == i), {})
        pr = next((p for p in photo_checks if int(p.get("candidate_index", -1)) == i), {})
        sr = next((s for s in sem_checks if int(s.get("candidate_index", -1)) == i), {})
        fs = _fit_score(wr, pr, sr)
        if best_alt_fit is None or fs > best_alt_fit:
            best_alt_fit = fs
            best_alt_idx = i

    sel_wiki = next((w for w in wiki_checks if int(w.get("candidate_index", -1)) == idx), primary_wiki)
    sel_photo = next((p for p in photo_checks if int(p.get("candidate_index", -1)) == idx), primary_photo)

    articles: List[Dict[str, Any]] = []
    titles_seen: set[str] = set()

    photo_top = (sel_photo.get("top_matches") or []) if isinstance(sel_photo, dict) else []
    for pm in photo_top[:6]:
        tit = pm.get("title") or ""
        key = tit.lower()
        if not tit or key in titles_seen:
            continue
        titles_seen.add(key)
        sim = pm.get("photo_similarity")
        articles.append(
            {
                "title": tit,
                "url": pm.get("page_url") or wikipedia_article_url(tit),
                "distance_m": pm.get("distance_m"),
                "relevance_score": round(float(sim) * 100, 1) if sim is not None else None,
                "photo_similarity": sim,
                "photo_match_url": pm.get("image_url"),
                "overlap_cues": ["photo_match"] if sim is not None else [],
                "extract": f"CLIP photo match {float(sim):.3f} to Wikimedia image."
                if sim is not None
                else "",
            }
        )

    nearest = sel_wiki.get("nearest_title")
    if nearest and nearest.lower() not in titles_seen:
        articles.insert(
            0,
            {
                "title": nearest,
                "url": wikipedia_article_url(nearest),
                "distance_m": sel_wiki.get("nearest_distance_m"),
                "relevance_score": primary_fit,
                "overlap_cues": ["geosearch_nearest"],
                "extract": "",
            },
        )

    photo_sim = sel_photo.get("best_similarity")
    photo_detail = (sel_photo.get("detail") or "").strip()
    wiki_quality = "strong" if ev.proof_satisfied else "weak"
    if photo_sim is not None and float(photo_sim) >= 0.75:
        wiki_quality = "strong_photo"
    elif photo_sim is not None and float(photo_sim) >= 0.68:
        wiki_quality = "moderate_photo"

    syn_parts = [ev.summary_note or ""]
    if photo_detail:
        syn_parts.append(f"**Photo match:** {photo_detail}")
    if sel_photo.get("best_match"):
        bm = sel_photo["best_match"]
        if bm.get("title"):
            syn_parts.append(f"Best Wikimedia image: **{bm.get('title')}**")

    return WikipediaPlaceContext(
        enabled=True,
        note="English Wikipedia geosearch, OpenTopoData relief, CLIP vs article text, and CLIP vs Commons/Wikipedia photos.",
        synthesized_summary="\n\n".join(p for p in syn_parts if p),
        primary_wikipedia_fit_score=primary_fit,
        best_alternative_wikipedia_fit_score=best_alt_fit,
        best_alternative_wikipedia_index=best_alt_idx,
        primary_photo_similarity=float(photo_sim) if photo_sim is not None else None,
        wiki_match_quality=wiki_quality,
        primary_pin_adjusted=bool(ev.pin_adjusted),
        pin_adjustment_note=(
            f"Primary pin promoted to candidate #{idx} after Wikipedia open-data + photo checks."
            if ev.pin_adjusted
            else ""
        ),
        articles=articles,
    )
