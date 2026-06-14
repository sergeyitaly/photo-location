"""
Human-readable “lenses” on a prediction: camera/view, built form, Wikipedia open-data check.

Not coordinates — interpretation to read results alongside pixel stats + external_validation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from app.models.schemas import (
    ExternalValidationSummary,
    GeolocationReadingAxes,
    SceneGeolocationCues,
)


def _pick_stats(scene: Optional[SceneGeolocationCues], image_rgb: Optional[np.ndarray]) -> Dict[str, Any]:
    if scene and scene.pixel_stats:
        return dict(scene.pixel_stats)
    if image_rgb is not None and getattr(image_rgb, "size", 0) > 0:
        from app.features.scene_geolocation_cues import compute_view_geometry_pixel_stats

        return dict(compute_view_geometry_pixel_stats(image_rgb))
    return {}


def _axis_perspective(stats: Dict[str, Any], coordinate_source: str) -> str:
    pre = ""
    if coordinate_source == "exif_gps":
        pre = "Coordinates are trusted EXIF GPS; framing notes describe the photo only. "
    elif coordinate_source == "filename_hint":
        pre = "Pin is filename-derived; framing notes describe pixels only. "

    ar = float(stats.get("frame_aspect_ratio") or 0)
    sky = float(stats.get("sky_brightness_top_fraction") or 0)

    if ar <= 0:
        return pre + "Frame geometry was not summarised (missing pixel stats)."

    line = []
    if ar >= 1.25:
        line.append(f"Landscape-oriented frame (aspect {ar:.2f}) — wider field of view, often street or horizon.")
    elif ar <= 0.92:
        line.append(f"Portrait-oriented frame (aspect {ar:.2f}) — more vertical span; façades or tall forms may fill the view.")
    else:
        line.append(f"Near-square framing (aspect {ar:.2f}).")

    if sky >= 0.52:
        line.append("Bright upper band suggests open sky / horizon visibility (weather and exposure affect this).")
    elif sky <= 0.35:
        line.append("Dim upper band — canopy, dense buildings overhead, heavy weather, or exposure — not proof of indoors.")

    return pre + " ".join(line)


def _axis_building(stats: Dict[str, Any], coordinate_source: str) -> str:
    vert = float(stats.get("facade_verticality_proxy") or 0)
    edge = float(stats.get("edge_density_gray") or 0)

    if vert <= 0 and edge <= 0:
        return "Built-form proxies unavailable without pixel statistics (enable scene cue bundle or provide image array)."

    bits = []
    if vert >= 0.55:
        bits.append(
            f"Strong lower-frame vertical edge energy (score {vert:.2f}) — consistent with street-facing façades, columns, or towers."
        )
    elif vert >= 0.35:
        bits.append(
            f"Moderate vertical emphasis (score {vert:.2f}) — mixed street geometry or partial buildings."
        )
    else:
        bits.append(
            f"Low vertical-edge bias (score {vert:.2f}) — open ground, distant views, or smooth surfaces more likely."
        )

    if edge >= 0.42:
        bits.append(f"High texture / edges overall ({edge:.2f}) — cluttered urban grain or foliage.")
    elif edge <= 0.22:
        bits.append(f"Low edge density ({edge:.2f}) — smooth surfaces, fog, blur, or open plain.")

    if coordinate_source == "exif_gps":
        bits.append("This does not correct the GPS pin; it only describes the photograph.")
    return " ".join(bits)


def _axis_wikipedia(ev: Optional[ExternalValidationSummary]) -> str:
    if ev is None:
        return "No Wikipedia / relief summary was attached to this response."
    if not ev.enabled:
        return (ev.summary_note or "").strip() or "English Wikipedia geosearch and relief checks were off for this run."
    parts: list[str] = [(ev.summary_note or "").strip()]

    idx = int(ev.selected_candidate_index or 0)
    sem_list = ev.wikipedia_semantic_checks or []
    row: Optional[Dict[str, Any]] = None
    for s in sem_list:
        if isinstance(s, dict) and int(s.get("candidate_index", -1)) == idx:
            row = s
            break
    if row is None and sem_list and isinstance(sem_list[0], dict):
        row = sem_list[0]

    if row and (row.get("detail") or "").strip():
        det = str(row["detail"]).strip()
        title = row.get("best_semantic_title")
        sim = row.get("similarity")
        if title and sim is not None:
            parts.append(
                f"Estimated article match for the selected pin: CLIP vs lead for “{title}” (similarity {float(sim):.3f}). {det}"
            )
        else:
            parts.append(f"Semantic gate: {det}")

    photo_list = getattr(ev, "wikipedia_photo_checks", None) or []
    photo_row: Optional[Dict[str, Any]] = None
    for p in photo_list:
        if isinstance(p, dict) and int(p.get("candidate_index", -1)) == idx:
            photo_row = p
            break
    if photo_row and (photo_row.get("detail") or "").strip():
        pdet = str(photo_row["detail"]).strip()
        psim = photo_row.get("best_similarity")
        bm = photo_row.get("best_match") or {}
        btitle = bm.get("title") if isinstance(bm, dict) else None
        if psim is not None and btitle:
            parts.append(
                f"Wikimedia photo match: CLIP {float(psim):.3f} vs “{btitle}”. {pdet}"
            )
        else:
            parts.append(f"Photo gate: {pdet}")

    text = " ".join(p for p in parts if p)
    return text[:900] + ("…" if len(text) > 900 else "")


def build_geolocation_reading_axes(
    image_rgb: Optional[np.ndarray],
    scene: Optional[SceneGeolocationCues],
    external_validation: Optional[ExternalValidationSummary],
    *,
    coordinate_source: str,
) -> GeolocationReadingAxes:
    stats = _pick_stats(scene, image_rgb)
    return GeolocationReadingAxes(
        perspective_of_view=_axis_perspective(stats, coordinate_source),
        building_proportions=_axis_building(stats, coordinate_source),
        estimated_wikipedia=_axis_wikipedia(external_validation),
    )
