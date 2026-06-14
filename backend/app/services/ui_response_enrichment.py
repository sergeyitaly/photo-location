"""
Map inference_debug + scene bundles into top-level UI fields the frontend expects.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.models.schemas import (
    ExternalValidationSummary,
    GeolocationReadingAxes,
    LocationPrediction,
    MLImageRecognition,
    SceneGeolocationCues,
)


def _cue_items(cues: Any) -> List[Dict[str, Any]]:
    if not cues:
        return []
    out: List[Dict[str, Any]] = []
    for c in cues:
        if isinstance(c, dict):
            label = c.get("label") or ""
            conf = c.get("score", c.get("confidence", 0.0))
        else:
            label = getattr(c, "label", "") or ""
            conf = getattr(c, "score", getattr(c, "confidence", 0.0))
        try:
            conf_f = float(conf)
        except (TypeError, ValueError):
            conf_f = 0.0
        if label:
            out.append({"label": str(label), "confidence": conf_f})
    return out


def _pred_dicts(rows: Any) -> List[LocationPrediction]:
    if not rows:
        return []
    out: List[LocationPrediction] = []
    for row in rows:
        if isinstance(row, LocationPrediction):
            out.append(row)
            continue
        if not isinstance(row, dict):
            continue
        try:
            out.append(LocationPrediction.model_validate(row))
        except Exception:
            continue
    return out


def build_geoclip_ranked_predictions(inference_debug: Dict[str, Any]) -> List[LocationPrediction]:
    src = (inference_debug or {}).get("source_predictions") or {}
    geo = src.get("geoclip") or []
    return _pred_dicts(geo)


def build_identified_elements(ml: Optional[MLImageRecognition]) -> List[Dict[str, Any]]:
    if not ml:
        return []
    labels = ml.scene_and_object_labels or []
    out: List[Dict[str, Any]] = []
    for item in labels:
        if isinstance(item, dict):
            label = item.get("label") or ""
            score = item.get("score", item.get("confidence", 0.0))
        else:
            label = getattr(item, "label", "") or ""
            score = getattr(item, "score", getattr(item, "confidence", 0.0))
        if label:
            try:
                conf = float(score)
            except (TypeError, ValueError):
                conf = 0.0
            out.append({"label": str(label), "confidence": conf})
    return out


def build_architecture_hints(scene: Optional[SceneGeolocationCues]) -> Optional[Dict[str, Any]]:
    if not scene:
        return None
    built = _cue_items(scene.built_environment)
    palette = _cue_items(scene.palette_and_finish)
    scale = _cue_items(scene.design_and_upkeep_proxy)
    density = built[:4] if built else []
    if not any([built, palette, scale, density]):
        return None
    return {
        "structural_edges": built,
        "color_palette": palette,
        "construction_scale": scale,
        "building_density": density,
    }


def build_plant_geo_hints(scene: Optional[SceneGeolocationCues]) -> List[Dict[str, Any]]:
    if not scene:
        return []
    hints: List[Dict[str, Any]] = []
    for item in _cue_items(scene.vegetation)[:8]:
        hints.append(
            {
                "plant_prompt": item["label"],
                "native_region": "scene cue (not a verified range map)",
                "confidence": item["confidence"],
                "latitude": None,
                "longitude": None,
            }
        )
    return hints


def build_season_time_bundle(
    scene: Optional[SceneGeolocationCues],
    astronomy: Any,
) -> Dict[str, Any]:
    pixel = (scene.pixel_stats if scene else None) or {}
    sky_metrics: Optional[Dict[str, Any]] = None
    if pixel:
        sky_brightness = pixel.get("sky_brightness_top_fraction")
        if sky_brightness is not None:
            sky_metrics = {
                "mean_rgb_upper": None,
                "mean_brightness_upper": float(sky_brightness),
                "hue_bucket": "derived",
            }

    season_hints: Optional[Dict[str, Any]] = None
    if astronomy and isinstance(astronomy, dict):
        season = astronomy.get("season_hint") or ""
        tod = astronomy.get("time_of_day_hint") or ""
        summary = astronomy.get("summary") or ""
        if season or tod or summary:
            season_hints = {
                "summary": summary,
                "season_hint": season,
                "time_of_day_hint": tod,
                "month_band_scores": [],
                "month_scores": [],
            }
    elif scene and scene.climate_and_light:
        items = _cue_items(scene.climate_and_light)
        if items:
            season_hints = {
                "summary": ", ".join(f"{i['label']} ({i['confidence']*100:.0f}%)" for i in items[:3]),
                "month_band_scores": [],
                "month_scores": [],
            }

    visual_tod: Optional[Dict[str, Any]] = None
    if astronomy and isinstance(astronomy, dict) and astronomy.get("time_of_day_hint"):
        visual_tod = {
            "bucket": astronomy.get("time_of_day_hint"),
            "summary": astronomy.get("summary") or "",
            "confidence": float(astronomy.get("latitude_confidence") or 0.3),
        }

    return {
        "season_time_hints": season_hints,
        "sky_image_metrics": sky_metrics,
        "visual_time_of_day": visual_tod,
    }


def build_flower_bush_road_hints(scene: Optional[SceneGeolocationCues]) -> Optional[Dict[str, Any]]:
    if not scene:
        return None
    veg = _cue_items(scene.vegetation)
    built = _cue_items(scene.built_environment)
    flowers = [v for v in veg if any(k in v["label"].lower() for k in ("flower", "bush", "garden", "shrub"))]
    roads = [b for b in built if any(k in b["label"].lower() for k in ("road", "street", "pavement", "asphalt"))]
    if not flowers and not roads:
        return None
    return {"flowers_bushes": flowers, "road_surface": roads}


def build_integrated_estimate(
    primary: Optional[LocationPrediction],
    axes: Optional[GeolocationReadingAxes],
    scene: Optional[SceneGeolocationCues],
    external: Optional[ExternalValidationSummary],
    model_used: str,
) -> Optional[Dict[str, Any]]:
    if not primary:
        return None
    place = ", ".join(p for p in [primary.city, primary.country] if p) or "predicted coordinates"
    conf = float(primary.confidence or 0.0)
    headline = f"**{place}** ({conf * 100:.1f}% fusion confidence)"
    geo_narrative = (axes.perspective_of_view if axes else "") or ""
    scene_narrative = (scene.interpretive_summary if scene else "") or ""
    if axes and axes.building_proportions:
        scene_narrative = (scene_narrative + " " + axes.building_proportions).strip()
    wiki_line = (axes.estimated_wikipedia if axes else "") or ""
    rec = (external.summary_note if external else "") or wiki_line or (
        "Treat region-scale cues as directional; verify exact place names in satellite and street imagery."
    )
    agreement: List[str] = []
    tension: List[str] = []
    if external and external.proof_satisfied:
        agreement.append("Open-data cross-check (Wikipedia + relief) passed for the selected candidate.")
    elif external and external.enabled:
        tension.append("Open-data proof was incomplete — pin may not match Wikipedia + relief gates.")
    if model_used and "streetclip" in model_used.lower():
        agreement.append("StreetCLIP gazetteer contributed a named city hypothesis.")
    limitations = [
        "Exact village or street identification is not verified without manual imagery checks.",
        "CLIP and pixel cues can misread lakes, sky bands, and forest texture on any continent.",
    ]
    return {
        "headline": headline,
        "geo_narrative": geo_narrative,
        "scene_narrative": scene_narrative,
        "recommended_interpretation": rec,
        "agreement_signals": agreement,
        "tension_signals": tension,
        "limitations": limitations,
    }


def build_inference_models(
    inference_debug: Dict[str, Any],
    model_used: str,
) -> List[Dict[str, Any]]:
    catalog = {
        "geoclip": ("GeoCLIP GPS retrieval", "geolocation", "geolocal/GeoCLIP"),
        "streetclip": ("StreetCLIP gazetteer", "geolocation", "geolocal/StreetCLIP"),
        "clip_zs": ("CLIP country + landmark softmax", "geolocation", "openai/clip-vit-base-patch32"),
        "grid_search": ("Multi-resolution map grid", "geolocation", ""),
        "clip_classifier": ("CLIP country classifier", "auxiliary", ""),
        "clip_retrieval": ("CLIP similar-place retrieval", "auxiliary", ""),
    }
    sources = list((inference_debug or {}).get("fusion_sources") or [])
    if not sources and model_used:
        sources = [s.strip() for s in model_used.replace("fusion[", "").replace("]", "").split("+") if s.strip()]
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for key in sources:
        if key in seen:
            continue
        seen.add(key)
        name, category, ident = catalog.get(key, (key, "geolocation", ""))
        rows.append({"name": name, "category": category, "identifier": ident})
    if not rows and model_used and model_used != "none":
        rows.append({"name": model_used, "category": "geolocation", "identifier": ""})
    return rows


def build_streetview_refinement(inference_debug: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sv = (inference_debug or {}).get("streetview_verification")
    if not sv or not isinstance(sv, dict):
        return None
    if not sv.get("enabled", True) and not sv.get("api_configured"):
        return None
    return {
        "attempted": bool(sv.get("enabled") or sv.get("api_configured")),
        "swapped_primary": bool(sv.get("swapped_primary")),
        "chosen_candidate_index": sv.get("chosen_candidate_index"),
        "best_similarity": sv.get("best_similarity"),
        "similarity_threshold": sv.get("similarity_threshold", 0.72),
        "candidates_evaluated": len(sv.get("per_candidate") or []),
        "detail": sv.get("detail") or sv.get("summary") or "",
    }


def enrich_prediction_ui_fields(
    *,
    primary_prediction: Optional[LocationPrediction],
    scene_geolocation_cues: Optional[SceneGeolocationCues],
    geolocation_reading_axes: Optional[GeolocationReadingAxes],
    external_validation: Optional[ExternalValidationSummary],
    ml_image_recognition: Optional[MLImageRecognition],
    inference_debug: Dict[str, Any],
    model_used: str,
    astronomy_constraints: Any = None,
) -> Dict[str, Any]:
    """Return optional top-level UI fields to merge into PredictionResponse."""
    season_bundle = build_season_time_bundle(scene_geolocation_cues, astronomy_constraints)
    return {
        "geoclip_ranked_predictions": build_geoclip_ranked_predictions(inference_debug),
        "identified_elements": build_identified_elements(ml_image_recognition),
        "architecture_hints": build_architecture_hints(scene_geolocation_cues),
        "plant_geo_hints": build_plant_geo_hints(scene_geolocation_cues),
        "flower_bush_road_hints": build_flower_bush_road_hints(scene_geolocation_cues),
        "integrated_estimate": build_integrated_estimate(
            primary_prediction,
            geolocation_reading_axes,
            scene_geolocation_cues,
            external_validation,
            model_used,
        ),
        "inference_models": build_inference_models(inference_debug, model_used),
        "streetview_refinement": build_streetview_refinement(inference_debug),
        **season_bundle,
    }
