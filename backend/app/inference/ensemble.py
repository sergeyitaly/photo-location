"""Ensemble inference: GeoCLIP + StreetCLIP gazetteer + CLIP zero-shot country/landmark fusion."""
import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from app.config import settings
from app.data.gazetteer_loader import streetclip_gazetteer_json_resolved
from app.models.schemas import LocationPrediction
from app.inference.classifier import CountryClassifier
from app.inference.retrieval import ImageRetrieval
from app.inference.geoclip_inference import predict_locations_geoclip
from app.inference.multi_resolution_grid import predict_locations_multi_resolution_grid
from app.inference.streetclip_inference import predict_locations_streetclip_gazetteer
from app.inference.location_fusion import fuse_weighted_predictions
from app.inference.hybrid_geoclip_streetclip import reconcile_fusion_with_geoclip_streetclip
from app.inference.country_gazetteer import gazetteer_country_allowlist
from app.inference.fusion_tuning import (
    should_run_fast_confidence_grid,
    tune_fusion_source_weights,
)
from app.services.pipeline_progress import get_progress_tracker
from app.services.pipeline_live import (
    candidates_from_predictions,
    format_coord_short,
    format_place,
    merge_candidate_lists,
    region_hint_from_prior,
    sample_places_from_gazetteer_rows,
    streetclip_scope_note,
)
from app.services.place_display import display_place_label
from app.data.gazetteer_loader import (
    filter_gazetteer_for_streetclip,
    geoclip_prior_bbox_half_degrees,
    load_gazetteer_rows_from_disk,
)

logger = logging.getLogger(__name__)


def _report_progress(step: str, detail: str = "", **live: Any) -> None:
    try:
        tracker = get_progress_tracker()
        tracker.update_step(step, detail)
        if live:
            tracker.set_live(**live)
    except Exception:
        pass
    if detail:
        logger.info("Vision [%s]: %s", step, detail)
    else:
        logger.info("Vision [%s]", step)


def _lead_line(candidates: List[Dict[str, Any]], *, prefix: str) -> str:
    if not candidates:
        return prefix
    top = candidates[0]
    place = top.get("place") or format_place(top.get("city"), top.get("country"))
    pct = top.get("confidence_pct")
    if pct is not None:
        return f"{prefix}: {place} ({pct}%)"
    return f"{prefix}: {place}"


def _serialize_prediction_list(preds: List[LocationPrediction], limit: int = 8) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for pred in preds[: max(1, limit)]:
        out.append(
            {
                "latitude": float(pred.latitude),
                "longitude": float(pred.longitude),
                "country": pred.country,
                "city": pred.city,
                "confidence": float(pred.confidence),
                "distance_confidence_km": pred.distance_confidence_km,
            }
        )
    return out


class EnsembleInference:
    """
    Multi-model fusion (real checkpoints, downloaded on demand):
    - GeoCLIP: contrastive GPS over gallery (package `geoclip`)
    - StreetCLIP: zero-shot over gazetteer JSON (GeoNames-scale) or embedded fallback; bbox around GeoCLIP when enabled
    - CLIP (base model id): country + landmark softmax from `zero_shot_geo`
    """

    def __init__(self):
        self.classifier = CountryClassifier()
        self.retrieval = ImageRetrieval()
        logger.info("EnsembleInference initialized (GeoCLIP + StreetCLIP + CLIP-ZS pipeline)")

    def predict(
        self,
        image_array: np.ndarray,
        include_retrieval: bool = True,
        top_k: int = 5,
        clip_model_id: Optional[str] = None,
        fast: bool = False,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        timings_ms: Dict[str, float] = {}

        eff_settings = settings
        if fast:
            eff_settings = settings.model_copy(
                update={
                    "use_multi_resolution_grid_search": False,
                    "streetclip_gazetteer_max_labels": min(
                        int(settings.streetclip_gazetteer_max_labels), 2500
                    ),
                }
            )

        _report_progress("vision_inference")

        t0 = time.perf_counter()
        _report_progress("clip_classifier")
        classifier_preds = self.classifier.predict_from_image(
            image_array,
            clip_model_id=clip_model_id,
            confidence_threshold=0.008,
            top_k=max(top_k, 8),
        )
        timings_ms["clip_classifier"] = round((time.perf_counter() - t0) * 1000.0, 3)

        landmark_preds: List[LocationPrediction] = []
        if settings.ensemble_use_clip_zero_shot and clip_model_id:
            t_lm = time.perf_counter()
            try:
                from app.inference.zero_shot_geo import clip_landmark_predictions

                landmark_preds = clip_landmark_predictions(
                    image_array,
                    clip_model_id,
                    top_k=3,
                    min_prob=0.07,
                )
                timings_ms["clip_landmarks"] = round((time.perf_counter() - t_lm) * 1000.0, 3)
            except Exception as exc:
                logger.debug("CLIP landmark pass skipped: %s", exc)

        country_preds_for_filter = list(classifier_preds)
        country_allowlist = gazetteer_country_allowlist(country_preds_for_filter, settings)

        results["classifier_predictions"] = classifier_preds[:top_k]
        if landmark_preds:
            results["landmark_predictions"] = landmark_preds[:top_k]
        if country_allowlist:
            results["streetclip_country_allowlist"] = country_allowlist

        clip_cands = candidates_from_predictions(
            classifier_preds + landmark_preds, source="CLIP country", limit=5
        )
        _report_progress(
            "clip_classifier",
            candidates=clip_cands,
            processing_note=_lead_line(clip_cands, prefix="Country / landmark guess"),
        )

        retrieval_preds: List[LocationPrediction] = []
        if include_retrieval and not fast:
            t0 = time.perf_counter()
            _report_progress("clip_retrieval")
            retrieval_preds = self.retrieval.retrieve_from_image(
                image_array,
                clip_model_id=clip_model_id,
                k=top_k,
                confidence_threshold=0.05,
            )
            timings_ms["clip_retrieval"] = round((time.perf_counter() - t0) * 1000.0, 3)
            results["retrieval_predictions"] = retrieval_preds[:top_k]
            retr_cands = candidates_from_predictions(retrieval_preds, source="CLIP similar places", limit=4)
            merged = merge_candidate_lists(retr_cands, clip_cands, limit=6)
            _report_progress(
                "clip_retrieval",
                candidates=merged,
                processing_note=_lead_line(merged, prefix="Similar-place match"),
            )

        t0 = time.perf_counter()
        clip_fused = self._merge_clip_branches(
            classifier_preds + landmark_preds,
            retrieval_preds,
            include_retrieval=include_retrieval,
        )
        timings_ms["clip_merge"] = round((time.perf_counter() - t0) * 1000.0, 3)

        sources: List[tuple[str, float, List[LocationPrediction]]] = []

        geo_preds: List[LocationPrediction] = []
        sc_preds: List[LocationPrediction] = []
        grid_preds: List[LocationPrediction] = []

        if clip_fused and settings.fusion_weight_clip_zs > 0:
            sources.append(("clip_zs", float(settings.fusion_weight_clip_zs), clip_fused))

        # GeoCLIP before grid + StreetCLIP so gazetteer rows can be bbox-filtered (avoids
        # scoring six-figure city lists on CPU when a GPS prior exists).
        if settings.use_geoclip and settings.fusion_weight_geoclip > 0:
            geo_predict_k = max(
                top_k,
                int(settings.geoclip_merge_max_ranks),
                int(settings.hybrid_alt_geoclip_scan_top),
            )
            t0 = time.perf_counter()
            _report_progress("geoclip")
            geo_preds = predict_locations_geoclip(image_array, top_k=geo_predict_k)
            timings_ms["geoclip"] = round((time.perf_counter() - t0) * 1000.0, 3)
            if geo_preds:
                sources.append(
                    ("geoclip", float(settings.fusion_weight_geoclip), geo_preds),
                )
            geo_cands = candidates_from_predictions(geo_preds, source="GeoCLIP GPS", limit=5)
            lead = geo_preds[0] if geo_preds else None
            prior_note = (
                f"GPS estimate {format_coord_short(float(lead.latitude), float(lead.longitude))}"
                if lead
                else "Estimating coordinates from image…"
            )
            geo_region_hint = None
            if lead:
                lat_m, lon_m = geoclip_prior_bbox_half_degrees(geo_preds, settings)
                geo_region_hint = region_hint_from_prior(
                    (float(lead.latitude), float(lead.longitude)),
                    lat_margin_deg=lat_m,
                    lon_margin_deg=lon_m,
                )
            _report_progress(
                "geoclip",
                candidates=merge_candidate_lists(geo_cands, clip_cands, limit=6),
                region_hint=geo_region_hint,
                processing_note=prior_note,
                lead_place=display_place_label(lead) if lead else None,
            )

        geo_prior: Optional[Tuple[float, float]] = None
        gaz_settings = eff_settings
        if settings.use_geoclip and geo_preds:
            geo_prior = (float(geo_preds[0].latitude), float(geo_preds[0].longitude))
            lat_m, lon_m = geoclip_prior_bbox_half_degrees(geo_preds, settings)
            gaz_settings = eff_settings.model_copy(
                update={
                    "streetclip_gazetteer_bbox_lat_deg": lat_m,
                    "streetclip_gazetteer_bbox_lon_deg": lon_m,
                }
            )

        run_grid = (
            gaz_settings.use_multi_resolution_grid_search
            and gaz_settings.fusion_weight_grid_search > 0
        )
        if fast and not run_grid:
            run_grid = should_run_fast_confidence_grid(
                fast=True,
                geo_preds=geo_preds,
                country_predictions=country_preds_for_filter,
                settings=settings,
            )
            if run_grid:
                gaz_settings = gaz_settings.model_copy(
                    update={
                        "use_multi_resolution_grid_search": True,
                        "grid_search_top_coarse_cells": int(
                            settings.fast_grid_top_coarse_cells
                        ),
                        "grid_search_top_fine_cells": int(settings.fast_grid_top_fine_cells),
                    }
                )
                logger.info(
                    "Fast mode: running confidence-gated mini grid (GeoCLIP/CLIP signal weak)"
                )

        if run_grid:
            t0 = time.perf_counter()
            _report_progress("grid_search")
            grid_report = predict_locations_multi_resolution_grid(
                image_array,
                settings=gaz_settings,
                top_k=top_k,
                geo_prior=geo_prior,
                country_allowlist=country_allowlist,
            )
            timings_ms["grid_search"] = round((time.perf_counter() - t0) * 1000.0, 3)
            if grid_report.get("timings_ms"):
                for key, value in dict(grid_report["timings_ms"]).items():
                    timings_ms[f"grid_search_{key}"] = float(value)
            grid_preds = list(grid_report.get("predictions") or [])
            if grid_preds:
                sources.append(("grid_search", float(settings.fusion_weight_grid_search), grid_preds))
            grid_cands = candidates_from_predictions(grid_preds, source="Map grid", limit=4)
            tracker = get_progress_tracker()
            prev = (tracker.get_current().get("live") or {}).get("candidates") or []
            _report_progress(
                "grid_search",
                candidates=merge_candidate_lists(grid_cands, prev, limit=6),
                processing_note=_lead_line(grid_cands, prefix="Best map-grid cell")
                if grid_cands
                else "Scanning coarse world regions…",
            )
            results["grid_search_debug"] = {
                "coarse_cell_count": int(grid_report.get("coarse_cell_count") or 0),
                "fine_cell_count": int(grid_report.get("fine_cell_count") or 0),
                "top_coarse_cells": list(grid_report.get("top_coarse_cells") or []),
                "top_fine_cells": list(grid_report.get("top_fine_cells") or []),
            }

        if gaz_settings.use_streetclip and gaz_settings.fusion_weight_streetclip > 0:
            t0 = time.perf_counter()
            sc_rows = filter_gazetteer_for_streetclip(
                load_gazetteer_rows_from_disk(gaz_settings),
                settings=gaz_settings,
                geo_prior=geo_prior,
                country_allowlist=country_allowlist,
            )
            sample_places = sample_places_from_gazetteer_rows(sc_rows, limit=6)
            _report_progress(
                "streetclip",
                processing_note=streetclip_scope_note(sc_rows, geo_prior=geo_prior, settings=gaz_settings),
                sample_places=sample_places,
                region_hint=region_hint_from_prior(geo_prior, settings=gaz_settings),
            )
            def _streetclip_chunk_progress(
                preds: List[LocationPrediction], info: Dict[str, Any]
            ) -> None:
                if not preds:
                    return
                cands = candidates_from_predictions(preds, source="StreetCLIP city", limit=6)
                tracker = get_progress_tracker()
                prev = (tracker.get_current().get("live") or {}).get("candidates") or []
                merged = merge_candidate_lists(cands, prev, limit=8)
                country = info.get("country") or "region"
                scored = int(info.get("labels_scored") or 0)
                total = int(info.get("labels_total") or 0)
                note = (
                    f"City search in {country} ({scored:,} / {total:,} labels checked)"
                )
                if info.get("early_stopped"):
                    note = (
                        "Best match peaked — skipping weaker city batches; "
                        "keeping top guesses as anchors"
                    )
                _report_progress(
                    "streetclip",
                    candidates=merged,
                    processing_note=note,
                    region_hint=region_hint_from_prior(geo_prior, settings=gaz_settings),
                )

            sc_preds = predict_locations_streetclip_gazetteer(
                image_array,
                settings=gaz_settings,
                top_k=top_k,
                geo_prior=geo_prior,
                country_allowlist=country_allowlist,
                progress_callback=_streetclip_chunk_progress,
            )
            timings_ms["streetclip"] = round((time.perf_counter() - t0) * 1000.0, 3)
            if sc_preds:
                sources.append(("streetclip", float(settings.fusion_weight_streetclip), sc_preds))
            sc_cands = candidates_from_predictions(sc_preds, source="StreetCLIP city", limit=5)
            tracker = get_progress_tracker()
            prev = (tracker.get_current().get("live") or {}).get("candidates") or []
            _report_progress(
                "streetclip",
                candidates=merge_candidate_lists(sc_cands, prev, limit=8),
                processing_note=_lead_line(sc_cands, prefix="Best city match")
                if sc_cands
                else "Matching city names in region…",
            )

        if not sources and clip_fused:
            sources.append(("clip_zs_fallback", 1.0, clip_fused))

        sources = tune_fusion_source_weights(
            sources,
            geo_preds=geo_preds,
            sc_preds=sc_preds,
            settings=settings,
            image_rgb=image_array,
        )

        t0 = time.perf_counter()
        _report_progress("fusion_merge")
        merged_predictions = (
            fuse_weighted_predictions(sources, dedupe_decimals=settings.fusion_dedupe_decimals)
            if sources
            else []
        )
        timings_ms["fusion_merge"] = round((time.perf_counter() - t0) * 1000.0, 3)

        # Full GeoCLIP rank list for hybrid reconcile (needs more than top_k hypotheses).
        if merged_predictions and geo_preds and sc_preds:
            t0 = time.perf_counter()
            merged_predictions = reconcile_fusion_with_geoclip_streetclip(
                merged_predictions,
                geo_preds,
                sc_preds,
                settings=settings,
            )
            timings_ms["fusion_reconcile"] = round((time.perf_counter() - t0) * 1000.0, 3)

        model_used_override: Optional[str] = None
        if not merged_predictions:
            for label, raw in (
                ("clip_zs_raw", clip_fused),
                ("grid_search_raw", grid_preds),
                ("geoclip_raw", geo_preds),
                ("streetclip_raw", sc_preds),
            ):
                if not raw:
                    continue
                merged_predictions = fuse_weighted_predictions(
                    [(label, 1.0, raw)],
                    dedupe_decimals=settings.fusion_dedupe_decimals,
                )
                if merged_predictions:
                    model_used_override = "emergency[" + label + "]"
                    logger.warning(
                        "Ensemble: weighted fusion produced no pin; using unweighted %s (%d hypotheses)",
                        label,
                        len(raw),
                    )
                    break

        tag_parts = [name for name, _w, preds in sources if preds]
        results["model_used"] = (
            model_used_override
            if model_used_override
            else ("fusion[" + "+".join(tag_parts) + "]" if tag_parts else "none")
        )
        results["fusion_sources"] = tag_parts
        results["timings_ms"] = timings_ms
        results["source_counts"] = {
            "clip_classifier": len(classifier_preds),
            "clip_retrieval": len(retrieval_preds),
            "clip_fused": len(clip_fused),
            "grid_search": len(grid_preds),
            "geoclip": len(geo_preds),
            "streetclip": len(sc_preds),
            "merged": len(merged_predictions),
        }
        results["source_predictions"] = {
            "clip_classifier": _serialize_prediction_list(classifier_preds, limit=top_k),
            "clip_retrieval": _serialize_prediction_list(retrieval_preds, limit=top_k),
            "clip_fused": _serialize_prediction_list(clip_fused, limit=top_k),
            "grid_search": _serialize_prediction_list(grid_preds, limit=top_k),
            "geoclip": _serialize_prediction_list(geo_preds, limit=top_k),
            "streetclip": _serialize_prediction_list(sc_preds, limit=top_k),
            "merged": _serialize_prediction_list(merged_predictions, limit=top_k),
        }

        results["geoclip_predictions"] = geo_preds
        results["streetclip_predictions"] = sc_preds
        results["primary_prediction"] = merged_predictions[0] if merged_predictions else None
        results["alternative_predictions"] = merged_predictions[1:top_k] if len(merged_predictions) > 1 else []
        results["ensemble_size"] = len(tag_parts)

        if merged_predictions:
            fused_cands = candidates_from_predictions(merged_predictions, source="Fused estimate", limit=6)
            lead = merged_predictions[0]
            get_progress_tracker().set_live(
                candidates=fused_cands,
                processing_note=_lead_line(
                    fused_cands,
                    prefix="Combined best guess",
                ),
                lead_place=format_place(lead.city, lead.country),
                region_hint=region_hint_from_prior(
                    (float(lead.latitude), float(lead.longitude)),
                ),
            )

        return results

    def _merge_clip_branches(
        self,
        classifier_preds: List[LocationPrediction],
        retrieval_preds: List[LocationPrediction],
        include_retrieval: bool = True,
    ) -> List[LocationPrediction]:
        if not classifier_preds:
            return retrieval_preds if include_retrieval else []

        if not include_retrieval or not retrieval_preds:
            return classifier_preds

        merged: dict = {}
        for pred in classifier_preds + retrieval_preds:
            key = (round(pred.latitude, 3), round(pred.longitude, 3))
            if key not in merged:
                merged[key] = {
                    "latitude": pred.latitude,
                    "longitude": pred.longitude,
                    "country": pred.country,
                    "city": pred.city,
                    "confidences": [],
                }
            merged[key]["confidences"].append(pred.confidence)

        result_preds: List[LocationPrediction] = []
        for _key, data in merged.items():
            avg_confidence = float(np.mean(data["confidences"]))
            result_preds.append(
                LocationPrediction(
                    latitude=data["latitude"],
                    longitude=data["longitude"],
                    country=data["country"],
                    city=data["city"],
                    confidence=avg_confidence,
                    distance_confidence_km=10.0,
                )
            )
        return sorted(result_preds, key=lambda x: x.confidence, reverse=True)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "ensemble_type": "GeoCLIP + StreetCLIP + CLIP zero-shot (weighted fusion)",
            "classifier_info": self.classifier.get_confidence_distribution(),
            "retrieval_info": self.retrieval.get_vector_db_stats(),
            "fusion_weights": {
                "geoclip": settings.fusion_weight_geoclip,
                "streetclip": settings.fusion_weight_streetclip,
                "clip_zs": settings.fusion_weight_clip_zs,
                "grid_search": settings.fusion_weight_grid_search,
            },
            "streetclip_model_id": settings.streetclip_model_id,
            "streetclip_gazetteer_source": (
                "file" if streetclip_gazetteer_json_resolved(settings) is not None else "embedded"
            ),
            "merge_strategy": "Per-source max-normalization then weighted fuse + lat/lon dedupe",
        }
