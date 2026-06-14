"""
Structured flora / fauna / architecture / palette *cues* for interpretive geolocation help.

Combines fast pixel heuristics with optional CLIP softmax over curated English prompts.
Does **not** output latitude/longitude by itself — use as prior context alongside real geo models.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from app.data.scene_clip_prompts import SCENE_CUE_BANKS
from app.data.cultural_economic_clip_prompts import (
    CULTURAL_ECONOMIC_CLIP_BANKS,
    DISCLAIMER_CULTURAL_ECONOMIC,
    METHODOLOGY_CULTURAL_ECONOMIC,
)
from app.inference.clip_common import clip_softmax_for_prompts, is_clip_runtime_available

logger = logging.getLogger(__name__)

_METHODOLOGY = (
    "Pixel statistics describe color, vegetation index, edges, and sky brightness; "
    "optional CLIP softmax ranks hand-written scene phrases within each bank only. "
    "All scores are indicative and can misfire across continents."
)


def _rgb_float(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = arr[:, :, 0].astype(np.float64) / 255.0
    g = arr[:, :, 1].astype(np.float64) / 255.0
    b = arr[:, :, 2].astype(np.float64) / 255.0
    return r, g, b


def _excess_green(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> float:
    exg = 2.0 * g - r - b
    return float(np.clip(np.mean(exg), 0.0, 1.0))


def _luminance(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.299 * r + 0.587 * g + 0.114 * b


def _saturation_approx(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    return np.where(mx > 1e-6, (mx - mn) / (mx + 1e-6), np.zeros_like(mx))


def _edge_density_gray(gray: np.ndarray) -> float:
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx * gx + gy * gy)
    return float(np.clip(np.mean(mag) * 2.0, 0.0, 1.0))


def _facade_verticality_proxy(gray: np.ndarray) -> float:
    """
    Lower-frame vertical vs horizontal gradient energy (0..1).
    Higher values often correlate with façades, street corridors, towers — not region-specific proof.
    """
    h = gray.shape[0]
    if h < 8:
        return 0.0
    lo = gray[int(h * 0.35) :, :].astype(np.float64)
    gy, gx = np.gradient(lo)
    v_en = float(np.mean(np.abs(gy)))
    h_en = float(np.mean(np.abs(gx)))
    ratio = v_en / (h_en + 1e-8)
    return float(np.clip((ratio - 0.55) / 2.2, 0.0, 1.0))


def compute_view_geometry_pixel_stats(image_rgb: np.ndarray) -> Dict[str, float]:
    """
    Cheap framing + façade proxies for interpretive text (also merged into scene pixel_stats).
    Safe to call when scene cue bundle is disabled.
    """
    r, g, b = _rgb_float(image_rgb)
    lum = _luminance(r, g, b)
    gray = lum
    h0, w0 = image_rgb.shape[:2]
    return {
        "frame_aspect_ratio": round(w0 / float(max(1, h0)), 4),
        "facade_verticality_proxy": round(_facade_verticality_proxy(gray), 4),
        "sky_brightness_top_fraction": round(_sky_brightness_top(image_rgb), 4),
        "edge_density_gray": round(_edge_density_gray(gray), 4),
    }


def _sky_brightness_top(arr: np.ndarray) -> float:
    h = arr.shape[0]
    top = arr[: max(1, int(h * 0.28)), :, :]
    r, g, b = _rgb_float(top)
    return float(np.clip(np.mean(_luminance(r, g, b)), 0.0, 1.0))


def _green_bottom_heavy(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> float:
    """Positive if lower frame is greener than upper (street trees)."""
    h = r.shape[0]
    lo = slice(int(h * 0.55), h)
    hi = slice(0, int(h * 0.35))
    ex_lo = np.mean(2.0 * g[lo] - r[lo] - b[lo])
    ex_hi = np.mean(2.0 * g[hi] - r[hi] - b[hi])
    return float(np.clip((ex_lo - ex_hi + 0.2) / 0.6, 0.0, 1.0))


def _cue(label: str, score: float, source: str) -> Dict[str, Any]:
    return {"label": label, "score": float(np.clip(score, 0.0, 1.0)), "source": source}


def _pixel_vegetation_cues(r, g, b, green_bottom: float) -> List[Dict[str, Any]]:
    exg_m = _excess_green(r, g, b)
    sat = float(np.mean(_saturation_approx(r, g, b)))
    cues: List[Dict[str, Any]] = []
    cues.append(_cue("Vegetation index (excess green)", exg_m, "pixel_heuristic"))
    cues.append(_cue("Street-tree / lower-frame vegetation emphasis", green_bottom, "pixel_heuristic"))
    if exg_m > 0.18 and sat > 0.22:
        cues.append(_cue("Lush green + color — possible humid / growing season", min(1.0, exg_m * 3.5), "derived"))
    if exg_m < 0.06:
        cues.append(_cue("Sparse vegetation — arid or built-heavy scene", 1.0 - exg_m * 8.0, "derived"))
    return cues


def _pixel_built_cues(gray: np.ndarray, edge_d: float, sat_mean: float) -> List[Dict[str, Any]]:
    cues: List[Dict[str, Any]] = []
    cues.append(_cue("Built-environment edge complexity (urban texture proxy)", edge_d, "pixel_heuristic"))
    if edge_d > 0.35:
        cues.append(_cue("High texture — dense urban or cluttered facade", min(1.0, edge_d * 1.8), "derived"))
    else:
        cues.append(_cue("Lower texture — open landscape or smooth surfaces", 1.0 - edge_d * 1.5, "derived"))
    if sat_mean > 0.35:
        cues.append(_cue("Colorful scene — signage or varied paint (not proof of region)", min(1.0, sat_mean * 1.5), "derived"))
    return cues


def _pixel_palette_cues(r, g, b, sat_mean: float, warmth: float) -> List[Dict[str, Any]]:
    cues: List[Dict[str, Any]] = []
    cues.append(_cue("Mean color saturation", sat_mean, "pixel_heuristic"))
    cues.append(_cue("Warm vs cool bias (R−B channel balance)", float(np.clip((warmth + 0.15) / 0.35, 0.0, 1.0)), "pixel_heuristic"))
    if sat_mean < 0.18:
        cues.append(_cue("Muted palette — concrete, overcast, or faded paint possible", 1.0 - sat_mean * 4.0, "derived"))
    return cues


def _pixel_climate_light(sky_b: float, lum_mean: float, sat_mean: float) -> List[Dict[str, Any]]:
    cues: List[Dict[str, Any]] = []
    cues.append(_cue("Sky strip brightness (top of frame)", sky_b, "pixel_heuristic"))
    cues.append(_cue("Overall scene luminance", lum_mean, "pixel_heuristic"))
    if sky_b > 0.72 and lum_mean > 0.55:
        cues.append(_cue("Bright sky / daylight — clear or high sun conditions likely", min(1.0, sky_b * lum_mean * 1.4), "derived"))
    if sky_b < 0.38:
        cues.append(_cue("Dark sky strip — night, deep shade, or heavy cloud possible", 1.0 - sky_b, "derived"))
    if sat_mean < 0.15 and sky_b > 0.45:
        cues.append(_cue("Low saturation + bright grey — overcast-like lighting possible", min(1.0, (1.0 - sat_mean) * sky_b), "derived"))
    return cues


def _design_spend_proxy(edge_d: float, sat_mean: float, lum_mean: float) -> List[Dict[str, Any]]:
    """Very weak heuristic: smooth regions + accents — **not** economic ground truth."""
    smooth = 1.0 - edge_d
    accent = sat_mean
    score = float(np.clip(0.35 * smooth + 0.35 * accent + 0.2 * lum_mean, 0.0, 1.0))
    return [
        _cue(
            "Speculative “designed / maintained built scene” proxy (smooth + color emphasis)",
            score,
            "pixel_heuristic",
        ),
        _cue(
            "Low confidence: “design spend” cannot be inferred reliably from a single photo",
            0.95,
            "derived",
        ),
    ]


def _interpretive_summary(
    pixel_stats: Dict[str, float],
    clip_banks: List[Dict[str, Any]],
) -> str:
    parts: List[str] = []
    exg = pixel_stats.get("vegetation_excess_green_mean", 0.0)
    if exg > 0.2:
        parts.append("Strong green channel signal suggests vegetation or parks in frame.")
    elif exg < 0.07:
        parts.append("Little excess green — built environment, arid ground, or season may dominate.")

    edge = pixel_stats.get("edge_density_gray", 0.0)
    if edge > 0.38:
        parts.append("High edge density often correlates with urban texture or intricate facades.")
    elif edge < 0.18:
        parts.append("Low edge density may indicate open sky, smooth walls, or distant horizon.")

    if clip_banks:
        top_lines: List[str] = []
        for bank in clip_banks[:2]:
            cats = bank.get("categories") or []
            if not cats:
                continue
            first = cats[0].get("items") or []
            if first:
                top_lines.append(f"{bank.get('title', 'Cue')}: “{first[0].get('label', '')[:56]}…”")
        if top_lines:
            parts.append("CLIP top cues — " + "; ".join(top_lines))

    if not parts:
        return "No strong single cue; combine with dedicated geo models and metadata when available."
    return " ".join(parts)


def compute_scene_geolocation_cues(
    image_rgb: np.ndarray,
    *,
    model_id: str,
    clip_top_n: int = 5,
    include_cultural_economic_visual: bool = True,
) -> Dict[str, Any]:
    if image_rgb is None or image_rgb.size == 0:
        return {
            "methodology": _METHODOLOGY,
            "pixel_stats": {},
            "vegetation": [],
            "built_environment": [],
            "palette_and_finish": [],
            "climate_and_light": [],
            "design_and_upkeep_proxy": [],
            "clip_banks_detail": [],
            "clip_available": False,
            "clip_model_id": None,
            "interpretive_summary": "Empty image.",
            "cultural_economic_visual": None,
        }

    r, g, b = _rgb_float(image_rgb)
    lum = _luminance(r, g, b)
    gray = lum
    sat = _saturation_approx(r, g, b)
    sat_mean = float(np.mean(sat))
    lum_mean = float(np.mean(lum))
    warmth = float(np.mean(r - b))
    edge_d = _edge_density_gray(gray)
    sky_b = _sky_brightness_top(image_rgb)
    green_bottom = _green_bottom_heavy(r, g, b)

    h0, w0 = image_rgb.shape[:2]
    pixel_stats = {
        "vegetation_excess_green_mean": round(_excess_green(r, g, b), 4),
        "mean_saturation": round(sat_mean, 4),
        "mean_luminance": round(lum_mean, 4),
        "warmth_r_minus_b_mean": round(warmth, 4),
        "edge_density_gray": round(edge_d, 4),
        "sky_brightness_top_fraction": round(sky_b, 4),
        "green_bottom_frame_bias": round(green_bottom, 4),
        "frame_aspect_ratio": round(w0 / float(max(1, h0)), 4),
        "facade_verticality_proxy": round(_facade_verticality_proxy(gray), 4),
    }

    veg = _pixel_vegetation_cues(r, g, b, green_bottom)
    built = _pixel_built_cues(gray, edge_d, sat_mean)
    pal = _pixel_palette_cues(r, g, b, sat_mean, warmth)
    cli = _pixel_climate_light(sky_b, lum_mean, sat_mean)
    design = _design_spend_proxy(edge_d, sat_mean, lum_mean)

    clip_banks: List[Dict[str, Any]] = []
    clip_ok = False
    cultural_economic_visual: Dict[str, Any] | None = None

    if is_clip_runtime_available():
        try:
            for bank in SCENE_CUE_BANKS:
                prompts = bank["prompts"]
                pairs = clip_softmax_for_prompts(image_rgb, prompts, model_id=model_id)
                rows = pairs[:clip_top_n]
                clip_banks.append(
                    {
                        "bank_id": bank["id"],
                        "title": bank["title"],
                        "categories": [
                            {
                                "category_id": bank["id"],
                                "title": bank["title"],
                                "items": [{"label": x[0], "confidence": x[1], "source": "clip_softmax"} for x in rows],
                            }
                        ],
                    }
                )
            clip_ok = True
        except Exception as e:
            logger.warning("Scene CLIP cues failed: %s", e, exc_info=True)

        if include_cultural_economic_visual:
            ce_banks: List[Dict[str, Any]] = []
            ce_ok = False
            try:
                for bank in CULTURAL_ECONOMIC_CLIP_BANKS:
                    prompts = bank["prompts"]
                    pairs = clip_softmax_for_prompts(image_rgb, prompts, model_id=model_id)
                    rows = pairs[:clip_top_n]
                    ce_banks.append(
                        {
                            "bank_id": bank["id"],
                            "title": bank["title"],
                            "categories": [
                                {
                                    "category_id": bank["id"],
                                    "title": bank["title"],
                                    "items": [
                                        {"label": x[0], "confidence": x[1], "source": "clip_softmax"} for x in rows
                                    ],
                                }
                            ],
                        }
                    )
                ce_ok = True
            except Exception as e:
                logger.warning("Cultural-economic CLIP cues failed: %s", e, exc_info=True)
            cultural_economic_visual = {
                "methodology": METHODOLOGY_CULTURAL_ECONOMIC,
                "disclaimer": DISCLAIMER_CULTURAL_ECONOMIC,
                "clip_banks_detail": ce_banks,
                "clip_available": ce_ok,
            }

    summary = _interpretive_summary(pixel_stats, clip_banks)

    # Map CLIP banks to legacy flat buckets for API compatibility
    veg_clip: List[Dict[str, Any]] = []
    built_clip: List[Dict[str, Any]] = []
    pal_clip: List[Dict[str, Any]] = []
    cli_clip: List[Dict[str, Any]] = []

    for bank in clip_banks:
        bid = bank.get("bank_id")
        cats = bank.get("categories") or []
        items_flat: List[Dict[str, Any]] = []
        for c in cats:
            for it in c.get("items") or []:
                items_flat.append(
                    {"label": it["label"], "score": it["confidence"], "source": it.get("source", "clip_softmax")}
                )
        if bid == "vegetation_trees":
            veg_clip.extend(items_flat)
        elif bid == "architecture_built":
            built_clip.extend(items_flat)
        elif bid == "palette_finish":
            pal_clip.extend(items_flat)
        elif bid == "climate_light":
            cli_clip.extend(items_flat)

    def merge_pixel_clip(pixel_list: List[Dict[str, Any]], clip_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged = pixel_list + clip_list
        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged[:14]

    return {
        "methodology": _METHODOLOGY,
        "pixel_stats": pixel_stats,
        "vegetation": merge_pixel_clip(veg, veg_clip),
        "built_environment": merge_pixel_clip(built, built_clip),
        "palette_and_finish": merge_pixel_clip(pal, pal_clip),
        "climate_and_light": merge_pixel_clip(cli, cli_clip),
        "design_and_upkeep_proxy": design,
        "clip_banks_detail": clip_banks,
        "clip_available": clip_ok,
        "clip_model_id": model_id if clip_ok else None,
        "interpretive_summary": summary,
        "cultural_economic_visual": cultural_economic_visual,
    }
