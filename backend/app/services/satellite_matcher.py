"""
Satellite + Aerial Reverse Matching.

From predicted coordinates, fetch satellite/aerial tiles and compare
basic visual properties against the uploaded photo:
  - Color histogram (vegetation = green dominance, desert = red/brown)
  - Texture / edge density (urban vs rural)
  - Brightness / contrast

This catches mismatches like:
  - Photo shows dense forest, satellite shows desert -> prediction wrong
  - Photo shows flat farmland, satellite shows mountains -> prediction wrong

Uses free tile sources (NASA GIBS, configurable) or commercial
(Bing/Google/Mapbox) when keys are provided.

No heavy AI — pure pixel statistics + correlation.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
from PIL import Image
from io import BytesIO

from app.config import Settings

logger = logging.getLogger(__name__)

# NASA GIBS — Web Mercator tiles (must match _latlon_to_tile_xyz, which is EPSG:3857 / slippy map math).
# Using epsg4326 in the URL with Mercator x,y produces 400 Bad Request from GIBS.
NASA_GIBS_BASE = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best"

# Sentinel-2 L2A true color (lower res, but free)
S2_LAYER = "S2L2A_AllBands_TrueColor"

# MODIS true color (daily, 250m resolution)
MODIS_LAYER = "MODIS_Terra_CorrectedReflectance_TrueColor"


def _latlon_to_tile_xyz(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """Convert lat/lon to web mercator tile x,y at given zoom."""
    import math

    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def _tile_url_nasa_gibs(zoom: int, x: int, y: int, layer: str = MODIS_LAYER) -> str:
    """Build NASA GIBS WMTS tile URL."""
    return (
        f"{NASA_GIBS_BASE}/{layer}/default/GoogleMapsCompatible_Level9/"
        f"{zoom}/{y}/{x}.jpeg"
    )


def _tile_url_bing(
    lat: float, lon: float, zoom: int, api_key: str
) -> str:
    """Bing Maps aerial imagery (higher res, requires key)."""
    return (
        f"https://dev.virtualearth.net/REST/v1/Imagery/Map/Aerial/"
        f"{lat},{lon}/{zoom}?mapSize=256,256&key={api_key}"
    )


async def _fetch_tile_async(
    url: str,
    timeout: float = 5.0,
    headers: Optional[dict[str, str]] = None,
) -> Optional[np.ndarray]:
    """Fetch tile from URL asynchronously and return as RGB numpy array (H, W, 3)."""
    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers=headers,
        ) as client:
            r = await client.get(url)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
            return np.array(img, dtype=np.uint8)
    except Exception as e:
        logger.warning("Tile fetch failed for %s: %s", url[:120], e)
        return None


def _color_histogram_similarity(
    img_a: np.ndarray, img_b: np.ndarray
) -> float:
    """
    Compare normalized RGB histograms.
    Returns cosine similarity of histogram vectors (0..1).
    """
    def _hist_vec(img: np.ndarray) -> np.ndarray:
        # 8 bins per channel -> 24-dim vector
        vec = []
        for c in range(3):
            h, _ = np.histogram(img[:, :, c], bins=8, range=(0, 256))
            vec.extend(h.tolist())
        v = np.array(vec, dtype=np.float64)
        norm = np.linalg.norm(v)
        return v / (norm + 1e-9)

    ha = _hist_vec(img_a)
    hb = _hist_vec(img_b)
    sim = float(np.dot(ha, hb))
    return max(0.0, min(1.0, sim))


def _vegetation_index_score(img: np.ndarray) -> float:
    """
    Simple greenness score: (G - R) / (G + R + 1).
    Returns -1..1 where higher = more vegetation.
    """
    r = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    b = img[:, :, 2].astype(np.float32)
    # Simple greenness
    score = float(np.mean((g - r) / (g + r + 1.0)))
    return score


def _edge_density(img: np.ndarray) -> float:
    """
    Rough urban detector: standard deviation of grayscale = proxy for edge density.
    Higher = more texture/buildings.
    """
    gray = np.mean(img, axis=2)
    return float(np.std(gray))


def _brightness(img: np.ndarray) -> float:
    """Mean pixel value normalized to 0..1."""
    return float(np.mean(img) / 255.0)


def _compare_photo_to_tile(
    photo_rgb: np.ndarray,
    tile_rgb: np.ndarray,
) -> Dict[str, Any]:
    """
    Compare uploaded photo against satellite tile.
    Returns dict with match scores and interpretation.
    """
    # Resize both to common size for fair comparison
    h, w = min(photo_rgb.shape[0], tile_rgb.shape[0]), min(
        photo_rgb.shape[1], tile_rgb.shape[1]
    )
    h = max(h, 64)
    w = max(w, 64)

    from PIL import Image

    p = Image.fromarray(photo_rgb).resize((w, h), Image.Resampling.LANCZOS)
    t = Image.fromarray(tile_rgb).resize((w, h), Image.Resampling.LANCZOS)
    p_arr = np.array(p)
    t_arr = np.array(t)

    hist_sim = _color_histogram_similarity(p_arr, t_arr)

    photo_veg = _vegetation_index_score(p_arr)
    tile_veg = _vegetation_index_score(t_arr)
    veg_diff = abs(photo_veg - tile_veg)

    photo_edge = _edge_density(p_arr)
    tile_edge = _edge_density(t_arr)
    edge_diff = abs(photo_edge - tile_edge)

    photo_bright = _brightness(p_arr)
    tile_bright = _brightness(t_arr)
    bright_diff = abs(photo_bright - tile_bright)

    # Overall match score: 1.0 = perfect match, 0.0 = total mismatch
    # Weights: color 40%, vegetation 30%, texture 15%, brightness 15%
    veg_match = max(0.0, 1.0 - veg_diff / 0.5)  # normalize
    edge_match = max(0.0, 1.0 - edge_diff / 60.0)
    bright_match = max(0.0, 1.0 - bright_diff / 0.4)

    overall = (
        hist_sim * 0.40
        + veg_match * 0.30
        + edge_match * 0.15
        + bright_match * 0.15
    )

    interpretation = "match"
    if overall < 0.35:
        interpretation = "strong_mismatch"
    elif overall < 0.55:
        interpretation = "weak_mismatch"
    elif overall < 0.75:
        interpretation = "moderate_match"

    return {
        "overall_match_score": round(overall, 3),
        "interpretation": interpretation,
        "histogram_similarity": round(hist_sim, 3),
        "vegetation_match": round(veg_match, 3),
        "photo_vegetation_index": round(photo_veg, 3),
        "tile_vegetation_index": round(tile_veg, 3),
        "edge_density_match": round(edge_match, 3),
        "brightness_match": round(bright_match, 3),
        "photo_brightness": round(photo_bright, 3),
        "tile_brightness": round(tile_bright, 3),
    }


async def satellite_reverse_match(
    photo_rgb: np.ndarray,
    lat: float,
    lon: float,
    settings: Settings,
) -> Dict[str, Any]:
    """
    Fetch satellite tile at predicted coordinates and compare with photo.
    Returns match scores + interpretation.
    """
    enabled = getattr(settings, "use_satellite_matching", True)
    if not enabled:
        return {
            "enabled": False,
            "skipped_reason": "disabled_in_settings",
            "summary": "Satellite matching disabled.",
        }

    zoom = int(getattr(settings, "satellite_match_zoom", 14))
    api_key = getattr(settings, "bing_maps_api_key", "") or ""

    tile_rgb: Optional[np.ndarray] = None
    source = "nasa_gibs"

    # Try Bing first if key available (higher resolution)
    hdrs = settings.outbound_http_headers()

    if api_key:
        try:
            url = _tile_url_bing(lat, lon, zoom, api_key)
            tile_rgb = await _fetch_tile_async(url, timeout=5.0, headers=hdrs)
            if tile_rgb is not None:
                source = "bing_aerial"
        except Exception:
            pass

    # Fallback to NASA GIBS (free, no key)
    # NASA GIBS GoogleMapsCompatible_Level9 only supports zoom 0-9
    if tile_rgb is None:
        try:
            gibs_zoom = min(zoom, 9)
            x, y = _latlon_to_tile_xyz(lat, lon, gibs_zoom)
            url = _tile_url_nasa_gibs(gibs_zoom, x, y, layer=MODIS_LAYER)
            tile_rgb = await _fetch_tile_async(url, timeout=5.0, headers=hdrs)
        except Exception as e:
            logger.warning("NASA GIBS tile fetch failed: %s", e)

    if tile_rgb is None:
        return {
            "enabled": False,
            "skipped_reason": "tile_fetch_failed",
            "summary": (
                "Could not fetch satellite tile. "
                "Check internet connection or configure Bing Maps API key for higher resolution."
            ),
        }

    comparison = _compare_photo_to_tile(photo_rgb, tile_rgb)

    summary_parts = [
        f"Satellite match score: {comparison['overall_match_score']:.2f} ({comparison['interpretation']}).",
    ]
    if comparison["interpretation"] == "strong_mismatch":
        summary_parts.append(
            "Photo terrain/building appearance differs strongly from satellite at predicted coordinates. "
            "Consider alternative predictions or verify location."
        )
    elif comparison["interpretation"] == "weak_mismatch":
        summary_parts.append(
            "Some visual differences between photo and satellite. Treat with moderate caution."
        )
    else:
        summary_parts.append(
            "Photo visual properties broadly consistent with satellite imagery at predicted coordinates."
        )

    return {
        "enabled": True,
        "source": source,
        "zoom": zoom,
        "coordinates": {"lat": round(lat, 5), "lon": round(lon, 5)},
        "skipped_reason": None,
        "summary": " ".join(summary_parts),
        "comparison": comparison,
    }
