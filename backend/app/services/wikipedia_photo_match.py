"""
Wikipedia / Wikimedia Commons photo matching for external validation.

1. GeoSearch on Commons (namespace 6) for images near candidate lat/lon.
2. Fallback: English Wikipedia page lead image (pageimages) for geosearch article titles.
3. CLIP image–image cosine similarity vs the uploaded photo.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
from PIL import Image

from app.config import Settings
from app.inference.clip_common import clip_image_image_cosine_similarity
from app.services.throttled_http import OutboundHttpPolicy, throttled_get_bytes

logger = logging.getLogger(__name__)

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
WIKI_API = "https://en.wikipedia.org/w/api.php"


async def _fetch_image_rgb(
    url: str,
    client: httpx.AsyncClient,
    policy: OutboundHttpPolicy,
) -> Optional[np.ndarray]:
    if not url or not str(url).strip():
        return None
    try:
        content, err = await throttled_get_bytes(
            client, str(url).strip(), policy=policy, timeout=20.0
        )
        if err or not content:
            return None
        img = Image.open(io.BytesIO(content)).convert("RGB")
        return np.array(img, dtype=np.uint8)
    except Exception as e:
        logger.debug("Wikipedia image fetch failed (%s): %s", url[:80], e)
        return None


async def _commons_geosearch_files(
    lat: float,
    lon: float,
    *,
    radius_m: int,
    limit: int,
    client: httpx.AsyncClient,
    policy: OutboundHttpPolicy,
) -> List[Dict[str, Any]]:
    params = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "list": "geosearch",
        "gscoord": f"{lat}|{lon}",
        "gsradius": min(int(radius_m), 10000),
        "gsnamespace": 6,
        "gslimit": limit,
    }
    try:
        data, err = await policy.get_json(client, COMMONS_API, params=params)
        if err or data is None:
            raise RuntimeError(err or "empty response")
        pages = (data.get("query") or {}).get("geosearch") or []
        out: List[Dict[str, Any]] = []
        for p in pages:
            title = p.get("title")
            if not title:
                continue
            out.append(
                {
                    "title": str(title),
                    "distance_m": float(p["dist"]) if p.get("dist") is not None else None,
                    "source": "commons_geosearch",
                }
            )
        return out
    except Exception as e:
        logger.warning("Commons geosearch failed: %s", e)
        return []


async def _commons_thumbnail_url(
    file_title: str,
    *,
    width: int,
    client: httpx.AsyncClient,
    policy: OutboundHttpPolicy,
) -> Optional[str]:
    params = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "titles": file_title,
        "prop": "imageinfo",
        "iiprop": "url|thumburl",
        "iiurlwidth": max(120, min(width, 1280)),
    }
    try:
        data, err = await policy.get_json(client, COMMONS_API, params=params)
        if err or data is None:
            return None
        pages = (data.get("query") or {}).get("pages") or []
        if not pages:
            return None
        page = pages[0]
        if page.get("missing"):
            return None
        infos = page.get("imageinfo") or []
        if not infos:
            return None
        info = infos[0]
        return info.get("thumburl") or info.get("url")
    except Exception as e:
        logger.debug("Commons imageinfo failed for %s: %s", file_title, e)
        return None


async def _wikipedia_pageimage_url(
    article_title: str,
    *,
    width: int,
    client: httpx.AsyncClient,
    policy: OutboundHttpPolicy,
) -> Optional[str]:
    params = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "titles": article_title,
        "prop": "pageimages",
        "piprop": "thumbnail",
        "pithumbsize": max(120, min(width, 1280)),
        "redirects": 1,
    }
    try:
        data, err = await policy.get_json(client, WIKI_API, params=params)
        if err or data is None:
            return None
        pages = (data.get("query") or {}).get("pages") or []
        if not pages or pages[0].get("missing"):
            return None
        thumb = pages[0].get("thumbnail") or {}
        return thumb.get("source")
    except Exception as e:
        logger.debug("Wikipedia pageimage failed for %s: %s", article_title, e)
        return None


def commons_file_page_url(file_title: str) -> str:
    from urllib.parse import quote

    name = (file_title or "").replace(" ", "_")
    return f"https://commons.wikimedia.org/wiki/{quote(name)}"


def wikipedia_article_url(title: str) -> str:
    from urllib.parse import quote

    return f"https://en.wikipedia.org/wiki/{quote((title or '').replace(' ', '_'))}"


async def score_wikipedia_photo_match(
    image_rgb: np.ndarray,
    lat: float,
    lon: float,
    *,
    settings: Settings,
    ranked_articles: List[Dict[str, Any]],
    client: httpx.AsyncClient,
    policy: OutboundHttpPolicy,
) -> Dict[str, Any]:
    """
    Find nearby Commons/Wikipedia images and return best CLIP photo similarity.
    """
    model_id = settings.globe_clip_model_id
    thumb_w = int(getattr(settings, "wikipedia_photo_thumb_width", 640))
    max_files = max(1, int(getattr(settings, "wikipedia_photo_max_files", 8)))
    max_articles = max(0, int(getattr(settings, "wikipedia_photo_max_article_images", 4)))
    radius = int(getattr(settings, "wikipedia_commons_geosearch_radius_m", settings.wikipedia_geosearch_radius_m))

    candidates: List[Dict[str, Any]] = []

    commons_files = await _commons_geosearch_files(
        lat, lon, radius_m=radius, limit=max_files, client=client, policy=policy
    )
    for cf in commons_files:
        url = await _commons_thumbnail_url(
            cf["title"], width=thumb_w, client=client, policy=policy
        )
        if url:
            candidates.append(
                {
                    "title": cf["title"],
                    "image_url": url,
                    "page_url": commons_file_page_url(cf["title"]),
                    "distance_m": cf.get("distance_m"),
                    "source": "commons_geosearch",
                }
            )

    for art in (ranked_articles or [])[:max_articles]:
        tit = art.get("title")
        if not tit:
            continue
        url = await _wikipedia_pageimage_url(
            str(tit), width=thumb_w, client=client, policy=policy
        )
        if url:
            candidates.append(
                {
                    "title": str(tit),
                    "image_url": url,
                    "page_url": wikipedia_article_url(str(tit)),
                    "distance_m": art.get("distance_m"),
                    "source": "wikipedia_pageimage",
                }
            )

    best_sim: Optional[float] = None
    best_row: Optional[Dict[str, Any]] = None
    scored: List[Dict[str, Any]] = []

    for cand in candidates:
        rgb = await _fetch_image_rgb(cand["image_url"], client, policy)
        if rgb is None:
            continue
        sim = clip_image_image_cosine_similarity(image_rgb, rgb, model_id)
        row = {
            **cand,
            "photo_similarity": sim,
        }
        scored.append(row)
        if sim is not None and (best_sim is None or sim > best_sim):
            best_sim = float(sim)
            best_row = row

    thr = float(getattr(settings, "wikipedia_photo_min_similarity", 0.68))
    images_tried = len(scored)
    proven = False
    if images_tried == 0:
        detail = "no Commons/Wikipedia images found near this pin to compare"
        proven = True
    elif best_sim is None:
        detail = "images found but CLIP comparison unavailable"
        proven = True
    else:
        proven = best_sim >= thr
        best_title = (best_row or {}).get("title") or "?"
        detail = (
            f"best photo CLIP {best_sim:.3f} vs {best_title} (≥ {thr:.3f})"
            if proven
            else f"best photo CLIP {best_sim:.3f} ({best_title}) < {thr:.3f}"
        )

    return {
        "images_found": len(candidates),
        "images_scored": images_tried,
        "best_similarity": best_sim,
        "best_match": best_row,
        "threshold": thr,
        "proven": proven,
        "detail": detail,
        "top_matches": sorted(
            [s for s in scored if s.get("photo_similarity") is not None],
            key=lambda x: float(x["photo_similarity"]),
            reverse=True,
        )[:5],
    }
