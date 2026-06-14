"""Proxy Google Maps Static + Street View images (keeps API key server-side)."""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/maps", tags=["maps"])


def _maps_api_key() -> str:
    key = (settings.google_maps_api_key or "").strip()
    if not key or key.lower() in ("", "none", "null"):
        return ""
    return key


@router.get("/streetview")
async def proxy_streetview(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
):
    api_key = _maps_api_key()
    if not api_key:
        raise HTTPException(status_code=503, detail="GOOGLE_MAPS_API_KEY not configured")
    url = (
        "https://maps.googleapis.com/maps/api/streetview"
        f"?size=640x400&location={latitude},{longitude}&key={api_key}"
    )
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url)
    except httpx.HTTPError as exc:
        logger.warning("Street View proxy fetch failed: %s", exc)
        raise HTTPException(status_code=502, detail="Street View upstream fetch failed") from exc
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail="Street View image unavailable")
    media = r.headers.get("content-type", "image/jpeg")
    return Response(content=r.content, media_type=media)


@router.get("/staticmap")
async def proxy_staticmap(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
):
    api_key = _maps_api_key()
    if not api_key:
        raise HTTPException(status_code=503, detail="GOOGLE_MAPS_API_KEY not configured")
    url = (
        "https://maps.googleapis.com/maps/api/staticmap"
        f"?center={latitude},{longitude}&zoom=17&size=640x400&maptype=satellite"
        f"&markers=color:red%7C{latitude},{longitude}&key={api_key}"
    )
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url)
    except httpx.HTTPError as exc:
        logger.warning("Static map proxy fetch failed: %s", exc)
        raise HTTPException(status_code=502, detail="Static map upstream fetch failed") from exc
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail="Static map image unavailable")
    media = r.headers.get("content-type", "image/png")
    return Response(content=r.content, media_type=media)
