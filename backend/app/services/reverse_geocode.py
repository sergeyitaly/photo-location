"""Resolve coordinates to city/town/village names worldwide via OpenStreetMap Nominatim (reverse geocode)."""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional, Tuple
from urllib.parse import urlencode

import httpx

from app.config import Settings
from app.models.schemas import LocationPrediction, PlaceResolution

logger = logging.getLogger(__name__)


def effective_accept_language_for_nominatim(
    header_accept_language: Optional[str],
    body_override: Optional[str],
    default_lang: str,
) -> str:
    """Pick Nominatim ``accept-language`` — body wins, then browser header, then server default."""
    if body_override and str(body_override).strip():
        return str(body_override).strip()[:48]
    if header_accept_language:
        first = header_accept_language.split(",")[0].strip().split(";")[0].strip()
        if first:
            return first[:48]
    return (default_lang or "en").strip()[:48]


def _pick_locality(address: dict) -> Tuple[str | None, str | None]:
    """Prefer city → town → village → hamlet → locality → municipality."""
    order = ("city", "town", "village", "hamlet", "locality", "municipality")
    for key in order:
        v = address.get(key)
        if v and str(v).strip():
            return str(v).strip(), key.replace("_", " ")
    return None, None


def _build_user_agent(settings: Settings) -> str:
    ua = (getattr(settings, "nominatim_http_user_agent", None) or "").strip()
    if ua:
        return ua
    return f"{settings.app_name}/{settings.app_version} (+https://wiki.openstreetmap.org/wiki/Nominatim)"


async def nominatim_reverse(
    lat: float,
    lon: float,
    *,
    settings: Settings,
    accept_language: str = "en",
) -> PlaceResolution:
    """
    Reverse-geocode a single point. Retries transient HTTP failures; safe on total failure — ``error`` set.
    """
    base = (getattr(settings, "nominatim_base_url", None) or "").strip().rstrip("/")
    if not base:
        return PlaceResolution(
            source="openstreetmap_nominatim",
            error="nominatim_base_url not configured",
        )

    params = {
        "lat": lat,
        "lon": lon,
        "format": "jsonv2",
        "accept-language": accept_language,
        "zoom": str(getattr(settings, "nominatim_reverse_zoom", 14)),
        "addressdetails": "1",
        "namedetails": "1",
    }
    url = f"{base}/reverse?{urlencode(params)}"
    headers = {"User-Agent": _build_user_agent(settings)}

    timeout = float(getattr(settings, "reverse_geocode_timeout_s", 10.0))
    max_attempts = max(1, int(getattr(settings, "reverse_geocode_retry_attempts", 2)) + 1)
    backoff = float(getattr(settings, "reverse_geocode_retry_backoff_s", 0.75))

    data: Optional[dict] = None
    last_err: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(url, headers=headers)
            if r.status_code == 429:
                return PlaceResolution(
                    source="openstreetmap_nominatim",
                    error="rate_limited_by_nominatim",
                )
            if r.status_code in (502, 503, 504):
                if attempt < max_attempts - 1:
                    logger.warning(
                        "Nominatim reverse HTTP %s (attempt %s/%s), retrying…",
                        r.status_code,
                        attempt + 1,
                        max_attempts,
                    )
                    await asyncio.sleep(backoff * (2**attempt))
                    continue
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            last_err = e
            if attempt < max_attempts - 1:
                logger.warning("Nominatim reverse attempt %s failed: %s", attempt + 1, e)
                await asyncio.sleep(backoff * (2**attempt))
                continue
            logger.warning("Nominatim reverse failed: %s", e)
            return PlaceResolution(source="openstreetmap_nominatim", error=str(e))

    if data is None:
        return PlaceResolution(
            source="openstreetmap_nominatim",
            error=str(last_err) if last_err else "nominatim_empty_response",
        )

    addr = data.get("address") or {}
    locality, kind = _pick_locality(addr)
    admin = addr.get("state") or addr.get("region") or addr.get("county")
    country = addr.get("country")
    county = addr.get("county") if isinstance(addr.get("county"), str) else None
    cc = addr.get("country_code")
    if cc is not None:
        cc = str(cc).strip().upper()[:2] or None

    return PlaceResolution(
        locality=locality,
        locality_kind=kind,
        administrative_area=str(admin).strip() if admin else None,
        county=county,
        country=str(country).strip() if country else None,
        country_code=cc,
        display_name=(data.get("display_name") or "").strip() or None,
        source="openstreetmap_nominatim",
        attribution="Data © OpenStreetMap contributors, ODbL — https://www.openstreetmap.org/copyright",
    )


async def enrich_predictions_with_reverse_geocode(
    primary: LocationPrediction,
    alternatives: List[LocationPrediction],
    *,
    settings: Settings,
    accept_language: str = "en",
) -> Tuple[LocationPrediction, List[LocationPrediction]]:
    """Attach ``place_resolution`` from Nominatim to primary and optionally first N alternatives."""
    if not getattr(settings, "reverse_geocode_enabled", False):
        return primary, alternatives

    pr = await nominatim_reverse(
        primary.latitude,
        primary.longitude,
        settings=settings,
        accept_language=accept_language,
    )
    primary_out = primary.model_copy(update={"place_resolution": pr})

    max_alt = int(getattr(settings, "reverse_geocode_max_alternatives", 0))
    if max_alt <= 0 or not alternatives:
        return primary_out, list(alternatives)

    out_alts: List[LocationPrediction] = []
    delay_s = float(getattr(settings, "reverse_geocode_inter_request_delay_s", 1.1))

    for i, alt in enumerate(alternatives):
        if i >= max_alt:
            out_alts.append(alt)
            continue
        if i > 0:
            await asyncio.sleep(delay_s)
        ar = await nominatim_reverse(
            alt.latitude,
            alt.longitude,
            settings=settings,
            accept_language=accept_language,
        )
        out_alts.append(alt.model_copy(update={"place_resolution": ar}))

    return primary_out, out_alts
