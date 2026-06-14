"""GeoNames → StreetCLIP gazetteer JSON (optional UI builder)."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import settings
from app.models.schemas import GazetteerBuildRequest
from app.services.geonames_gazetteer_build import (
    GEONAMES_CITY_DUMPS,
    build_gazetteer_json,
    list_countries,
    resolve_safe_gazetteer_download,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["gazetteer"])


def _dump_meta() -> list[dict]:
    return [
        {"id": k, "zip_name": v[0], "description": v[1]}
        for k, v in sorted(GEONAMES_CITY_DUMPS.items(), key=lambda kv: kv[0])
    ]


@router.get("/gazetteer/meta")
async def gazetteer_meta():
    """Dump kinds + enable flag for the web UI."""
    return {
        "enabled": settings.gazetteer_build_enabled,
        "dumps": _dump_meta(),
        "license_note": "GeoNames data CC BY 4.0 — https://www.geonames.org/",
    }


@router.get("/gazetteer/countries")
async def gazetteer_countries():
    """ISO2 + English country name (from GeoNames countryInfo.txt; may download once)."""
    if not settings.gazetteer_build_enabled:
        return {"enabled": False, "countries": []}
    try:
        rows = await asyncio.to_thread(list_countries, settings)
    except Exception as e:
        logger.exception("gazetteer countries failed")
        raise HTTPException(status_code=502, detail=str(e)) from e
    return {"enabled": True, "countries": rows}


@router.post("/gazetteer/build")
async def gazetteer_build(body: GazetteerBuildRequest):
    """Download/cache GeoNames zip + countryInfo, emit JSON under app/data/generated."""
    if not settings.gazetteer_build_enabled:
        raise HTTPException(status_code=403, detail="Gazetteer build is disabled on this server.")
    try:
        result = await asyncio.to_thread(
            lambda: build_gazetteer_json(
                settings,
                dump_key=body.dump.strip(),
                country_iso=body.country_iso.strip(),
            ),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("gazetteer build failed")
        raise HTTPException(status_code=502, detail=str(e)) from e
    result["download_url_path"] = f"/gazetteer/download/{result['filename']}"
    result["env_hint"] = (
        f"Set STREETCLIP_GAZETTEER_PATH={result['relative_path']} in backend/.env and restart "
        "to use this gazetteer for StreetCLIP."
    )
    return result


@router.get("/gazetteer/download/{filename}")
async def gazetteer_download(filename: str):
    """Fetch a previously built gazetteer JSON (same origin as POST /gazetteer/build)."""
    if not settings.gazetteer_build_enabled:
        raise HTTPException(status_code=403, detail="Gazetteer build is disabled on this server.")
    try:
        path = resolve_safe_gazetteer_download(settings, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found — build it first.")
    return FileResponse(path, filename=filename, media_type="application/json")
