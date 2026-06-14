"""
Background GeoNames → StreetCLIP gazetteer JSON at server startup (optional).

UI-free alternative to POST /gazetteer/build; exposes phase/message for polling via /config.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from app.config import Settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_public: Dict[str, Any] = {
    "enabled": False,
    "phase": "idle",
    "message": "",
    "rows": None,
    "filename": None,
    "relative_path": None,
    "error": None,
}


def get_gazetteer_autoload_status() -> Dict[str, Any]:
    """JSON-safe snapshot for GET /config."""
    with _lock:
        return {**_public}


def _set(**kwargs: Any) -> None:
    with _lock:
        _public.update(kwargs)


def _default_output_relpath(settings: Settings) -> str:
    dump = (getattr(settings, "streetclip_gazetteer_autoload_dump", "cities1000") or "cities1000").strip().lower()
    return f"app/data/generated/streetclip_gazetteer_{dump}_world.json"


def _run_autoload(settings: Settings) -> None:
    """Blocking: download GeoNames dump + write JSON (may take many minutes)."""
    from pathlib import Path

    from app.data.gazetteer_loader import clear_gazetteer_json_cache
    from app.services.geonames_gazetteer_build import build_gazetteer_json, GEONAMES_CITY_DUMPS

    rel = _default_output_relpath(settings)
    dump_key = (getattr(settings, "streetclip_gazetteer_autoload_dump", "cities1000") or "cities1000").strip().lower()

    if dump_key not in GEONAMES_CITY_DUMPS:
        _set(
            phase="error",
            message=f"Invalid streetclip_gazetteer_autoload_dump: {dump_key}",
            error="bad_dump_key",
        )
        return

    skip = bool(getattr(settings, "streetclip_gazetteer_autoload_skip_if_exists", True))
    backend_root = Path(__file__).resolve().parents[2]
    cand = backend_root / "app" / "data" / "generated" / f"streetclip_gazetteer_{dump_key}_world.json"
    if skip and cand.is_file() and cand.stat().st_size > 1000:
        _set(
            phase="ready",
            message=f"Using existing gazetteer ({cand.name}, {cand.stat().st_size // 1024} KB on disk).",
            rows=None,
            filename=cand.name,
            relative_path=rel,
            error=None,
        )
        logger.info("Gazetteer autoload: skipped — %s already present", cand.name)
        return

    try:
        _set(
            phase="downloading",
            message=f"Downloading GeoNames {GEONAMES_CITY_DUMPS[dump_key][0]} + countryInfo (first run can take several minutes)…",
            rows=None,
            filename=None,
            relative_path=None,
            error=None,
        )
        _set(phase="building", message="Building worldwide city JSON from GeoNames dump…")
        result = build_gazetteer_json(settings, dump_key=dump_key, country_iso="ALL")
        clear_gazetteer_json_cache()
        _set(
            phase="ready",
            message=(
                f"StreetCLIP gazetteer ready: {result['rows']} rows → {result['filename']}. "
                "GeoNames CC BY 4.0 — attribute geonames.org when redistributing."
            ),
            rows=int(result.get("rows") or 0),
            filename=result.get("filename"),
            relative_path=result.get("relative_path"),
            error=None,
        )
        logger.info("Gazetteer autoload: complete — %s rows", result.get("rows"))
    except Exception as e:
        logger.exception("Gazetteer autoload failed")
        _set(
            phase="error",
            message=f"Gazetteer build failed: {e}",
            rows=None,
            filename=None,
            relative_path=None,
            error=str(e),
        )


def start_gazetteer_autoload_background(settings: Settings) -> None:
    """Spawn daemon thread if autoload enabled and StreetCLIP is used."""
    enabled = bool(getattr(settings, "streetclip_gazetteer_autoload_at_startup", False))
    _set(
        enabled=enabled,
        phase="skipped" if not enabled else "queued",
        message=(
            "Gazetteer autoload disabled (STREETCLIP_GAZETTEER_AUTOLOAD_AT_STARTUP=false)."
            if not enabled
            else "Queued — starting GeoNames download/build shortly…"
        ),
        rows=None,
        filename=None,
        relative_path=None,
        error=None,
    )

    if not enabled:
        return
    if not getattr(settings, "use_streetclip", False):
        _set(
            phase="skipped",
            message="StreetCLIP disabled — gazetteer autoload not needed.",
            error=None,
        )
        return

    def run() -> None:
        _set(phase="starting", message="Starting gazetteer build thread…")
        try:
            _run_autoload(settings)
        except Exception as e:
            logger.exception("Gazetteer autoload thread crashed")
            _set(phase="error", message=str(e), error=str(e))

    t = threading.Thread(target=run, name="gazetteer-autoload", daemon=True)
    t.start()
    logger.info("Gazetteer autoload thread started")
