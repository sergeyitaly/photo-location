"""
Disk-based prediction cache for identical image re-uploads.

Key: SHA-256 of raw image bytes (deterministic, collision-resistant).
Storage: JSON files under app/data/generated/prediction_cache/.
TTL: Entries older than prediction_cache_ttl_seconds are ignored and deleted on read.
Eviction: When max entries exceeded, oldest files (by mtime) are removed on write.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import Settings

logger = logging.getLogger(__name__)


def _default_cache_dir() -> Path:
    """Fallback cache directory relative to this module."""
    return Path(__file__).resolve().parents[2] / "app" / "data" / "generated" / "prediction_cache"


def _resolve_cache_dir(settings: Settings) -> Path:
    explicit = getattr(settings, "prediction_cache_dir", "") or ""
    if explicit:
        p = Path(explicit).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p
    p = _default_cache_dir()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _compute_image_hash(image_bytes: bytes) -> str:
    """SHA-256 hex digest of raw image bytes."""
    return hashlib.sha256(image_bytes).hexdigest()


def _cache_path(cache_dir: Path, image_hash: str) -> Path:
    return cache_dir / f"{image_hash}.json"


def _is_stale(path: Path, ttl_seconds: int) -> bool:
    if not path.is_file():
        return True
    if ttl_seconds <= 0:
        return False
    try:
        mtime = path.stat().st_mtime
        return (time.time() - mtime) > ttl_seconds
    except OSError:
        return True


def _evict_oldest(cache_dir: Path, max_entries: int) -> None:
    """Remove oldest files by mtime when over max_entries."""
    try:
        files = [f for f in cache_dir.iterdir() if f.is_file() and f.suffix == ".json"]
    except OSError:
        return
    if len(files) <= max_entries:
        return
    files.sort(key=lambda p: p.stat().st_mtime)
    to_remove = files[: len(files) - max_entries]
    for f in to_remove:
        try:
            f.unlink()
        except OSError:
            pass
    if to_remove:
        logger.debug("Evicted %d old prediction cache entries", len(to_remove))


def get_cached_prediction(
    image_bytes: bytes,
    settings: Settings,
) -> Optional[Dict[str, Any]]:
    """
    Check disk cache for a prediction result keyed by image hash.
    Returns None if disabled, missing, or stale (stale files are deleted).
    """
    if not getattr(settings, "use_prediction_cache", True):
        return None

    image_hash = _compute_image_hash(image_bytes)
    cache_dir = _resolve_cache_dir(settings)
    path = _cache_path(cache_dir, image_hash)

    ttl = int(getattr(settings, "prediction_cache_ttl_seconds", 86400))
    if _is_stale(path, ttl):
        if path.exists():
            try:
                path.unlink()
                logger.debug("Removed stale prediction cache entry %s", image_hash[:12])
            except OSError:
                pass
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        logger.info("Prediction cache HIT for %s… (%s)", image_hash[:12], path.name)
        return data
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Prediction cache read failed for %s: %s", image_hash[:12], e)
        return None


def set_cached_prediction(
    image_bytes: bytes,
    result: Dict[str, Any],
    settings: Settings,
) -> None:
    """
    Store a prediction result on disk keyed by image hash.
    Evicts oldest entries if over max.
    """
    if not getattr(settings, "use_prediction_cache", True):
        return

    image_hash = _compute_image_hash(image_bytes)
    cache_dir = _resolve_cache_dir(settings)
    path = _cache_path(cache_dir, image_hash)

    try:
        payload = {
            "cached_at": time.time(),
            "result": result,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        logger.info("Prediction cache STORED for %s… (%s)", image_hash[:12], path.name)
    except OSError as e:
        logger.warning("Prediction cache write failed for %s: %s", image_hash[:12], e)
        return

    max_entries = int(getattr(settings, "prediction_cache_max_entries", 1000))
    if max_entries > 0:
        _evict_oldest(cache_dir, max_entries)


def delete_cached_prediction(image_bytes: bytes, settings: Settings) -> bool:
    """Remove the disk cache entry for this image hash, if present."""
    if not image_bytes:
        return False
    cache_dir = _resolve_cache_dir(settings)
    image_hash = _compute_image_hash(image_bytes)
    path = _cache_path(cache_dir, image_hash)
    if not path.is_file():
        return False
    try:
        path.unlink()
        logger.info("Prediction cache deleted for %s…", image_hash[:12])
        return True
    except OSError as e:
        logger.warning("Prediction cache delete failed for %s: %s", image_hash[:12], e)
        return False


def clear_prediction_cache(settings: Settings) -> int:
    """Remove all prediction cache entries. Returns count removed."""
    cache_dir = _resolve_cache_dir(settings)
    count = 0
    try:
        for f in cache_dir.iterdir():
            if f.is_file() and f.suffix == ".json":
                try:
                    f.unlink()
                    count += 1
                except OSError:
                    pass
    except OSError:
        pass
    logger.info("Cleared %d prediction cache entries from %s", count, cache_dir)
    return count

