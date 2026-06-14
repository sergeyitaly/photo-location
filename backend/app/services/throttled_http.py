"""
Throttled outbound HTTP for Wikipedia, Wikimedia Commons, and OpenTopoData.

Free/public APIs return 429 when hit too fast. This module enforces per-host minimum
spacing, retries with exponential backoff (honouring Retry-After when present), short
TTL caching, and a circuit breaker after repeated rate limits.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)


class RateLimitCircuitOpen(Exception):
    """Host circuit breaker tripped after repeated 429 responses."""

    def __init__(self, host: str, retry_after_s: float = 0.0):
        self.host = host
        self.retry_after_s = retry_after_s
        super().__init__(host)


def _host_from_url(url: str) -> str:
    return (urlparse(url).netloc or "").lower() or "default"


def _cache_key(url: str, params: Optional[Dict[str, Any]]) -> str:
    if not params:
        return url
    items = tuple(sorted((k, str(v)) for k, v in params.items()))
    return f"{url}?{items}"


class OutboundHttpPolicy:
    """Per-host spacing, retries, cache, and circuit breaker from Settings."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._locks: Dict[str, asyncio.Lock] = {}
        self._last_request: Dict[str, float] = {}
        self._circuit_open_until: Dict[str, float] = {}
        self._consecutive_429: Dict[str, int] = {}
        self._json_cache: Dict[str, Tuple[float, Any]] = {}
        self._intervals = {
            "en.wikipedia.org": float(
                getattr(settings, "wikipedia_min_request_interval_s", 0.25)
            ),
            "commons.wikimedia.org": float(
                getattr(settings, "commons_min_request_interval_s", 0.35)
            ),
            "api.opentopodata.org": float(
                getattr(settings, "opentopodata_min_request_interval_s", 1.15)
            ),
        }
        self._default_interval = float(
            getattr(settings, "outbound_http_default_interval_s", 0.2)
        )

    def interval_for_host(self, host: str) -> float:
        return self._intervals.get(host, self._default_interval)

    def _lock_for(self, host: str) -> asyncio.Lock:
        if host not in self._locks:
            self._locks[host] = asyncio.Lock()
        return self._locks[host]

    def is_circuit_open(self, host: str) -> bool:
        return time.monotonic() < self._circuit_open_until.get(host, 0.0)

    def circuit_retry_after_s(self, host: str) -> float:
        until = self._circuit_open_until.get(host, 0.0)
        return max(0.0, until - time.monotonic())

    def _trip_circuit(self, host: str, retry_after_s: float) -> None:
        trips = int(getattr(self.settings, "outbound_http_circuit_trip_after_429", 3))
        self._consecutive_429[host] = self._consecutive_429.get(host, 0) + 1
        if self._consecutive_429[host] >= trips:
            cooldown = float(
                getattr(self.settings, "outbound_http_circuit_cooldown_s", 90.0)
            )
            pause = max(cooldown, retry_after_s)
            self._circuit_open_until[host] = time.monotonic() + pause
            logger.warning(
                "HTTP circuit open for %s (%.0fs) after %s consecutive 429s",
                host,
                pause,
                self._consecutive_429[host],
            )

    def _clear_429_streak(self, host: str) -> None:
        self._consecutive_429[host] = 0

    async def _wait_spacing(self, host: str) -> None:
        if self.is_circuit_open(host):
            raise RateLimitCircuitOpen(host, self.circuit_retry_after_s(host))
        lock = self._lock_for(host)
        async with lock:
            if self.is_circuit_open(host):
                raise RateLimitCircuitOpen(host, self.circuit_retry_after_s(host))
            interval = self.interval_for_host(host)
            now = time.monotonic()
            last = self._last_request.get(host, 0.0)
            wait = interval - (now - last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request[host] = time.monotonic()

    def _get_cached_json(self, key: str) -> Optional[Any]:
        ttl = float(getattr(self.settings, "outbound_http_cache_ttl_s", 600.0))
        row = self._json_cache.get(key)
        if not row:
            return None
        ts, data = row
        if time.monotonic() - ts > ttl:
            del self._json_cache[key]
            return None
        return data

    def _set_cached_json(self, key: str, data: Any) -> None:
        self._json_cache[key] = (time.monotonic(), data)

    async def get_json(
        self,
        client: httpx.AsyncClient,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 25.0,
        use_cache: bool = True,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        GET JSON with throttle + retry. Returns (data, error_message).
        On circuit open, error describes skip without hitting the network.
        """
        host = _host_from_url(url)
        cache_key = _cache_key(url, params)
        if use_cache:
            cached = self._get_cached_json(cache_key)
            if cached is not None:
                return cached, None

        max_retries = int(getattr(self.settings, "outbound_http_429_max_retries", 5))
        backoff = float(getattr(self.settings, "outbound_http_429_backoff_base_s", 2.0))

        last_err: Optional[str] = None
        for attempt in range(max_retries + 1):
            try:
                await self._wait_spacing(host)
            except RateLimitCircuitOpen as e:
                return None, (
                    f"Rate limited on {host} — pausing outbound requests "
                    f"({e.retry_after_s:.0f}s cooldown)"
                )

            try:
                r = await client.get(url, params=params, timeout=timeout)
                if r.status_code == 429:
                    retry_after = _parse_retry_after(r)
                    self._trip_circuit(host, retry_after)
                    last_err = f"429 Too Many Requests (attempt {attempt + 1}/{max_retries + 1})"
                    if attempt < max_retries:
                        sleep_s = max(retry_after, backoff * (2**attempt))
                        logger.info(
                            "HTTP 429 %s — sleeping %.1fs before retry",
                            host,
                            sleep_s,
                        )
                        await asyncio.sleep(sleep_s)
                        continue
                    if self.is_circuit_open(host):
                        return None, (
                            f"Rate limited on {host} after {max_retries + 1} retries "
                            f"(cooldown {self.circuit_retry_after_s(host):.0f}s)"
                        )
                    return None, f"Client error '429 Too Many Requests' for url '{r.url}'"

                if r.status_code in (502, 503, 504):
                    last_err = f"{r.status_code} upstream error"
                    if attempt < max_retries:
                        await asyncio.sleep(backoff * (2**attempt))
                        continue
                    return None, last_err

                r.raise_for_status()
                data = r.json()
                self._clear_429_streak(host)
                if use_cache:
                    self._set_cached_json(cache_key, data)
                return data, None
            except httpx.HTTPStatusError as e:
                last_err = str(e)
                if e.response.status_code == 429 and attempt < max_retries:
                    continue
                return None, last_err
            except Exception as e:
                last_err = str(e)
                if attempt < max_retries:
                    await asyncio.sleep(backoff * (2**attempt))
                    continue
                return None, last_err

        return None, last_err or "request failed"


def _parse_retry_after(response: httpx.Response) -> float:
    raw = (response.headers.get("Retry-After") or "").strip()
    if not raw:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


async def throttled_get_bytes(
    client: httpx.AsyncClient,
    url: str,
    *,
    policy: OutboundHttpPolicy,
    timeout: float = 20.0,
) -> Tuple[Optional[bytes], Optional[str]]:
    """GET binary body (e.g. thumbnail) with spacing; no JSON cache."""
    host = _host_from_url(url)
    max_retries = int(getattr(policy.settings, "outbound_http_429_max_retries", 5))
    backoff = float(getattr(policy.settings, "outbound_http_429_backoff_base_s", 2.0))
    last_err: Optional[str] = None

    for attempt in range(max_retries + 1):
        try:
            await policy._wait_spacing(host)
        except RateLimitCircuitOpen as e:
            return None, f"Rate limited on {host} ({e.retry_after_s:.0f}s cooldown)"

        try:
            r = await client.get(url, timeout=timeout, follow_redirects=True)
            if r.status_code == 429:
                retry_after = _parse_retry_after(r)
                policy._trip_circuit(host, retry_after)
                last_err = "429 Too Many Requests"
                if attempt < max_retries:
                    await asyncio.sleep(max(retry_after, backoff * (2**attempt)))
                    continue
                return None, last_err
            if r.status_code != 200:
                return None, f"HTTP {r.status_code}"
            policy._clear_429_streak(host)
            return r.content, None
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries:
                await asyncio.sleep(backoff * (2**attempt))
                continue
            return None, last_err
    return None, last_err
