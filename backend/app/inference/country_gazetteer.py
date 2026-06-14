"""
CLIP country softmax → StreetCLIP gazetteer country filter.

Reduces wrong-continent city matches by scoring only rows whose country matches
the top CLIP country hypotheses (with alias normalization).
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Sequence, Set

from app.config import Settings
from app.models.schemas import LocationPrediction

# CLIP country name → gazetteer variants (GeoNames English names).
_COUNTRY_ALIASES: dict[str, set[str]] = {
    "united states": {"united states", "united states of america", "usa", "u.s.a.", "us"},
    "united kingdom": {"united kingdom", "great britain", "uk", "u.k.", "england", "scotland", "wales"},
    "russia": {"russia", "russian federation"},
    "south korea": {"south korea", "korea, republic of", "republic of korea"},
    "north korea": {"north korea", "korea, democratic people's republic of"},
    "czechia": {"czechia", "czech republic"},
    "ivory coast": {"ivory coast", "côte d'ivoire", "cote d'ivoire"},
    "turkey": {"turkey", "türkiye", "turkiye"},
    "vietnam": {"vietnam", "viet nam"},
    "laos": {"laos", "lao people's democratic republic"},
    "bolivia": {"bolivia", "bolivia (plurinational state of)"},
    "venezuela": {"venezuela", "venezuela (bolivarian republic of)"},
    "tanzania": {"tanzania", "united republic of tanzania"},
    "moldova": {"moldova", "republic of moldova"},
    "iran": {"iran", "iran (islamic republic of)"},
    "syria": {"syria", "syrian arab republic"},
    "palestine": {"palestine", "palestinian territory"},
    "taiwan": {"taiwan", "taiwan, province of china"},
    "hong kong": {"hong kong", "hong kong sar"},
    "macau": {"macau", "macao", "macao sar"},
}


def _normalize_country_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", (name or "").strip().lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _alias_keys_for(name: str) -> set[str]:
    norm = _normalize_country_name(name)
    if not norm:
        return set()
    keys = {norm}
    for _canonical, variants in _COUNTRY_ALIASES.items():
        if norm in variants or norm == _canonical:
            keys |= variants
            keys.add(_canonical)
    return keys


def country_names_match(a: str, b: str) -> bool:
    """True if two country strings refer to the same territory (fuzzy)."""
    ka = _alias_keys_for(a)
    kb = _alias_keys_for(b)
    if not ka or not kb:
        return False
    return bool(ka & kb)


def top_clip_countries(
    country_predictions: Sequence[LocationPrediction],
    *,
    max_countries: int = 3,
    min_confidence: float = 0.01,
) -> List[str]:
    """Unique country names from CLIP country softmax, highest confidence first."""
    seen: set[str] = set()
    ordered: List[str] = []
    for pred in sorted(country_predictions, key=lambda p: float(p.confidence), reverse=True):
        name = (pred.country or "").strip()
        if not name or name in seen:
            continue
        if float(pred.confidence) < min_confidence:
            continue
        seen.add(name)
        ordered.append(name)
        if len(ordered) >= max_countries:
            break
    return ordered


def gazetteer_country_allowlist(
    country_predictions: Sequence[LocationPrediction],
    settings: Settings,
) -> Optional[List[str]]:
    """
    Countries to keep in StreetCLIP search. None = no country filter.
    """
    if not getattr(settings, "streetclip_country_filter_enabled", True):
        return None
    if not country_predictions:
        return None
    max_n = int(getattr(settings, "streetclip_country_filter_max_countries", 3))
    min_conf = float(getattr(settings, "streetclip_country_filter_min_confidence", 0.012))
    allow = top_clip_countries(
        country_predictions,
        max_countries=max_n,
        min_confidence=min_conf,
    )
    return allow or None


def row_country_in_allowlist(row_country: str, allowlist: Sequence[str]) -> bool:
    return any(country_names_match(row_country, allowed) for allowed in allowlist)
