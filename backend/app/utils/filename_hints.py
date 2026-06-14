"""Optional filename-based location hints for demos (not real vision)."""
import re
import unicodedata
from typing import Optional

from app.models.schemas import LocationPrediction

# Ordered: first match wins (put more specific phrases before generic ones).
_FILENAME_PLACES: list[tuple[str, float, float, str, str]] = [
    ("beverly hills", 34.0736, -118.4004, "USA", "Beverly Hills"),
    ("los angeles", 34.0522, -118.2437, "USA", "Los Angeles"),
    ("manhattan", 40.7831, -73.9712, "USA", "New York"),
    ("new york", 40.7128, -74.0060, "USA", "New York"),
    ("nyc", 40.7128, -74.0060, "USA", "New York"),
    ("eiffel", 48.8584, 2.2945, "France", "Paris"),
    ("paris", 48.8566, 2.3522, "France", "Paris"),
    ("big ben", 51.5007, -0.1246, "United Kingdom", "London"),
    ("london", 51.5074, -0.1278, "United Kingdom", "London"),
]


def _normalize_filename(name: str) -> str:
    """Lowercase, strip accents, collapse separators for substring match."""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower()
    name = name.rsplit(".", 1)[0] if "." in name else name
    name = re.sub(r"[_\-]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def location_from_filename(filename: Optional[str]) -> Optional[LocationPrediction]:
    """
    If the filename clearly names a known place, return a weak prior location.

    This does not analyze image pixels; it only helps demos when EXIF is missing.
    """
    if not filename or not str(filename).strip():
        return None

    n = _normalize_filename(str(filename))

    for phrase, lat, lon, country, city in _FILENAME_PLACES:
        if phrase in n:
            return LocationPrediction(
                latitude=lat,
                longitude=lon,
                country=country,
                city=city,
                confidence=0.82,
                distance_confidence_km=25.0,
            )

    return None
