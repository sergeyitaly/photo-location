"""
Embedded fallback only — used when ``STREETCLIP_GAZETTEER_PATH`` is unset or missing.

For production, generate a GeoNames-scale JSON via ``scripts/build_streetclip_gazetteer.py``
and point ``streetclip_gazetteer_path`` at it (StreetCLIP filters by GeoCLIP bbox + cap).
"""

from __future__ import annotations

from typing import Any, Dict, List

# Short global spread (~35) — not a substitute for a full gazetteer file.
WORLD_CITIES_EMBEDDED_FALLBACK: List[Dict[str, Any]] = [
    {"city": "Paris", "country": "France", "lat": 48.8566, "lon": 2.3522},
    {"city": "London", "country": "United Kingdom", "lat": 51.5074, "lon": -0.1278},
    {"city": "Berlin", "country": "Germany", "lat": 52.5200, "lon": 13.4050},
    {"city": "Rome", "country": "Italy", "lat": 41.9028, "lon": 12.4964},
    {"city": "Madrid", "country": "Spain", "lat": 40.4168, "lon": -3.7038},
    {"city": "Amsterdam", "country": "Netherlands", "lat": 52.3676, "lon": 4.9041},
    {"city": "Vienna", "country": "Austria", "lat": 48.2082, "lon": 16.3738},
    {"city": "Warsaw", "country": "Poland", "lat": 52.2297, "lon": 21.0122},
    {"city": "Moscow", "country": "Russia", "lat": 55.7558, "lon": 37.6173},
    {"city": "Istanbul", "country": "Turkey", "lat": 41.0082, "lon": 28.9784},
    {"city": "Dubai", "country": "United Arab Emirates", "lat": 25.2048, "lon": 55.2708},
    {"city": "Tel Aviv", "country": "Israel", "lat": 32.0853, "lon": 34.7818},
    {"city": "Cairo", "country": "Egypt", "lat": 30.0444, "lon": 31.2357},
    {"city": "Lagos", "country": "Nigeria", "lat": 6.5244, "lon": 3.3792},
    {"city": "Nairobi", "country": "Kenya", "lat": -1.286389, "lon": 36.817223},
    {"city": "Cape Town", "country": "South Africa", "lat": -33.9249, "lon": 18.4241},
    {"city": "New York", "country": "USA", "lat": 40.7128, "lon": -74.0060},
    {"city": "Los Angeles", "country": "USA", "lat": 34.0522, "lon": -118.2437},
    {"city": "Chicago", "country": "USA", "lat": 41.8781, "lon": -87.6298},
    {"city": "San Francisco", "country": "USA", "lat": 37.7749, "lon": -122.4194},
    {"city": "Mexico City", "country": "Mexico", "lat": 19.4326, "lon": -99.1332},
    {"city": "São Paulo", "country": "Brazil", "lat": -23.5505, "lon": -46.6333},
    {"city": "Buenos Aires", "country": "Argentina", "lat": -34.6037, "lon": -58.3816},
    {"city": "Toronto", "country": "Canada", "lat": 43.6532, "lon": -79.3832},
    {"city": "Vancouver", "country": "Canada", "lat": 49.2827, "lon": -123.1207},
    {"city": "Tokyo", "country": "Japan", "lat": 35.6762, "lon": 139.6503},
    {"city": "Seoul", "country": "South Korea", "lat": 37.5665, "lon": 126.9780},
    {"city": "Beijing", "country": "China", "lat": 39.9042, "lon": 116.4074},
    {"city": "Shanghai", "country": "China", "lat": 31.2304, "lon": 121.4737},
    {"city": "Singapore", "country": "Singapore", "lat": 1.3521, "lon": 103.8198},
    {"city": "Bangkok", "country": "Thailand", "lat": 13.7563, "lon": 100.5018},
    {"city": "Delhi", "country": "India", "lat": 28.6139, "lon": 77.2090},
    {"city": "Mumbai", "country": "India", "lat": 19.0760, "lon": 72.8777},
    {"city": "Sydney", "country": "Australia", "lat": -33.8688, "lon": 151.2093},
    {"city": "Melbourne", "country": "Australia", "lat": -37.8136, "lon": 144.9631},
]

# Back-compat name (same reference).
WORLD_CITIES = WORLD_CITIES_EMBEDDED_FALLBACK
