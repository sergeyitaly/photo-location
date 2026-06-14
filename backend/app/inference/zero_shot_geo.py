"""
CLIP zero-shot helpers for coarse geolocation: country priors + landmark prompts.

Uses contrastive softmax over hand-written prompts — not a trained GeoCLIP index.
"""

from __future__ import annotations

from typing import List

import numpy as np

from app.models.schemas import LocationPrediction
from app.inference.clip_common import clip_softmax_probs_ordered, is_clip_runtime_available
from app.data.world_country_clip_centroids import COUNTRY_ENTRIES_WORLDWIDE

# Worldwide ISO-level English names + centroids (~250). Regenerate `world_country_clip_centroids.py` if needed.
COUNTRY_ENTRIES: List[tuple[str, float, float]] = COUNTRY_ENTRIES_WORLDWIDE

LANDMARK_ENTRIES: List[dict] = [
    {
        "prompt": "a photo of the Eiffel Tower in Paris France",
        "lat": 48.8584,
        "lon": 2.2945,
        "city": "Eiffel Tower",
        "country": "France",
    },
    {
        "prompt": "a photo of the Statue of Liberty in New York",
        "lat": 40.6892,
        "lon": -74.0445,
        "city": "Statue of Liberty",
        "country": "USA",
    },
    {
        "prompt": "a photo of Big Ben and the Houses of Parliament in London",
        "lat": 51.5007,
        "lon": -0.1246,
        "city": "Big Ben",
        "country": "United Kingdom",
    },
    {
        "prompt": "a photo of the Taj Mahal in India",
        "lat": 27.1751,
        "lon": 78.0421,
        "city": "Taj Mahal",
        "country": "India",
    },
    {
        "prompt": "a photo of the Great Wall of China",
        "lat": 40.4319,
        "lon": 116.5704,
        "city": "Great Wall",
        "country": "China",
    },
    {
        "prompt": "a photo of the Colosseum in Rome Italy",
        "lat": 41.8902,
        "lon": 12.4922,
        "city": "Colosseum",
        "country": "Italy",
    },
    {
        "prompt": "a photo of the Sydney Opera House Australia",
        "lat": -33.8568,
        "lon": 151.2153,
        "city": "Sydney Opera House",
        "country": "Australia",
    },
    {
        "prompt": "a photo of the Golden Gate Bridge San Francisco",
        "lat": 37.8199,
        "lon": -122.4783,
        "city": "Golden Gate Bridge",
        "country": "USA",
    },
    {
        "prompt": "a photo of the Brandenburg Gate in Berlin Germany",
        "lat": 52.5163,
        "lon": 13.3777,
        "city": "Brandenburg Gate",
        "country": "Germany",
    },
    {
        "prompt": "a photo of the Acropolis in Athens Greece",
        "lat": 37.9715,
        "lon": 23.7267,
        "city": "Acropolis",
        "country": "Greece",
    },
    {
        "prompt": "a photo of Saint Basil's Cathedral Red Square Moscow",
        "lat": 55.7525,
        "lon": 37.6231,
        "city": "Saint Basil's Cathedral",
        "country": "Russia",
    },
    {
        "prompt": "a photo of Christ the Redeemer statue Rio de Janeiro Brazil",
        "lat": -22.9519,
        "lon": -43.2105,
        "city": "Christ the Redeemer",
        "country": "Brazil",
    },
    {
        "prompt": "a photo of the Pyramids of Giza Egypt",
        "lat": 29.9792,
        "lon": 31.1342,
        "city": "Pyramids of Giza",
        "country": "Egypt",
    },
    {
        "prompt": "a photo of the Burj Khalifa Dubai UAE",
        "lat": 25.1972,
        "lon": 55.2744,
        "city": "Burj Khalifa",
        "country": "United Arab Emirates",
    },
    {
        "prompt": "a photo of the Sagrada Familia Barcelona Spain",
        "lat": 41.4036,
        "lon": 2.1744,
        "city": "Sagrada Familia",
        "country": "Spain",
    },
]


def clip_country_predictions(
    image_rgb: np.ndarray,
    model_id: str,
    *,
    top_k: int = 8,
    min_prob: float = 0.006,
) -> List[LocationPrediction]:
    """
    Zero-shot country softmax over ``a photograph taken in {country}`` for every loaded territory.

    With ~250 labels, uniform chance is ~0.004 — use a low ``min_prob`` or rely on ``top_k`` only.
    """
    if not is_clip_runtime_available():
        return []
    prompts = [f"a photograph taken in {name}" for name, _, _ in COUNTRY_ENTRIES]
    probs = clip_softmax_probs_ordered(image_rgb, prompts, model_id)
    if probs is None or len(probs) != len(COUNTRY_ENTRIES):
        return []

    n = len(COUNTRY_ENTRIES)
    uniform = 1.0 / float(max(n, 1))
    # Ignore mass barely above random unless user sets min_prob higher
    floor = max(min_prob, uniform * 1.25)

    order = sorted(range(len(probs)), key=lambda i: float(probs[i]), reverse=True)
    out: List[LocationPrediction] = []
    for i in order[:top_k]:
        if float(probs[i]) < floor:
            continue
        name, lat, lon = COUNTRY_ENTRIES[i]
        p = float(probs[i])
        out.append(
            LocationPrediction(
                latitude=lat,
                longitude=lon,
                country=name,
                city=name,
                confidence=min(p, 1.0),
                distance_confidence_km=float(350.0 + (1.0 - p) * 1700.0),
            )
        )
    return out


def clip_landmark_predictions(
    image_rgb: np.ndarray,
    model_id: str,
    *,
    top_k: int = 5,
    min_prob: float = 0.05,
) -> List[LocationPrediction]:
    """Landmark softmax over curated text prompts with known coordinates."""
    if not is_clip_runtime_available():
        return []
    prompts = [e["prompt"] for e in LANDMARK_ENTRIES]
    probs = clip_softmax_probs_ordered(image_rgb, prompts, model_id)
    if probs is None or len(probs) != len(LANDMARK_ENTRIES):
        return []

    order = sorted(range(len(probs)), key=lambda i: float(probs[i]), reverse=True)
    out: List[LocationPrediction] = []
    for i in order[:top_k]:
        if float(probs[i]) < min_prob:
            continue
        e = LANDMARK_ENTRIES[i]
        p = float(probs[i])
        out.append(
            LocationPrediction(
                latitude=e["lat"],
                longitude=e["lon"],
                country=e["country"],
                city=e["city"],
                confidence=min(p * 0.98, 1.0),
                distance_confidence_km=12.0,
            )
        )
    return out
