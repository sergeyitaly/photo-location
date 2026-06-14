"""CLIP softmax prompts for general scene / object image recognition (interpretive)."""

from __future__ import annotations

# Curated open-vocabulary labels — softmax is only over this list for one image.
ML_RECOGNITION_PROMPTS: list[str] = [
    "a photo of a city street with buildings",
    "a photo of a highway or rural road",
    "a photo of mountains or hills",
    "a photo of a beach or coastline",
    "a photo of a desert landscape",
    "a photo of a forest or dense trees",
    "a photo of farmland or fields",
    "a photo taken indoors",
    "a photo of a historic church or cathedral exterior",
    "a photo of a mosque",
    "a photo of a temple",
    "a photo of a bridge",
    "a photo of a skyline or tall buildings",
    "a photo of snow or winter scene",
    "a photo of people in an urban plaza",
    "a photo of water such as a lake or river",
    "a photo of residential houses",
    "a photo of power lines or pylons",
    "a photo of tropical palm trees",
    "a photo of signage or storefronts",
    "a photo at night with artificial lighting",
    "a photo of a stadium or arena",
    "a photo of a monument or statue",
]
