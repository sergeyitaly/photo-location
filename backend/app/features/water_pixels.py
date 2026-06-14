"""
Generic open-water pixel heuristic (any geography).

Used to avoid mis-labeling lake/ocean scenes as arid when green vegetation is low.
Does not move map pins or prefer any country or city.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple


def water_fraction_central(image_rgb) -> float:
    """Share of blue water-like pixels in the central band (wide lake/sea shots)."""
    if image_rgb is None or image_rgb.size == 0:
        return 0.0
    import numpy as np

    h, w = image_rgb.shape[:2]
    y0, y1 = int(h * 0.22), int(h * 0.78)
    x0, x1 = int(w * 0.12), int(w * 0.88)
    band = image_rgb[y0:y1, x0:x1]
    if band.size == 0:
        return 0.0
    r = band[:, :, 0].astype(np.float32)
    g = band[:, :, 1].astype(np.float32)
    b = band[:, :, 2].astype(np.float32)
    water = (b > r + 12.0) & (b > g + 6.0) & (b > 45.0)
    sky = (b > 95.0) & (g > 75.0) & (r > 60.0) & (np.abs(b - g) < 35.0)
    valid = water & ~sky
    return float(np.mean(valid))


def open_water_scene_score(
    image_rgb,
    *,
    ml_labels: Optional[Sequence[Tuple[str, float]]] = None,
) -> float:
    """0–1 score that the frame may show open water; does not identify which lake or sea."""
    score = water_fraction_central(image_rgb) if image_rgb is not None else 0.0
    if ml_labels:
        for label, prob in ml_labels[:6]:
            low = (label or "").lower()
            if any(k in low for k in ("lake", "river", "water", "reservoir", "shore", "coast", "sea", "ocean")):
                score = max(score, float(prob) * 0.85)
    return min(1.0, score)
