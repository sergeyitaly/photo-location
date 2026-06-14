"""
Specialist visual detectors for elite geolocation cues.

Detects:
  1. Shadow geometry -> sun elevation, latitude band, hemisphere
  2. Utility pole proxies -> rough pole type classification from edges/texture
  3. Road line proxies -> detect linear features in road-like regions
  4. Driving side proxies -> vehicle position in lane, road edge bias

All of these are lightweight pixel heuristics (no heavy ML models).
They provide DetectedCue objects for the CountryEliminationEngine and BayesianGeoReasoner.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.reasoning.country_elimination import DetectedCue
from app.reasoning.astronomy_solver import AstronomyConstraints, AstronomySolver

logger = logging.getLogger(__name__)


def _connected_components_label(binary_mask: np.ndarray) -> tuple[np.ndarray, int] | None:
    """
    Label connected regions on a boolean mask.

    Returns (labeled_array, num_features) or None if scipy is not installed.
    Pole/road proxy cues are skipped when scipy is missing so /predict still succeeds.
    """
    try:
        from scipy import ndimage

        return ndimage.label(binary_mask)
    except ImportError:
        logger.warning(
            "scipy is not installed — skipping pole/road-line blob grouping "
            "(install with: pip install 'scipy>=1.11')"
        )
        return None


# ---------------------------------------------------------------------------
# Shadow detection
# ---------------------------------------------------------------------------

def detect_shadow_features(image_rgb: np.ndarray) -> Dict[str, Optional[float]]:
    """
    Detect shadow-related features from pixel statistics.

    Returns:
        {
            "shadow_direction_deg": approximate shadow direction (0=N, 90=E),
            "shadow_ratio_estimate": shadow length / object height estimate,
            "dark_region_ratio": fraction of image that is very dark,
            "confidence": 0..1 confidence in these estimates
        }
    """
    if image_rgb is None or image_rgb.size == 0:
        return {
            "shadow_direction_deg": None,
            "shadow_ratio_estimate": None,
            "dark_region_ratio": None,
            "confidence": 0.0,
        }

    h, w = image_rgb.shape[:2]
    gray = np.mean(image_rgb, axis=2)

    # Identify dark regions (potential shadows)
    # Shadows are dark but often not pure black; they also tend to be cooler (bluer)
    r = image_rgb[:, :, 0].astype(np.float32)
    g = image_rgb[:, :, 1].astype(np.float32)
    b = image_rgb[:, :, 2].astype(np.float32)

    # Shadow mask: dark + relatively blue/cool
    dark_threshold = np.percentile(gray, 25)
    shadow_mask = (gray < dark_threshold) & (b > r * 0.9) & (b > g * 0.8)
    dark_ratio = float(np.mean(shadow_mask))

    # Estimate shadow direction from gradient of dark regions
    shadow_dir = None
    if dark_ratio > 0.05:
        # Find centroids of dark regions in lower half (ground shadows)
        lower_half = shadow_mask[h // 2 :, :]
        if np.any(lower_half):
            y_coords, x_coords = np.where(lower_half)
            # Project to angle from vertical (simplified)
            if len(y_coords) > 10:
                # Centroid of dark region
                cy = np.mean(y_coords) + h // 2
                cx = np.mean(x_coords)
                # Vector from image center to centroid
                dx = cx - w / 2
                dy = cy - h / 2
                angle_rad = np.arctan2(dx, -dy)  # 0 = up (north), pi/2 = right (east)
                shadow_dir = float(np.degrees(angle_rad) % 360)

    # Shadow ratio: compare dark region extent to bright region height
    shadow_ratio = None
    bright_mask = gray > np.percentile(gray, 75)
    if np.any(bright_mask) and np.any(shadow_mask):
        bright_height = np.max(np.where(bright_mask)[0]) - np.min(np.where(bright_mask)[0])
        shadow_height = np.max(np.where(shadow_mask)[0]) - np.min(np.where(shadow_mask)[0])
        if bright_height > 0:
            shadow_ratio = float(shadow_height) / float(bright_height)

    confidence = min(1.0, dark_ratio * 5.0)  # more dark = higher confidence

    return {
        "shadow_direction_deg": shadow_dir,
        "shadow_ratio_estimate": shadow_ratio,
        "dark_region_ratio": dark_ratio,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Utility pole proxy detection
# ---------------------------------------------------------------------------

def detect_pole_proxies(image_rgb: np.ndarray) -> List[Dict[str, float]]:
    """
    Detect vertical pole-like structures using edge analysis.

    Returns a list of detected pole proxies with confidence scores.
    NOT a real pole classifier — uses edge density + verticality heuristics.
    """
    if image_rgb is None or image_rgb.size == 0:
        return []

    h, w = image_rgb.shape[:2]
    gray = np.mean(image_rgb, axis=2).astype(np.float32)

    # Vertical edge detection
    gy, gx = np.gradient(gray)
    vertical_edges = np.abs(gy)
    horizontal_edges = np.abs(gx)

    # Find vertical line regions (poles have strong vertical edges, weak horizontal)
    pole_mask = (vertical_edges > np.percentile(vertical_edges, 85)) & (
        horizontal_edges < np.percentile(horizontal_edges, 70)
    )

    # Segment into connected vertical regions
    cc = _connected_components_label(pole_mask)
    if cc is None:
        return []
    labeled, num_features = cc

    poles: List[Dict[str, float]] = []
    for i in range(1, min(num_features + 1, 20)):  # limit to top 20
        region = labeled == i
        coords = np.where(region)
        if len(coords[0]) < 20:
            continue

        ry_min, ry_max = coords[0].min(), coords[0].max()
        rx_min, rx_max = coords[1].min(), coords[1].max()
        height = ry_max - ry_min
        width = rx_max - rx_min

        if height < h * 0.15:  # too short
            continue
        if width > h * 0.08:  # too wide (building, not pole)
            continue
        if height / (width + 1) < 3:  # not vertical enough
            continue

        # Classify by texture (very rough)
        region_rgb = image_rgb[ry_min:ry_max, rx_min:rx_max, :]
        mean_color = np.mean(region_rgb, axis=(0, 1))
        color_std = np.std(region_rgb, axis=(0, 1)).mean()

        # Wood: brown-ish, higher texture variance
        # Concrete: gray, lower texture variance
        # Metal: uniform, often darker
        is_wood = mean_color[0] > mean_color[2] and color_std > 30
        is_concrete = np.abs(mean_color[0] - mean_color[1]) < 15 and np.abs(mean_color[1] - mean_color[2]) < 15 and color_std < 25
        is_metal = np.mean(mean_color) < 100 and color_std < 20

        pole_type = "unknown"
        if is_wood:
            pole_type = "wooden"
        elif is_concrete:
            pole_type = "concrete"
        elif is_metal:
            pole_type = "metal"

        poles.append({
            "type": pole_type,
            "height_px": int(height),
            "width_px": int(width),
            "aspect_ratio": float(height / (width + 1)),
            "confidence": min(0.9, float(height / h) * 2.0),
        })

    # Sort by confidence
    poles.sort(key=lambda x: x["confidence"], reverse=True)
    return poles[:5]


# ---------------------------------------------------------------------------
# Road line proxy detection
# ---------------------------------------------------------------------------

def detect_road_line_proxies(image_rgb: np.ndarray) -> List[Dict[str, float]]:
    """
    Detect horizontal line features in lower portion of image (road region).

    Uses simple edge + color analysis to find potential lane markings.
    """
    if image_rgb is None or image_rgb.size == 0:
        return []

    h, w = image_rgb.shape[:2]
    # Focus on lower 40% of image (road surface)
    road_region = image_rgb[int(h * 0.6) :, :, :]
    road_gray = np.mean(road_region, axis=2)

    # Find bright line-like features (white/yellow markings)
    bright_mask = road_gray > np.percentile(road_gray, 80)

    # Horizontal line detection
    gy, gx = np.gradient(road_gray)
    horizontal_edges = np.abs(gx) > np.percentile(np.abs(gx), 75)

    # Line mask: bright + horizontal edge
    line_mask = bright_mask & horizontal_edges

    cc = _connected_components_label(line_mask)
    if cc is None:
        return []
    labeled, num_features = cc

    lines: List[Dict[str, float]] = []
    for i in range(1, min(num_features + 1, 10)):
        region = labeled == i
        coords = np.where(region)
        if len(coords[0]) < 50:
            continue

        ry_min, ry_max = coords[0].min(), coords[0].max()
        rx_min, rx_max = coords[1].min(), coords[1].max()
        length = rx_max - rx_min
        width = ry_max - ry_min

        if length < w * 0.1:  # too short
            continue
        if width > length * 0.5:  # not line-like
            continue

        # Determine color
        line_rgb = road_region[ry_min:ry_max, rx_min:rx_max, :]
        mean_color = np.mean(line_rgb, axis=(0, 1))

        r, g, b = mean_color
        is_yellow = r > 180 and g > 150 and b < 120
        is_white = r > 180 and g > 180 and b > 180 and np.std(mean_color) < 30
        is_red = r > 150 and g < 100 and b < 100

        color = "unknown"
        if is_yellow:
            color = "yellow"
        elif is_white:
            color = "white"
        elif is_red:
            color = "red"

        lines.append({
            "color": color,
            "length_px": int(length),
            "width_px": int(width),
            "aspect_ratio": float(length / (width + 1)),
            "confidence": min(0.85, float(length / w) * 1.5),
        })

    lines.sort(key=lambda x: x["confidence"], reverse=True)
    return lines[:5]


# ---------------------------------------------------------------------------
# Driving side proxy
# ---------------------------------------------------------------------------

def detect_driving_side_proxy(image_rgb: np.ndarray) -> Optional[Dict[str, float]]:
    """
    Very rough proxy: if vehicles are visible, check which side of road they're on.
    Uses asymmetry in lower portion of image.

    Returns {"side": "left" or "right", "confidence": 0..1} or None.
    """
    if image_rgb is None or image_rgb.size == 0:
        return None

    h, w = image_rgb.shape[:2]
    road_region = image_rgb[int(h * 0.5) :, :, :]

    # Compare left vs right halves for vehicle-like features (dark blobs with symmetry)
    left_half = road_region[:, : w // 2, :]
    right_half = road_region[:, w // 2 :, :]

    left_dark = np.mean(left_half) < np.mean(right_half)
    right_dark = np.mean(right_half) < np.mean(left_half)

    # Heuristic: in countries with left-hand traffic, vehicles drive on left side of image
    # (when camera faces forward on the road). This is extremely weak.
    if abs(np.mean(left_half) - np.mean(right_half)) < 20:
        return None  # too symmetric

    if left_dark:
        return {"side": "left", "confidence": 0.3}
    else:
        return {"side": "right", "confidence": 0.3}


# ---------------------------------------------------------------------------
# Main aggregation: convert all detections to DetectedCue objects
# ---------------------------------------------------------------------------

def extract_all_specialist_cues(
    image_rgb: np.ndarray,
    scene_cues: Optional[Dict] = None,
) -> List[DetectedCue]:
    """
    Run all specialist detectors and return a unified list of DetectedCue objects.

    This is the main entry point called by the feature extractor.
    """
    cues: List[DetectedCue] = []

    # 1. Shadow / astronomy cues
    shadow = detect_shadow_features(image_rgb)
    if shadow["confidence"] > 0.2:
        if shadow["shadow_direction_deg"] is not None:
            # Map direction to hemisphere cue
            d = shadow["shadow_direction_deg"]
            if 315 <= d or d <= 45:
                cues.append(DetectedCue(
                    cue_type="latitude_band",
                    value="temperate",
                    confidence=shadow["confidence"] * 0.5,
                    source="shadow_heuristic",
                ))
            elif 135 <= d <= 225:
                cues.append(DetectedCue(
                    cue_type="latitude_band",
                    value="subtropical",
                    confidence=shadow["confidence"] * 0.5,
                    source="shadow_heuristic",
                ))

        if shadow["shadow_ratio_estimate"] is not None:
            ratio = shadow["shadow_ratio_estimate"]
            if ratio > 2.0:
                cues.append(DetectedCue(
                    cue_type="latitude_band",
                    value="temperate",
                    confidence=shadow["confidence"] * 0.4,
                    source="shadow_ratio_heuristic",
                ))

    # 2. Pole cues
    poles = detect_pole_proxies(image_rgb)
    for pole in poles:
        if pole["confidence"] > 0.4:
            if pole["type"] == "wooden":
                cues.append(DetectedCue(
                    cue_type="pole_type",
                    value="wooden_crossarm_us_style",
                    confidence=pole["confidence"] * 0.6,
                    source="pixel_heuristic",
                ))
            elif pole["type"] == "concrete":
                cues.append(DetectedCue(
                    cue_type="pole_type",
                    value="concrete_spindle_europe",
                    confidence=pole["confidence"] * 0.5,
                    source="pixel_heuristic",
                ))
            elif pole["type"] == "metal":
                cues.append(DetectedCue(
                    cue_type="pole_type",
                    value="steel_tubular_modern",
                    confidence=pole["confidence"] * 0.5,
                    source="pixel_heuristic",
                ))

    # 3. Road line cues
    lines = detect_road_line_proxies(image_rgb)
    for line in lines:
        if line["confidence"] > 0.4:
            if line["color"] == "yellow":
                cues.append(DetectedCue(
                    cue_type="road_marking",
                    value="yellow_center_white_edge_us",
                    confidence=line["confidence"] * 0.6,
                    source="pixel_heuristic",
                ))
            elif line["color"] == "white":
                cues.append(DetectedCue(
                    cue_type="road_marking",
                    value="white_center_white_edge_europe",
                    confidence=line["confidence"] * 0.5,
                    source="pixel_heuristic",
                ))
            elif line["color"] == "red":
                cues.append(DetectedCue(
                    cue_type="road_marking",
                    value="red_white_curb_asia",
                    confidence=line["confidence"] * 0.5,
                    source="pixel_heuristic",
                ))

    # 4. Driving side cue (very weak)
    drive_proxy = detect_driving_side_proxy(image_rgb)
    if drive_proxy:
        cues.append(DetectedCue(
            cue_type="drive_side",
            value=drive_proxy["side"],
            confidence=drive_proxy["confidence"],
            source="pixel_heuristic",
        ))

    logger.info("Specialist detectors: extracted %d cues from image", len(cues))
    return cues


def get_astronomy_constraints(
    image_rgb: np.ndarray,
    scene_cues: Optional[Dict] = None,
) -> AstronomyConstraints:
    """Get astronomy constraints from shadow analysis."""
    solver = AstronomySolver()
    return solver.quick_solve_from_pixels(image_rgb, scene_cues)
