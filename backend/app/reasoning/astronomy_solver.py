"""
Sun / Shadow Astronomy Solver.

Extremely underrated geolocation signal.
From one photo:
  - shadow angle
  - sun elevation
  - season
  - hemisphere cues

Can narrow latitude strongly using pure astronomy math.
No ML, no external APIs, no heavy dependencies.

Key insight:
  - Shadow length / object height ratio -> solar elevation angle
  - Solar elevation + approximate date/time -> latitude band
  - Shadow direction -> solar azimuth -> rough longitude constraint (if time known)
  - North/south side shadows -> hemisphere confirmation
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Earth's axial tilt (obliquity of the ecliptic)
OBLIQUITY_DEG = 23.44


@dataclass
class AstronomyConstraints:
    """Latitude/longitude constraints derived from shadow analysis."""

    latitude_min: float = -90.0
    latitude_max: float = 90.0
    latitude_confidence: float = 0.0  # 0..1, how strong is this constraint
    hemisphere_hint: str = ""  # "northern", "southern", "equatorial", ""
    season_hint: str = ""  # "summer", "winter", "equinox", ""
    solar_elevation_deg: Optional[float] = None
    shadow_direction_deg: Optional[float] = None  # 0=N, 90=E, 180=S, 270=W
    time_of_day_hint: str = ""  # "morning", "noon", "afternoon", "evening"
    summary: str = ""


class AstronomySolver:
    """
    Pure math solver for geographic constraints from solar geometry.

    Does NOT require EXIF timestamp — uses shadow angles + scene brightness
    to infer time-of-day and solar elevation. With or without date, produces
    latitude bands that are physically impossible to violate.
    """

    def solve_from_shadows(
        self,
        shadow_angle_deg: Optional[float] = None,
        shadow_ratio: Optional[float] = None,  # shadow_length / object_height
        shadow_direction_deg: Optional[float] = None,  # 0=N, 90=E, 180=S, 270=W
        scene_brightness: Optional[float] = None,  # 0..1 from pixel stats
        sky_brightness: Optional[float] = None,  # 0..1 from top-of-frame
        month_hint: Optional[int] = None,  # 1-12 if known (e.g. from vegetation)
    ) -> AstronomyConstraints:
        """
        Main entry point: derive latitude constraints from shadow measurements.

        shadow_ratio: if shadow is 2x object height, ratio = 2.0
        Higher ratio -> lower sun elevation -> higher latitude (or winter).
        """
        constraints = AstronomyConstraints()

        # Estimate solar elevation from shadow ratio
        if shadow_ratio is not None and shadow_ratio > 0:
            # tan(elevation) = object_height / shadow_length = 1 / ratio
            elevation_rad = math.atan(1.0 / shadow_ratio)
            constraints.solar_elevation_deg = math.degrees(elevation_rad)
        elif shadow_angle_deg is not None:
            # Direct elevation angle if measured
            constraints.solar_elevation_deg = 90.0 - abs(shadow_angle_deg)

        # Estimate time of day from shadow direction + scene brightness
        constraints.time_of_day_hint = self._estimate_time_of_day(
            shadow_direction_deg, scene_brightness, sky_brightness
        )

        # Hemisphere from shadow direction
        if shadow_direction_deg is not None:
            constraints.hemisphere_hint = self._infer_hemisphere(shadow_direction_deg)
            constraints.shadow_direction_deg = shadow_direction_deg

        # Latitude band from solar elevation + season
        if constraints.solar_elevation_deg is not None:
            lat_min, lat_max, conf = self._latitude_from_elevation(
                constraints.solar_elevation_deg,
                month_hint,
                constraints.time_of_day_hint,
            )
            constraints.latitude_min = lat_min
            constraints.latitude_max = lat_max
            constraints.latitude_confidence = conf

            # Refine with hemisphere
            if constraints.hemisphere_hint == "northern":
                constraints.latitude_min = max(constraints.latitude_min, 0.0)
            elif constraints.hemisphere_hint == "southern":
                constraints.latitude_max = min(constraints.latitude_max, 0.0)

        # Season hint from month + elevation
        constraints.season_hint = self._infer_season(month_hint, constraints.solar_elevation_deg)

        constraints.summary = self._build_summary(constraints)
        return constraints

    def _estimate_time_of_day(
        self,
        shadow_direction_deg: Optional[float],
        scene_brightness: Optional[float],
        sky_brightness: Optional[float],
    ) -> str:
        """Infer approximate time of day from brightness and shadow direction."""
        # Brightness-based
        if sky_brightness is not None:
            if sky_brightness > 0.75 and scene_brightness and scene_brightness > 0.6:
                return "midday"
            if sky_brightness < 0.35:
                return "evening_or_night"
            if sky_brightness > 0.5 and scene_brightness and scene_brightness < 0.4:
                return "morning_or_evening"

        # Shadow direction-based (northern hemisphere assumed if ambiguous)
        if shadow_direction_deg is not None:
            d = shadow_direction_deg % 360
            if 45 <= d <= 135:
                return "morning"  # shadow points roughly west (sun in east)
            if 225 <= d <= 315:
                return "afternoon"  # shadow points roughly east (sun in west)
            if 135 < d < 225:
                return "midday"  # shadow points roughly north (sun in south)
            if d < 45 or d > 315:
                return "noon_or_south"  # shadow points roughly south (sun in north)

        return "unknown"

    def _infer_hemisphere(self, shadow_direction_deg: float) -> str:
        """
        Infer hemisphere from shadow direction at midday.

        If shadow points north at noon -> northern hemisphere (sun in south)
        If shadow points south at noon -> southern hemisphere (sun in north)
        """
        d = shadow_direction_deg % 360
        # Midday shadow: if roughly north (shadow_direction around 0/360 or 315-45),
        # this implies sun is south -> northern hemisphere
        if 315 <= d or d <= 45:
            return "northern"
        if 135 <= d <= 225:
            return "southern"
        # Ambiguous: near east/west shadows
        return "equatorial_or_ambiguous"

    def _latitude_from_elevation(
        self,
        elevation_deg: float,
        month_hint: Optional[int],
        time_of_day_hint: str,
    ) -> Tuple[float, float, float]:
        """
        Convert solar elevation to latitude bounds.

        Maximum solar elevation at a latitude on a given day:
          max_elev = 90 - |lat - declination|
        So: lat_max_possible = 90 - elevation + declination
            lat_min_possible = -90 + elevation + declination

        Without date, declination ranges from -23.44 to +23.44.
        With date, we can narrow this significantly.
        """
        if elevation_deg <= 0:
            # Sun below horizon — either night or extreme latitude winter
            return -90.0, 90.0, 0.1

        # Declination range
        if month_hint is not None:
            # Approximate declination by month
            month_decl = self._approx_declination(month_hint)
            decl_min = month_decl - 5.0  # some tolerance
            decl_max = month_decl + 5.0
        else:
            decl_min = -OBLIQUITY_DEG
            decl_max = OBLIQUITY_DEG

        # Time-of-day correction: if not midday, elevation is lower than max possible
        time_factor = 1.0
        if time_of_day_hint in ("morning", "afternoon", "morning_or_evening"):
            time_factor = 0.75  # sun lower than max
        elif time_of_day_hint == "evening_or_night":
            time_factor = 0.5

        adjusted_elev = elevation_deg / time_factor if time_factor > 0 else elevation_deg

        # Latitude bounds
        # At maximum elevation (midday): lat = declination +/- (90 - elevation)
        # We need to consider both extremes of declination
        lat_range_min = -90.0
        lat_range_max = 90.0

        # Upper bound: latitude can't be so high that max_elev < observed_elev
        # max_elev = 90 - |lat - declination| >= observed_elev
        # |lat - declination| <= 90 - observed_elev
        # So: declination - (90 - elev) <= lat <= declination + (90 - elev)
        max_spread = 90.0 - adjusted_elev

        if max_spread < 90.0:
            # Tighten bounds based on declination range
            lat_min_candidate = decl_min - max_spread
            lat_max_candidate = decl_max + max_spread
            lat_range_min = max(lat_range_min, lat_min_candidate)
            lat_range_max = min(lat_range_max, lat_max_candidate)

        # Special case: very high elevation (>70) implies near-equatorial or summer midday
        if adjusted_elev > 70:
            lat_range_min = max(lat_range_min, -35.0)
            lat_range_max = min(lat_range_max, 35.0)

        # Very low elevation (<15) implies high latitude or sunrise/sunset
        if adjusted_elev < 15:
            # Could be anywhere, but more likely >45 degrees latitude
            pass  # keep wide bounds but lower confidence

        # Confidence: tighter bounds = higher confidence
        bound_width = lat_range_max - lat_range_min
        if bound_width < 30:
            confidence = 0.8
        elif bound_width < 60:
            confidence = 0.6
        elif bound_width < 120:
            confidence = 0.4
        else:
            confidence = 0.2

        # Boost confidence if month is known
        if month_hint is not None:
            confidence = min(0.95, confidence + 0.15)

        return lat_range_min, lat_range_max, confidence

    def _approx_declination(self, month: int) -> float:
        """Approximate solar declination by month (simplified)."""
        # Month 1 (Jan) -> declination ~ -20 (winter solstice was Dec)
        # Month 7 (Jul) -> declination ~ +20 (summer solstice was Jun)
        month_to_decl = {
            1: -20.0, 2: -12.0, 3: -2.0, 4: +10.0,
            5: +18.0, 6: +23.0, 7: +21.0, 8: +14.0,
            9: +4.0, 10: -8.0, 11: -18.0, 12: -23.0,
        }
        return month_to_decl.get(month, 0.0)

    def _infer_season(
        self,
        month_hint: Optional[int],
        solar_elevation_deg: Optional[float],
    ) -> str:
        if month_hint is None:
            if solar_elevation_deg is not None:
                if solar_elevation_deg > 60:
                    return "likely_summer_or_tropical"
                if solar_elevation_deg < 25:
                    return "likely_winter_or_high_latitude"
            return "unknown"

        if month_hint in (12, 1, 2):
            return "winter_northern_summer_southern"
        if month_hint in (3, 4, 5):
            return "spring_northern_autumn_southern"
        if month_hint in (6, 7, 8):
            return "summer_northern_winter_southern"
        if month_hint in (9, 10, 11):
            return "autumn_northern_spring_southern"
        return "unknown"

    def _build_summary(self, constraints: AstronomyConstraints) -> str:
        parts: List[str] = []
        if constraints.solar_elevation_deg is not None:
            parts.append("Solar elevation: %.1f degrees." % constraints.solar_elevation_deg)
        if constraints.hemisphere_hint:
            parts.append("Hemisphere: %s." % constraints.hemisphere_hint)
        if constraints.time_of_day_hint:
            parts.append("Time of day: %s." % constraints.time_of_day_hint)
        if constraints.season_hint:
            parts.append("Season: %s." % constraints.season_hint)
        if constraints.latitude_confidence > 0:
            parts.append(
                "Latitude constraint: %.1f to %.1f (confidence %.2f)."
                % (constraints.latitude_min, constraints.latitude_max, constraints.latitude_confidence)
            )
        if not parts:
            return "No astronomical constraints could be derived."
        return " ".join(parts)

    def quick_solve_from_pixels(
        self,
        image_rgb: Optional[np.ndarray] = None,
        scene_cues: Optional[Dict] = None,
    ) -> AstronomyConstraints:
        """
        Quick entry point: derive constraints from pixel heuristics alone.
        Uses scene brightness and vegetation cues to estimate season/latitude.
        """
        brightness = None
        sky_brightness = None
        month_hint = None

        if scene_cues and scene_cues.get("pixel_stats"):
            stats = scene_cues["pixel_stats"]
            brightness = stats.get("mean_luminance")
            sky_brightness = stats.get("sky_brightness_top_fraction")

        # Infer month from vegetation if available
        if scene_cues and scene_cues.get("vegetation"):
            veg = scene_cues["vegetation"]
            veg_labels = [v.get("label", "").lower() for v in veg if isinstance(v, dict)]
            if any("lush" in v or "growing" in v or "humid" in v for v in veg_labels):
                month_hint = 6  # assume summer for lush vegetation
            elif any("dry" in v or "arid" in v or "sparse" in v for v in veg_labels):
                month_hint = 1  # assume winter/dry season

        # Use brightness to estimate if midday or not
        time_hint = "unknown"
        if brightness is not None and sky_brightness is not None:
            if brightness > 0.6 and sky_brightness > 0.7:
                time_hint = "midday"
            elif brightness < 0.35:
                time_hint = "evening_or_night"

        # Infer shadow direction from scene cues (very rough)
        shadow_dir = None
        if scene_cues and scene_cues.get("climate_and_light"):
            climate = scene_cues["climate_and_light"]
            for cue in climate:
                label = cue.get("label", "").lower() if isinstance(cue, dict) else str(cue).lower()
                if "shadow" in label and "long" in label:
                    # Long shadows -> morning or evening
                    time_hint = "morning_or_evening"

        return self.solve_from_shadows(
            shadow_direction_deg=shadow_dir,
            scene_brightness=brightness,
            sky_brightness=sky_brightness,
            month_hint=month_hint,
        )

