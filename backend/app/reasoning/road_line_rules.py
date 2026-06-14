"""
Road Line / Marking Style Country Signal Database.

Elite GeoGuessr players use road markings as strong country signals:
  - Yellow center + white edge: USA, Canada, most of Americas
  - White center + white edge: UK, Ireland, most of Europe
  - Yellow center: Russia, CIS countries
  - Dashed white EU motorway: Germany, France, Netherlands, etc.
  - No center line on rural roads: Scandinavia, NZ, rural areas
  - Red/white curb: Japan, South Korea, much of Asia
  - Yellow curb no-parking: France, Spain, Mediterranean EU

This module maps detected road marking styles to country likelihoods.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Set

from app.data.country_geo_rules import ROAD_MARKING_STYLE_COUNTRIES

logger = logging.getLogger(__name__)


ROAD_MARKING_DESCRIPTIONS: Dict[str, str] = {
    "yellow_center_white_edge_us": 
        "Yellow center line with white edge lines. Characteristic of USA, Canada, "
        "Mexico, and most of Central and South America.",
    "white_center_white_edge_europe": 
        "White center line with white edge lines. Standard in UK, Ireland, France, "
        "Germany, and most of continental Europe.",
    "yellow_double_solid_no_passing": 
        "Double solid yellow lines indicating no passing. Common in USA, Canada, "
        "Australia, New Zealand, Japan, and South Korea.",
    "dashed_white_eu_motorway": 
        "Dashed white lines on multi-lane motorway. Standard in EU countries "
        "including Germany, France, Netherlands, Italy, Spain.",
    "yellow_center_russia_cis": 
        "Yellow center line (often with white edge). Distinctive of Russia, Ukraine, "
        "Belarus, Kazakhstan, and other CIS/post-Soviet states.",
    "white_dashed_center_uk_commonwealth": 
        "White dashed center line with white edge. Common in UK, Ireland, Australia, "
        "New Zealand, India, and Commonwealth countries.",
    "no_center_line_rural": 
        "No painted center line on rural road. Typical of Scandinavia (Norway, Sweden, "
        "Finland), New Zealand rural roads, Chile, Argentina.",
    "red_white_curb_asia": 
        "Red and white painted curb markings. Common in Japan, South Korea, Taiwan, "
        "Thailand, Singapore, Hong Kong, and urban Asia.",
    "yellow_curb_no_parking_eu": 
        "Yellow painted curb indicating no parking. Standard in France, Spain, Italy, "
        "Portugal, Belgium, Netherlands, and much of EU.",
}


def get_road_marking_countries(road_marking: str) -> Set[str]:
    """Return the set of countries associated with a road marking style."""
    return ROAD_MARKING_STYLE_COUNTRIES.get(road_marking, set())


def get_all_road_marking_styles() -> List[str]:
    """Return all known road marking style identifiers."""
    return list(ROAD_MARKING_STYLE_COUNTRIES.keys())


def describe_road_marking(road_marking: str) -> str:
    """Return human-readable description of a road marking style."""
    return ROAD_MARKING_DESCRIPTIONS.get(road_marking, "Unknown road marking style.")


def get_road_markings_for_country(country: str) -> List[str]:
    """Return all road marking styles associated with a given country."""
    result: List[str] = []
    for marking, countries in ROAD_MARKING_STYLE_COUNTRIES.items():
        if country in countries:
            result.append(marking)
    return result


def rank_countries_by_road_evidence(
    detected_markings: List[str],
    confidence: float = 0.7,
) -> Dict[str, float]:
    """
    Score countries by how many detected road marking styles support them.
    Returns a dict of {country: score}.
    """
    country_scores: Dict[str, float] = {}
    for marking in detected_markings:
        countries = get_road_marking_countries(marking)
        for country in countries:
            country_scores[country] = country_scores.get(country, 0.0) + confidence
    return country_scores


# CLIP prompt bank for road line detection
ROAD_MARKING_CLIP_PROMPTS: List[str] = [
    # Yellow center / white edge (Americas)
    "road with yellow center line and white edge lines",
    "highway with double yellow center line",
    "American style road marking yellow center",
    # White center / white edge (Europe)
    "road with white center line and white edge lines",
    "European style road marking white lines",
    "motorway with white dashed center line",
    # Double solid yellow
    "double solid yellow lines no passing zone",
    "road with double yellow center lines",
    # EU motorway dashed white
    "European motorway dashed white lane markings",
    "German autobahn lane markings white dashed",
    "multi-lane highway with white dashed lines",
    # Yellow center (Russia/CIS)
    "road with yellow center line post-soviet style",
    "Russian highway yellow center marking",
    # White dashed (UK/Commonwealth)
    "British style road white dashed center line",
    "road with white painted center line UK",
    # No center line rural
    "rural road with no painted center line",
    "unmarked country road no lines",
    "Scandinavian rural road without center marking",
    # Red/white curb
    "red and white painted curb",
    "Japanese style curb marking red white",
    "urban street with red white curb lines",
    # Yellow curb
    "yellow painted curb no parking",
    "French style yellow curb marking",
    "European no-parking yellow curb",
    # Other markings
    "zebra crossing white stripes",
    "pedestrian crossing marked with yellow stripes",
    "roundabout road markings directional arrows",
    "bus stop yellow markings on road",
    "bicycle lane green paint markings",
]
