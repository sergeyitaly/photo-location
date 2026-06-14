"""
Utility Pole Country Signal Database.

Elite GeoGuessr players consider utility poles one of the strongest country signals.
Different countries have distinct pole types:
  - Wooden crossarm (US style): USA, Canada, Philippines, Japan
  - Wooden triangle crossarm: UK, Ireland, Australia, NZ, South Africa
  - Concrete spindle: Germany, France, Netherlands, Central Europe
  - Concrete tubular: Mediterranean (Spain, Italy, Greece, Balkans)
  - Steel lattice: High-tension lines worldwide
  - Steel tubular: Modern Asian cities
  - Wooden pole no crossarm: Scandinavia, NZ, Chile, Argentina

This module maps detected pole types to country likelihoods for the reasoning engine.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Set

from app.data.country_geo_rules import UTILITY_POLE_TYPE_COUNTRIES

logger = logging.getLogger(__name__)


# Human-readable descriptions for each pole type
POLE_TYPE_DESCRIPTIONS: Dict[str, str] = {
    "wooden_crossarm_us_style": 
        "Wooden pole with crossarm, typical of North America and parts of Asia. "
        "Often has multiple insulators and guy wires.",
    "wooden_triangle_crossarm": 
        "Wooden pole with triangular crossarm arrangement, common in UK, Ireland, "
        "Australia, New Zealand, South Africa, and former British colonies.",
    "concrete_spindle_europe": 
        "Concrete spindle pole with rounded top, common in Germany, Netherlands, "
        "Belgium, Austria, Switzerland, and Central Europe.",
    "concrete_tubular_mediterranean": 
        "Concrete tubular pole, widespread in Mediterranean countries including "
        "Spain, Italy, Portugal, Greece, and the Balkans.",
    "steel_lattice_high_tension": 
        "Steel lattice tower for high-voltage transmission, found worldwide but "
        "distinctive designs vary by region.",
    "steel_tubular_modern": 
        "Modern steel tubular pole, common in East Asian cities, Gulf states, "
        "and newer developments globally.",
    "wooden_pole_no_crossarm_single_wire": 
        "Simple wooden pole without crossarm, often with single wire. Common in "
        "rural Scandinavia, New Zealand, Chile, and Argentina.",
}


def get_pole_type_countries(pole_type: str) -> Set[str]:
    """Return the set of countries associated with a pole type."""
    return UTILITY_POLE_TYPE_COUNTRIES.get(pole_type, set())


def get_all_pole_types() -> List[str]:
    """Return all known pole type identifiers."""
    return list(UTILITY_POLE_TYPE_COUNTRIES.keys())


def describe_pole_type(pole_type: str) -> str:
    """Return human-readable description of a pole type."""
    return POLE_TYPE_DESCRIPTIONS.get(pole_type, "Unknown pole type.")


def get_pole_types_for_country(country: str) -> List[str]:
    """Return all pole types associated with a given country."""
    result: List[str] = []
    for pole_type, countries in UTILITY_POLE_TYPE_COUNTRIES.items():
        if country in countries:
            result.append(pole_type)
    return result


def rank_countries_by_pole_evidence(
    detected_pole_types: List[str],
    confidence: float = 0.7,
) -> Dict[str, float]:
    """
    Score countries by how many detected pole types support them.
    Returns a dict of {country: score}.
    """
    country_scores: Dict[str, float] = {}
    for pole_type in detected_pole_types:
        countries = get_pole_type_countries(pole_type)
        for country in countries:
            country_scores[country] = country_scores.get(country, 0.0) + confidence
    return country_scores


# CLIP prompt bank for utility pole detection
UTILITY_POLE_CLIP_PROMPTS: List[str] = [
    # Wooden crossarm (US style)
    "wooden utility pole with crossarm and insulators",
    "wooden electrical pole multiple wires crossarm",
    "utility pole with horizontal crossarm wooden",
    # Wooden triangle (British/Commonwealth)
    "wooden utility pole triangular crossarm arrangement",
    "British style wooden electricity pole with triangular top",
    # Concrete spindle (Central Europe)
    "concrete spindle utility pole rounded top",
    "German style concrete electricity pole",
    "concrete utility pole with mushroom top",
    # Concrete tubular (Mediterranean)
    "concrete tubular utility pole smooth surface",
    "Mediterranean concrete electricity pole",
    "Spanish style concrete utility pole",
    # Steel lattice
    "steel lattice transmission tower high voltage",
    "metal grid electricity pylon",
    "lattice tower power line",
    # Steel tubular
    "steel tubular utility pole modern",
    "smooth metal electricity pole urban",
    "steel pipe utility pole with lights",
    # Simple wooden
    "simple wooden pole single wire no crossarm",
    "rural wooden pole with one wire",
    "Scandinavian style wooden utility pole",
    # Generic / other
    "utility pole with transformer box",
    "electrical pole with street light attached",
    "concrete utility pole with climbing steps",
    "guy wire anchor concrete block utility pole",
]
