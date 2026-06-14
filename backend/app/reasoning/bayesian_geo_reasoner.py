"""
Bayesian Geographic Reasoning Layer.

Instead of: palm trees + Cyrillic + mountains = maybe Balkans
Need:       Palm trees rare above X latitude
            Cyrillic common in BG/RS/UA/RU
            Dry mountains + Mediterranean roofs = Montenegro likely

Uses a weighted evidence graph:
  - Each cue provides a likelihood per country
  - Country priors (population-weighted or uniform)
  - Contradiction penalties when cues conflict
  - Posterior = prior * product(likelihoods) * penalty

This is lightweight (no neural nets) and can outperform raw AI on ambiguous scenes.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from app.data.country_geo_rules import (
    ALL_COUNTRIES,
    ARCTIC_ALPINE_COUNTRIES,
    DESERT_COUNTRIES,
    EU_MEMBER_COUNTRIES,
    LEFT_HAND_DRIVE_COUNTRIES,
    MEDITERRANEAN_CLIMATE_COUNTRIES,
    RIGHT_HAND_DRIVE_COUNTRIES,
    SCRIPT_TO_COUNTRIES,
    SUBTROPICAL_COUNTRIES,
    TEMPERATE_COUNTRIES,
    TROPICAL_COUNTRIES,
    UTILITY_POLE_TYPE_COUNTRIES,
    ROAD_MARKING_STYLE_COUNTRIES,
    get_country_set,
)
from app.reasoning.country_elimination import DetectedCue

logger = logging.getLogger(__name__)


@dataclass
class GeoReasoningResult:
    """Output of the Bayesian geo reasoner."""

    country_posteriors: Dict[str, float] = field(default_factory=dict)
    top_country: str = ""
    top_confidence: float = 0.0
    evidence_breakdown: List[Dict] = field(default_factory=list)
    contradiction_penalties_applied: List[str] = field(default_factory=list)
    summary: str = ""


class BayesianGeoReasoner:
    """
    Probabilistic reasoner over country hypotheses.

    Each DetectedCue maps to a likelihood function over countries.
    The posterior P(country | cues) is proportional to P(country) * product of P(cue | country).

    Contradictions (e.g. palm trees + arctic climate) apply a soft penalty
    rather than zeroing out, to gracefully handle noisy cues.
    """

    # Population-approximate priors (higher weight for more commonly photographed countries)
    # These are rough relative weights, not true probabilities.
    COUNTRY_PRIOR_WEIGHTS: Dict[str, float] = {
        "United States": 1.0, "China": 0.95, "India": 0.9, "Japan": 0.85,
        "Germany": 0.8, "United Kingdom": 0.8, "France": 0.8, "Italy": 0.75,
        "Brazil": 0.75, "Russia": 0.75, "Canada": 0.7, "Australia": 0.7,
        "Spain": 0.7, "Mexico": 0.7, "South Korea": 0.65, "Turkey": 0.65,
        "Netherlands": 0.65, "Saudi Arabia": 0.6, "Switzerland": 0.6,
        "Sweden": 0.6, "Poland": 0.6, "Belgium": 0.55, "Thailand": 0.55,
        "Austria": 0.55, "Norway": 0.55, "United Arab Emirates": 0.55,
        "Israel": 0.55, "Singapore": 0.55, "Malaysia": 0.55, "Ireland": 0.5,
        "South Africa": 0.5, "Philippines": 0.5, "Denmark": 0.5, "Hong Kong": 0.5,
        "Finland": 0.5, "Colombia": 0.5, "New Zealand": 0.5, "Greece": 0.5,
        "Portugal": 0.5, "Qatar": 0.45, "Czechia": 0.45, "Hungary": 0.45,
        "Romania": 0.45, "Chile": 0.45, "Peru": 0.45, "Iraq": 0.4,
        "Kazakhstan": 0.4, "Kuwait": 0.4, "Ukraine": 0.4, "Morocco": 0.4,
        "Slovakia": 0.4, "Ecuador": 0.4, "Puerto Rico": 0.4, "Kenya": 0.4,
        "Ethiopia": 0.4, "Algeria": 0.4, "Venezuela": 0.4, "Uzbekistan": 0.35,
        "Costa Rica": 0.35, "Luxembourg": 0.35, "Croatia": 0.35, "Slovenia": 0.35,
        "Lithuania": 0.35, "Serbia": 0.35, "Bulgaria": 0.35, "Azerbaijan": 0.35,
        "Panama": 0.35, "Latvia": 0.35, "Cyprus": 0.35, "Estonia": 0.35,
        "Ghana": 0.35, "Uruguay": 0.35, "Belarus": 0.35, "Sri Lanka": 0.35,
        "Bahrain": 0.35, "Dominican Republic": 0.35, "Guatemala": 0.3,
        "Tunisia": 0.3, "Nepal": 0.3, "Cambodia": 0.3, "Jordan": 0.3,
        "Myanmar": 0.3, "Mongolia": 0.3, "North Macedonia": 0.3, "Jamaica": 0.3,
        "Tanzania": 0.3, "Georgia": 0.3, "Brunei": 0.3, "Armenia": 0.3,
        "Albania": 0.3, "Moldova": 0.3, "Bosnia and Herzegovina": 0.3,
        "Kyrgyzstan": 0.25, "Botswana": 0.25, "Senegal": 0.25, "Zimbabwe": 0.25,
        "Sudan": 0.25, "Uganda": 0.25, "Zambia": 0.25, "Honduras": 0.25,
        "Paraguay": 0.25, "Laos": 0.25, "El Salvador": 0.25, "Trinidad and Tobago": 0.25,
        "Nicaragua": 0.25, "Madagascar": 0.25, "Mali": 0.25, "Papua New Guinea": 0.25,
        "Mauritius": 0.25, "Namibia": 0.25, "Bahamas": 0.25, "New Caledonia": 0.25,
        "Rwanda": 0.2, "Montenegro": 0.2, "Mauritania": 0.2, "Togo": 0.2,
        "Fiji": 0.2, "Eswatini": 0.2, "Suriname": 0.2, "Bhutan": 0.2,
        "Lesotho": 0.2, "Guyana": 0.2, "Maldives": 0.2, "Barbados": 0.2,
        "Cape Verde": 0.2, "Belize": 0.2, "Malta": 0.2, "Iceland": 0.2,
        "Vanuatu": 0.2, "Gabon": 0.2, "Liberia": 0.2, "Guinea": 0.2,
        "Andorra": 0.2, "Liechtenstein": 0.2, "Monaco": 0.2, "San Marino": 0.2,
        "Gibraltar": 0.2, "Vatican City": 0.2, "Faroe Islands": 0.15,
        "Greenland": 0.15, "Svalbard and Jan Mayen": 0.15, "Antarctica": 0.1,
    }

    def __init__(self) -> None:
        self.all_countries = ALL_COUNTRIES.copy()
        self.priors = self._build_priors()
        logger.info("BayesianGeoReasoner: %d countries, prior entropy %.3f", len(self.all_countries), self._entropy(self.priors))

    def _build_priors(self) -> Dict[str, float]:
        priors: Dict[str, float] = {}
        baseline = 0.2
        total = 0.0
        for c in self.all_countries:
            w = self.COUNTRY_PRIOR_WEIGHTS.get(c, baseline)
            priors[c] = w
            total += w
        if total > 0:
            priors = {c: w / total for c, w in priors.items()}
        return priors

    @staticmethod
    def _entropy(probs: Dict[str, float]) -> float:
        e = 0.0
        for p in probs.values():
            if p > 0:
                e -= p * math.log(p)
        return e

    def reason(self, cues: List[DetectedCue]) -> GeoReasoningResult:
        if not cues:
            top = max(self.priors, key=lambda c: self.priors[c])
            return GeoReasoningResult(
                country_posteriors=dict(self.priors),
                top_country=top,
                top_confidence=self.priors[top],
                evidence_breakdown=[],
                contradiction_penalties_applied=[],
                summary="No evidence cues — returning population-approximate priors.",
            )

        posteriors: Dict[str, float] = dict(self.priors)
        evidence_breakdown: List[Dict] = []
        contradiction_penalties: List[str] = []

        for cue in cues:
            likelihood = self._likelihood_for_cue(cue)
            if not likelihood:
                continue

            for country in posteriors:
                if country in likelihood:
                    posteriors[country] *= likelihood[country]
                else:
                    posteriors[country] *= 0.3

            top3 = sorted(likelihood, key=lambda c: likelihood[c], reverse=True)[:3]
            evidence_breakdown.append({
                "cue_type": cue.cue_type,
                "value": cue.value,
                "confidence": cue.confidence,
                "source": cue.source,
                "countries_supported": len(likelihood),
                "top_3_countries": top3,
            })

        # Detect contradictions
        has_tropical = any(c.cue_type == "latitude_band" and c.value.lower() == "tropical" for c in cues)
        has_arctic = any(c.cue_type == "latitude_band" and c.value.lower() == "arctic" for c in cues)
        if has_tropical and has_arctic:
            contradiction_penalties.append("Contradiction: tropical + arctic latitude cues")
            for c in posteriors:
                if c in ARCTIC_ALPINE_COUNTRIES:
                    posteriors[c] *= 0.7

        # Normalize
        total = sum(posteriors.values())
        if total > 0:
            posteriors = {c: p / total for c, p in posteriors.items()}

        top_country = max(posteriors, key=lambda c: posteriors[c])
        top_conf = posteriors[top_country]

        summary = self._build_summary(cues, posteriors, top_country, top_conf, contradiction_penalties)

        return GeoReasoningResult(
            country_posteriors=posteriors,
            top_country=top_country,
            top_confidence=top_conf,
            evidence_breakdown=evidence_breakdown,
            contradiction_penalties_applied=contradiction_penalties,
            summary=summary,
        )

    def _likelihood_for_cue(self, cue: DetectedCue) -> Dict[str, float]:
        ct = cue.cue_type.lower()
        val = cue.value.lower()
        conf = max(0.01, min(1.0, cue.confidence))

        def _make_likelihood(supporting: Set[str], strength: float = 0.9) -> Dict[str, float]:
            baseline = 0.1
            like: Dict[str, float] = {}
            for c in self.all_countries:
                if c in supporting:
                    like[c] = baseline + strength * conf
                else:
                    like[c] = baseline * (1.0 - conf * 0.5)
            return like

        if ct == "script":
            countries = SCRIPT_TO_COUNTRIES.get(val, set())
            if countries:
                return _make_likelihood(countries, 0.85)
            return {}

        if ct == "drive_side":
            if val == "left":
                return _make_likelihood(LEFT_HAND_DRIVE_COUNTRIES, 0.8)
            if val == "right":
                return _make_likelihood(RIGHT_HAND_DRIVE_COUNTRIES, 0.8)
            return {}

        if ct == "climate":
            countries = get_country_set("climate", val)
            if countries:
                return _make_likelihood(countries, 0.75)
            return {}

        if ct == "latitude_band":
            if val == "tropical":
                return _make_likelihood(TROPICAL_COUNTRIES | SUBTROPICAL_COUNTRIES, 0.7)
            if val == "subtropical":
                return _make_likelihood(SUBTROPICAL_COUNTRIES | TROPICAL_COUNTRIES, 0.6)
            if val == "temperate":
                return _make_likelihood(TEMPERATE_COUNTRIES | SUBTROPICAL_COUNTRIES, 0.6)
            if val == "arctic":
                return _make_likelihood(ARCTIC_ALPINE_COUNTRIES | TEMPERATE_COUNTRIES, 0.7)
            return {}

        if ct == "pole_type":
            countries = get_country_set("pole_type", cue.value)
            if countries:
                return _make_likelihood(countries, 0.8)
            return {}

        if ct == "road_marking":
            countries = get_country_set("road_marking", cue.value)
            if countries:
                return _make_likelihood(countries, 0.75)
            return {}

        if ct == "vegetation":
            if val in ("palm", "tropical"):
                return _make_likelihood(TROPICAL_COUNTRIES | SUBTROPICAL_COUNTRIES, 0.7)
            if val in ("pine", "spruce", "fir"):
                return _make_likelihood(TEMPERATE_COUNTRIES | ARCTIC_ALPINE_COUNTRIES, 0.65)
            if val in ("cactus", "succulent", "dry_scrub"):
                return _make_likelihood(DESERT_COUNTRIES | MEDITERRANEAN_CLIMATE_COUNTRIES, 0.7)
            return {}

        if ct == "language":
            lang_to_script = {
                "english": "latin", "spanish": "latin", "french": "latin",
                "german": "latin", "portuguese": "latin", "italian": "latin",
                "russian": "cyrillic", "ukrainian": "cyrillic", "bulgarian": "cyrillic",
                "serbian": "cyrillic", "arabic": "arabic", "hebrew": "hebrew",
                "chinese": "chinese", "japanese": "japanese", "korean": "korean",
                "hindi": "devanagari", "thai": "thai", "greek": "greek",
                "armenian": "armenian", "georgian": "georgian",
            }
            script = lang_to_script.get(val)
            if script:
                countries = SCRIPT_TO_COUNTRIES.get(script, set())
                if countries:
                    return _make_likelihood(countries, 0.8)
            return {}

        if ct == "eu_member":
            if val in ("true", "yes", "1"):
                return _make_likelihood(EU_MEMBER_COUNTRIES, 0.75)
            return _make_likelihood(ALL_COUNTRIES - EU_MEMBER_COUNTRIES, 0.6)

        return {}

    def _build_summary(
        self,
        cues: List[DetectedCue],
        posteriors: Dict[str, float],
        top_country: str,
        top_conf: float,
        penalties: List[str],
    ) -> str:
        parts: List[str] = []
        parts.append("Bayesian reasoning over %d cues." % len(cues))
        if penalties:
            parts.append("Contradictions: %s" % "; ".join(penalties))
        parts.append("Top country: %s (confidence %.3f)." % (top_country, top_conf))
        # Show top 5
        top5 = sorted(posteriors.items(), key=lambda x: x[1], reverse=True)[:5]
        top5_str = ", ".join(["%s %.2f" % (c, p) for c, p in top5])
        parts.append("Top 5: %s" % top5_str)
        return " ".join(parts)

