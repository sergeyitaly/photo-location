"""
Country Elimination Engine.

Instead of predicting one place, first eliminate impossible regions.
Example: right-hand driving + EU plate + dry grass + Cyrillic → remove 180 countries.

This is a deterministic rule engine that intersects country sets from detected cues.
Each cue produces a "keep set"; the intersection narrows candidates.
Contradictions (e.g. Cyrillic + Japanese script) produce penalties rather than hard elimination
when the intersection would be empty, to avoid over-confidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from app.data.country_geo_rules import (
    ALL_COUNTRIES,
    ARCTIC_ALPINE_COUNTRIES,
    DESERT_COUNTRIES,
    EU_MEMBER_COUNTRIES,
    EUROPEAN_PLATE_COUNTRIES,
    LEFT_HAND_DRIVE_COUNTRIES,
    MEDITERRANEAN_CLIMATE_COUNTRIES,
    RIGHT_HAND_DRIVE_COUNTRIES,
    SCRIPT_TO_COUNTRIES,
    SUBTROPICAL_COUNTRIES,
    TEMPERATE_COUNTRIES,
    TROPICAL_COUNTRIES,
    get_country_set,
)

logger = logging.getLogger(__name__)


@dataclass
class DetectedCue:
    """A single piece of evidence extracted from the image or metadata."""

    cue_type: str  # e.g. "script", "drive_side", "climate", "pole_type", "road_marking"
    value: str  # e.g. "cyrillic", "right", "desert", "wooden_crossarm_us_style"
    confidence: float = 1.0  # 0..1, how sure we are about this cue
    source: str = "rule"  # "pixel_heuristic", "clip_softmax", "ocr", "manual"


@dataclass
class CountryEliminationResult:
    """Output of the elimination engine."""

    remaining_countries: Set[str] = field(default_factory=set)
    eliminated_countries: Set[str] = field(default_factory=set)
    country_scores: Dict[str, float] = field(default_factory=dict)  # posterior-like score per country
    applied_cues: List[DetectedCue] = field(default_factory=list)
    contradiction_penalties: List[str] = field(default_factory=list)
    summary: str = ""
    num_remaining: int = 0
    num_eliminated: int = 0


class CountryEliminationEngine:
    """
    Rule-based country elimination using intersecting sets.

    Strategy:
      1. Start with ALL_COUNTRIES.
      2. For each DetectedCue, get the keep-set (countries consistent with the cue).
      3. Intersect keep-sets. If intersection becomes empty, apply contradiction
         penalties instead of hard elimination (keep the most likely subset).
      4. Score remaining countries by how many cues support them.
    """

    def __init__(self) -> None:
        self.all_countries = ALL_COUNTRIES.copy()
        logger.info("CountryEliminationEngine: %d countries in universe", len(self.all_countries))

    def eliminate(self, cues: List[DetectedCue]) -> CountryEliminationResult:
        if not cues:
            return CountryEliminationResult(
                remaining_countries=self.all_countries.copy(),
                eliminated_countries=set(),
                country_scores={c: 1.0 for c in self.all_countries},
                applied_cues=[],
                contradiction_penalties=[],
                summary="No cues provided — all countries remain possible.",
                num_remaining=len(self.all_countries),
                num_eliminated=0,
            )

        # Build per-cue keep sets
        cue_sets: List[Tuple[DetectedCue, Set[str]]] = []
        for cue in cues:
            keep = self._resolve_keep_set(cue)
            if keep:
                cue_sets.append((cue, keep))

        if not cue_sets:
            return CountryEliminationResult(
                remaining_countries=self.all_countries.copy(),
                eliminated_countries=set(),
                country_scores={c: 1.0 for c in self.all_countries},
                applied_cues=cues,
                contradiction_penalties=["No cue mapped to country sets"],
                summary="Cues provided but none mapped to known country sets.",
                num_remaining=len(self.all_countries),
                num_eliminated=0,
            )

        # Try strict intersection first
        intersection: Set[str] = self.all_countries.copy()
        for _cue, keep in cue_sets:
            intersection &= keep

        contradiction_penalties: List[str] = []

        if not intersection:
            # Contradiction: cues conflict. Use soft voting instead of hard elimination.
            # Score each country by weighted sum of supporting cues.
            logger.warning("CountryElimination: strict intersection empty — switching to soft voting")
            contradiction_penalties.append(
                f"Strict intersection empty across {len(cue_sets)} cues; using soft voting."
            )
            country_scores = self._soft_vote(cue_sets)
            # Keep top-scoring countries above a dynamic threshold
            if country_scores:
                max_score = max(country_scores.values())
                threshold = max_score * 0.35  # keep countries within 35% of best
                remaining = {c for c, s in country_scores.items() if s >= threshold}
            else:
                remaining = self.all_countries.copy()
                country_scores = {c: 1.0 for c in remaining}
        else:
            # Strict intersection succeeded — score by cue support within intersection
            country_scores = self._score_within_set(intersection, cue_sets)
            remaining = intersection

        eliminated = self.all_countries - remaining

        # Build human-readable summary
        summary = self._build_summary(cue_sets, remaining, eliminated, contradiction_penalties)

        return CountryEliminationResult(
            remaining_countries=remaining,
            eliminated_countries=eliminated,
            country_scores=country_scores,
            applied_cues=cues,
            contradiction_penalties=contradiction_penalties,
            summary=summary,
            num_remaining=len(remaining),
            num_eliminated=len(eliminated),
        )

    def _resolve_keep_set(self, cue: DetectedCue) -> Set[str]:
        """Map a DetectedCue to the set of countries consistent with it."""
        ct = cue.cue_type.lower()
        val = cue.value.lower()

        if ct == "script":
            return SCRIPT_TO_COUNTRIES.get(val, set())

        if ct == "drive_side":
            if val == "left":
                return LEFT_HAND_DRIVE_COUNTRIES.copy()
            if val == "right":
                return RIGHT_HAND_DRIVE_COUNTRIES.copy()
            return set()

        if ct == "climate":
            return get_country_set("climate", val)

        if ct == "eu_member":
            if val in ("true", "yes", "1"):
                return EU_MEMBER_COUNTRIES.copy()
            return ALL_COUNTRIES - EU_MEMBER_COUNTRIES

        if ct == "plate_region":
            if val == "eu":
                return EUROPEAN_PLATE_COUNTRIES.copy()
            if val == "north_america":
                from app.data.country_geo_rules import NORTH_AMERICAN_PLATE_COUNTRIES
                return NORTH_AMERICAN_PLATE_COUNTRIES.copy()
            return set()

        if ct == "pole_type":
            return get_country_set("pole_type", cue.value)

        if ct == "road_marking":
            return get_country_set("road_marking", cue.value)

        if ct == "latitude_band":
            if val == "tropical":
                return TROPICAL_COUNTRIES.copy()
            if val == "subtropical":
                return SUBTROPICAL_COUNTRIES.copy()
            if val == "temperate":
                return TEMPERATE_COUNTRIES.copy()
            if val == "arctic":
                return ARCTIC_ALPINE_COUNTRIES.copy()
            return set()

        if ct == "vegetation":
            # Palm trees → tropical/subtropical; pine/needle → temperate/boreal
            if val in ("palm", "tropical", "banana", "mangrove"):
                return TROPICAL_COUNTRIES | SUBTROPICAL_COUNTRIES
            if val in ("pine", "spruce", "fir", "needle", "taiga"):
                return TEMPERATE_COUNTRIES | ARCTIC_ALPINE_COUNTRIES
            if val in ("deciduous", "oak", "beech", "maple"):
                return TEMPERATE_COUNTRIES | SUBTROPICAL_COUNTRIES
            if val in ("cactus", "succulent", "dry_scrub"):
                return DESERT_COUNTRIES | MEDITERRANEAN_CLIMATE_COUNTRIES
            return set()

        if ct == "language":
            # Map language names to scripts
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
                return SCRIPT_TO_COUNTRIES.get(script, set())
            return set()

        return set()

    def _soft_vote(self, cue_sets: List[Tuple[DetectedCue, Set[str]]]) -> Dict[str, float]:
        """When strict intersection fails, score each country by weighted cue support."""
        scores: Dict[str, float] = {c: 0.0 for c in self.all_countries}
        for cue, keep in cue_sets:
            weight = cue.confidence
            for country in keep:
                if country in scores:
                    scores[country] += weight
        # Normalize to 0..1
        max_s = max(scores.values()) if scores else 1.0
        if max_s > 0:
            scores = {c: min(1.0, s / max_s) for c, s in scores.items()}
        return scores

    def _score_within_set(
        self,
        remaining: Set[str],
        cue_sets: List[Tuple[DetectedCue, Set[str]]],
    ) -> Dict[str, float]:
        """Score countries within a non-empty intersection."""
        scores: Dict[str, float] = {c: 0.0 for c in remaining}
        for cue, keep in cue_sets:
            weight = cue.confidence
            for country in remaining:
                if country in keep:
                    scores[country] += weight
        max_s = max(scores.values()) if scores else 1.0
        if max_s > 0:
            scores = {c: min(1.0, s / max_s) for c, s in scores.items()}
        return scores

    def _build_summary(
        self,
        cue_sets: List[Tuple[DetectedCue, Set[str]]],
        remaining: Set[str],
        eliminated: Set[str],
        penalties: List[str],
    ) -> str:
        parts: List[str] = []
        parts.append(f"Applied {len(cue_sets)} elimination cues.")
        if penalties:
            parts.append(f"Contradictions: {'; '.join(penalties)}")
        parts.append(f"Remaining: {len(remaining)} countries; eliminated: {len(eliminated)}.")
        if remaining and len(remaining) <= 12:
            parts.append(f"Candidates: {', '.join(sorted(remaining))}.")
        elif remaining:
            top = sorted(remaining)[:10]
            parts.append(f"Top candidates include: {', '.join(top)}…")
        return " ".join(parts)


# Convenience: build cues from scene analysis outputs

def cues_from_detected_text(texts: List[str], confidence: float = 0.7) -> List[DetectedCue]:
    """Generate elimination cues from OCR-detected text snippets."""
    cues: List[DetectedCue] = []
    if not texts:
        return cues

    text_lower = " ".join(t.lower() for t in texts)

    # Script detection heuristics
    script_signals = {
        "cyrillic": ["а", "б", "в", "г", "д", "е", "ё", "ж", "з", "и", "й", "к", "л", "м", "н", "о", "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я"],
        "arabic": ["ا", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر", "ز", "س", "ش", "ص", "ض", "ط", "ظ", "ع", "غ", "ف", "ق", "ك", "ل", "م", "ن", "ه", "و", "ي"],
        "greek": ["α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ", "λ", "μ", "ν", "ξ", "ο", "π", "ρ", "σ", "τ", "υ", "φ", "χ", "ψ", "ω"],
        "hebrew": ["א", "ב", "ג", "ד", "ה", "ו", "ז", "ח", "ט", "י", "כ", "ל", "מ", "נ", "ס", "ע", "פ", "צ", "ק", "ר", "ש", "ת"],
        "thai": ["ก", "ข", "ค", "ฅ", "ฆ", "ง", "จ", "ฉ", "ช", "ซ", "ฌ", "ญ", "ฎ", "ฏ", "ฐ", "ฑ", "ฒ", "ณ", "ด", "ต", "ถ", "ท", "ธ", "น", "บ", "ป", "ผ", "ฝ", "พ", "ฟ", "ภ", "ม", "ย", "ร", "ล", "ว", "ศ", "ษ", "ส", "ห", "ฬ", "อ", "ฮ"],
        "japanese": ["あ", "い", "う", "え", "お", "か", "き", "く", "け", "こ", "さ", "し", "す", "せ", "そ", "た", "ち", "つ", "て", "と", "な", "に", "ぬ", "ね", "の", "は", "ひ", "ふ", "へ", "ほ", "ま", "み", "む", "め", "も", "や", "ゆ", "よ", "ら", "り", "る", "れ", "ろ", "わ", "を", "ん"],
        "korean": ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ", "ㅏ", "ㅑ", "ㅓ", "ㅕ", "ㅗ", "ㅛ", "ㅜ", "ㅠ", "ㅡ", "ㅣ"],
        "devanagari": ["अ", "आ", "इ", "ई", "उ", "ऊ", "ए", "ऐ", "ओ", "औ", "क", "ख", "ग", "घ", "ङ", "च", "छ", "ज", "झ", "ञ", "ट", "ठ", "ड", "ढ", "ण", "त", "थ", "द", "ध", "न", "प", "फ", "ब", "भ", "म", "य", "र", "ल", "व", "श", "ष", "स", "ह"],
    }

    detected_scripts: Set[str] = set()
    for script, chars in script_signals.items():
        if any(ch in text_lower for ch in chars):
            detected_scripts.add(script)

    for script in detected_scripts:
        cues.append(DetectedCue(cue_type="script", value=script, confidence=confidence, source="ocr"))

    # Language-specific keywords
    lang_keywords = {
        "english": ["street", "road", "ave", "blvd", "highway", "county", "state"],
        "spanish": ["calle", "avenida", "carrera", "plaza", "pueblo", "provincia"],
        "french": ["rue", "avenue", "boulevard", "place", "chemin", "route"],
        "german": ["straße", "strasse", "weg", "platz", "allee", "gasse"],
        "portuguese": ["rua", "avenida", "praça", "estrada", "travessa"],
        "italian": ["via", "viale", "piazza", "corso", "strada"],
        "russian": ["улица", "проспект", "переулок", "площадь", "шоссе"],
        "ukrainian": ["вулиця", "проспект", "площа", "шосе"],
        "polish": ["ulica", "aleja", "plac", "osiedle"],
        "dutch": ["straat", "weg", "plein", "laan", "dijk"],
        "swedish": ["gatan", "vägen", "torget", "gränd"],
        "japanese": ["通り", "町", "区", "県", "市", "道"],
        "chinese": ["路", "街", "大道", "区", "市", "省"],
        "korean": ["로", "길", "동", "구", "시", "도"],
        "arabic": ["شارع", "طريق", "ساحة", "حي"],
        "hebrew": ["רחוב", "שדרות", "כיכר"],
        "greek": ["οδός", "λεωφόρος", "πλατεία"],
        "turkish": ["sokak", "cadde", "bulvar", "mahalle"],
        "hindi": ["रोड", "मार्ग", "चौक"],
        "thai": ["ถนน", "ซอย", "ตำบล", "อำเภอ"],
    }

    for lang, keywords in lang_keywords.items():
        if any(kw in text_lower for kw in keywords):
            cues.append(DetectedCue(cue_type="language", value=lang, confidence=confidence * 0.9, source="ocr"))

    return cues


def cues_from_vegetation(vegetation_types: List[str], confidence: float = 0.6) -> List[DetectedCue]:
    """Generate elimination cues from vegetation classification."""
    cues: List[DetectedCue] = []
    if not vegetation_types:
        return cues

    veg_lower = [v.lower() for v in vegetation_types]

    palm_signals = ["palm", "tropical", "banana", "mangrove", "coconut"]
    if any(s in v for v in veg_lower for s in palm_signals):
        cues.append(DetectedCue(cue_type="vegetation", value="palm", confidence=confidence, source="pixel_heuristic"))
        cues.append(DetectedCue(cue_type="latitude_band", value="tropical", confidence=confidence * 0.8, source="derived"))

    pine_signals = ["pine", "spruce", "fir", "needle", "conifer", "taiga", "boreal"]
    if any(s in v for v in veg_lower for s in pine_signals):
        cues.append(DetectedCue(cue_type="vegetation", value="pine", confidence=confidence, source="pixel_heuristic"))
        cues.append(DetectedCue(cue_type="latitude_band", value="temperate", confidence=confidence * 0.7, source="derived"))

    dry_signals = ["dry", "arid", "desert", "cactus", "succulent", "scrub", "savanna"]
    if any(s in v for v in veg_lower for s in dry_signals):
        cues.append(DetectedCue(cue_type="climate", value="desert", confidence=confidence * 0.7, source="derived"))

    med_signals = ["mediterranean", "olive", "cypress", "lavender", "vineyard"]
    if any(s in v for v in veg_lower for s in med_signals):
        cues.append(DetectedCue(cue_type="climate", value="mediterranean", confidence=confidence, source="derived"))

    return cues


def cues_from_infrastructure(
    infrastructure_type: Optional[str],
    pole_cues: Optional[List[str]] = None,
    road_cues: Optional[List[str]] = None,
    confidence: float = 0.6,
) -> List[DetectedCue]:
    """Generate elimination cues from infrastructure detections."""
    cues: List[DetectedCue] = []

    if infrastructure_type:
        it = infrastructure_type.lower()
        if "urban" in it or "highway" in it or "street" in it:
            # No strong elimination from generic urban
            pass

    if pole_cues:
        for pole in pole_cues:
            cues.append(DetectedCue(cue_type="pole_type", value=pole, confidence=confidence, source="clip_softmax"))

    if road_cues:
        for road in road_cues:
            cues.append(DetectedCue(cue_type="road_marking", value=road, confidence=confidence, source="clip_softmax"))

    return cues

