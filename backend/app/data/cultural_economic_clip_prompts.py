"""
CLIP softmax banks about **built environment & street commerce** — not people, not GDP.

Used only as relative phrase rankings within each bank. Easy to stereotype; treat as weak priors.
"""

from __future__ import annotations

from typing import List, TypedDict


class CulturalClipBank(TypedDict):
    id: str
    title: str
    prompts: List[str]


def _p(*lines: str) -> List[str]:
    return [s.strip() for s in lines if s.strip()]


CULTURAL_ECONOMIC_CLIP_BANKS: List[CulturalClipBank] = [
    {
        "id": "urban_housing_form",
        "title": "Urban form & housing typology (visual)",
        "prompts": _p(
            "high-rise apartment towers filling the skyline",
            "single-family suburban houses setbacks and lawns",
            "dense historic low-rise masonry street wall continuous",
            "informal corrugated metal roof clusters hillside",
            "gated compound wall residential security entrance",
            "modern glass condo tower podium retail base",
            "narrow alley between brick row houses continuous facade",
            "soviet-era prefab panel apartment slab facade grid",
            "mixed mid-rise plaster buildings continuous street edge",
            "single-storey detached plots agricultural fringe town",
        ),
    },
    {
        "id": "street_commerce_upkeep",
        "title": "Commerce, signage & maintenance (visual)",
        "prompts": _p(
            "luxury retail storefront polished glass windows",
            "informal sidewalk vendor stalls tarps crowded",
            "modern indoor shopping mall atrium escalators",
            "street market fabric awnings dense signage",
            "freshly paved sidewalk with painted lane markings",
            "cracked pavement potholes unrepaired curb",
            "dense neon vertical signage night commercial strip",
            "minimal signage quiet residential street only",
            "construction hoarding scaffolding renovation wrap",
            "ornate historic shopfront carved stone trim",
        ),
    },
    {
        "id": "architectural_idiom_hints",
        "title": "Architectural idioms (stylized phrases only)",
        "prompts": _p(
            "islamic geometric tile facade mashrabiya lattice hints",
            "east asian curved ceramic roof ridge temple gate silhouette",
            "iberian colonial stone arcade plaza arcades walkway",
            "neoclassical columns pediment civic building entrance",
            "brutalist raw concrete geometric mass public building",
            "art deco streamline vertical ribs theatre facade",
            "timber frame balcony mountain chalet steep roof",
            "red sandstone mughal arch institutional gateway",
            "bauhaus flat roof ribbon window white render",
            "veranda wraparound wooden colonial tropical house",
        ),
    },
]

METHODOLOGY_CULTURAL_ECONOMIC = (
    "Three independent CLIP softmax banks over English phrases describing buildings, streets, and commerce. "
    "Scores sum to 1 within each bank; they are **not** country labels, GDP estimates, or cultural facts."
)

DISCLAIMER_CULTURAL_ECONOMIC = (
    "This output must not be used to judge people, nations, or economies. "
    "CLIP matches generic English stereotypes to pixels and will false-positive globally. "
    "Geographic coordinates from vision alone are not reliable at very high accuracy — prefer EXIF GPS when available."
)
