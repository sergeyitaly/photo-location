"""
CLIP text prompts for scene-level geolocation *cues* (softmax readouts, not coordinates).

Each bank is softmax-normalized independently — scores are comparable only within that bank.
"""

from __future__ import annotations

from typing import List, TypedDict


class SceneCueBank(TypedDict):
    id: str
    title: str
    prompts: List[str]


def _p(*lines: str) -> List[str]:
    return [s.strip() for s in lines if s.strip()]


SCENE_CUE_BANKS: List[SceneCueBank] = [
    {
        "id": "vegetation_trees",
        "title": "Vegetation & trees",
        "prompts": _p(
            "palm trees along street or tropical vegetation",
            "conifer pine or spruce forest edge",
            "deciduous oak or maple temperate street trees",
            "dry yellow grassland or savanna with scattered trees",
            "lush rice paddy or flooded green agriculture",
            "desert scrub with almost no green plants",
            "suburban manicured lawn and ornamental shrubs",
            "dense tropical jungle wall of green foliage",
            "mediterranean olive grove or dry orchard rows",
            "eucalyptus or silver gum trees dry climate",
        ),
    },
    {
        "id": "architecture_built",
        "title": "Architecture & built form",
        "prompts": _p(
            "white stucco mediterranean plaster facade",
            "red brick victorian row houses continuous street wall",
            "modern glass curtain wall tower reflective",
            "north american wood lap siding suburban house",
            "east asian tiled curved roof temple or gate",
            "corrugated zinc or metal informal rooftop patchwork",
            "colonial stone arcade plaza arcades pastel paint",
            "brutalist raw concrete slab architecture",
            "islamic dome minaret prayer hall silhouette",
            "timber frame alpine chalet steep pitched roof",
        ),
    },
    {
        "id": "palette_finish",
        "title": "Paint, materials & upkeep",
        "prompts": _p(
            "fresh saturated painted facade recently maintained",
            "faded peeling paint weathered plaster wall",
            "neutral grey concrete glass corporate minimal finish",
            "bright chaotic shop signage many competing colors",
            "terracotta roof tiles orange clay visible",
            "pastel washed colors sun bleached walls",
            "high contrast trim colorful wooden shutters",
            "rust stained industrial metal aged exterior",
        ),
    },
    {
        "id": "climate_light",
        "title": "Light, sky & atmosphere",
        "prompts": _p(
            "harsh midday sun strong shadows on ground",
            "overcast grey diffuse sky flat lighting",
            "golden hour warm low angle side lighting",
            "humid haze soft distant skyline reduced contrast",
            "clear deep blue dry sky high altitude feel",
            "fog or heavy mist obscuring background",
            "dusty murky air desert or pollution haze",
        ),
    },
]
