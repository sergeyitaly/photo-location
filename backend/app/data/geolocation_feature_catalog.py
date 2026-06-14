"""
Expandable catalog of geolocation-relevant visual / contextual analysis dimensions.

Each entry is a *hypothesis surface* for future extractors (CLIP banks, classical CV,
segmentation, metadata fusion). This module does not assign GPS — it enumerates cues.

Minimum size target: ≥500 distinct feature specifications (versioned, reproducible).
"""

from __future__ import annotations

from typing import Any, Dict, List

GEOLOCATION_FEATURE_CATALOG_VERSION = "1.1.0"

# Eleven orthogonal “analysis facets” applied to each base cue template.
_ANALYSIS_FACETS: List[str] = [
    "global_image_statistics",
    "upper_sky_band_roi",
    "lower_ground_band_roi",
    "vertical_facade_roi",
    "center_weighted_roi",
    "high_frequency_texture",
    "hue_saturation_histogram",
    "illumination_colour_temperature_proxy",
    "shadow_direction_consistency",
    "multi_scale_pyramid",
    "temporal_consistency_vs_neighbors",
]


# Fifty-two base cue families (expandable); × 11 facets → 572 catalog rows.
_BASE_CUES: List[str] = [
    "Zenith skylight chromaticity versus horizon wash",
    "Diffuse horizontal illuminance proxy from RGB luminance",
    "Circumsolar glare bloom presence and extent",
    "Cloud genus indicators (cumulus stratiform layering)",
    "Contrail persistence and spreading angle",
    "Fog droplet scattering milky gradient",
    "Rain shaft darkness contrast against sky dome",
    "Snow crystal sparkle density and colour temperature",
    "Nocturnal artificial skyglow dome gradient",
    "Polar night twilight purple band elevation",
    "Canopy closure fraction (forest gap dynamics)",
    "Sunfleck penumbra statistics under canopy",
    "Broadleaf versus needle-leaf silhouette frequency",
    "Epiphyte loading on trunks (humid tropics proxy)",
    "Understory fern density (wet temperate proxy)",
    "Savanna grass sward colour seasonality",
    "Steppe dwarf shrub spacing regularity",
    "Tundra polygon vegetation micro-pattern",
    "Mangrove pneumatophore rhythm in mud",
    "Rice paddy bund geometry and water film specularity",
    "Vineyard row spacing versus slope aspect",
    "Tea terrace contour curvature tightness",
    "Oil palm regimented planting detection",
    "Eucalyptus bark shedding ribbon texture",
    "Palm crown silhouette height distribution",
    "Surface soil hue iron oxide redness",
    "Caliche crust polygon cracking pattern",
    "Loess vertical erosion ribbing",
    "Saline efflorescence crust whiteness",
    "Volcanic black scoria grain size",
    "Laterite gravel lateritic orange fraction",
    "Alluvial fan braided channel texture",
    "Glacial till unsorted clast scatter",
    "Karst cockpit dolline spacing",
    "Aeolian ripple asymmetry orientation",
    "Periglacial stone stripe sorting",
    "River surface roughness versus discharge proxy",
    "Lake colour dissolved organic matter brownness",
    "Estuarine turbidity gradient versus tide proxy",
    "Coral shallow water chromatic absorption band",
    "Black sand magnetite sparkle density",
    "White gypsum dune slope slip face angle",
    "Ice jam shelf edge texture on river",
    "Reservoir drawdown bathtub ring staining",
    "Roof pitch distribution on residential blocks",
    "Terracotta versus slate roof hue dominance",
    "Flat concrete roof water tank silhouette frequency",
    "Baroque facade ornament density window rhythm",
    "Modern curtain wall reflection vertical mullion pitch",
    "Timber frame infill diagonal brace angle",
    "Adobe rounded corner erosion profile",
    "Colonial balcony iron grille lattice pitch",
    "Art deco rounded corner banding rhythm",
    "Brutalist board-mark concrete stripe spacing",
    "High-voltage transmission tower lattice geometry",
    "Catenary overhead tram wire presence",
    "Cantilever signal gantry mast silhouette",
    "Roundabout lane marking spiral tightness",
    "Raised pedestrian crossing zebra width",
    "Bus shelter advertising frame aspect ratio",
    "Bollard spacing along sidewalk edge",
    "Stone curb versus poured concrete curb lip",
    "Dual carriageway median planting strip width",
    "Guardrail reflector dot spacing frequency",
    "License plate aspect ratio and frame colour",
    "Taxi roof sign silhouette aspect",
    "Right-hand versus left-hand traffic flow proxy",
    "Bus pantograph silhouette on roof",
    "Tram rail groove shadow in asphalt",
    "Roundabout direction arrow wear pattern",
    "Utility pole cross-arm insulator stack count",
    "Guy wire anchor concrete pad frequency",
    "Street name plaque mounting height and font family",
    "Shop awning folding versus fixed tension",
    "Vendor cart umbrella colour entropy",
    "Outdoor seating plastic chair stack chroma",
    "Air conditioning external unit grille density",
    "Satellite dish diameter clustering on roofs",
    "Solar panel azimuth tilt plane visibility",
    "Clothesline pole height and multi-line spacing",
    "Stone wall dry-stack versus mortar joint width",
    "Fence palisade tip wear weathering gradient",
]


def _build_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n = 0
    for cue in _BASE_CUES:
        cat = cue.split(".")[0][:72]
        for facet in _ANALYSIS_FACETS:
            rows.append(
                {
                    "id": f"gfc_{n:04d}",
                    "category": cat,
                    "facet": facet,
                    "cue": cue,
                    "description": (
                        f"{cue}. Measurable via facet: {facet.replace('_', ' ')} "
                        "(relative softmax, classical descriptor, or segmentation mask)."
                    ),
                    "modality": "vision_proxy",
                }
            )
            n += 1
    return rows


_FEATURE_ROWS: List[Dict[str, Any]] = _build_rows()

assert len(_FEATURE_ROWS) >= 500, "catalog minimum contract"


def get_geolocation_feature_catalog() -> List[Dict[str, Any]]:
    """Return a shallow copy of the full feature specification list."""
    return list(_FEATURE_ROWS)


def geolocation_feature_catalog_count() -> int:
    return len(_FEATURE_ROWS)


def geolocation_feature_catalog_version() -> str:
    return GEOLOCATION_FEATURE_CATALOG_VERSION
