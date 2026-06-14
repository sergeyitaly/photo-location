"""
Regional “globe” cues — macro-region visual stereotypes for CLIP softmax only.

Scores are relative within each category’s prompt list and do **not** move the map pin.
"""

from __future__ import annotations

from typing import List, TypedDict


class GlobeRegionalCategoryDef(TypedDict):
    id: str
    title: str
    prompts: List[str]


class GlobeRegionalRegionDef(TypedDict):
    id: str
    title: str
    categories: List[GlobeRegionalCategoryDef]


def _p(*lines: str) -> List[str]:
    return [s.strip() for s in lines if s.strip()]


GLOBE_REGION_DEFINITIONS: List[GlobeRegionalRegionDef] = [
    {
        "id": "eastern_europe_ex_ussr",
        "title": "Eastern Europe / former USSR",
        "categories": [
            {
                "id": "post_soviet_gas_and_utilities",
                "title": "Yellow outdoor utilities",
                "prompts": _p(
                    "yellow painted gas pipes along a sidewalk",
                    "above-ground yellow steel gas main beside a road",
                    "soviet-era outdoor pipes painted yellow on a facade",
                    "narrow trench repair exposing yellow plumbing pipe",
                    "plastic garden hose on lawn",
                    "no yellow outdoor utility piping visible",
                ),
            },
            {
                "id": "steppe_edge_terrain",
                "title": "Steppe-edge relief",
                "prompts": _p(
                    "gentle grassy hills fading into horizon",
                    "shallow erosion gully in rolling dry grassland",
                    "flat boreal plain almost no hills",
                    "snow-capped alpine peaks",
                    "dense rainforest canopy",
                ),
            },
            {
                "id": "rural_eastern_slavic_settlement",
                "title": "Rural roads & plots",
                "prompts": _p(
                    "narrow asphalt village road with wooden utility poles",
                    "detached rural houses behind metal gates",
                    "vegetable gardens along roadside fence",
                    "american suburb with mailboxes and lawns",
                    "medieval stone village lane",
                ),
            },
        ],
    },
    {
        "id": "western_northern_europe",
        "title": "Western & Northern Europe",
        "categories": [
            {
                "id": "narrow_stone_street_hist_core",
                "title": "Historic cores",
                "prompts": _p(
                    "narrow cobbled lane between stone buildings",
                    "brick row houses with sash windows street perspective",
                    "glass curtain wall skyline only modern towers",
                    "latin american plaza arcades pastel buildings",
                ),
            },
            {
                "id": "temperate_farmland_hedges",
                "title": "Patchwork farmland",
                "prompts": _p(
                    "patchwork green fields hedgerows under grey sky",
                    "straight crop rows across flat plateau",
                    "terraced rice paddies flooded reflections",
                    "semi-arid scrub no hedgerows",
                ),
            },
            {
                "id": "wet_climate_vegetation",
                "title": "Cool wet vegetation",
                "prompts": _p(
                    "mossy stone wall damp temperate forest edge",
                    "temperate deciduous trees roadside canopy",
                    "dry desert thorn bush sparse vegetation",
                    "tropical palm-lined boulevard humid",
                ),
            },
            {
                "id": "baltic_nordic_built_cues",
                "title": "Baltic & Nordic built form (soft CLiP hints only)",
                "prompts": _p(
                    "falun red wood house scandinavian rural",
                    "painted clapboard house finnish lake summer",
                    "baltic art nouveau ornamental plaster city block",
                    "oslo or stockholm modern stucco long street frontage",
                ),
            },
        ],
    },
    {
        "id": "north_america",
        "title": "North America",
        "categories": [
            {
                "id": "suburban_grid_car_scale",
                "title": "Suburban car scale",
                "prompts": _p(
                    "wide arterial road turning lane traffic lights suburban",
                    "mailbox at curb detached house setback lawn",
                    "dense medieval alley no setback lawns",
                    "east asian neon signage canyon street night",
                ),
            },
            {
                "id": "wood_frame_siding",
                "title": "Wood framing & siding",
                "prompts": _p(
                    "wood lap siding suburban house chimney",
                    "brick victorian townhouse row",
                    "rendered mediterranean stucco flat roof cluster",
                    "corrugated zinc informal roofing patchwork",
                ),
            },
            {
                "id": "continental_plains_horizon",
                "title": "Interior plains horizon",
                "prompts": _p(
                    "dead straight highway vanishing toward flat horizon corn belt",
                    "irrigation pivot circle crops from aerial hint",
                    "dense european hedge maze fields",
                    "alp mountain valley no corn horizon",
                ),
            },
        ],
    },
    {
        "id": "latin_america_caribbean",
        "title": "Latin America & Caribbean",
        "categories": [
            {
                "id": "colonial_plaza_arcades",
                "title": "Colonial squares",
                "prompts": _p(
                    "arcaded walkway around shaded plaza colonial stone",
                    "pastel plaster facade wooden shutters balconies",
                    "narrow europe cobble alley gothic arches",
                    "minimalist brutalist plaza concrete only",
                ),
            },
            {
                "id": "tropical_vegetation_urban_edge",
                "title": "Tropical urban greenery",
                "prompts": _p(
                    "bougainvillea over masonry wall humid sun",
                    "dense palm mixed broadleaf roadside tropical city",
                    "cold climate bare deciduous trees only",
                    "dry highland scrub minimal understory",
                ),
            },
            {
                "id": "andes_or_meso_vernacular",
                "title": "Mountain / volcanic vernacular",
                "prompts": _p(
                    "steep hillside settlement corrugated roofs terraces",
                    "earthquake-ready heavy masonry colonial church plain facade",
                    "fjord glacier granite northern europe slope",
                    "sandstone mesa american southwest red rock only",
                ),
            },
        ],
    },
    {
        "id": "middle_east_north_africa",
        "title": "Middle East & North Africa",
        "categories": [
            {
                "id": "flat_roofs_satellites",
                "title": "Flat roofs & satellites",
                "prompts": _p(
                    "flat roofline rim satellite dishes clustered urban",
                    "steep alpine chalet snow pitched roof loads",
                    "east asian tiled temple roof silhouette",
                    "wet tropical corrugated rural housing only",
                ),
            },
            {
                "id": "desert_aridity",
                "title": "Arid ambience",
                "prompts": _p(
                    "dust haze pale sky sand-colored walls dissolve distance",
                    "lush green rainy temperate pasture sky",
                    "snow covered ground boreal winter street",
                    "dense jungle humid dark green canopy wall",
                ),
            },
            {
                "id": "oasis_palms_irrigation",
                "title": "Palms & irrigation",
                "prompts": _p(
                    "date palms ringed settlement irrigation ditch berm",
                    "wet rice paddy terraces flooded tropical agriculture",
                    "cold desert sagebrush without palms",
                    "mediterranean olive grove dry stone terraces only",
                ),
            },
        ],
    },
    {
        "id": "subsaharan_africa",
        "title": "Sub-Saharan Africa",
        "categories": [
            {
                "id": "red_laterite_roads",
                "title": "Red soil roadsides",
                "prompts": _p(
                    "laterite gravel shoulder vivid orange roadside earth exposed",
                    "fresh grey asphalt northern suburb verge grass",
                    "snow lined icy shoulder boreal road",
                    "wet black mud volcanic tropical track",
                ),
            },
            {
                "id": "savanna_wide_sky",
                "title": "Savanna horizon",
                "prompts": _p(
                    "umbrella acacia scattered grass beige dry season skyline",
                    "dense rainforest canopy wall humid green oppressive",
                    "english pasture hedgerow muted green temperate",
                    "american corn monoculture infinite rows",
                ),
            },
            {
                "id": "informal_roofscapes",
                "title": "Informal roofscapes",
                "prompts": _p(
                    "patchwork corrugated zinc rooftop clusters hillside view",
                    "american rv park orderly paved lanes trailers",
                    "dense europe slate uniform roofline historic core",
                    "modern glass waterfront towers marina",
                ),
            },
        ],
    },
    {
        "id": "south_asia",
        "title": "South Asia",
        "categories": [
            {
                "id": "dense_commerce_lane",
                "title": "Dense commerce lanes",
                "prompts": _p(
                    "stacked storefront shutters tarp awnings crowded sidewalk",
                    "minimalist nordic storefront empty sidewalk signage sparse",
                    "shopping mall interior atrium escalators climate controlled",
                    "wet market southeast asia tropical fruit stacks open hall",
                ),
            },
            {
                "id": "motor_traffic_wire_tangle",
                "title": "Traffic & aerial wires",
                "prompts": _p(
                    "auto rickshaw mix weaving dense intersection faded paint",
                    "underground utilities clean avenue no aerial clutter tokyo",
                    "horse cart cobble europe historic core",
                    "desert highway clear sky minimal poles",
                ),
            },
            {
                "id": "monsoon_greenery",
                "title": "Monsoon roadside green",
                "prompts": _p(
                    "lush roadside hedge giant leaves vines damp wall stain",
                    "arid rajasthan sand colored buildings thorn scrub",
                    "snow dusted conifers himalayan slope winter",
                    "australian eucalyptus dry sclerophyll silver blue",
                ),
            },
        ],
    },
    {
        "id": "east_asia",
        "title": "East Asia",
        "categories": [
            {
                "id": "neon_vertical_signage",
                "title": "Neon signage stacks",
                "prompts": _p(
                    "dense vertical neon kanji hangul layers night canyon glow",
                    "historic europe low stone signage minimal night electricity",
                    "latin plaza colonial pastel low buildings night soft",
                    "middle eastern warm sodium street single story shops",
                ),
            },
            {
                "id": "curved_tile_roof_elements",
                "title": "Curved tile roofs",
                "prompts": _p(
                    "grey curved ceramic roof ridge temple gate silhouette",
                    "flat islamic dome minaret prayer hall outline",
                    "steep alpine slate roof church spire europe",
                    "corrugated shed industrial shed flat compound",
                ),
            },
            {
                "id": "bamboo_scaffold_poles",
                "title": "Bamboo scaffold & poles",
                "prompts": _p(
                    "bamboo scaffolding lashed across facade renovation dense city",
                    "tubular steel scaffold uniform grid western construction site",
                    "timber bamboo forest agriculture slope only no buildings",
                    "clear desert boulevard minimal poles overhead",
                ),
            },
        ],
    },
    {
        "id": "oceania_se_asia_insular",
        "title": "Oceania & maritime Southeast Asia",
        "categories": [
            {
                "id": "humid_insular_street_green",
                "title": "Humid street planting",
                "prompts": _p(
                    "lush roadside trees vines humid tropical island town edge",
                    "dry mediterranean scrub sparse maquis coastal",
                    "cold nz pasture sheep fence rolling green temperate",
                    "american southwest desert succulent only",
                ),
            },
            {
                "id": "timber_vernacular_elevated",
                "title": "Timber / elevated builds",
                "prompts": _p(
                    "weatherboard timber bungalow on stilts coastal humid",
                    "concrete brutalist slab tropical housing minimal pilotis",
                    "central europe timber frame plaster infill dense village",
                    "steel glass high-rise waterfront generic global city",
                ),
            },
            {
                "id": "coral_karst_limestone",
                "title": "Karst / limestone hints",
                "prompts": _p(
                    "tower karst limestone peaks humid bay backdrop",
                    "dark volcanic plugs steep jungle ridges ocean island",
                    "sandstone desert mesa american southwest monument shapes",
                    "norwegian fjord granite walls narrow cold sea",
                ),
            },
        ],
    },
    {
        "id": "central_asia_caucasus",
        "title": "Central Asia & Caucasus",
        "categories": [
            {
                "id": "wide_soviet_avenue_panel",
                "title": "Wide avenues & Soviet-era slabs",
                "prompts": _p(
                    "massive brutalist apartment slab long shadow wide boulevard",
                    "ornate stalinist facade columns central asian capital",
                    "narrow winding lane wood balcony caucasus hillside town",
                    "american strip mall parking sea asphalt low retail",
                    "colonial spanish plaza single story arcade",
                ),
            },
            {
                "id": "inland_desert_steppe_strip",
                "title": "Steppe & dry interior",
                "prompts": _p(
                    "featureless dry steppe horizon irrigation canal strip",
                    "salt flat cracked earth distant mountain chain",
                    "rolling green pasture hedge temperate oceanic",
                    "dense jungle canopy wet equatorial",
                ),
            },
            {
                "id": "high_valley_pasture_mountains",
                "title": "High valleys & pasture",
                "prompts": _p(
                    "steep grassy slope cattle dirt track mountain valley",
                    "glacial u-shaped valley bare rock moraine palette",
                    "flat floodplain rice paddies tropical",
                    "red desert canyon sandstone american southwest",
                ),
            },
        ],
    },
    {
        "id": "mainland_southeast_asia",
        "title": "Mainland Southeast Asia",
        "categories": [
            {
                "id": "motorbike_dense_street",
                "title": "Motorbike urban fabric",
                "prompts": _p(
                    "dense motorbike parking sidewalk jungle of scooters",
                    "wet tropical street tar steam after rain reflections",
                    "wide north american suburban arterial no scooters",
                    "empty nordic bicycle lane winter bare trees",
                ),
            },
            {
                "id": "temple_gold_stupa_roofline",
                "title": "Temple rooflines & gold leaf",
                "prompts": _p(
                    "golden spire ornate temple roof glitter tropical sun",
                    "white plaster stupa rounded dome buddhist compound",
                    "gothic cathedral pointed arch europe stone",
                    "glass office tower skyline only generic global",
                ),
            },
            {
                "id": "wet_market_canopy_street",
                "title": "Covered markets & tropical produce",
                "prompts": _p(
                    "open sided metal roof market hall tropical fruit piles",
                    "plastic tarp stall canopy crowded alley humidity",
                    "climate controlled supermarket fluorescent aisles only",
                    "farmers wooden crates temperate europe square",
                ),
            },
        ],
    },
    {
        "id": "polar_high_latitude",
        "title": "Polar & high latitude",
        "categories": [
            {
                "id": "snow_dominant_ground",
                "title": "Snowpack & cleared streets",
                "prompts": _p(
                    "snow banks piled beside plowed asphalt road winter",
                    "bare birch trees snow heavy sky low winter light",
                    "dry cracked earth desert summer no snow",
                    "lush green meadow spring flower temperate",
                ),
            },
            {
                "id": "low_sun_angle_light",
                "title": "Low sun / long shadows (season cue)",
                "prompts": _p(
                    "very long shadows across snow low golden hour arctic spring",
                    "overhead harsh noon tropical sun short shadow",
                    "grey overcast diffuse light no shadows northern winter",
                    "dramatic sunset silhouette desert horizon",
                ),
            },
            {
                "id": "tundra_scrub_horizon",
                "title": "Tundra & sparse boreal scrub",
                "prompts": _p(
                    "low stunted shrubs moss carpet exposed rock arctic tundra",
                    "dense boreal forest dark evergreen wall",
                    "savanna grass umbrella acacia dry season africa",
                    "alpine meadow wildflowers steep slope europe",
                ),
            },
        ],
    },
]
