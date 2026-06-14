"""
CLIP softmax banks for **visible** infrastructure, energy hardware, and coarse economic *visual* proxies.

Does not detect buried pipes without surface clues, measure GDP, or infer utility ownership.
Scores are softmax-normalized **within each bank only**.
"""

from __future__ import annotations

from typing import List, TypedDict


class InfraEnergyBank(TypedDict):
    id: str
    title: str
    prompts: List[str]


def _p(*lines: str) -> List[str]:
    return [s.strip() for s in lines if s.strip()]


INFRASTRUCTURE_ENERGY_CLIP_BANKS: List[InfraEnergyBank] = [
    {
        "id": "economic_activity_visual",
        "title": "Economic activity (visual proxies only — not GDP or income)",
        "prompts": _p(
            "dense central business district skyscrapers and corporate towers",
            "informal street vending stalls tarps sidewalk economy",
            "shipping container port cranes logistics waterfront",
            "large industrial factory smokestack manufacturing plant",
            "open pit mine or quarry heavy machinery brown terrain",
            "large mechanized agriculture tractor crop rows industrial farm",
            "luxury retail shopping district polished storefronts pedestrian",
            "quiet residential suburb single family houses low density",
            "construction site cranes new high-rise development",
            "rust belt abandoned industrial decay empty factory",
        ),
    },
    {
        "id": "fossil_gas_infrastructure",
        "title": "Gas & hydrocarbon surface cues (when visible)",
        "prompts": _p(
            "above ground industrial gas pipeline corridor steel pipes",
            "pipeline route marker posts along road or field edge",
            "spherical LNG storage tanks refinery petrochemical yard",
            "natural gas compressor station fenced machinery yard",
            "fuel petrol station canopy pumps forecourt",
            "oil or gas flare stack flame industrial night sky",
            "refinery distillation column complex pipes maze",
            "methane venting equipment oilfield pumpjack nodding donkey",
            "gas meter cabinet residential building exterior wall",
            "yellow buried utility warning tape excavation trench",
        ),
    },
    {
        "id": "electrical_grid_infrastructure",
        "title": "Electrical transmission & distribution (visible)",
        "prompts": _p(
            "high voltage lattice transmission tower power line corridor",
            "wooden utility pole multiple crossarms neighborhood wires",
            "electrical substation yard transformers fenced concrete pad",
            "insulator strings on high voltage suspension tower",
            "streetlight mast with overhead power cables urban",
            "underground electrical vault manhole utility cover street",
            "distribution box pad mounted transformer green metal cabinet",
            "offshore wind farm export cable landing station coastal",
            "electrical relay hut concrete block equipment shelter",
            "dense catenary wires tram or trolley electric transit",
        ),
    },
    {
        "id": "solar_photovoltaic",
        "title": "Solar PV (rooftop & utility-scale appearances)",
        "prompts": _p(
            "rooftop solar photovoltaic panel array blue rectangles tilted",
            "large ground mounted solar farm rows from low altitude",
            "carport solar canopy parking shade panels overhead",
            "floating solar panels reservoir lake installation",
            "building integrated photovoltaic glass facade solar",
            "solar thermal flat plate collectors hot water roof",
            "agricultural land dual use solar panels above crops",
            "industrial rooftop vast flat solar installation aerial",
            "small portable solar panel camping van roof",
            "no solar panels visible mostly conventional roof surface",
        ),
    },
    {
        "id": "wind_power",
        "title": "Wind generation (turbines & farms)",
        "prompts": _p(
            "large white horizontal axis wind turbine tower three blades",
            "offshore wind farm multiple turbines on horizon sea",
            "ridgeline wind turbines spaced along hill crest",
            "single small residential wind turbine tower property",
            "wind turbine blade transportation truck oversized load road",
            "wind farm service road gravel between turbine bases",
            "wind turbine nacelle close-up maintenance lift",
            "no wind turbines visible rolling hills without turbines",
            "coal or gas power plant cooling towers steam not wind",
            "run of river hydroelectric dam spillway concrete",
        ),
    },
]

METHODOLOGY_INFRASTRUCTURE_ENERGY = (
    "Five independent CLIP softmax banks over English phrases for infrastructure and energy hardware. "
    "Each bank sums to 1.0 internally. "
    "Cannot see underground pipes unless excavation or markers show; cannot estimate economics numerically."
)

DISCLAIMER_INFRASTRUCTURE_ENERGY = (
    "Visual similarity only — not safety inspection, not regulatory compliance, not precise identification "
    "of utilities. Do not use for operational decisions about gas, electrical, or industrial sites."
)
