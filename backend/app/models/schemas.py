"""Pydantic schemas for API requests/responses"""
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal

from pydantic import BaseModel, Field, ConfigDict

CoordinateSource = Literal["exif_gps", "filename_hint", "vision_estimate"]
CueSource = Literal["pixel_heuristic", "clip_softmax", "derived"]


class FeatureAnalysis(BaseModel):
    """Analysis of visual features extracted from an image"""
    landmarks: Optional[List[Dict[str, Any]]] = Field(default=None)
    vegetation_types: Optional[List[str]] = None
    architecture_style: Optional[str] = None
    detected_text: Optional[List[str]] = None
    weather_condition: Optional[str] = None
    time_of_day: Optional[str] = None
    infrastructure_type: Optional[str] = None
    detected_poles: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Specialist detector: utility pole proxies from edge analysis."
    )
    detected_road_lines: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Specialist detector: road marking line proxies from lower-frame analysis."
    )
    shadow_analysis: Optional[Dict[str, Any]] = Field(
        default=None, description="Specialist detector: shadow direction, ratio, and dark region stats."
    )

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "example": {
                "landmarks": [{"name": "Eiffel Tower", "confidence": 0.95}],
                "vegetation_types": ["deciduous trees", "grass"],
                "architecture_style": "European",
                "detected_text": ["PARIS", "FRANCE"],
                "weather_condition": "sunny",
                "time_of_day": "afternoon",
                "infrastructure_type": "urban",
                "detected_poles": [{"type": "concrete", "confidence": 0.65}],
                "shadow_analysis": {"shadow_direction_deg": 45.0, "confidence": 0.4},
            }
        },
    )


class GlobeRegionalItem(BaseModel):
    label: str
    confidence: float


class GlobeRegionalCategoryHints(BaseModel):
    category_id: str
    title: str
    items: List[GlobeRegionalItem] = Field(default_factory=list)


class GlobeRegionalRegionHints(BaseModel):
    region_id: str
    title: str
    categories: List[GlobeRegionalCategoryHints] = Field(default_factory=list)


class SceneCueItem(BaseModel):
    label: str
    score: float = Field(..., ge=0.0, le=1.0)
    source: CueSource = "pixel_heuristic"


class CulturalEconomicVisualCues(BaseModel):
    """Built-form / commerce / façade-idiom CLIP readouts — not GDP or culture measurement."""

    methodology: str = ""
    disclaimer: str = ""
    clip_banks_detail: List[Dict[str, Any]] = Field(default_factory=list)
    clip_available: bool = False


class SceneGeolocationCues(BaseModel):
    """
    Flora/fauna/architecture/palette/light cues — interpretive only.
    Pixel block is cheap; CLIP banks require torch+transformers.
    """

    methodology: str = ""
    pixel_stats: Dict[str, float] = Field(default_factory=dict)
    vegetation: List[SceneCueItem] = Field(default_factory=list)
    built_environment: List[SceneCueItem] = Field(default_factory=list)
    palette_and_finish: List[SceneCueItem] = Field(default_factory=list)
    climate_and_light: List[SceneCueItem] = Field(default_factory=list)
    design_and_upkeep_proxy: List[SceneCueItem] = Field(default_factory=list)
    clip_banks_detail: Optional[List[Dict[str, Any]]] = None
    clip_available: bool = False
    clip_model_id: Optional[str] = None
    interpretive_summary: str = ""
    cultural_economic_visual: Optional[CulturalEconomicVisualCues] = None


class GlobeRegionalHintsResult(BaseModel):
    """CLIP softmax over hand-written regional prompt sets; does not set coordinates."""

    clip_available: bool
    note: Optional[str] = None
    model_id: Optional[str] = None
    regions: List[GlobeRegionalRegionHints] = Field(default_factory=list)


class RecognitionLabel(BaseModel):
    """Single softmax label from a fixed CLIP prompt list."""

    label: str
    score: float = Field(..., ge=0.0, le=1.0)


class MLImageRecognition(BaseModel):
    """Neural image recognition summary (CLIP softmax over curated prompts)."""

    clip_available: bool = False
    model_id: Optional[str] = None
    methodology: str = ""
    scene_and_object_labels: List[RecognitionLabel] = Field(default_factory=list)
    note: Optional[str] = None


class InfrastructureEnergyVisualCues(BaseModel):
    """Gas, electrical grid, solar, wind & visual economic-activity proxies — CLIP softmax only."""

    enabled: bool = True
    skipped_reason: Optional[str] = Field(
        default=None,
        description="Present when bundle skipped (settings off, missing torch, empty image).",
    )
    methodology: str = ""
    disclaimer: str = ""
    clip_banks_detail: List[Dict[str, Any]] = Field(default_factory=list)
    clip_available: bool = False
    clip_model_id: Optional[str] = None
    interpretive_summary: str = ""


class GeolocationReadingAxes(BaseModel):
    """
    Three interpretive lenses for the same pin — not alternate coordinates.

    Perspective + building lines are cheap pixel proxies; Wikipedia line summarises open-data checks.
    """

    perspective_of_view: str = Field(default="", description="Camera framing / sky band / aspect — descriptive only.")
    building_proportions: str = Field(
        default="",
        description="Vertical vs horizontal edge energy in the lower frame + texture — façade / street cues.",
    )
    estimated_wikipedia: str = Field(
        default="",
        description="English Wikipedia geosearch + relief + optional CLIP–lead semantic gate.",
    )


class WikipediaPlaceContext(BaseModel):
    """Wikipedia / Commons context for the results UI (articles, photo match, pin adjustment)."""

    enabled: bool = False
    note: str = ""
    synthesized_summary: str = ""
    physical_setting_summary: str = ""
    alternative_wikipedia_note: str = ""
    primary_wikipedia_fit_score: Optional[float] = None
    best_alternative_wikipedia_fit_score: Optional[float] = None
    best_alternative_wikipedia_index: Optional[int] = None
    primary_photo_similarity: Optional[float] = None
    wiki_match_quality: str = ""
    primary_pin_adjusted: bool = False
    pin_adjustment_note: str = ""
    articles: List[Dict[str, Any]] = Field(default_factory=list)


class ExternalValidationSummary(BaseModel):
    """English Wikipedia geosearch + OpenTopoData relief + CLIP text/photo checks across candidates."""

    enabled: bool = False
    skipped_reason: Optional[str] = None
    selected_candidate_index: int = 0
    pin_adjusted: bool = False
    proof_satisfied: bool = Field(
        default=True,
        description="True when some candidate passed all active gates (wiki + relief + semantic + photo if enabled).",
    )
    wikipedia_checks: List[Dict[str, Any]] = Field(default_factory=list)
    relief_checks: List[Dict[str, Any]] = Field(default_factory=list)
    wikipedia_semantic_checks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="CLIP vs Wikipedia lead extract per candidate when semantic gate is on.",
    )
    wikipedia_photo_checks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="CLIP vs Wikimedia Commons / Wikipedia lead images near each candidate.",
    )
    summary_note: str = ""


class CrossReferenceDatabaseSummary(BaseModel):
    """Local gazetteer/database cross-check across fused candidates."""

    enabled: bool = False
    skipped_reason: Optional[str] = None
    selected_candidate_index: int = 0
    pin_adjusted: bool = False
    matched_place_name: Optional[str] = None
    matched_country: Optional[str] = None
    candidate_checks: List[Dict[str, Any]] = Field(default_factory=list)
    summary_note: str = ""


class PlaceResolution(BaseModel):
    """
    Human-readable place from reverse geocoding at the predicted coordinates (any city/town/village OSM knows).
    Independent of neural model labels — use when the pin is plausible but the classifier city name is generic.
    """

    locality: Optional[str] = Field(
        default=None,
        description="City, town, village, hamlet, or similar from OpenStreetMap.",
    )
    locality_kind: Optional[str] = Field(
        default=None,
        description="Which address field supplied locality (city, town, village, …).",
    )
    administrative_area: Optional[str] = Field(default=None, description="State, oblast, region, etc.")
    county: Optional[str] = None
    country: Optional[str] = Field(default=None, description="Country name from OSM.")
    country_code: Optional[str] = Field(
        default=None,
        description="ISO 3166-1 alpha-2 when provided by Nominatim (stable for globe-wide logic).",
    )
    display_name: Optional[str] = Field(default=None, description="Full Nominatim display line.")
    source: str = "openstreetmap_nominatim"
    attribution: str = Field(
        default="Data © OpenStreetMap contributors",
        description="Keep visible when showing resolved names (ODbL).",
    )
    error: Optional[str] = Field(default=None, description="Set when lookup failed.")


class LocationPrediction(BaseModel):
    """Single location prediction"""
    latitude: float = Field(..., description="Predicted latitude")
    longitude: float = Field(..., description="Predicted longitude")
    country: str = Field(..., description="Predicted country")
    city: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    distance_confidence_km: Optional[float] = None
    place_resolution: Optional[PlaceResolution] = Field(
        default=None,
        description="Reverse-geocoded place at lat/lon (covers worldwide settlements when enabled).",
    )


class CountryEliminationResult(BaseModel):
    """Rule-based country elimination output."""

    remaining_countries: List[str] = Field(default_factory=list)
    eliminated_countries: List[str] = Field(default_factory=list)
    country_scores: Dict[str, float] = Field(default_factory=dict)
    applied_cue_count: int = 0
    contradiction_penalties: List[str] = Field(default_factory=list)
    summary: str = ""
    num_remaining: int = 0
    num_eliminated: int = 0


class GeoReasoningResult(BaseModel):
    """Bayesian geographic reasoning output."""

    country_posteriors: Dict[str, float] = Field(default_factory=dict)
    top_country: str = ""
    top_confidence: float = 0.0
    evidence_breakdown: List[Dict[str, Any]] = Field(default_factory=list)
    contradiction_penalties_applied: List[str] = Field(default_factory=list)
    summary: str = ""


class AstronomyConstraints(BaseModel):
    """Latitude/longitude constraints from shadow/sun analysis."""

    latitude_min: float = -90.0
    latitude_max: float = 90.0
    latitude_confidence: float = 0.0
    hemisphere_hint: str = ""
    season_hint: str = ""
    solar_elevation_deg: Optional[float] = None
    shadow_direction_deg: Optional[float] = None
    time_of_day_hint: str = ""
    summary: str = ""


class PredictionResponse(BaseModel):
    """Complete prediction response with analysis and results"""
    status: str = "success"
    image_id: str
    primary_prediction: LocationPrediction
    alternative_predictions: List[LocationPrediction] = Field(default_factory=list)
    feature_analysis: Optional[FeatureAnalysis] = None
    processing_time_ms: float
    model_used: str = "ensemble"
    has_exif_gps: bool = False
    globe_regional_hints: Optional[GlobeRegionalHintsResult] = None
    coordinate_source: CoordinateSource = Field(
        default="vision_estimate",
        description="exif_gps ≈ trusting embedded coordinates; vision_estimate ≈ pixels/heuristics.",
    )
    geoposition_accuracy_note: str = Field(
        default="",
        description="Honest limitation text for this response (never implies 99.9% worldwide vision accuracy).",
    )
    scene_geolocation_cues: Optional[SceneGeolocationCues] = None
    geolocation_reading_axes: Optional[GeolocationReadingAxes] = Field(
        default=None,
        description="Short copy: view / built form / Wikipedia estimate — read with scene_geolocation_cues + external_validation.",
    )
    external_validation: Optional[ExternalValidationSummary] = None
    cross_reference_database: Optional[CrossReferenceDatabaseSummary] = None
    ml_image_recognition: Optional[MLImageRecognition] = None
    infrastructure_energy_cues: Optional[InfrastructureEnergyVisualCues] = None
    country_elimination: Optional[CountryEliminationResult] = Field(
        default=None,
        description="Rule-based elimination of impossible countries from detected cues.",
    )
    geo_reasoning: Optional[GeoReasoningResult] = Field(
        default=None,
        description="Bayesian posterior probabilities over countries from evidence fusion.",
    )
    astronomy_constraints: Optional[AstronomyConstraints] = Field(
        default=None,
        description="Latitude/hemisphere constraints derived from shadow/sun geometry.",
    )
    fast_prediction_applied: bool = False
    wikipedia_enabled_in_request: bool = Field(
        default=False,
        description="True when external validation (Wikipedia + relief + photo) was requested and not fast-skipped.",
    )
    wikipedia_place_context: Optional[WikipediaPlaceContext] = Field(
        default=None,
        description="Nearby Wikipedia/Commons articles and photo-match scores for the UI.",
    )
    from_cache: bool = Field(
        default=False,
        description="True when the response was returned from disk cache (identical image re-upload).",
    )
    timings_ms: Dict[str, float] = Field(default_factory=dict)
    inference_debug: Dict[str, Any] = Field(default_factory=dict)
    # UI enrichment (derived from inference_debug + scene bundles; not alternate coordinates)
    geoclip_ranked_predictions: Optional[List[LocationPrediction]] = Field(
        default=None,
        description="GeoCLIP rank list for the results drill-down panel.",
    )
    identified_elements: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="CLIP scene/object labels for the identified-elements panel.",
    )
    architecture_hints: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Built-form softmax groups mapped for the architecture hints panel.",
    )
    plant_geo_hints: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Vegetation scene cues (interpretive, not a verified range map).",
    )
    season_time_hints: Optional[Dict[str, Any]] = Field(default=None)
    sky_image_metrics: Optional[Dict[str, Any]] = Field(default=None)
    visual_time_of_day: Optional[Dict[str, Any]] = Field(default=None)
    flower_bush_road_hints: Optional[Dict[str, Any]] = Field(default=None)
    integrated_estimate: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Synthesized narrative for the integrated estimate card.",
    )
    inference_models: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Grouped model checklist for the primary prediction card.",
    )
    streetview_refinement: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Street View verification summary for the refinement row.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "image_id": "img_12345",
                "primary_prediction": {
                    "latitude": 48.8584,
                    "longitude": 2.2945,
                    "country": "France",
                    "city": "Paris",
                    "confidence": 0.92,
                    "distance_confidence_km": 5.0,
                },
                "alternative_predictions": [
                    {
                        "latitude": 48.8566,
                        "longitude": 2.3522,
                        "country": "France",
                        "city": "Paris",
                        "confidence": 0.85,
                        "distance_confidence_km": 8.0,
                    }
                ],
                "feature_analysis": {
                    "landmarks": [{"name": "Eiffel Tower", "confidence": 0.95}],
                    "architecture_style": "Parisian",
                    "detected_text": ["PARIS"],
                },
                "processing_time_ms": 234.5,
                "model_used": "ensemble",
                "has_exif_gps": False,
            }
        }


class ImageUploadRequest(BaseModel):
    """Request model for image upload"""

    image_url: Optional[str] = None
    base64_image: Optional[str] = None
    original_filename: Optional[str] = Field(
        default=None,
        description="Original client filename; optional hint when EXIF GPS is absent (demo).",
    )
    use_cloud_inference: bool = False
    fast_prediction: bool = Field(
        default=False,
        description="If true, skip slower optional enrichments and return the core pin faster.",
    )
    clear_prediction_cache: bool = Field(
        default=False,
        description=(
            "If true, delete any cached result for this image and run a full fresh prediction "
            "(do not return a previous cached response)."
        ),
    )
    include_llm_detective: bool = Field(
        default=True,
        description=(
            "Run local Ollama LLM detective after vision (key thoughts). "
            "Independent of Fast prediction — only needs Ollama running."
        ),
    )
    include_feature_analysis: bool = True
    include_globe_regional_hints: bool = Field(
        default=True,
        description="If true, run multi-region CLIP softmax readouts (optional torch+transformers).",
    )
    include_scene_geolocation_cues: bool = Field(
        default=True,
        description="Vegetation/architecture/palette pixel heuristics + optional CLIP softmax banks.",
    )
    include_cultural_economic_visual_cues: bool = Field(
        default=True,
        description=(
            "Optional CLIP banks for built-form / street-commerce / façade-idiom phrases (speculative; not GDP/culture)."
        ),
    )
    include_external_validation: bool = Field(
        default=True,
        description="Cross-check ensemble candidates with Wikipedia geosearch + OpenTopoData elevation grid.",
    )
    include_ml_image_recognition: bool = Field(
        default=True,
        description="CLIP softmax over curated scene/object prompts (requires torch+transformers).",
    )
    include_infrastructure_energy_cues: bool = Field(
        default=True,
        description=(
            "Gas, grid, solar, wind & visual economic-activity CLIP banks "
            "(requires torch+transformers; toggled with USE_INFRASTRUCTURE_ENERGY_CLIP server-side)."
        ),
    )
    include_reverse_geocode: bool = Field(
        default=True,
        description=(
            "Resolve pin coordinates to OpenStreetMap city/town/village name (server: REVERSE_GEOCODE_ENABLED)."
        ),
    )
    reverse_geocode_accept_language: Optional[str] = Field(
        default=None,
        description=(
            "Preferred locale for display names (e.g. uk, en-GB). If omitted, uses HTTP Accept-Language "
            "then server default — worldwide naming via Nominatim."
        ),
    )

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "example": {
                "base64_image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                "use_cloud_inference": False,
                "include_feature_analysis": True,
            }
        },
    )


class GazetteerBuildRequest(BaseModel):
    """Build StreetCLIP gazetteer JSON from a GeoNames city dump + optional country filter."""

    dump: str = Field(default="cities15000", description="cities1000 | cities5000 | cities15000")
    country_iso: str = Field(
        default="ALL",
        description="ISO-3166-1 alpha-2 (e.g. UA) or ALL / empty for worldwide rows from that dump.",
    )

