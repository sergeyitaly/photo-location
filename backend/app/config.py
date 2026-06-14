"""Configuration settings for the Photo Geolocation system"""
import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration from environment variables"""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App Info
    app_name: str = "Photo Geolocation System"
    app_version: str = "0.1.0"
    debug: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # Server
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", 8000))
    
    # ML Model Config
    model_device: str = os.getenv("MODEL_DEVICE", "cpu")  # cpu or cuda
    use_on_device_model: bool = os.getenv("USE_ON_DEVICE_MODEL", "True").lower() == "true"
    globe_clip_model_id: str = Field(
        default="openai/clip-vit-base-patch32",
        description="Hugging Face CLIP model for regional softmax cue panels.",
    )
    ensemble_use_clip_zero_shot: bool = Field(
        default=True,
        description="When torch is available, use CLIP zero-shot country + landmark softmax for ensemble geolocation.",
    )
    ml_recognition_top_n: int = Field(default=15, ge=3, le=40, description="Top softmax labels for ML recognition panel.")

    # Multi-model fusion (GeoCLIP + StreetCLIP + CLIP zero-shot)
    use_geoclip: bool = Field(
        default=True,
        description="Neural GPS via pip package geoclip (downloads weights on first run).",
    )
    use_streetclip: bool = Field(
        default=True,
        description="StreetCLIP HF model zero-shot over gazetteer JSON or embedded fallback (large ViT download).",
    )
    streetclip_model_id: str = Field(
        default="geolocal/StreetCLIP",
        description="Hugging Face repo id for StreetCLIP (ViT-L/14 @336).",
    )
    streetclip_gazetteer_chunk_size: int = Field(
        default=48,
        ge=8,
        le=96,
        description="Labels per StreetCLIP forward (VRAM trade-off).",
    )
    streetclip_early_stop_enabled: bool = Field(
        default=True,
        description=(
            "While scoring city labels in chunks, stop when a batch peaks below earlier batches "
            "(search is moving away from the best match)."
        ),
    )
    streetclip_early_stop_logit_margin: float = Field(
        default=0.35,
        ge=0.05,
        le=3.0,
        description="Chunk/country peak logit must stay within this of the global best or search stops.",
    )
    streetclip_early_stop_weak_chunks: int = Field(
        default=2,
        ge=1,
        le=8,
        description="Consecutive weak chunks (below margin) before aborting remaining city labels.",
    )
    streetclip_country_ordered_search: bool = Field(
        default=True,
        description="Score cities country-by-country (nearest countries first when GeoCLIP prior exists).",
    )
    streetclip_skip_country_on_decline: bool = Field(
        default=True,
        description="Skip remaining countries when a country's best logit is weaker than the global peak.",
    )
    streetclip_search_top_heap: int = Field(
        default=32,
        ge=8,
        le=128,
        description="How many best city logits to retain as anchors while scanning chunks.",
    )
    streetclip_gazetteer_path: str = Field(
        default="",
        description=(
            "Path to JSON array [{city, country, lat, lon, pop?}] from GeoNames (see scripts/build_streetclip_gazetteer.py). "
            "Empty = small embedded fallback only."
        ),
    )
    streetclip_gazetteer_geo_filter: bool = Field(
        default=True,
        description="When GeoCLIP rank-1 exists, restrict StreetCLIP labels to a lat/lon box around it.",
    )
    streetclip_gazetteer_bbox_lat_deg: float = Field(
        default=2.0,
        ge=0.25,
        le=45.0,
        description="Minimum half-height (deg latitude) of filter box around GeoCLIP prior (~220 km at 2°).",
    )
    streetclip_gazetteer_bbox_lon_deg: float = Field(
        default=2.5,
        ge=0.25,
        le=60.0,
        description="Minimum half-width (deg longitude; scaled by cos(lat)) of filter box around prior.",
    )
    streetclip_gazetteer_bbox_lat_max_deg: float = Field(
        default=6.0,
        ge=1.0,
        le=45.0,
        description="Upper cap on lat half-box after GeoCLIP spread expansion.",
    )
    streetclip_gazetteer_bbox_lon_max_deg: float = Field(
        default=8.0,
        ge=1.0,
        le=60.0,
        description="Upper cap on lon half-box after GeoCLIP spread expansion.",
    )
    geoclip_bbox_spread_multiplier: float = Field(
        default=1.4,
        ge=1.0,
        le=3.0,
        description="Expand bbox by this factor × max km spread among top GeoCLIP ranks.",
    )
    geoclip_bbox_spread_pad_km: float = Field(
        default=40.0,
        ge=5.0,
        le=500.0,
        description="Minimum radius (km) around GeoCLIP rank-1 when sizing the gazetteer filter box.",
    )
    gazetteer_bbox_adaptive_widen: bool = Field(
        default=True,
        description="If geo-filter yields too few cities, widen the box in steps until enough rows exist.",
    )
    gazetteer_bbox_min_rows_after_filter: int = Field(
        default=60,
        ge=10,
        le=5000,
        description="Target minimum gazetteer rows before StreetCLIP / grid scoring.",
    )
    streetclip_gazetteer_prioritize_distance: bool = Field(
        default=True,
        description="When trimming to max_labels inside the bbox, prefer nearest cities to GeoCLIP prior.",
    )
    streetclip_country_filter_enabled: bool = Field(
        default=True,
        description="Restrict StreetCLIP gazetteer rows to top CLIP country softmax hypotheses.",
    )
    streetclip_country_filter_max_countries: int = Field(
        default=3,
        ge=1,
        le=8,
        description="How many CLIP countries to keep in the gazetteer allowlist.",
    )
    streetclip_country_filter_min_confidence: float = Field(
        default=0.012,
        ge=0.0,
        le=0.5,
        description="Minimum CLIP country softmax prob to include in allowlist.",
    )
    streetclip_country_filter_min_rows: int = Field(
        default=40,
        ge=10,
        le=2000,
        description="If country filter leaves fewer rows, fall back to bbox-only trim.",
    )
    streetclip_confident_margin_threshold: float = Field(
        default=0.10,
        ge=0.02,
        le=0.5,
        description="StreetCLIP top1−top2 margin above which GeoCLIP weight is reduced in fusion.",
    )
    geoclip_downweight_when_streetclip_confident: float = Field(
        default=0.55,
        ge=0.2,
        le=1.0,
        description="Multiply GeoCLIP fusion weight when StreetCLIP margin is strong.",
    )
    streetclip_boost_when_confident: float = Field(
        default=1.12,
        ge=1.0,
        le=2.0,
        description="Multiply StreetCLIP fusion weight when margin is strong.",
    )
    geoclip_scatter_spread_km_threshold: float = Field(
        default=350.0,
        ge=50.0,
        le=3000.0,
        description="Down-weight GeoCLIP when top ranks disagree beyond this distance.",
    )
    geoclip_downweight_when_scattered: float = Field(
        default=0.72,
        ge=0.2,
        le=1.0,
        description="GeoCLIP weight multiplier when top hypotheses are geographically scattered.",
    )
    fast_mode_confidence_gated_grid: bool = Field(
        default=True,
        description="In fast mode, run a reduced grid search when GeoCLIP/CLIP country confidence is weak.",
    )
    fast_grid_geoclip_max_confidence: float = Field(
        default=0.14,
        ge=0.05,
        le=0.5,
        description="Run fast grid when GeoCLIP top-1 confidence is at or below this.",
    )
    fast_grid_geoclip_spread_km: float = Field(
        default=100.0,
        ge=20.0,
        le=800.0,
        description="Run fast grid when top GeoCLIP ranks spread beyond this (km).",
    )
    fast_grid_clip_country_max_confidence: float = Field(
        default=0.10,
        ge=0.03,
        le=0.4,
        description="Run fast grid when best CLIP country softmax is at or below this.",
    )
    fast_grid_top_coarse_cells: int = Field(
        default=4,
        ge=2,
        le=12,
        description="Coarse cells in fast confidence-gated grid (smaller than full pipeline).",
    )
    fast_grid_top_fine_cells: int = Field(
        default=4,
        ge=2,
        le=12,
        description="Fine cells in fast confidence-gated grid.",
    )
    feature_analysis_clip_landmarks: bool = Field(
        default=True,
        description="Run CLIP landmark softmax into feature_analysis.landmarks.",
    )
    feature_analysis_clip_architecture: bool = Field(
        default=True,
        description="Run CLIP architecture style hint into feature_analysis.architecture_style.",
    )
    feature_analysis_ocr_enabled: bool = Field(
        default=True,
        description="Run Tesseract OCR for detected_text when pytesseract is installed.",
    )
    place_promote_named_primary: bool = Field(
        default=False,
        description=(
            "If true, replace a GeoCLIP-labelled primary with a nearby named city only when that "
            "city has higher fusion confidence than the primary (default off — keeps highest score)."
        ),
    )
    place_promote_max_distance_km: float = Field(
        default=85.0,
        ge=5.0,
        le=500.0,
        description="Max distance (km) for optional named-primary promotion.",
    )
    place_promote_min_confidence_ratio: float = Field(
        default=1.0,
        ge=1.0,
        le=2.0,
        description="Named alt must meet primary_confidence × this ratio to replace a GeoCLIP primary.",
    )
    streetclip_gazetteer_max_labels: int = Field(
        default=6000,
        ge=200,
        le=200000,
        description="Hard cap on labels scored per image after geo-filter (population / distance trim).",
    )
    streetclip_gazetteer_min_population: int = Field(
        default=0,
        ge=0,
        description="If >0 and gazetteer rows include pop, drop smaller settlements before trim.",
    )
    gazetteer_build_enabled: bool = Field(
        default=True,
        description="Expose POST /gazetteer/build (GeoNames download). Disable on untrusted networks.",
    )
    streetclip_gazetteer_autoload_at_startup: bool = Field(
        default=True,
        description=(
            "Background-download GeoNames city dump + build StreetCLIP gazetteer JSON when the server starts. "
            "Uses streetclip_gazetteer_autoload_dump (worldwide). Poll /config gazetteer_autoload for phase."
        ),
    )
    streetclip_gazetteer_autoload_dump: str = Field(
        default="cities1000",
        description="GeoNames dump key: cities1000 | cities5000 | cities15000 (worldwide ALL).",
    )
    streetclip_gazetteer_autoload_skip_if_exists: bool = Field(
        default=True,
        description="If autoload JSON already on disk, skip download/build on startup.",
    )
    gazetteer_data_dir: str = Field(
        default="",
        description="Optional absolute path for generated gazetteer JSON (default: app/data/generated).",
    )
    faiss_geotag_index_path: str = Field(
        default="",
        description=(
            "Optional Faiss index file built offline from CLIP embeddings (same model as globe_clip_model_id). "
            "Requires faiss-cpu and matching faiss_geotag_coords_npy_path."
        ),
    )
    faiss_geotag_coords_npy_path: str = Field(
        default="",
        description="Optional float32 .npy shaped (N, 2) with lat, lon aligned to Faiss vectors.",
    )
    fusion_weight_geoclip: float = Field(default=0.45, ge=0.0, le=1.0)
    fusion_weight_streetclip: float = Field(default=0.35, ge=0.0, le=1.0)
    fusion_weight_clip_zs: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Weight for base CLIP country + landmark softmax ensemble.",
    )
    use_multi_resolution_grid_search: bool = Field(
        default=True,
        description="Run a coarse-to-fine StreetCLIP grid search over the gazetteer before city refinement.",
    )
    fusion_weight_grid_search: float = Field(
        default=0.22,
        ge=0.0,
        le=1.0,
        description="Weight for the multi-resolution gazetteer grid-search source in fusion.",
    )
    grid_search_coarse_lat_deg: float = Field(
        default=12.0,
        ge=1.0,
        le=90.0,
        description="Coarse grid latitude cell height when no GeoCLIP prior (worldwide search).",
    )
    grid_search_coarse_lon_deg: float = Field(
        default=12.0,
        ge=1.0,
        le=180.0,
        description="Coarse grid longitude cell width when no GeoCLIP prior.",
    )
    grid_search_fine_lat_deg: float = Field(
        default=2.0,
        ge=0.25,
        le=30.0,
        description="Fine grid latitude cell height when no GeoCLIP prior.",
    )
    grid_search_fine_lon_deg: float = Field(
        default=2.0,
        ge=0.25,
        le=30.0,
        description="Fine grid longitude cell width when no GeoCLIP prior.",
    )
    grid_search_prior_coarse_lat_deg: float = Field(
        default=1.5,
        ge=0.25,
        le=30.0,
        description="Coarse grid cell height when a GeoCLIP prior bbox is active.",
    )
    grid_search_prior_coarse_lon_deg: float = Field(
        default=1.5,
        ge=0.25,
        le=30.0,
        description="Coarse grid cell width when a GeoCLIP prior bbox is active.",
    )
    grid_search_prior_fine_lat_deg: float = Field(
        default=0.35,
        ge=0.1,
        le=10.0,
        description="Fine grid cell height with GeoCLIP prior (~40 km).",
    )
    grid_search_prior_fine_lon_deg: float = Field(
        default=0.35,
        ge=0.1,
        le=10.0,
        description="Fine grid cell width with GeoCLIP prior.",
    )
    grid_search_top_coarse_cells: int = Field(
        default=6,
        ge=1,
        le=24,
        description="How many coarse grid cells advance to the fine search stage.",
    )
    grid_search_top_fine_cells: int = Field(
        default=8,
        ge=1,
        le=32,
        description="How many fine grid cells advance to city-level refinement.",
    )
    grid_search_representatives_per_cell: int = Field(
        default=3,
        ge=1,
        le=8,
        description="Representative gazetteer labels scored for each grid cell.",
    )
    grid_search_city_limit_per_fine_cell: int = Field(
        default=24,
        ge=4,
        le=128,
        description="Max city labels scored inside each winning fine cell.",
    )

    hybrid_streetclip_alt_geoclip_reconcile: bool = Field(
        default=True,
        description=(
            "After fusion, if StreetCLIP top-1 names a gazetteer city, prefer the GeoCLIP rank whose "
            "GPS is closest to that city when rank-1 GPS is farther (capital-vs-regional mismatch)."
        ),
    )
    hybrid_alt_geoclip_scan_top: int = Field(default=8, ge=2, le=24)
    hybrid_alt_geoclip_min_improvement_km: float = Field(
        default=35.0,
        ge=0.0,
        description="Min km closer to StreetCLIP city centroid vs GeoCLIP rank-1 to allow promotion.",
    )
    hybrid_alt_geoclip_min_rank_sep_km: float = Field(
        default=28.0,
        ge=0.0,
        description="Min separation km between GeoCLIP rank-1 and promoted rank (avoid jitter).",
    )
    hybrid_alt_geoclip_min_softmax_alt_conf: float = Field(
        default=0.035,
        ge=0.0,
        le=1.0,
        description="StreetCLIP softmax floor on top-1 city before hybrid promotion.",
    )
    hybrid_alt_geoclip_min_geoclip_rank_conf: float = Field(
        default=0.006,
        ge=0.0,
        le=1.0,
        description="GeoCLIP probability floor on the promoted rank.",
    )
    fusion_open_water_streetclip_boost: bool = Field(
        default=True,
        description=(
            "Slightly increase StreetCLIP fusion weight when open water is visible (any region). "
            "Does not force a specific city — only reweights gazetteer vs GPS models."
        ),
    )
    fusion_open_water_fraction_threshold: float = Field(default=0.08, ge=0.02, le=0.5)
    fusion_open_water_streetclip_weight_multiplier: float = Field(
        default=1.06,
        ge=1.0,
        le=1.35,
        description="Multiply StreetCLIP fusion weight when open-water pixels exceed threshold.",
    )
    fusion_dedupe_decimals: int = Field(
        default=3,
        ge=2,
        le=6,
        description=(
            "Lat/lon decimal places when merging fusion candidates from different sources. "
            "3 ≈ ~100 m latitude bins — finer than 2 (~1 km), reduces wrongful merging of nearby hypotheses."
        ),
    )
    geoclip_merge_max_ranks: int = Field(
        default=16,
        ge=6,
        le=48,
        description=(
            "How many GeoCLIP gallery ranks to retrieve and merge into weighted fusion "
            "(more ranks compete with StreetCLIP / CLIP-ZS for the primary pin)."
        ),
    )

    geoposition_note_exif: str = Field(
        default=(
            "Coordinates come from embedded GPS (EXIF), not from scene analysis. "
            "Primary confidence 0.999 reflects trust in those tags when present; "
            "real-world handset error is often ~5–50 m and metadata can be missing or edited."
        ),
    )
    geoposition_note_filename: str = Field(
        default=(
            "Location is inferred from the filename keyword list only (demo heuristic). "
            "It is not visual geolocation and is not suitable as ground truth."
        ),
    )
    geoposition_note_vision: str = Field(
        default=(
            "Estimated from pixels / internal models — not verified ground truth. "
            "Region-scale cues (climate, relief, broad built form, infrastructure style) are often useful; "
            "exact village or street-level labels are unverified and can be wrong by tens to hundreds of km. "
            "Confirm the pin in satellite and street imagery (road layout, poles, pipes, roof spacing) before trusting a name. "
            "Use EXIF GPS when you need trustworthy coordinates."
        ),
    )
    
    # Feature Extraction
    enable_landmark_detection: bool = True
    enable_vegetation_analysis: bool = True
    enable_architecture_analysis: bool = True
    enable_text_ocr: bool = True
    
    # Inference
    confidence_threshold: float = 0.3
    top_k_predictions: int = Field(
        default=8,
        ge=3,
        le=16,
        description="Fusion alternatives returned; raise for richer GeoCLIP rank coverage + hybrid reconcile.",
    )
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./geolocations.db")
    
    # API Keys (optional for enhanced features)
    google_maps_api_key: str = os.getenv("GOOGLE_MAPS_API_KEY", "")

    # OpenStreetMap Nominatim — resolve any pin to city/town/village (worldwide, ODbL)
    reverse_geocode_enabled: bool = Field(
        default=True,
        description="After prediction, call Nominatim reverse to name the place at lat/lon (any settlement).",
    )
    reverse_geocode_max_alternatives: int = Field(
        default=0,
        ge=0,
        le=8,
        description="How many alternative pins to resolve (sequential; add delay to respect OSM policy).",
    )
    reverse_geocode_inter_request_delay_s: float = Field(
        default=1.1,
        ge=0.0,
        description="Pause between Nominatim calls (public instance: be polite; use your own Nominatim for batch).",
    )
    reverse_geocode_timeout_s: float = Field(default=10.0, ge=2.0, le=60.0)
    reverse_geocode_retry_attempts: int = Field(
        default=2,
        ge=0,
        le=8,
        description="Retry count for transient Nominatim failures (502/503/504/timeouts); actual tries = 1 + this.",
    )
    reverse_geocode_retry_backoff_s: float = Field(
        default=0.75,
        ge=0.0,
        le=30.0,
        description="Base delay for exponential backoff between retries (worldwide CDN / self-hosted variance).",
    )
    reverse_geocode_default_accept_language: str = Field(
        default="en",
        description="Fallback Nominatim accept-language when client sends none (use en, local, or multivalue).",
    )
    nominatim_base_url: str = Field(
        default="https://nominatim.openstreetmap.org",
        description="Nominatim API base (set to your self-hosted instance for production / high volume).",
    )
    nominatim_http_user_agent: str = Field(
        default="",
        description="Required: identify your app to OSM. If empty, app_name + version are used (set a real contact in production).",
    )
    nominatim_reverse_zoom: int = Field(
        default=14,
        ge=3,
        le=18,
        description="Nominatim reverse detail level (higher = more local).",
    )

    # Wikipedia + OpenTopoData cross-check (ensemble candidates only)
    wikipedia_geosearch_radius_m: int = Field(default=10000, description="MediaWiki geosearch radius in meters.")
    wikipedia_validation_max_distance_m: int = Field(
        default=12000,
        description="Nearest article must be within this distance (m) to count as proven.",
    )
    wikipedia_min_articles_for_proof: int = Field(default=1, ge=1)
    wikipedia_require_title_city_match: bool = Field(
        default=False,
        description="If true, nearest article titles must loosely match predicted city name (stricter).",
    )
    wikipedia_semantic_gate_enabled: bool = Field(
        default=True,
        description=(
            "After geosearch + relief, require CLIP similarity vs English Wikipedia leads (nearest titles); "
            "otherwise try the next fusion candidate until proof or exhaustion."
        ),
    )
    wikipedia_semantic_min_similarity: float = Field(
        default=0.14,
        ge=0.0,
        le=1.0,
        description="Minimum mapped CLIP image–text score vs best Wikipedia lead among scanned titles.",
    )
    wikipedia_semantic_eval_top_titles: int = Field(
        default=5,
        ge=1,
        le=12,
        description="How many nearest geo-articles per candidate to extract and score with CLIP (best wins).",
    )
    wikipedia_photo_gate_enabled: bool = Field(
        default=True,
        description=(
            "After geosearch + relief + text CLIP, require CLIP image similarity vs nearby "
            "Wikimedia Commons / Wikipedia lead photos."
        ),
    )
    wikipedia_photo_min_similarity: float = Field(
        default=0.68,
        ge=0.0,
        le=1.0,
        description="Minimum CLIP cosine similarity between upload and best nearby Wikimedia image.",
    )
    wikipedia_photo_max_files: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Max Commons geosearch files to fetch and score per candidate.",
    )
    wikipedia_photo_max_article_images: int = Field(
        default=3,
        ge=0,
        le=8,
        description="Max Wikipedia article lead images (pageimages) to score per candidate.",
    )
    wikipedia_photo_thumb_width: int = Field(
        default=640,
        ge=200,
        le=1280,
        description="Thumbnail width when downloading Wikimedia images for CLIP comparison.",
    )
    wikipedia_commons_geosearch_radius_m: int = Field(
        default=10000,
        description="Commons File: namespace geosearch radius (meters) around candidate pin.",
    )
    opentopodata_dataset: str = Field(default="srtm90m", description="OpenTopoData dataset id (e.g. srtm90m).")
    opentopodata_grid_step_deg: float = Field(default=0.02, description="Spacing for 3×3 relief grid in degrees.")
    opentopodata_min_samples: int = Field(default=4, ge=1, description="Min valid elevation samples to accept relief row.")
    opentopodata_min_request_interval_s: float = Field(
        default=1.15,
        ge=0.5,
        le=10.0,
        description="Min seconds between OpenTopoData requests (free tier ≈1 req/s).",
    )
    wikipedia_min_request_interval_s: float = Field(
        default=0.28,
        ge=0.05,
        le=5.0,
        description="Min seconds between English Wikipedia API requests.",
    )
    commons_min_request_interval_s: float = Field(
        default=0.35,
        ge=0.05,
        le=5.0,
        description="Min seconds between Wikimedia Commons API requests.",
    )
    outbound_http_default_interval_s: float = Field(
        default=0.2,
        ge=0.0,
        le=5.0,
        description="Default min spacing for other outbound GET hosts.",
    )
    outbound_http_429_max_retries: int = Field(
        default=5,
        ge=0,
        le=12,
        description="Retries after HTTP 429 before giving up on a single URL.",
    )
    outbound_http_429_backoff_base_s: float = Field(
        default=2.0,
        ge=0.5,
        le=30.0,
        description="Exponential backoff base (seconds) for 429/5xx retries.",
    )
    outbound_http_cache_ttl_s: float = Field(
        default=900.0,
        ge=0.0,
        le=86400.0,
        description="TTL for cached Wikipedia/OpenTopo JSON responses (same coords reuse).",
    )
    outbound_http_circuit_trip_after_429: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Consecutive 429s on one host before pausing all requests to that host.",
    )
    outbound_http_circuit_cooldown_s: float = Field(
        default=90.0,
        ge=10.0,
        le=600.0,
        description="Seconds to pause a host after the circuit breaker trips.",
    )
    external_validation_max_candidates: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Max fusion candidates to hit Wikipedia/OpenTopo (rest are skipped).",
    )
    external_validation_coord_cache_decimals: int = Field(
        default=3,
        ge=2,
        le=5,
        description="Round lat/lon for deduping identical geosearch/relief lookups (~111 m at 3).",
    )

    # Local gazetteer/database cross-reference
    use_cross_reference_database: bool = Field(
        default=True,
        description="Cross-check fused candidates against the local gazetteer database before open-data validation.",
    )
    cross_reference_search_radius_km: float = Field(
        default=80.0,
        ge=5.0,
        le=500.0,
        description="Radius used to look for nearby gazetteer places around each candidate.",
    )
    cross_reference_nearest_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="How many nearby gazetteer rows to inspect per candidate.",
    )
    cross_reference_promote_min_score_delta: float = Field(
        default=0.12,
        ge=0.0,
        le=2.0,
        description="Min support-score improvement vs candidate #0 before promoting another candidate.",
    )

    # Infrastructure / energy / visual economic proxies (CLIP banks)
    use_infrastructure_energy_clip: bool = Field(
        default=True,
        description=(
            "CLIP softmax banks for gas/power/solar/wind/grid cues and visual economic-activity proxies "
            "(see USE_INFRASTRUCTURE_ENERGY_CLIP in .env.example)."
        ),
    )
    infrastructure_energy_clip_top_n: int = Field(
        default=8,
        ge=3,
        le=20,
        description="Top softmax labels returned per infrastructure-energy bank.",
    )

    # Geo-Reasoning Engine (new — rule-based + Bayesian + astronomy)
    use_country_elimination: bool = Field(
        default=True,
        description="Rule-based country elimination from detected cues (script, climate, poles, road markings).",
    )
    use_bayesian_reasoning: bool = Field(
        default=True,
        description="Bayesian evidence fusion over country hypotheses with priors and contradiction penalties.",
    )
    use_astronomy_solver: bool = Field(
        default=True,
        description="Derive latitude/hemisphere constraints from shadow angles and sun elevation (pure math).",
    )
    use_specialist_detectors: bool = Field(
        default=True,
        description="Run pixel-heuristic detectors for utility poles, road lines, and shadow geometry.",
    )
    reasoning_fusion_boost_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight given to Bayesian reasoning when re-ranking fusion candidates (0=ignore).",
    )
    reasoning_latitude_penalty_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight for penalizing fusion candidates outside astronomy-derived latitude bounds.",
    )

    preload_torch_models_at_startup: bool = Field(
        default=True,
        description=(
            "At FastAPI startup, load configured CLIP + GeoCLIP + StreetCLIP weights into RAM "
            "(longer boot; first /predict stays fast). Set PRELOAD_TORCH_MODELS_AT_STARTUP=false to skip."
        ),
    )

    use_satellite_matching: bool = Field(
        default=True,
        description="Run satellite tile reverse-match on full (non-fast) vision predictions.",
    )
    use_streetview_verification: bool = Field(
        default=True,
        description="Run Street View CLIP verification on full (non-fast) vision predictions.",
    )

    # Service timeouts (seconds) — prevent pipeline hangs
    satellite_match_timeout_seconds: float = Field(
        default=12.0,
        ge=3.0,
        le=60.0,
        description="Hard timeout for satellite tile fetch + comparison per candidate.",
    )
    streetview_verify_timeout_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=120.0,
        description="Hard timeout for Street View verification (primary + alternatives).",
    )
    use_llm_detective: bool = Field(
        default=True,
        description="Run optional local Ollama reasoning step after vision fusion.",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama HTTP API base URL (no trailing slash).",
    )
    ollama_model: str = Field(
        default="qwen2.5:7b",
        description="Ollama model tag for the detective layer (e.g. qwen2.5:7b, llama3.2:3b).",
    )
    ollama_warmup_at_startup: bool = Field(
        default=True,
        description="On server boot, ping Ollama with a 1-token request so the model is loaded before /predict.",
    )
    ollama_http_timeout_seconds: float = Field(
        default=180.0,
        ge=30.0,
        le=600.0,
        description="HTTP client timeout for a single Ollama chat request (includes cold model load on CPU).",
    )
    llm_detective_timeout_seconds: float = Field(
        default=240.0,
        ge=15.0,
        le=600.0,
        description="Pipeline wait for the full LLM detective step (should be ≤ ollama_http_timeout_seconds).",
    )
    external_validation_timeout_seconds: float = Field(
        default=120.0,
        ge=15.0,
        le=300.0,
        description="Hard timeout for Wikipedia + OpenTopoData cross-check (throttled requests need more time).",
    )
    reverse_geocode_batch_timeout_seconds: float = Field(
        default=15.0,
        ge=3.0,
        le=60.0,
        description="Hard timeout for Nominatim reverse-geocode of primary + alternatives.",
    )
    predict_endpoint_timeout_seconds: float = Field(
        default=300.0,
        ge=30.0,
        le=600.0,
        description="Overall server-side ceiling for POST /predict (should match frontend AbortController).",
    )

    # Prediction result cache (disk-based, keyed by SHA-256 of image bytes)
    use_prediction_cache: bool = Field(
        default=True,
        description="Cache prediction responses on disk so identical image re-uploads return instantly.",
    )
    prediction_cache_dir: str = Field(
        default="",
        description="Absolute path for prediction cache JSON files. Empty = app/data/generated/prediction_cache.",
    )
    prediction_cache_ttl_seconds: int = Field(
        default=86400,
        ge=60,
        le=2592000,
        description="How long (seconds) to keep a cached prediction before treating it as stale.",
    )
    prediction_cache_max_entries: int = Field(
        default=1000,
        ge=10,
        le=10000,
        description="Max number of cached prediction files before oldest are evicted.",
    )

    def outbound_http_headers(self) -> dict[str, str]:
        """
        Identifies this app to public HTTP APIs (Wikimedia, OpenTopoData, NASA GIBS, …).
        Wikimedia returns 403 without a descriptive User-Agent; set NOMINATIM_HTTP_USER_AGENT to a real contact.
        """
        ua = (self.nominatim_http_user_agent or "").strip()
        if not ua:
            ua = (
                f"{self.app_name}/{self.app_version} "
                "(photo-geolocation; configure NOMINATIM_HTTP_USER_AGENT with app URL or email)"
            )
        return {"User-Agent": ua}


settings = Settings()
