"""UI response enrichment helpers."""

from app.models.schemas import LocationPrediction
from app.services.ui_response_enrichment import (
    build_geoclip_ranked_predictions,
    build_identified_elements,
    build_integrated_estimate,
    enrich_prediction_ui_fields,
)


def test_geoclip_ranked_from_inference_debug():
    dbg = {
        "source_predictions": {
            "geoclip": [
                {
                    "latitude": 45.8,
                    "longitude": 9.1,
                    "country": "Italy",
                    "city": "Como area",
                    "confidence": 0.12,
                }
            ]
        }
    }
    ranks = build_geoclip_ranked_predictions(dbg)
    assert len(ranks) == 1
    assert ranks[0].country == "Italy"


def test_enrich_prediction_ui_fields_minimal():
    primary = LocationPrediction(
        latitude=45.8,
        longitude=9.1,
        country="Italy",
        city="Como",
        confidence=0.4,
    )
    fields = enrich_prediction_ui_fields(
        primary_prediction=primary,
        scene_geolocation_cues=None,
        geolocation_reading_axes=None,
        external_validation=None,
        ml_image_recognition=None,
        inference_debug={"fusion_sources": ["geoclip", "streetclip"]},
        model_used="fusion[geoclip+streetclip]",
    )
    assert fields["integrated_estimate"] is not None
    assert len(fields["inference_models"]) >= 2
