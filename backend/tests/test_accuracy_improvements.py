"""Unit tests for accuracy-oriented inference helpers."""

from app.inference.country_gazetteer import (
    country_names_match,
    gazetteer_country_allowlist,
    row_country_in_allowlist,
    top_clip_countries,
)
from app.inference.fusion_tuning import (
    should_run_fast_confidence_grid,
    streetclip_confidence_margin,
    tune_fusion_source_weights,
)
from app.models.schemas import LocationPrediction
from app.config import Settings


def test_country_alias_match():
    assert country_names_match("USA", "United States")
    assert country_names_match("UK", "United Kingdom")
    assert country_names_match("Russia", "Russian Federation")
    assert not country_names_match("France", "Japan")


def test_gazetteer_country_allowlist():
    settings = Settings()
    preds = [
        LocationPrediction(
            latitude=48.8,
            longitude=2.3,
            country="France",
            city="France",
            confidence=0.08,
        ),
        LocationPrediction(
            latitude=52.5,
            longitude=13.4,
            country="Germany",
            city="Germany",
            confidence=0.04,
        ),
    ]
    allow = gazetteer_country_allowlist(preds, settings)
    assert allow is not None
    assert "France" in allow
    assert row_country_in_allowlist("France", allow)
    assert not row_country_in_allowlist("Japan", allow)


def test_top_clip_countries_dedupes():
    preds = [
        LocationPrediction(latitude=0, longitude=0, country="Spain", city="Spain", confidence=0.2),
        LocationPrediction(latitude=0, longitude=0, country="Spain", city="Spain", confidence=0.1),
    ]
    assert top_clip_countries(preds, max_countries=3) == ["Spain"]


def test_streetclip_margin():
    preds = [
        LocationPrediction(latitude=0, longitude=0, country="X", city="A", confidence=0.4),
        LocationPrediction(latitude=0, longitude=0, country="X", city="B", confidence=0.2),
    ]
    assert streetclip_confidence_margin(preds) == 0.2


def test_tune_fusion_downweights_geoclip_when_streetclip_confident():
    settings = Settings()
    geo = [
        LocationPrediction(latitude=51.5, longitude=-0.1, country="G", city="g1", confidence=0.2),
    ]
    sc = [
        LocationPrediction(latitude=51.5, longitude=-0.1, country="UK", city="London", confidence=0.5),
        LocationPrediction(latitude=48.8, longitude=2.3, country="FR", city="Paris", confidence=0.2),
    ]
    sources = [
        ("geoclip", 0.45, geo),
        ("streetclip", 0.35, sc),
    ]
    tuned = tune_fusion_source_weights(sources, geo_preds=geo, sc_preds=sc, settings=settings)
    geo_w = next(w for n, w, _ in tuned if n == "geoclip")
    sc_w = next(w for n, w, _ in tuned if n == "streetclip")
    assert geo_w < 0.45
    assert sc_w > 0.35


def test_fast_grid_trigger_on_low_geoclip_confidence():
    settings = Settings()
    geo = [
        LocationPrediction(latitude=40.0, longitude=-74.0, country="G", city="g1", confidence=0.11),
    ]
    assert should_run_fast_confidence_grid(
        fast=True,
        geo_preds=geo,
        country_predictions=[],
        settings=settings,
    )
