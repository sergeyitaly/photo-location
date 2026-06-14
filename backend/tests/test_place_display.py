"""Place display labels for UI."""

from app.models.schemas import LocationPrediction, PlaceResolution
from app.services.place_display import (
    display_place_label,
    enrich_prediction_display_labels,
    is_geoclip_placeholder,
    is_named_gazetteer_place,
    promote_named_primary_if_available,
    sort_predictions_by_confidence,
)


def test_geoclip_placeholder():
    assert is_geoclip_placeholder("GeoCLIP rank 1", "GeoCLIP gallery")
    assert not is_geoclip_placeholder("Kyiv", "Ukraine")


def test_country_only_centroid_not_named_city():
    assert not is_named_gazetteer_place("Italy", "Italy")
    assert is_named_gazetteer_place("Como", "Italy")


def test_display_named_place():
    pred = LocationPrediction(
        latitude=50.45,
        longitude=30.52,
        country="Ukraine",
        city="Kharkivskyi Masyv",
        confidence=0.35,
    )
    assert display_place_label(pred) == "Kharkivskyi Masyv, Ukraine"


def test_display_with_osm_resolution():
    pred = LocationPrediction(
        latitude=50.45,
        longitude=30.52,
        country="GeoCLIP gallery",
        city="GeoCLIP rank 1",
        confidence=0.45,
        place_resolution=PlaceResolution(
            locality="Kyiv",
            locality_kind="city",
            country="Ukraine",
            display_name="Kyiv, Ukraine",
            source="openstreetmap_nominatim",
        ),
    )
    assert "Kyiv" in display_place_label(pred)


def test_sort_predictions_puts_highest_confidence_first():
    primary = LocationPrediction(
        latitude=45.79,
        longitude=8.41,
        country="Italy",
        city="Como",
        confidence=0.066,
    )
    alt = LocationPrediction(
        latitude=45.80,
        longitude=8.41,
        country="Italy",
        city="Orta San Giulio",
        confidence=0.118,
    )
    new_primary, alts = sort_predictions_by_confidence(primary, [alt])
    assert new_primary.city == "Orta San Giulio"
    assert alts[0].city == "Como"


def test_promote_does_not_swap_when_named_alt_is_weaker():
    primary = LocationPrediction(
        latitude=50.451,
        longitude=30.522,
        country="GeoCLIP gallery",
        city="GeoCLIP rank 1",
        confidence=0.45,
    )
    alt = LocationPrediction(
        latitude=50.408,
        longitude=30.660,
        country="Ukraine",
        city="Kharkivskyi Masyv",
        confidence=0.35,
    )
    promoted, alts = promote_named_primary_if_available(primary, [alt])
    assert promoted.city == "GeoCLIP rank 1"
    assert alts[0].city == "Kharkivskyi Masyv"


def test_enrich_geoclip_placeholder_shows_coords_not_gallery():
    pred = LocationPrediction(
        latitude=45.7985,
        longitude=8.4051,
        country="GeoCLIP gallery",
        city="GeoCLIP rank 1",
        confidence=0.074,
    )
    out = enrich_prediction_display_labels(pred)
    assert "GeoCLIP gallery" not in (out.country or "")
    assert "45.80" in (out.city or "")
    assert "rank 1" in (out.country or "").lower()


def test_promote_with_stronger_named_alt_sorts_it_to_primary():
    """Higher-confidence named alts become primary via internal sort (promote is a no-op)."""
    primary = LocationPrediction(
        latitude=50.451,
        longitude=30.522,
        country="GeoCLIP gallery",
        city="GeoCLIP rank 1",
        confidence=0.40,
    )
    alt = LocationPrediction(
        latitude=50.452,
        longitude=30.523,
        country="Ukraine",
        city="Kharkivskyi Masyv",
        confidence=0.55,
    )
    promoted, alts = promote_named_primary_if_available(primary, [alt])
    assert promoted.city == "Kharkivskyi Masyv"
    assert promoted.confidence == 0.55
    assert any(a.city == "GeoCLIP rank 1" for a in alts)
