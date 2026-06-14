"""Static datasets and prompt banks."""

from app.data.geolocation_feature_catalog import (
    GEOLOCATION_FEATURE_CATALOG_VERSION,
    geolocation_feature_catalog_count,
    geolocation_feature_catalog_version,
    get_geolocation_feature_catalog,
)

__all__ = [
    "GEOLOCATION_FEATURE_CATALOG_VERSION",
    "geolocation_feature_catalog_count",
    "geolocation_feature_catalog_version",
    "get_geolocation_feature_catalog",
]
