"""Inference engines for geolocation prediction (lazy imports — avoids loading torch at import)."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["CountryClassifier", "ImageRetrieval", "EnsembleInference"]

if TYPE_CHECKING:
    from .classifier import CountryClassifier
    from .ensemble import EnsembleInference
    from .retrieval import ImageRetrieval


def __getattr__(name: str):
    if name == "CountryClassifier":
        from .classifier import CountryClassifier

        return CountryClassifier
    if name == "ImageRetrieval":
        from .retrieval import ImageRetrieval

        return ImageRetrieval
    if name == "EnsembleInference":
        from .ensemble import EnsembleInference

        return EnsembleInference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
