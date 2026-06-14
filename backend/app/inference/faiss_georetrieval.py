"""
Optional FAISS nearest-neighbour search over (CLIP embedding → lat/lon).

Build the index offline with the **same** CLIP model as ``globe_clip_model_id`` so dimensions match.

Expected files (see settings):
  - ``FAISS_GEOTAG_INDEX_PATH``: Faiss index file (e.g. IndexFlatIP or IVF)
  - ``FAISS_GEOTAG_COORDS_NPY_PATH``: float32 array shape (N, 2) with [lat, lon] per vector

If either path is missing or ``faiss`` is not installed, retrieval falls back to empty.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.config import Settings

logger = logging.getLogger(__name__)


def _faiss_mod():
    try:
        import faiss  # type: ignore

        return faiss
    except ImportError:
        return None


@lru_cache(maxsize=2)
def _load_bundle_paths(index_resolved: str, coords_resolved: str) -> Optional[Tuple[Any, np.ndarray, int]]:
    """Load FAISS index + coordinate table; cached by resolved paths."""
    faiss = _faiss_mod()
    if faiss is None:
        return None
    try:
        index = faiss.read_index(index_resolved)
        coords = np.load(coords_resolved).astype(np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2:
            logger.warning("faiss_georetrieval: coords must be (N, 2) lat/lon")
            return None
        n = int(index.ntotal)
        if coords.shape[0] != n:
            logger.warning(
                "faiss_georetrieval: index.ntotal=%s != coords rows=%s",
                n,
                coords.shape[0],
            )
            return None
        return (index, coords, n)
    except Exception as e:
        logger.warning("faiss_georetrieval: failed to load bundle: %s", e)
        return None


def load_faiss_bundle(settings: Settings) -> Optional[Tuple[Any, np.ndarray, int]]:
    """Returns (faiss_index, coords (N,2) float64, ntotal) or None."""
    idx_path = (getattr(settings, "faiss_geotag_index_path", None) or "").strip()
    coord_path = (getattr(settings, "faiss_geotag_coords_npy_path", None) or "").strip()
    if not idx_path or not coord_path:
        return None
    if _faiss_mod() is None:
        logger.debug("faiss_georetrieval: faiss not installed (pip install faiss-cpu)")
        return None

    ip = Path(idx_path).expanduser().resolve()
    cp = Path(coord_path).expanduser().resolve()
    if not ip.is_file() or not cp.is_file():
        logger.info("faiss_georetrieval: index or coords file missing — skipping ANN retrieval")
        return None

    return _load_bundle_paths(str(ip), str(cp))


def search_neighbors(
    settings: Settings,
    query_embedding: np.ndarray,
    *,
    k: int = 5,
) -> List[Tuple[float, float, float]]:
    """
    Returns list of (lat, lon, score) for k neighbours. Score is inner-product when index is IP;
    not calibrated to probability.
    """
    bundle = load_faiss_bundle(settings)
    if bundle is None:
        return []
    index, coords, _nt = bundle
    faiss = _faiss_mod()
    if faiss is None:
        return []

    q = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    if q.shape[1] != index.d:
        logger.warning(
            "faiss_georetrieval: embedding dim %s != index dim %s",
            q.shape[1],
            index.d,
        )
        return []

    faiss.normalize_L2(q)
    sims, idxs = index.search(q, min(k, index.ntotal))
    out: List[Tuple[float, float, float]] = []
    for j in range(idxs.shape[1]):
        ii = int(idxs[0, j])
        if ii < 0:
            continue
        lat, lon = float(coords[ii, 0]), float(coords[ii, 1])
        score = float(sims[0, j])
        out.append((lat, lon, score))
    return out


def faiss_bundle_stats(settings: Settings) -> Dict[str, Any]:
    b = load_faiss_bundle(settings)
    if b is None:
        return {"configured": False, "vectors": 0}
    _index, _coords, n = b
    return {"configured": True, "vectors": n, "dim": int(_index.d)}
