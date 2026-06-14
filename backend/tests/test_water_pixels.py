"""Generic open-water pixel heuristic (no regional pin overrides)."""

import importlib.util
from pathlib import Path

_wp_path = Path(__file__).resolve().parents[1] / "app" / "features" / "water_pixels.py"
_spec = importlib.util.spec_from_file_location("water_pixels", _wp_path)
_wp = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_wp)
open_water_scene_score = _wp.open_water_scene_score


def test_open_water_scene_score_from_ml_labels():
    score = open_water_scene_score(
        None,
        ml_labels=[("a photo of water such as a lake or river", 0.15)],
    )
    assert score >= 0.1
