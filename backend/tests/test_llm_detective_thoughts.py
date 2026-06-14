"""Ollama detective: placeholder filtering and vision-based synthesis."""

from app.services.llm_detective import (
    _detective_has_substance,
    _is_placeholder_detective_text,
    _sanitize_detective_dict,
    build_key_thoughts,
    synthesize_detective_from_inputs,
)


def test_placeholder_detection():
    assert _is_placeholder_detective_text("clue 1")
    assert _is_placeholder_detective_text("evidence 2")
    assert not _is_placeholder_detective_text("Dense vegetation and sunny weather")


def test_sanitize_strips_template_echo():
    raw = {
        "strongest_clues": ["clue 1", "Vegetation: palm trees"],
        "contradictions": ["contradiction 1"],
        "detective_summary": "Real summary about Eastern Europe.",
    }
    clean = _sanitize_detective_dict(raw)
    assert clean["strongest_clues"] == ["Vegetation: palm trees"]
    assert clean["contradictions"] == []
    assert _detective_has_substance(clean)


def test_synthesize_from_real_inputs():
    fa = {
        "vegetation_types": ["dense_vegetation"],
        "weather_condition": "sunny",
        "architecture_style": "soviet_panel",
    }
    preds = [
        {"city": "Kyiv", "country": "Ukraine", "confidence": 0.42},
        {"city": "GeoCLIP rank 2", "country": "GeoCLIP gallery", "confidence": 0.13},
    ]
    out = synthesize_detective_from_inputs(fa, preds)
    thoughts = build_key_thoughts(out)
    assert thoughts
    assert not any("clue 1" in t.lower() for t in thoughts)
    assert any("ukraine" in t.lower() or "kyiv" in t.lower() or "vision" in t.lower() for t in thoughts)
