"""
Human-Like Detective Layer (Open LLM Local).

Uses a free local LLM (Qwen / Mistral / Llama via Ollama) as a textual reasoning assistant —
NOT for final coordinates, but for clue consistency and contradictions.

Requires: Ollama running locally with a model such as `ollama pull qwen2.5:7b`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"

# When OLLAMA_MODEL is missing locally, pick the first installed name matching these prefixes.
_PREFERRED_MODEL_PREFIXES = (
    "qwen2.5",
    "qwen2",
    "llama3.2",
    "llama3",
    "mistral",
    "gemma2",
    "gemma",
    "phi3",
    "phi",
)

# Per base_url: True = /api/chat works, False = fall back to /api/generate only.
_chat_api_supported: Dict[str, bool] = {}

_DETECTIVE_SYSTEM_PROMPT = """You are a geography detective. Examine clues from a photograph and reason about where it might have been taken.

Rules:
1. You do NOT guess exact coordinates.
2. You reason from evidence to likely regions/countries.
3. You call out contradictions (e.g., palm trees + snow).
4. You state confidence levels for each conclusion.
5. You suggest what additional evidence would be most valuable.

Respond in valid JSON only, no markdown, no text outside the JSON object."""

_DETECTIVE_USER_TEMPLATE = """Computer vision detected these clues:

{clues_text}

Top predicted locations:
{predictions_text}

Write ONE JSON object only. Use the actual clue and place names above — never write generic placeholders like "clue 1" or "evidence 2".

Required keys:
- strongest_clues: 2–4 short strings citing real visual evidence from the clues section
- contradictions: array of strings (use [] if none); only real conflicts you see in the clues
- most_consistent_locations: 1–3 place names from the predictions that fit the clues best
- confidence_assessment: one sentence on how much you trust the vision guesses
- additional_evidence_needed: 1–3 concrete things that would narrow the location (e.g. readable signs, EXIF, wider view)
- detective_summary: 2–3 sentences tying clues to the most likely region"""

_DETECTIVE_USER_PLAIN_TEMPLATE = """You are helping interpret a photo's location. Use ONLY the facts below.

Visual clues:
{clues_text}

Vision model guesses:
{predictions_text}

Reply in exactly 4 lines (no JSON):
Line A: Most likely country or region and why (one sentence).
Line B: Strongest visual clue from the list (one sentence).
Line C: Biggest doubt or contradiction, or "None obvious" (one sentence).
Line D: One thing that would confirm the place (one sentence)."""

_PLACEHOLDER_ITEM_RE = re.compile(
    r"^(clue|contradiction|location|evidence|item)\s*#?\s*\d+$",
    re.IGNORECASE,
)
_PLACEHOLDER_PHRASES = frozenset(
    {
        "string explaining confidence",
        "2-3 sentence summary of reasoning",
        "one sentence",
        "most likely country or region and why (one sentence).",
        "strongest visual clue from the list (one sentence).",
        "biggest doubt or contradiction, or \"none obvious\" (one sentence).",
        "one thing that would confirm the place (one sentence).",
    }
)
_INSTRUCTION_PREFIXES = (
    "line a:",
    "line b:",
    "line c:",
    "line d:",
    "most likely country",
    "strongest visual clue",
    "biggest doubt",
    "one thing that would",
)


async def _is_ollama_available(base_url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{base_url.rstrip('/')}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


async def _ollama_model_names(base_url: str) -> List[str]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{base_url.rstrip('/')}/api/tags")
            if r.status_code != 200:
                return []
            models = r.json().get("models") or []
            return [str(m.get("name") or "") for m in models]
    except Exception:
        return []


def _model_is_installed(model: str, available: List[str]) -> bool:
    if not available:
        return True
    if model in available:
        return True
    base = model.split(":")[0].lower()
    for name in available:
        n_base = name.split(":")[0].lower()
        if name == model or name.startswith(f"{model}:") or n_base == base:
            return True
    return False


def _pick_installed_fallback(available: List[str], *, avoid_base: Optional[str] = None) -> Optional[str]:
    """Choose a reasonable installed model when the configured tag is missing."""
    if not available:
        return None
    avoid = (avoid_base or "").split(":")[0].lower()
    for prefix in _PREFERRED_MODEL_PREFIXES:
        for name in available:
            n_base = name.split(":")[0].lower()
            if avoid and n_base == avoid:
                continue
            if n_base.startswith(prefix) or prefix in n_base:
                return name
    for name in available:
        if avoid and name.split(":")[0].lower() == avoid:
            continue
        return name
    return available[0]


def _resolve_model_name(requested: str, available: List[str]) -> Tuple[str, Optional[str]]:
    """
    Match qwen2.5:7b / mistral to installed tags (with or without :latest).
    Returns (model_tag, fallback_note) where fallback_note explains auto-picks.
    """
    req = (requested or DEFAULT_MODEL).strip()
    if not available:
        return req, None
    if req in available:
        return req, None
    base = req.split(":")[0]
    for name in available:
        if name == req or name.startswith(f"{req}:") or name.startswith(f"{base}:"):
            return name, None
    for name in available:
        if base in name.split(":")[0]:
            return name, None
    fallback = _pick_installed_fallback(available, avoid_base=base)
    if fallback:
        note = (
            f"OLLAMA_MODEL={req!r} is not installed; using {fallback!r}. "
            f"Run `ollama pull {req}` or set OLLAMA_MODEL={fallback!r} in backend/.env."
        )
        logger.warning(note)
        return fallback, note
    return req, None


def _format_ollama_query_error(
    err: str,
    *,
    model: str,
    requested: str,
    available: List[str],
) -> str:
    text = (err or "").strip()
    lower = text.lower()
    if "404" in text and "not found" in lower:
        installed = ", ".join(available[:6]) if available else "(none)"
        return (
            f"Model {model!r} is not installed. Run: ollama pull {requested} "
            f"— installed models: {installed}"
        )
    return text


async def warmup_ollama_model(settings: Settings) -> Dict[str, Any]:
    """Load the configured model into Ollama RAM (1-token chat) before first /predict."""
    report: Dict[str, Any] = {"ok": False, "skipped": None, "model": None, "error": None}
    if not getattr(settings, "ollama_warmup_at_startup", True):
        report["skipped"] = "ollama_warmup_at_startup=false"
        return report
    if not getattr(settings, "use_llm_detective", True):
        report["skipped"] = "use_llm_detective=false"
        return report

    base_url = getattr(settings, "ollama_base_url", DEFAULT_OLLAMA_URL).rstrip("/")
    requested = getattr(settings, "ollama_model", DEFAULT_MODEL)
    if not await _is_ollama_available(base_url):
        report["skipped"] = "ollama_not_running"
        return report

    available = await _ollama_model_names(base_url)
    model, _fallback_note = _resolve_model_name(requested, available)
    report["model"] = model
    timeout = float(getattr(settings, "ollama_http_timeout_seconds", 180.0))
    _, err = await _ollama_complete_json(
        base_url=base_url,
        model=model,
        system="Reply with one word.",
        user="ok",
        http_timeout=timeout,
        num_predict=4,
        use_json_format=False,
    )
    if err:
        report["error"] = err
        logger.warning("Ollama warmup failed: %s", err)
    else:
        report["ok"] = True
        api = "chat" if _chat_api_supported.get(base_url, True) else "generate"
        logger.info("Ollama warmup ok for model %s via /api/%s", model, api)
    return report


def _build_clues_text(cues: Dict[str, Any], *, max_chars: int = 2400) -> str:
    lines: List[str] = []

    vegetation = cues.get("vegetation_types") or []
    if vegetation:
        lines.append(f"Vegetation: {', '.join(str(v) for v in vegetation[:6])}")

    architecture = cues.get("architecture_style")
    if architecture:
        lines.append(f"Architecture style: {architecture}")

    text = cues.get("detected_text") or []
    if text:
        lines.append(f"Detected text (OCR): {', '.join(str(t) for t in text[:5])}")

    weather = cues.get("weather_condition")
    if weather:
        lines.append(f"Weather: {weather}")

    time_of_day = cues.get("time_of_day")
    if time_of_day:
        lines.append(f"Estimated time of day: {time_of_day}")

    infrastructure = cues.get("infrastructure_type")
    if infrastructure:
        lines.append(f"Infrastructure type: {infrastructure}")

    poles = cues.get("detected_poles") or []
    if poles:
        pole_types = [str(p.get("type", "unknown")) for p in poles[:3]]
        lines.append(f"Utility poles detected: {', '.join(pole_types)}")

    roads = cues.get("detected_road_lines") or []
    if roads:
        road_colors = [str(r.get("color", "unknown")) for r in roads[:3]]
        lines.append(f"Road line colors: {', '.join(road_colors)}")

    shadows = cues.get("shadow_analysis")
    if shadows and isinstance(shadows, dict):
        conf = shadows.get("confidence", 0)
        if conf and float(conf) > 0.3:
            lines.append(f"Shadow analysis confidence: {float(conf):.2f}")

    ml_labels = cues.get("ml_labels") or []
    if ml_labels:
        top = [
            f"{l.get('label', '?')} ({float(l.get('score', 0)):.0%})"
            for l in ml_labels[:5]
        ]
        lines.append(f"ML scene labels: {', '.join(top)}")

    if not lines:
        lines.append("No strong visual cues detected.")

    body = "\n".join(f"- {line}" for line in lines)
    if len(body) > max_chars:
        body = body[: max_chars - 3] + "..."
    return body


def _build_predictions_text(predictions: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for i, p in enumerate(predictions[:3]):
        country = p.get("country", "unknown")
        city = p.get("city", "unknown")
        conf = float(p.get("confidence") or 0)
        lines.append(f"  {i + 1}. {city}, {country} — confidence {conf:.1%}")
    return "\n".join(lines) if lines else "  No predictions available."


def _is_placeholder_detective_text(text: str) -> bool:
    t = (text or "").strip()
    if not t or len(t) < 3:
        return True
    lower = t.lower()
    if lower in _PLACEHOLDER_PHRASES:
        return True
    if _PLACEHOLDER_ITEM_RE.match(t):
        return True
    if re.search(r"\b(clue|evidence|contradiction|location)\s+\d+\b", lower) and len(t) < 48:
        return True
    if any(lower.startswith(p) for p in _INSTRUCTION_PREFIXES):
        return True
    if "(one sentence)" in lower and len(t) < 100:
        return True
    if lower.startswith("fits: most likely") or lower.startswith("fits: geoclip"):
        return True
    if re.search(r"\bgeoclip\s+rank\s*\d", lower):
        return True
    if lower.startswith("need:") and "one sentence" in lower:
        return True
    if "geoclip gallery" in lower and len(t) < 90:
        return True
    return False


def _sanitize_string_list(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for item in items:
        s = str(item or "").strip()
        if s and not _is_placeholder_detective_text(s):
            out.append(s)
    return out


def _sanitize_detective_dict(parsed: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(parsed)
    for key in (
        "strongest_clues",
        "contradictions",
        "most_consistent_locations",
        "additional_evidence_needed",
    ):
        out[key] = _sanitize_string_list(out.get(key))
    for key in ("detective_summary", "confidence_assessment"):
        val = str(out.get(key) or "").strip()
        if _is_placeholder_detective_text(val):
            out[key] = ""
    return out


def _detective_has_substance(parsed: Dict[str, Any]) -> bool:
    summary = str(parsed.get("detective_summary") or "").strip()
    if summary and not _is_placeholder_detective_text(summary):
        return True
    if _sanitize_string_list(parsed.get("strongest_clues")):
        return True
    if _sanitize_string_list(parsed.get("most_consistent_locations")):
        return True
    conf = str(parsed.get("confidence_assessment") or "").strip()
    return bool(conf and not _is_placeholder_detective_text(conf))


def synthesize_detective_from_inputs(
    feature_analysis: Dict[str, Any],
    predictions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Deterministic detective bullets from real CV cues (used when small LLMs echo the JSON template)."""
    clues: List[str] = []
    for line in _build_clues_text(feature_analysis).splitlines():
        line = line.strip().lstrip("-").strip()
        if line and line != "No strong visual cues detected.":
            clues.append(line)

    landmarks = feature_analysis.get("landmarks") or []
    if isinstance(landmarks, list):
        for lm in landmarks[:2]:
            if isinstance(lm, dict):
                conf = float(lm.get("confidence") or 0.0)
                if conf and conf < 0.28:
                    continue
                name = lm.get("name") or lm.get("city")
                if name:
                    clues.append(f"Possible landmark (CLIP ≥28%): {name}")

    from app.services.place_display import is_geoclip_placeholder, is_named_gazetteer_place

    pred_lines: List[str] = []
    locs: List[str] = []
    for i, p in enumerate(predictions[:3]):
        city = str(p.get("city") or "unknown")
        country = str(p.get("country") or "unknown")
        conf = float(p.get("confidence") or 0)
        if is_geoclip_placeholder(city, country):
            pred_lines.append(f"GPS estimate rank {i + 1} ({conf:.0%})")
        else:
            pred_lines.append(f"{city}, {country} ({conf:.0%})")
        if is_named_gazetteer_place(city, country):
            locs.append(f"{city}, {country}")

    strongest = clues[:4] if clues else []
    if pred_lines and not strongest:
        strongest.append(f"Vision top guess: {pred_lines[0]}")

    contradictions: List[str] = []
    veg = feature_analysis.get("vegetation_types") or []
    weather = str(feature_analysis.get("weather_condition") or "").lower()
    if any("tropical" in str(v).lower() for v in veg) and "snow" in weather:
        contradictions.append("Tropical vegetation cues conflict with snowy weather reading")
    if len(predictions) >= 2:
        c0 = float(predictions[0].get("confidence") or 0)
        c1 = float(predictions[1].get("confidence") or 0)
        if c0 < 0.2 and c1 > 0.08 and abs(c0 - c1) < 0.06:
            contradictions.append("Vision models disagree — several guesses have similar low confidence")

    evidence: List[str] = []
    if not feature_analysis.get("detected_text"):
        evidence.append("Readable street signs or shop names (OCR)")
    evidence.append("Wider view or adjacent photos from the same place")
    if any(
        is_geoclip_placeholder(str(p.get("city")), str(p.get("country")))
        for p in predictions[:1]
    ):
        evidence.append("Stronger city-level StreetCLIP match in the gazetteer")

    summary_parts: List[str] = []
    if pred_lines:
        summary_parts.append(f"Vision ranks {pred_lines[0]} first")
    if clues:
        summary_parts.append(f"Scene cues include {clues[0].split(':')[0].lower()}")
    if contradictions:
        summary_parts.append(contradictions[0])
    summary = (
        ". ".join(summary_parts) + "."
        if summary_parts
        else "Insufficient cues; rely on the map pin from vision fusion."
    )

    conf_line = ""
    if predictions:
        top = float(predictions[0].get("confidence") or 0)
        if top >= 0.35:
            conf_line = "Moderate confidence in the leading vision guess."
        elif top >= 0.15:
            conf_line = "Low–moderate confidence — treat the pin as approximate."
        else:
            conf_line = "Very low confidence — multiple regions remain plausible."

    return {
        "strongest_clues": strongest,
        "contradictions": contradictions,
        "most_consistent_locations": locs[:3] or [p.split(" (")[0] for p in pred_lines[:2]],
        "confidence_assessment": conf_line,
        "additional_evidence_needed": evidence[:3],
        "detective_summary": summary,
        "synthesized": True,
    }


def _merge_detective(primary: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(fallback)
    for key in (
        "strongest_clues",
        "contradictions",
        "most_consistent_locations",
        "additional_evidence_needed",
    ):
        primary_list = _sanitize_string_list(primary.get(key))
        fallback_list = _sanitize_string_list(fallback.get(key))
        combined: List[str] = []
        seen: set[str] = set()
        for item in primary_list + fallback_list:
            k = item.lower()
            if k not in seen:
                seen.add(k)
                combined.append(item)
        merged[key] = combined[:6]
    for key in ("detective_summary", "confidence_assessment"):
        val = str(primary.get(key) or "").strip()
        if val and not _is_placeholder_detective_text(val):
            merged[key] = val
    merged["synthesized"] = bool(fallback.get("synthesized")) and not _detective_has_substance(
        primary
    )
    return merged


def _parse_plain_detective_response(raw: str) -> Dict[str, Any]:
    """Parse Line A/B/C/D plain-text format from small models."""
    lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
    if not lines:
        return {}
    buckets: Dict[str, str] = {}
    for line in lines:
        if _is_placeholder_detective_text(line):
            continue
        m = re.match(r"^line\s*([a-d])\s*[:.)-]\s*(.+)$", line, re.I)
        if m:
            content = m.group(2).strip()
            if not _is_placeholder_detective_text(content):
                buckets[m.group(1).upper()] = content
            continue
        if line.lower().startswith("a:"):
            buckets["A"] = line[2:].strip()
        elif line.lower().startswith("b:"):
            buckets["B"] = line[2:].strip()
        elif line.lower().startswith("c:"):
            buckets["C"] = line[2:].strip()
        elif line.lower().startswith("d:"):
            buckets["D"] = line[2:].strip()
    if not buckets and len(lines) >= 2:
        buckets = {"A": lines[0], "B": lines[1], "C": lines[2] if len(lines) > 2 else "", "D": lines[-1]}

    strongest = [buckets["B"]] if buckets.get("B") else []
    contradictions: List[str] = []
    c = buckets.get("C", "")
    if c and c.lower() not in ("none", "none obvious", "n/a"):
        contradictions.append(c)
    evidence = [buckets["D"]] if buckets.get("D") else []
    locations = [buckets["A"]] if buckets.get("A") else []
    return {
        "detective_summary": buckets.get("A", ""),
        "strongest_clues": strongest,
        "contradictions": contradictions,
        "most_consistent_locations": locations,
        "confidence_assessment": buckets.get("C", ""),
        "additional_evidence_needed": evidence,
    }


def build_key_thoughts(ld: Dict[str, Any]) -> List[str]:
    """Short bullets for UI: summary first, then clues, contradictions, locations."""
    thoughts: List[str] = []
    seen: set[str] = set()

    def add(text: Any) -> None:
        t = str(text or "").strip()
        if not t or len(t) < 3 or _is_placeholder_detective_text(t):
            return
        key = t.lower()
        if key in seen:
            return
        seen.add(key)
        thoughts.append(t)

    add(ld.get("detective_summary"))
    for item in ld.get("strongest_clues") or []:
        add(item)
    for item in ld.get("contradictions") or []:
        add(f"⚠ {item}")
    for item in ld.get("most_consistent_locations") or []:
        loc = str(item or "").strip()
        if loc.startswith("Fits:"):
            loc = loc[5:].strip()
        from app.services.place_display import is_geoclip_placeholder, is_named_gazetteer_place

        if "," in loc:
            parts = loc.split(",", 1)
            city, country = parts[0].strip(), parts[1].strip()
            if is_named_gazetteer_place(city, country):
                add(f"Fits: {loc}")
        elif not _is_placeholder_detective_text(loc) and not loc.lower().startswith("geoclip"):
            add(f"Fits: {loc}")
    add(ld.get("confidence_assessment"))
    for item in ld.get("additional_evidence_needed") or []:
        add(f"Need: {item}")
    if not thoughts and ld.get("summary"):
        add(ld.get("summary"))
    return thoughts[:10]


def _parse_detective_json(raw_response: str) -> Dict[str, Any]:
    raw = (raw_response or "").strip()
    if not raw:
        return {"detective_summary": "", "parse_error": True}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return {
            "detective_summary": raw[:400],
            "parse_error": True,
        }


def _extract_ollama_text(data: Dict[str, Any]) -> str:
    msg = data.get("message") or {}
    if isinstance(msg, dict) and msg.get("content"):
        return str(msg["content"])
    if data.get("response"):
        return str(data["response"])
    return ""


async def _ollama_post_once(
    client: httpx.AsyncClient,
    url: str,
    payload: Dict[str, Any],
) -> httpx.Response:
    return await client.post(url, json=payload)


def _ollama_request_payload(
    *,
    endpoint: str,
    model: str,
    system: str,
    user: str,
    options: Dict[str, Any],
    use_json_format: bool,
) -> Dict[str, Any]:
    if endpoint == "chat":
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": options,
        }
    else:
        payload = {
            "model": model,
            "prompt": user,
            "system": system,
            "stream": False,
            "options": options,
        }
    if use_json_format:
        payload["format"] = "json"
    return payload


async def _ollama_complete_json(
    *,
    base_url: str,
    model: str,
    system: str,
    user: str,
    http_timeout: float,
    num_predict: int = 280,
    use_json_format: bool = True,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Call Ollama for detective JSON. Tries /api/chat then /api/generate; on 500 retries
    without JSON format and switches endpoint (chat 500 often fixed by generate).
    """
    base = base_url.rstrip("/")
    chat_ok = _chat_api_supported.get(base, True)
    options = {
        "temperature": 0.25,
        "num_predict": int(num_predict),
        "num_ctx": 4096,
    }
    last_err: Optional[str] = None

    plan: List[Tuple[str, bool]] = []
    if chat_ok:
        plan.append(("chat", use_json_format))
        if use_json_format:
            plan.append(("chat", False))
    plan.append(("generate", use_json_format))
    if use_json_format:
        plan.append(("generate", False))

    for endpoint, json_fmt in plan:
        url = f"{base}/api/{endpoint}"
        for retry in range(2):
            try:
                payload = _ollama_request_payload(
                    endpoint=endpoint,
                    model=model,
                    system=system,
                    user=user,
                    options=options,
                    use_json_format=json_fmt,
                )
                async with httpx.AsyncClient(timeout=http_timeout) as client:
                    r = await _ollama_post_once(client, url, payload)
                if r.status_code == 404 and endpoint == "chat":
                    _chat_api_supported[base] = False
                    logger.info(
                        "Ollama %s has no /api/chat — using /api/generate",
                        base,
                    )
                    break
                if r.status_code in (500, 502, 503) and retry == 0:
                    logger.warning(
                        "Ollama %s %s (json=%s), retrying: %s",
                        endpoint,
                        r.status_code,
                        json_fmt,
                        r.text[:180],
                    )
                    await asyncio.sleep(2.0)
                    continue
                if r.status_code == 200:
                    if endpoint == "chat":
                        _chat_api_supported[base] = True
                    return _extract_ollama_text(r.json()), None
                last_err = f"Ollama HTTP {r.status_code}: {r.text[:240]}"
            except httpx.TimeoutException:
                last_err = (
                    f"Ollama request timed out after {http_timeout:.0f}s "
                    f"(model may still be loading on CPU — increase OLLAMA_HTTP_TIMEOUT_SECONDS)"
                )
                break
            except httpx.ConnectError:
                last_err = (
                    "Cannot connect to Ollama — run `ollama serve` in a terminal and keep it running."
                )
                break
            except Exception as e:
                last_err = str(e)
                if retry == 0:
                    await asyncio.sleep(1.5)
                    continue
                break
    return None, last_err


def _tinyllama_fallback(available: List[str]) -> Optional[str]:
    for name in available:
        if name.split(":")[0].lower().startswith("tinyllama"):
            return name
    return None


def _is_small_ollama_model(model: str) -> bool:
    m = (model or "").lower()
    return "tinyllama" in m or ":1.1b" in m or m.startswith("phi") and "mini" in m


async def _ollama_detective_with_fallbacks(
    *,
    base_url: str,
    model: str,
    available: List[str],
    system: str,
    user: str,
    user_plain: str,
    http_timeout: float,
) -> Tuple[Optional[str], Optional[str], str, bool]:
    """
    Try primary model, then tinyllama if needed.
    Returns (raw_text, error, model_used, used_plain_prompt).
    """
    models_to_try: List[str] = [model]
    tiny = _tinyllama_fallback(available)
    if tiny and tiny not in models_to_try:
        models_to_try.append(tiny)

    last_err: Optional[str] = None
    for idx, tag in enumerate(models_to_try):
        small = _is_small_ollama_model(tag)
        num_predict = 240 if small else 320
        prompt = user_plain if small else user
        raw, err = await _ollama_complete_json(
            base_url=base_url,
            model=tag,
            system=system,
            user=prompt,
            http_timeout=http_timeout,
            num_predict=num_predict,
            use_json_format=not small,
        )
        if not err:
            if idx > 0:
                logger.info("Ollama detective succeeded with fallback model %s", tag)
            return raw, None, tag, small
        last_err = err
        logger.warning("Ollama detective failed for model %s: %s", tag, err)
    return None, last_err, model, _is_small_ollama_model(model)


async def run_llm_detective(
    feature_analysis: Dict[str, Any],
    predictions: List[Dict[str, Any]],
    settings: Settings,
) -> Dict[str, Any]:
    """
    Query a local Ollama LLM with detected cues + predictions.
    Returns structured detective reasoning, or skip info if Ollama unavailable.
    """
    base_url = getattr(settings, "ollama_base_url", DEFAULT_OLLAMA_URL).rstrip("/")
    requested_model = getattr(settings, "ollama_model", DEFAULT_MODEL)
    enabled = getattr(settings, "use_llm_detective", True)
    http_timeout = float(getattr(settings, "ollama_http_timeout_seconds", 180.0))

    if not enabled:
        return {
            "enabled": False,
            "skipped_reason": "disabled_in_settings",
            "summary": "LLM detective layer disabled in settings.",
        }

    if not await _is_ollama_available(base_url):
        return {
            "enabled": False,
            "skipped_reason": "ollama_not_available",
            "summary": (
                "Local LLM detective unavailable. Install Ollama, run `ollama serve`, "
                f"then `ollama pull {requested_model}`."
            ),
        }

    available = await _ollama_model_names(base_url)
    model, model_fallback_note = _resolve_model_name(requested_model, available)
    if available and not _model_is_installed(model, available):
        installed = ", ".join(available[:6]) if available else "(none)"
        return {
            "enabled": False,
            "skipped_reason": "model_not_installed",
            "summary": (
                f"Ollama model {requested_model!r} is not installed. "
                f"Run: ollama pull {requested_model} — installed: {installed}"
            ),
            "model": requested_model,
            "key_thoughts": [
                f"Run: ollama pull {requested_model}",
                f"Or set OLLAMA_MODEL to an installed tag: {installed}",
            ],
        }

    clues_text = _build_clues_text(feature_analysis)
    predictions_text = _build_predictions_text(predictions)
    synth = synthesize_detective_from_inputs(feature_analysis, predictions)

    user_prompt = _DETECTIVE_USER_TEMPLATE.format(
        clues_text=clues_text,
        predictions_text=predictions_text,
    )
    user_plain = _DETECTIVE_USER_PLAIN_TEMPLATE.format(
        clues_text=clues_text,
        predictions_text=predictions_text,
    )

    raw_response, err, model_used, used_plain = await _ollama_detective_with_fallbacks(
        base_url=base_url,
        model=model,
        available=available,
        system=_DETECTIVE_SYSTEM_PROMPT,
        user=user_prompt,
        user_plain=user_plain,
        http_timeout=http_timeout,
    )
    model = model_used

    if err:
        friendly = _format_ollama_query_error(
            err, model=model, requested=requested_model, available=available
        )
        logger.warning("LLM detective query failed; using vision-based summary: %s", friendly)
        result = {
            "enabled": True,
            "model": model,
            "requested_model": requested_model if model != requested_model else None,
            "model_fallback_note": model_fallback_note,
            "skipped_reason": None,
            "llm_error": friendly,
            **{k: synth[k] for k in synth if k != "synthesized"},
        }
        result["key_thoughts"] = build_key_thoughts(result)
        if model_fallback_note:
            result["key_thoughts"] = [model_fallback_note, *result["key_thoughts"]][:10]
        return result

    if used_plain:
        parsed = _sanitize_detective_dict(_parse_plain_detective_response(raw_response or ""))
    else:
        parsed = _sanitize_detective_dict(_parse_detective_json(raw_response or ""))

    if not _detective_has_substance(parsed):
        logger.info(
            "Ollama returned template-like output from %s; using vision-synthesized detective bullets",
            model,
        )
        merged = dict(synth)
        merged["llm_enhanced"] = False
        merged["synthesized"] = True
    else:
        merged = _merge_detective(parsed, synth)
        merged["llm_enhanced"] = True
        merged["synthesized"] = False

    result = {
        "enabled": True,
        "model": model,
        "requested_model": requested_model if model != requested_model else None,
        "model_fallback_note": model_fallback_note,
        "skipped_reason": None,
        "strongest_clues": merged.get("strongest_clues", []),
        "contradictions": merged.get("contradictions", []),
        "most_consistent_locations": merged.get("most_consistent_locations", []),
        "confidence_assessment": merged.get("confidence_assessment", ""),
        "additional_evidence_needed": merged.get("additional_evidence_needed", []),
        "detective_summary": merged.get("detective_summary", ""),
        "llm_enhanced": merged.get("llm_enhanced", True),
    }
    result["key_thoughts"] = build_key_thoughts(result)
    if model_fallback_note:
        result["key_thoughts"] = [model_fallback_note, *result["key_thoughts"]][:10]
    if _is_small_ollama_model(model) and not result.get("llm_enhanced"):
        note = "Summary from vision cues (tiny model skipped generic JSON template)."
        if note not in result["key_thoughts"]:
            result["key_thoughts"] = [note, *result["key_thoughts"]][:10]
    return result
