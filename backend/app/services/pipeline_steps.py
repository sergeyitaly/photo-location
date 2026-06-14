"""
Human-readable labels and messages for GET /predict/progress.

Internal step ids (code) map to short user-facing copy (message) and a coarse checklist group.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class StepDef(TypedDict):
    group: str
    title: str
    default_message: str


# Coarse checklist groups shown in the UI (order matters).
CHECKLIST_GROUPS: List[Dict[str, str]] = [
    {"id": "prepare", "label": "Prepare your photo"},
    {"id": "features", "label": "Read visual cues"},
    {"id": "vision", "label": "Find location (AI models)"},
    {"id": "enrich", "label": "Place names & validation"},
    {"id": "finish", "label": "Build results"},
]

# Fine-grained steps reported by routes / ensemble.
STEP_DEFS: Dict[str, StepDef] = {
    "request_parse": {
        "group": "prepare",
        "title": "Receive upload",
        "default_message": "Receiving your photo…",
    },
    "image_decode": {
        "group": "prepare",
        "title": "Decode image",
        "default_message": "Decoding image pixels…",
    },
    "feature_analysis": {
        "group": "features",
        "title": "Visual cues",
        "default_message": "Reading vegetation, text, roads, and sky…",
    },
    "vision_inference": {
        "group": "vision",
        "title": "Vision models",
        "default_message": "Running location AI models…",
    },
    "clip_classifier": {
        "group": "vision",
        "title": "CLIP classifier",
        "default_message": "CLIP: estimating country and landmark…",
    },
    "clip_retrieval": {
        "group": "vision",
        "title": "CLIP retrieval",
        "default_message": "CLIP: comparing to known places…",
    },
    "geoclip": {
        "group": "vision",
        "title": "GeoCLIP",
        "default_message": "GeoCLIP: estimating GPS coordinates…",
    },
    "grid_search": {
        "group": "vision",
        "title": "Map grid search",
        "default_message": "Scanning the world map in coarse regions…",
    },
    "streetclip": {
        "group": "vision",
        "title": "StreetCLIP",
        "default_message": "StreetCLIP: matching city names…",
    },
    "fusion_merge": {
        "group": "vision",
        "title": "Fusion",
        "default_message": "Combining GeoCLIP, StreetCLIP, and CLIP…",
    },
    "cross_reference": {
        "group": "enrich",
        "title": "Gazetteer match",
        "default_message": "Cross-checking with local city database…",
    },
    "external_validation": {
        "group": "enrich",
        "title": "External validation",
        "default_message": "Wikipedia text + Wikimedia photo + elevation checks…",
    },
    "reverse_geocode": {
        "group": "enrich",
        "title": "Place names",
        "default_message": "Looking up street and town names (OpenStreetMap)…",
    },
    "analysis_panels": {
        "group": "enrich",
        "title": "Scene analysis",
        "default_message": "Regional and scene cue panels…",
    },
    "reasoning": {
        "group": "enrich",
        "title": "Geo reasoning",
        "default_message": "Country elimination and evidence fusion…",
    },
    "satellite_match": {
        "group": "enrich",
        "title": "Satellite check",
        "default_message": "Satellite imagery comparison…",
    },
    "streetview_verify": {
        "group": "enrich",
        "title": "Street View",
        "default_message": "Street View visual verification…",
    },
    "llm_detective": {
        "group": "enrich",
        "title": "LLM clues",
        "default_message": "Local LLM review of visual clues…",
    },
    "finish": {
        "group": "finish",
        "title": "Finish",
        "default_message": "Packaging your results…",
    },
}

PHASE_LABELS: Dict[str, str] = {
    "prepare": "Preparing",
    "features": "Reading cues",
    "vision": "Finding location",
    "enrich": "Refining",
    "finish": "Finishing",
    "done": "Done",
    "error": "Error",
}


def message_for_step(step_id: str, detail: Optional[str] = None) -> str:
    if detail and detail.strip():
        return detail.strip()
    info = STEP_DEFS.get(step_id)
    if info:
        return info["default_message"]
    return step_id.replace("_", " ").capitalize()


def title_for_step(step_id: str) -> str:
    info = STEP_DEFS.get(step_id)
    return info["title"] if info else step_id


def group_for_step(step_id: str) -> str:
    info = STEP_DEFS.get(step_id)
    return info["group"] if info else "vision"


def build_checklist(
    *,
    current_group: str,
    completed_groups: List[str],
    skipped_groups: List[str],
    sub_detail: str = "",
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for g in CHECKLIST_GROUPS:
        gid = g["id"]
        if gid in skipped_groups:
            state = "skipped"
        elif gid in completed_groups:
            state = "done"
        elif gid == current_group:
            state = "current"
        else:
            state = "pending"
        item: Dict[str, Any] = {"id": gid, "label": g["label"], "state": state}
        if state == "current" and sub_detail:
            item["detail"] = sub_detail
        out.append(item)
    return out


def estimate_percent(
    completed_groups: List[str],
    skipped_groups: List[str],
    current_group: str,
) -> int:
    total = len(CHECKLIST_GROUPS) - len(skipped_groups)
    if total <= 0:
        return 0
    done = len(completed_groups)
    # Credit half a step for the group currently in progress.
    if current_group and current_group not in completed_groups and current_group not in skipped_groups:
        done += 0.5
    return min(99, int(round(100.0 * done / total)))
