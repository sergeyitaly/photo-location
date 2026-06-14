"""
Real-time pipeline progress for GET /predict/progress.

Returns a small, human-oriented JSON payload (message, phase, checklist, percent)
instead of raw internal step ids.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from app.services.pipeline_steps import (
    PHASE_LABELS,
    build_checklist,
    estimate_percent,
    group_for_step,
    message_for_step,
    title_for_step,
)


@dataclass
class PipelineProgress:
    image_id: str = ""
    status: str = "idle"  # idle | running | completed | error
    current_step: str = ""
    step_detail: str = ""
    fast_mode: bool = False
    include_features: bool = True
    steps_completed: list[str] = field(default_factory=list)
    groups_completed: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    live: dict[str, Any] = field(default_factory=dict)
    started_at: float = 0.0
    last_updated: float = 0.0
    error: str = ""

    def to_public_dict(self) -> dict[str, Any]:
        """User-facing progress document for the web UI."""
        if self.status == "idle":
            return {
                "status": "idle",
                "message": "Waiting for a prediction request…",
                "phase": "idle",
                "phase_label": "Idle",
                "elapsed_s": 0,
                "percent": 0,
                "checklist": [],
                "live": {},
            }

        elapsed_s = round((time.time() - self.started_at), 1) if self.started_at > 0 else 0.0
        skipped = self._skipped_groups()
        current_group = group_for_step(self.current_step) if self.current_step else ""
        sub_detail = title_for_step(self.current_step) if self.current_step else ""

        if self.status == "completed":
            completed = [g["id"] for g in _active_groups(self.fast_mode, self.include_features)]
            return {
                "status": "completed",
                "message": "Location prediction finished.",
                "phase": "done",
                "phase_label": PHASE_LABELS["done"],
                "elapsed_s": elapsed_s,
                "percent": 100,
                "checklist": build_checklist(
                    current_group="",
                    completed_groups=completed,
                    skipped_groups=skipped,
                ),
                "fast_mode": self.fast_mode,
                "live": dict(self.live),
            }

        if self.status == "error":
            return {
                "status": "error",
                "message": self.error or "Prediction failed.",
                "phase": "error",
                "phase_label": PHASE_LABELS["error"],
                "elapsed_s": elapsed_s,
                "percent": estimate_percent(self.groups_completed, skipped, current_group),
                "checklist": build_checklist(
                    current_group=current_group,
                    completed_groups=self.groups_completed,
                    skipped_groups=skipped,
                    sub_detail=sub_detail,
                ),
                "fast_mode": self.fast_mode,
                "live": dict(self.live),
            }

        message = (
            (self.live.get("processing_note") or "").strip()
            or message_for_step(self.current_step, self.step_detail)
        )
        phase_label = PHASE_LABELS.get(current_group, "Working")

        return {
            "status": "running",
            "message": message,
            "phase": current_group or "vision",
            "phase_label": phase_label,
            "step": self.current_step,
            "step_title": title_for_step(self.current_step) if self.current_step else "",
            "elapsed_s": elapsed_s,
            "percent": estimate_percent(self.groups_completed, skipped, current_group),
            "checklist": build_checklist(
                current_group=current_group,
                completed_groups=self.groups_completed,
                skipped_groups=skipped,
                sub_detail=sub_detail,
            ),
            "fast_mode": self.fast_mode,
            "live": dict(self.live),
            # Legacy fields (older clients)
            "current_step": self.current_step,
            "step_detail": self.step_detail,
            "steps_completed": self.steps_completed,
            "timings_ms": self.timings_ms,
            "elapsed_ms": round(elapsed_s * 1000, 1),
        }

    def _skipped_groups(self) -> list[str]:
        skipped: list[str] = []
        if not self.include_features:
            skipped.append("features")
        if self.fast_mode:
            skipped.append("enrich")
        return skipped


def _active_groups(fast_mode: bool, include_features: bool) -> list[dict[str, str]]:
    from app.services.pipeline_steps import CHECKLIST_GROUPS

    skipped = set()
    if not include_features:
        skipped.add("features")
    if fast_mode:
        skipped.add("enrich")
    return [g for g in CHECKLIST_GROUPS if g["id"] not in skipped]


class PipelineProgressTracker:
    _instance: Optional["PipelineProgressTracker"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "PipelineProgressTracker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._progress = PipelineProgress()
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "_progress"):
            self._progress = PipelineProgress()

    def start_prediction(
        self,
        image_id: str,
        *,
        fast_mode: bool = False,
        include_features: bool = True,
    ) -> None:
        with self._lock:
            self._progress = PipelineProgress(
                image_id=image_id,
                status="running",
                current_step="request_parse",
                step_detail="",
                fast_mode=fast_mode,
                include_features=include_features,
                live={},
                started_at=time.time(),
                last_updated=time.time(),
            )

    def set_live(self, **kwargs: Any) -> None:
        """Merge location-identification context shown in the UI (candidates, region, etc.)."""
        with self._lock:
            if self._progress.status != "running":
                return
            for key, value in kwargs.items():
                if value is None:
                    self._progress.live.pop(key, None)
                else:
                    self._progress.live[key] = value
            self._progress.last_updated = time.time()

    def set_options(self, *, fast_mode: bool, include_features: bool) -> None:
        with self._lock:
            self._progress.fast_mode = fast_mode
            self._progress.include_features = include_features

    def update_step(
        self,
        step_name: str,
        detail: str = "",
        timings: Optional[dict[str, float]] = None,
    ) -> None:
        with self._lock:
            if self._progress.status != "running":
                return

            prev_step = self._progress.current_step
            if prev_step and prev_step != step_name:
                if prev_step not in self._progress.steps_completed:
                    self._progress.steps_completed.append(prev_step)
                prev_group = group_for_step(prev_step)
                new_group = group_for_step(step_name)
                if prev_group != new_group and prev_group not in self._progress.groups_completed:
                    self._progress.groups_completed.append(prev_group)

            self._progress.current_step = step_name
            self._progress.step_detail = detail
            self._progress.last_updated = time.time()
            if timings:
                self._progress.timings_ms.update(timings)

    def complete(self, final_timings: Optional[dict[str, float]] = None) -> dict[str, Any]:
        with self._lock:
            if self._progress.status == "running":
                if self._progress.current_step:
                    self._progress.steps_completed.append(self._progress.current_step)
                    g = group_for_step(self._progress.current_step)
                    if g not in self._progress.groups_completed:
                        self._progress.groups_completed.append(g)
                self._progress.status = "completed"
                self._progress.current_step = "finish"
                self._progress.step_detail = ""
                self._progress.last_updated = time.time()
                if final_timings:
                    self._progress.timings_ms.update(final_timings)
            return self._progress.to_public_dict()

    def error(self, error_message: str) -> dict[str, Any]:
        with self._lock:
            self._progress.status = "error"
            self._progress.error = error_message
            self._progress.last_updated = time.time()
            return self._progress.to_public_dict()

    def get_current(self) -> dict[str, Any]:
        with self._lock:
            return self._progress.to_public_dict()

    def reset(self) -> None:
        with self._lock:
            self._progress = PipelineProgress()


_progress_tracker: Optional[PipelineProgressTracker] = None


def get_progress_tracker() -> PipelineProgressTracker:
    global _progress_tracker
    if _progress_tracker is None:
        _progress_tracker = PipelineProgressTracker()
    return _progress_tracker
