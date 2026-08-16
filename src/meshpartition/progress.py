"""Lightweight progress/timing instrumentation shared by the CLI and web UI
entry points. A `ProgressReporter` records per-stage wall-clock duration
(always, for offline timing analysis) and, optionally, an intra-stage
substep signal (satellite index in Stage 2, Lloyd iteration/region index in
Stage 8) pushed through a throttled `on_update` sink for live progress
display.
"""

from __future__ import annotations

import time
from typing import Callable

PHASE_NAMES = [
    "normalize",
    "ingest",
    "triage",
    "bridges",
    "dual_graph",
    "seed",
    "grow",
    "repair",
    "relax",
    "extract",
]


class ProgressReporter:
    def __init__(
        self,
        phases: list[str] = PHASE_NAMES,
        on_update: Callable[[dict], None] | None = None,
        min_interval: float = 0.25,
    ) -> None:
        self.all_phases = list(phases)
        self._on_update = on_update
        self._min_interval = min_interval
        self._last_flush = 0.0

        self.stage_timings: list[dict] = []
        self.completed_phases: list[str] = []
        self.current_phase: str | None = None
        self.current_index: int | None = None
        self.substep_current: int | None = None
        self.substep_total: int | None = None
        self.substep_label: str | None = None

    def stage(self, name: str, index: int | None = None) -> "_StageScope":
        # Stages run strictly sequentially (never nested), so the count of
        # timings recorded so far is the next stage's index.
        if index is None:
            index = len(self.stage_timings)
        return _StageScope(self, name, index)

    def substep(self, current: int, total: int, label: str | None = None) -> None:
        self.substep_current = current
        self.substep_total = total
        self.substep_label = label
        self._flush()

    def snapshot(self) -> dict:
        return {
            "all_phases": self.all_phases,
            "completed_phases": list(self.completed_phases),
            "current_phase": self.current_phase,
            "current_index": self.current_index,
            "substep_current": self.substep_current,
            "substep_total": self.substep_total,
            "substep_label": self.substep_label,
        }

    def _flush(self, force: bool = False) -> None:
        if self._on_update is None:
            return
        now = time.monotonic()
        if not force and (now - self._last_flush) < self._min_interval:
            return
        self._last_flush = now
        self._on_update(self.snapshot())


class _StageScope:
    """Context manager for one pipeline stage. Records start/end wall time
    into reporter.stage_timings on exit (even on exception), and clears
    substep state so a finished stage doesn't leave stale progress behind."""

    def __init__(self, reporter: ProgressReporter, name: str, index: int) -> None:
        self._reporter = reporter
        self._name = name
        self._index = index
        self._t0 = 0.0

    def __enter__(self) -> ProgressReporter:
        r = self._reporter
        r.current_phase = self._name
        r.current_index = self._index
        r.substep_current = None
        r.substep_total = None
        r.substep_label = None
        self._t0 = time.perf_counter()
        r._flush(force=True)
        return r

    def __exit__(self, exc_type, exc, tb) -> bool:
        duration = time.perf_counter() - self._t0
        r = self._reporter
        r.stage_timings.append({
            "name": self._name,
            "index": self._index,
            "duration_seconds": round(duration, 4),
        })
        r.completed_phases.append(self._name)
        r.substep_current = None
        r.substep_total = None
        r.substep_label = None
        r._flush(force=True)
        return False
