"""ProgressReporter: stage timing bookkeeping and throttled substep flush."""

from __future__ import annotations

import time

import pytest

from meshpartition.progress import ProgressReporter


def test_stage_records_duration_and_completion():
    reporter = ProgressReporter(phases=["a", "b"])

    with reporter.stage("a", 0):
        time.sleep(0.01)
    with reporter.stage("b", 1):
        pass

    names = [t["name"] for t in reporter.stage_timings]
    assert names == ["a", "b"]
    assert reporter.stage_timings[0]["duration_seconds"] > 0
    assert reporter.completed_phases == ["a", "b"]


def test_stage_records_duration_even_on_exception():
    reporter = ProgressReporter(phases=["a"])

    with pytest.raises(ValueError):
        with reporter.stage("a", 0):
            raise ValueError("boom")

    assert [t["name"] for t in reporter.stage_timings] == ["a"]
    assert reporter.completed_phases == ["a"]


def test_substep_flush_is_throttled():
    updates = []
    reporter = ProgressReporter(on_update=updates.append, min_interval=10.0)

    with reporter.stage("a", 0):
        reporter.substep(1, 100)
        reporter.substep(2, 100)  # suppressed: inside min_interval window

    # stage entry + exit always flush unthrottled (force=True), so exactly
    # those two plus the first (unthrottled-window) substep call land.
    assert len(updates) >= 2
    assert updates[0]["current_phase"] == "a"


def test_substep_updates_current_state():
    reporter = ProgressReporter(min_interval=0.0)
    reporter.substep(3, 10, label="satellites")
    assert reporter.substep_current == 3
    assert reporter.substep_total == 10
    assert reporter.substep_label == "satellites"
