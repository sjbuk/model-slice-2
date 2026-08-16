"""End-to-end run of every implemented stage (0-8, plus the partial Stage 10
extraction), shared by the CLI script and the web UI so the stage-chaining
order lives in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass

from .mesh import RawMesh, WorkingMesh
from .normalize import normalize_mesh
from .progress import PHASE_NAMES, ProgressReporter
from .stage0 import IngestStats, ingest
from .stage1 import TriageResult, triage
from .stage2 import BridgeResult, build_bridges
from .stage4 import build_dual_graph
from .stage5 import seed
from .stage6 import grow
from .stage7 import repair
from .stage8 import RelaxationResult, relax
from .stage10 import extract


@dataclass
class PipelineResult:
    working: WorkingMesh
    ingest_stats: IngestStats
    triage: TriageResult
    bridges: BridgeResult
    relaxed: RelaxationResult
    pieces: list[RawMesh]
    stage_timings: list[dict]


def run_pipeline(
    raw: RawMesh,
    n_pieces: int,
    force_exact_count: bool = False,
    progress: ProgressReporter | None = None,
) -> PipelineResult:
    if progress is None:
        progress = ProgressReporter()

    with progress.stage(PHASE_NAMES[0]):
        raw = normalize_mesh(raw)
    with progress.stage(PHASE_NAMES[1]):
        working, stats = ingest(raw)
    with progress.stage(PHASE_NAMES[2]):
        tri = triage(working, n_pieces=n_pieces)
    with progress.stage(PHASE_NAMES[3]):
        bridges = build_bridges(working, tri, force_exact_count=force_exact_count, progress=progress)
    with progress.stage(PHASE_NAMES[4]):
        graph = build_dual_graph(working, bridges)
    with progress.stage(PHASE_NAMES[5]):
        seed_result = seed(graph, tri, bridges)
    with progress.stage(PHASE_NAMES[6]):
        growth = grow(working, graph, seed_result, abar=tri.abar)
    with progress.stage(PHASE_NAMES[7]):
        repaired = repair(working, graph, tri, bridges, growth)
    with progress.stage(PHASE_NAMES[8]):
        relaxed = relax(working, graph, tri, bridges, seed_result, growth, repaired, progress=progress)
    with progress.stage(PHASE_NAMES[9]):
        pieces = extract(working, relaxed.label)
    return PipelineResult(
        working=working,
        ingest_stats=stats,
        triage=tri,
        bridges=bridges,
        relaxed=relaxed,
        pieces=pieces,
        stage_timings=progress.stage_timings,
    )
