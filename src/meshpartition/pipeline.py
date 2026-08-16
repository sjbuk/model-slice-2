"""End-to-end run of every implemented stage (0-8, plus the partial Stage 10
extraction), shared by the CLI script and the web UI so the stage-chaining
order lives in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass

from .mesh import RawMesh, WorkingMesh
from .normalize import normalize_mesh
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


def run_pipeline(raw: RawMesh, n_pieces: int, force_exact_count: bool = False) -> PipelineResult:
    raw = normalize_mesh(raw)
    working, stats = ingest(raw)
    tri = triage(working, n_pieces=n_pieces)
    bridges = build_bridges(working, tri, force_exact_count=force_exact_count)
    graph = build_dual_graph(working, bridges)
    seed_result = seed(graph, tri, bridges)
    growth = grow(working, graph, seed_result, abar=tri.abar)
    repaired = repair(working, graph, tri, bridges, growth)
    relaxed = relax(working, graph, tri, bridges, seed_result, growth, repaired)
    pieces = extract(working, relaxed.label)
    return PipelineResult(
        working=working,
        ingest_stats=stats,
        triage=tri,
        bridges=bridges,
        relaxed=relaxed,
        pieces=pieces,
    )
