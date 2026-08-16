"""Runs one slice job to completion and writes result.json into the job
directory. Launched as a subprocess (not a thread) by app.py so the parent
Flask process holds a real OS process handle it can kill outright if a job
runs past the time budget -- Stage 8's relaxation re-seeding is O(faces^2)
per iteration (see stage8.py's _medoid) and can run long on inputs that
don't converge in a single pass, so a hard cutoff is the safety net.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from meshpartition.mesh_io import load_obj, save_obj
from meshpartition.output import write_puzzle_output
from meshpartition.pipeline import run_pipeline
from meshpartition.progress import PHASE_NAMES, ProgressReporter
from meshpartition.stage10 import write_preview_obj

# "load" isn't one of pipeline.py's stages -- it's parsing the (possibly
# very large, for a converted GLB/FBX) OBJ file before run_pipeline even
# starts. Without its own stage, the progress UI shows nothing at all for
# however long that parse takes, then jumps straight to (or past) the
# pipeline's first real stage the moment it finishes.
WORKER_PHASE_NAMES = ["load", *PHASE_NAMES]


def _write_progress(job_dir: Path, state: dict) -> None:
    tmp_path = job_dir / "progress.json.tmp"
    tmp_path.write_text(json.dumps(state))
    os.replace(tmp_path, job_dir / "progress.json")


def _source_file_size(job_dir: Path, obj_path: str, source_name: str) -> int:
    """The pipeline reads a converted pipeline.obj, not the original upload
    (see app.py's convert_to_obj) -- for non-OBJ uploads, prefer the
    original file's size so model_stats reflects what was actually
    uploaded, not the assimp-exported intermediate."""
    suffix = Path(source_name).suffix.lower()
    if suffix and suffix != ".obj":
        original = job_dir / f"original{suffix}"
        if original.exists():
            return original.stat().st_size
    return Path(obj_path).stat().st_size


def main() -> None:
    obj_path, n_pieces, job_dir = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    force_exact_count = len(sys.argv) > 4 and sys.argv[4] == "1"
    source_name = sys.argv[5] if len(sys.argv) > 5 else Path(obj_path).name
    result_path = job_dir / "result.json"

    reporter = ProgressReporter(
        phases=WORKER_PHASE_NAMES, on_update=lambda state: _write_progress(job_dir, state)
    )

    try:
        with reporter.stage("load"):
            raw = load_obj(obj_path)
        t0 = time.perf_counter()
        result = run_pipeline(
            raw, n_pieces=n_pieces, force_exact_count=force_exact_count, progress=reporter
        )
        elapsed = time.perf_counter() - t0

        for i, piece in enumerate(result.pieces):
            save_obj(str(job_dir / f"piece_{i:02d}.obj"), piece)
        write_preview_obj(
            str(job_dir / "preview.obj"),
            str(job_dir / "preview.mtl"),
            result.working,
            result.relaxed.label,
        )
        write_puzzle_output(job_dir, result.working, result.pieces)

        pieces_stats = [
            {
                "index": i,
                "area": float(area),
                "deviation": float((area - result.triage.abar) / result.triage.abar),
                "faces": int((result.relaxed.label == i).sum()),
            }
            for i, area in enumerate(result.relaxed.region_area)
        ]

        payload = {
            "status": "done",
            "elapsed_seconds": round(elapsed, 2),
            "source_faces": raw.face_count,
            "force_exact_count": force_exact_count,
            "converged": result.relaxed.converged,
            "max_area_deviation": result.relaxed.max_area_deviation,
            "iterations_run": result.relaxed.iterations_run,
            "piece_count": len(result.pieces),
            "pieces": pieces_stats,
            "stage_timings": reporter.stage_timings,
            "model_stats": {
                "source_name": source_name,
                "source_bytes": _source_file_size(job_dir, obj_path, source_name),
                "raw_faces": raw.face_count,
                "raw_vertices": len(raw.positions),
                "welded_faces": result.working.face_count,
                "welded_vertices": result.working.vertex_count,
                "degenerate_faces_removed": result.ingest_stats.degenerate_face_count,
                "major_count": len(result.triage.major_ids),
                "satellite_count": len(result.triage.satellite_ids),
                "n_pieces_requested": n_pieces,
                "force_exact_count": force_exact_count,
            },
        }
    except Exception as exc:
        payload = {
            "status": "error",
            "error": str(exc),
            "stage_timings": reporter.stage_timings,
        }

    result_path.write_text(json.dumps(payload))


if __name__ == "__main__":
    main()
