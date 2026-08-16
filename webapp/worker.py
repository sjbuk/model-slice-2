"""Runs one slice job to completion and writes result.json into the job
directory. Launched as a subprocess (not a thread) by app.py so the parent
Flask process holds a real OS process handle it can kill outright if a job
runs past the time budget -- Stage 8's relaxation re-seeding is O(faces^2)
per iteration (see stage8.py's _medoid) and can run long on inputs that
don't converge in a single pass, so a hard cutoff is the safety net.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from meshpartition.mesh_io import load_obj, save_obj
from meshpartition.output import write_puzzle_output
from meshpartition.pipeline import run_pipeline
from meshpartition.stage10 import write_preview_obj


def main() -> None:
    obj_path, n_pieces, job_dir = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    force_exact_count = len(sys.argv) > 4 and sys.argv[4] == "1"
    result_path = job_dir / "result.json"

    try:
        raw = load_obj(obj_path)
        t0 = time.perf_counter()
        result = run_pipeline(raw, n_pieces=n_pieces, force_exact_count=force_exact_count)
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
        }
    except Exception as exc:
        payload = {"status": "error", "error": str(exc)}

    result_path.write_text(json.dumps(payload))


if __name__ == "__main__":
    main()
