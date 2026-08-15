"""Drive an OBJ model through the implemented pipeline stages (0-8) and
report the result. Stage 9 (seam fairing) is not implemented yet; Stage 10
(extraction) is implemented in part -- just enough to write per-piece OBJs.

Usage: python scripts/run_pipeline.py <path-to-obj> [n_pieces]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from meshpartition.mesh_io import load_obj, save_obj
from meshpartition.pipeline import run_pipeline
from meshpartition.stage10 import write_preview_obj


def main() -> None:
    obj_path = sys.argv[1] if len(sys.argv) > 1 else "models/stanford-bunny.obj"
    n_pieces = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    t0 = time.perf_counter()

    print(f"Loading {obj_path} ...")
    raw = load_obj(obj_path)
    print(f"  {raw.face_count} faces, {len(raw.positions)} vertices")

    print(f"Running pipeline (n_pieces={n_pieces}) ...")
    result = run_pipeline(raw, n_pieces=n_pieces)
    elapsed = time.perf_counter() - t0

    working, tri, relaxed = result.working, result.triage, result.relaxed

    print(f"  welded to {working.vertex_count} vertices, {working.face_count} faces surviving "
          f"({result.ingest_stats.degenerate_face_count} degenerate removed)")
    print(f"  {len(tri.major_ids)} major component(s), {len(tri.satellite_ids)} satellite(s)")
    print(f"  abar (target area/piece): {tri.abar:.6f}")

    print()
    print("=== Result ===")
    print(f"Converged: {relaxed.converged} (max area deviation = {relaxed.max_area_deviation:.3%}, "
          f"target <= 25%)")
    print(f"Iterations run: {relaxed.iterations_run}")
    print(f"Score: {relaxed.score:.6f} (scores per iteration: {['%.4f' % s for s in relaxed.scores]})")
    print(f"Regions: {len(relaxed.region_area)}")
    for i, area in enumerate(relaxed.region_area):
        deviation = (area - tri.abar) / tri.abar
        print(f"  piece {i}: area={area:.6f}  deviation={deviation:+.1%}  "
              f"faces={int((relaxed.label == i).sum())}")
    print(f"Elapsed: {elapsed:.2f}s")

    out_dir = Path("output") / Path(obj_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, piece in enumerate(result.pieces):
        save_obj(str(out_dir / f"piece_{i:02d}.obj"), piece)

    write_preview_obj(
        str(out_dir / "preview.obj"),
        str(out_dir / "preview.mtl"),
        working,
        relaxed.label,
    )

    print()
    print(f"Wrote {len(result.pieces)} piece OBJ(s) + a colored combined preview to {out_dir}/")
    print(f"  Open {out_dir / 'preview.obj'} in Blender/MeshLab to see all pieces colored in one file.")
    print(f"  Individual piece_NN.obj files are ready to import into Unity.")


if __name__ == "__main__":
    main()
