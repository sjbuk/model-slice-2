"""Web UI: upload an FBX/GLB/OBJ model, choose a piece count, run the pipeline,
preview the result.

Slicing is dispatched as a subprocess (webapp/worker.py) rather than run
inline, so the request/response cycle stays fast even though slicing
itself can take a while.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from threading import Lock

from flask import Flask, jsonify, render_template, request, send_from_directory

APP_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = APP_ROOT / "output" / "web"
WORKER_SCRIPT = Path(__file__).resolve().parent / "worker.py"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".fbx", ".obj", ".glb"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB uploads

# job_id -> {"proc": Popen, "job_dir": Path, "started": float, "response": dict | None}
JOBS: dict[str, dict] = {}
JOBS_LOCK = Lock()


def convert_to_obj(source_path: Path, obj_path: Path) -> None:
    result = subprocess.run(
        ["assimp", "export", str(source_path), str(obj_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0 or not obj_path.exists():
        detail = result.stderr.strip() or result.stdout.strip() or "unknown assimp error"
        raise RuntimeError(f"{source_path.suffix.upper().lstrip('.')} conversion failed: {detail}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/slice", methods=["POST"])
def slice_model():
    file = request.files.get("model")
    if file is None or file.filename == "":
        return jsonify(error="No file uploaded."), 400

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify(error=f"Unsupported file type '{suffix}'. Upload .fbx, .glb, or .obj."), 400

    try:
        n_pieces = int(request.form.get("n_pieces", 8))
    except ValueError:
        return jsonify(error="n_pieces must be an integer."), 400
    if n_pieces < 2:
        return jsonify(error="n_pieces must be at least 2."), 400

    job_id = uuid.uuid4().hex[:12]
    job_dir = OUTPUT_ROOT / job_id
    job_dir.mkdir(parents=True)

    # pipeline_obj is what the slicer actually reads (mesh_io.load_obj only
    # understands OBJ). original_path is the untouched upload, kept around
    # purely so the frontend's "Original texture" preview can load it with a
    # format-native loader (GLTFLoader/FBXLoader) -- assimp's OBJ exporter
    # can't be used for that: it references embedded GLB/FBX textures with
    # its own internal `*N` index syntax instead of real image files, which
    # nothing outside assimp can resolve, so the material just renders flat
    # white.
    pipeline_obj = job_dir / "pipeline.obj"
    original_path = job_dir / f"original{suffix}"
    if suffix == ".obj":
        file.save(pipeline_obj)
        original_path = pipeline_obj
    else:
        file.save(original_path)
        try:
            convert_to_obj(original_path, pipeline_obj)
        except Exception as exc:
            return jsonify(error=str(exc)), 422

    proc = subprocess.Popen(
        [sys.executable, str(WORKER_SCRIPT), str(pipeline_obj), str(n_pieces), str(job_dir)]
    )
    with JOBS_LOCK:
        JOBS[job_id] = {
            "proc": proc,
            "job_dir": job_dir,
            "started": time.monotonic(),
            "response": None,
            "source_name": original_path.name,
        }

    return jsonify(job_id=job_id, status="running"), 202


@app.route("/api/jobs/<job_id>")
def job_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify(error="Unknown job id."), 404

    if job["response"] is not None:
        return jsonify(job["response"])

    proc: subprocess.Popen = job["proc"]
    elapsed = time.monotonic() - job["started"]

    if proc.poll() is None:
        return jsonify(status="running", elapsed_seconds=round(elapsed, 1))

    result_path = job["job_dir"] / "result.json"
    if proc.returncode != 0 or not result_path.exists():
        job["response"] = {
            "status": "error",
            "error": f"Worker process exited unexpectedly (code {proc.returncode}).",
        }
        return jsonify(job["response"])

    payload = json.loads(result_path.read_text())
    if payload.get("status") == "done":
        payload["job_id"] = job_id
        payload["preview_obj"] = f"/output/{job_id}/preview.obj"
        payload["preview_mtl"] = f"/output/{job_id}/preview.mtl"
        payload["piece_urls"] = [
            f"/output/{job_id}/piece_{i:02d}.obj" for i in range(payload["piece_count"])
        ]
        source_name = job["source_name"]
        payload["source_url"] = f"/output/{job_id}/{source_name}"
        payload["source_type"] = Path(source_name).suffix.lstrip(".")
    job["response"] = payload
    return jsonify(payload)


@app.route("/output/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_ROOT, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
