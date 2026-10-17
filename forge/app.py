"""Local-only HTTP application. Run with python -m forge serve."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .geography import DEMO_CENTER, centered_bbox, demo_osm, parse_location, validate_bbox
from .photos import analyze_photo
from .pipeline import generate
from .scene import build_scene

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
JOBS = DATA / "jobs"
UPLOADS = DATA / "uploads"
for folder in (JOBS, UPLOADS):
    folder.mkdir(parents=True, exist_ok=True)
Image.MAX_IMAGE_PIXELS = 40_000_000
app = FastAPI(title="RealityForge", version="0.1.0", docs_url="/api/docs")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver", "[::1]"])
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="realityforge")
state_lock = threading.Lock()
states: dict[str, dict] = {}
FILE_NAMES = {"world": "RealityForge-world.zip", "schematic": "realityforge.schem", "model": "scene.obj", "scene": "scene.json", "report": "report.json", "materials": "scene.mtl"}


@app.middleware("http")
async def local_origin(request: Request, call_next):
    origin = request.headers.get("origin")
    if request.method in {"POST", "PUT", "DELETE"} and origin and origin != str(request.base_url).rstrip("/"):
        return JSONResponse({"detail": "Use this app from its local browser address."}, status_code=403)
    length = request.headers.get("content-length", "0")
    if length.isdigit() and int(length) > 96 * 1024 * 1024:
        return JSONResponse({"detail": "Upload batch exceeds 96 MB."}, status_code=413)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


class PhotoSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str = Field(pattern=r"^[a-f0-9]{32}$")
    lat: float = Field(ge=-85, le=85)
    lon: float = Field(ge=-180, le=180)
    style: Literal["auto", "brick", "stone", "wood", "modern"] = "auto"
    radius_m: float = Field(default=80, ge=5, le=250)
    heading: float | None = Field(default=None, ge=0, lt=360)


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str = Field(default="RealityForge", min_length=1, max_length=80)
    bbox: list[float] = Field(min_length=4, max_length=4)
    meters_per_block: Literal[1, 2, 4] = 1
    source: Literal["live", "demo"] = "live"
    photos: list[PhotoSelection] = Field(default_factory=list, max_length=12)
    terrain: bool = False

    @field_validator("bbox")
    @classmethod
    def bbox_is_valid(cls, value):
        return validate_bbox(value)


class LocationRequest(BaseModel):
    value: str = Field(min_length=1, max_length=4000)


def identifier(value):
    if not re.fullmatch(r"[a-f0-9]{32}", value):
        raise HTTPException(404, "Not found")
    return value


def update(job_id, **fields):
    with state_lock:
        states[job_id].update(fields)
        snapshot = dict(states[job_id])
        path = JOBS / job_id / "status.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(snapshot), encoding="utf-8")
        temp.replace(path)


def run_job(job_id, payload):
    try:
        update(job_id, status="running")
        result = generate(payload, JOBS / job_id, DATA / "cache",
                          lambda progress, stage, message: update(job_id, progress=progress, stage=stage, message=message))
        files = {key: f"/api/jobs/{job_id}/files/{key}" for key, name in FILE_NAMES.items() if (JOBS / job_id / name).exists()}
        update(job_id, status="complete", progress=100, stage="complete", message="Your Java world is ready.",
               files=files, scene_url=f"/api/jobs/{job_id}/scene", stats=result["stats"], warnings=result["report"]["limitations"])
    except Exception as exc:
        update(job_id, status="failed", stage="failed", message="Generation stopped. Your original files are unchanged.", error=str(exc)[:600])


@app.get("/api/health")
def health():
    return {"status": "ok", "name": "RealityForge", "version": "0.1.0", "target": "Java 1.20.4"}


@app.get("/api/demo")
def demo():
    bbox = centered_bbox(*DEMO_CENTER)
    scene = build_scene(demo_osm(bbox), bbox)
    scene["source"] = "demo"
    scene["name"] = "Forge Gardens"
    scene["limitations"].append("Synthetic demonstration, not an actual Toronto neighbourhood.")
    return {"scene": scene, "bbox": bbox, "name": "Forge Gardens", "source": "demo"}


@app.post("/api/location")
def location(payload: LocationRequest):
    try:
        return parse_location(payload.value)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/photos")
def photos(files: list[UploadFile] = File(...)):
    if len(files) > 12:
        raise HTTPException(422, "Upload up to 12 photos at a time.")
    results = []
    for upload in files:
        raw = upload.file.read(16 * 1024 * 1024 + 1)
        if len(raw) > 16 * 1024 * 1024:
            raise HTTPException(413, "Each photo must be 16 MB or smaller.")
        import io
        try:
            with Image.open(io.BytesIO(raw)) as original:
                if original.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("Use JPEG, PNG or WebP. Convert HEIC photos to JPEG first.")
                if original.width * original.height > 40_000_000:
                    raise ValueError("Photo exceeds 40 megapixels. Resize it before uploading.")
                original.load()
                clean = ImageOps.exif_transpose(original).convert("RGB")
                clean.thumbnail((4096, 4096))
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            raise HTTPException(422, f"Could not read photo: {str(exc)[:150]}") from exc
        photo_id = uuid.uuid4().hex
        path = UPLOADS / f"{photo_id}.jpg"
        clean.save(path, "JPEG", quality=92)
        data = analyze_photo(path)
        data.update(id=photo_id, filename=Path(upload.filename or "photo.jpg").name[:120], url=f"/api/photos/{photo_id}")
        (UPLOADS / f"{photo_id}.json").write_text(json.dumps(data), encoding="utf-8")
        results.append(data)
    return {"photos": results}


@app.get("/api/photos/{photo_id}")
def photo(photo_id: str):
    path = UPLOADS / f"{identifier(photo_id)}.jpg"
    if not path.exists():
        raise HTTPException(404, "Photo not found")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/generate", status_code=202)
def start_generation(request: GenerationRequest):
    payload = request.model_dump()
    for photo in payload["photos"]:
        path = UPLOADS / f"{photo['id']}.json"
        if not path.exists():
            raise HTTPException(422, "A selected photo no longer exists. Upload it again.")
        photo.update({**json.loads(path.read_text(encoding="utf-8")), **photo})
    with state_lock:
        if sum(s["status"] in {"queued", "running"} for s in states.values()) >= 3:
            raise HTTPException(429, "The generation queue is full. Wait for a world to finish.")
        job_id = uuid.uuid4().hex
        (JOBS / job_id).mkdir()
        states[job_id] = {"id": job_id, "status": "queued", "progress": 0, "stage": "queued", "message": "Waiting for the local generator"}
    update(job_id)
    executor.submit(run_job, job_id, payload)
    return {"id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    identifier(job_id)
    with state_lock:
        if job_id in states:
            return dict(states[job_id])
    path = JOBS / job_id / "status.json"
    if not path.exists():
        raise HTTPException(404, "Generation not found")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved["status"] in {"running", "queued"}:
        saved.update(status="failed", error="Server restarted during generation. Generate again.")
    return saved


@app.get("/api/jobs/{job_id}/scene")
def job_scene(job_id: str):
    path = JOBS / identifier(job_id) / "scene.json"
    if not path.exists():
        raise HTTPException(404, "Scene is not ready yet")
    return FileResponse(path, media_type="application/json")


@app.get("/api/jobs/{job_id}/files/{kind}")
def artifact(job_id: str, kind: str):
    identifier(job_id)
    if kind not in FILE_NAMES:
        raise HTTPException(404, "Artifact not found")
    path = JOBS / job_id / FILE_NAMES[kind]
    if not path.exists():
        raise HTTPException(404, "Artifact is not ready yet")
    return FileResponse(path, filename=path.name)


@app.post("/api/jobs/{job_id}/install")
def install_world(job_id: str):
    """Install into a fresh, uniquely named save; never replace an existing world."""
    job = get_job(job_id)
    if job["status"] != "complete":
        raise HTTPException(409, "Finish generating this world first.")
    if sys.platform == "win32":
        saves = Path(os.environ.get("APPDATA", str(Path.home() / "AppData/Roaming"))) / ".minecraft/saves"
    elif sys.platform == "darwin":
        saves = Path.home() / "Library/Application Support/minecraft/saves"
    else:
        saves = Path.home() / ".minecraft/saves"
    source = JOBS / identifier(job_id) / "RealityForge"
    if not (source / "level.dat").exists():
        raise HTTPException(404, "The generated world folder is missing.")
    destination = saves / f"RealityForge-{job_id[:8]}"
    if destination.exists():
        raise HTTPException(409, "This world has already been installed. Use the ZIP to make another copy.")
    try:
        shutil.copytree(source, destination)
    except OSError as exc:
        raise HTTPException(500, f"Could not install the save: {exc}") from exc
    return {"path": str(destination), "message": "Installed as a new save. Open Minecraft Java 1.20.4 → Singleplayer."}


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
