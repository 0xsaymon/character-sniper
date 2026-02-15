"""
Character Sniper — Web UI (FastAPI + HTMX + SSE).

Run:
    python server.py
    # or: uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
    FileResponse,
)
from fastapi.templating import Jinja2Templates

from character_sniper import (
    FaceAnalyzer,
    CLIPEncoder,
    DummyCLIP,
    ImageScore,
    discover_folders,
    list_images,
    normalise_weights,
    prepare_original,
    process_folder,
    write_csv,
    copy_selected,
    IMAGE_EXTENSIONS,
)
from session_store import SessionStore

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Session persistence (SQLite)
# ---------------------------------------------------------------------------

store = SessionStore()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    """Restore latest completed job from SQLite on startup."""
    latest = store.load_latest_job()
    if latest and store.is_job_valid(latest["id"]):
        _restore_job_from_db(latest)
    elif latest:
        print(
            f"[session] Latest job {latest['id']} is outdated (input changed), skipping"
        )
    yield


app = FastAPI(title="Character Sniper", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


def _url_folder_filter(folder: str) -> str:
    """Jinja2 filter: convert internal folder name to URL-safe segment."""
    return "_flat" if folder == "." else folder


templates.env.filters["url_folder"] = _url_folder_filter

# ---------------------------------------------------------------------------
# Global model cache  (loaded once, reused across jobs)
# ---------------------------------------------------------------------------

_model_cache: dict[str, object] = {}


def load_models(clip_weight: float) -> tuple[FaceAnalyzer, object]:
    """Load or return cached models."""
    if "face_analyzer" not in _model_cache:
        print("[server] Loading FaceAnalyzer …")
        _model_cache["face_analyzer"] = FaceAnalyzer()
    face_analyzer = _model_cache["face_analyzer"]

    if clip_weight > 0:
        if "clip_encoder" not in _model_cache or isinstance(
            _model_cache["clip_encoder"], DummyCLIP
        ):
            print("[server] Loading CLIPEncoder …")
            _model_cache["clip_encoder"] = CLIPEncoder()
        clip_encoder = _model_cache["clip_encoder"]
    else:
        clip_encoder = DummyCLIP()

    return face_analyzer, clip_encoder


# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------

jobs: dict[str, dict] = {}


def _restore_job_from_db(saved: dict) -> None:
    """Hydrate a saved DB job into the in-memory *jobs* dict."""
    job_id = saved["id"]
    jobs[job_id] = {
        "status": saved["status"],
        "progress": 100 if saved["status"] == "complete" else 0,
        "current_file": "",
        "current_folder": "",
        "folders_done": 0,
        "folders_total": 0,
        "images_done": 0,
        "images_total": 0,
        "results": saved["results"],
        "all_results": [],  # not needed for display — only for export
        "original_path": saved["original_path"],
        "input_path": saved["input_path"],
        "params": saved["params"],
        "error": saved.get("error"),
        "is_recursive": saved.get("is_recursive", False),
        "start_time": time.time() - (saved.get("elapsed") or 0),
        "end_time": time.time() if saved["status"] == "complete" else None,
    }
    print(f"[session] Restored job {job_id} from database")


def new_job(original_path: str, input_path: str, params: dict) -> str:
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "current_file": "",
        "current_folder": "",
        "folders_done": 0,
        "folders_total": 0,
        "images_done": 0,
        "images_total": 0,
        "results": {},  # folder_name -> [ImageScore.to_dict(), ...]
        "all_results": [],  # flat list of ImageScore objects (for export)
        "original_path": original_path,
        "input_path": input_path,
        "params": params,
        "error": None,
        "is_recursive": False,
        "start_time": time.time(),
        "end_time": None,
    }
    return job_id


# ---------------------------------------------------------------------------
# Background ML processing
# ---------------------------------------------------------------------------


async def run_job(job_id: str):
    job = jobs[job_id]
    try:
        job["status"] = "loading_models"

        params = job["params"]
        face_w = params["face_weight"]
        clip_w = params["clip_weight"]

        # Load models in thread
        face_analyzer, clip_encoder = await asyncio.to_thread(load_models, clip_w)

        # Prepare original embeddings
        original_path = Path(job["original_path"])
        orig_face_emb, orig_clip_emb = await asyncio.to_thread(
            prepare_original,
            original_path,
            face_analyzer,
            clip_encoder,
        )

        # Discover folders
        input_dir = Path(job["input_path"])
        is_recursive, folders = await asyncio.to_thread(discover_folders, input_dir)
        job["is_recursive"] = is_recursive

        # Count total images
        all_image_lists: list[tuple[Path, list[Path]]] = []
        total_images = 0
        for folder in folders:
            imgs = list_images(folder)
            if imgs:
                all_image_lists.append((folder, imgs))
                total_images += len(imgs)

        job["folders_total"] = len(all_image_lists)
        job["images_total"] = total_images
        job["status"] = "running"

        all_results: list[ImageScore] = []

        for folder_idx, (folder, images) in enumerate(all_image_lists):
            folder_name = folder.name if is_recursive else "."
            job["current_folder"] = folder_name

            def on_progress(current: int, total: int, filename: str):
                job["images_done"] += 1
                job["current_file"] = filename
                if job["images_total"] > 0:
                    job["progress"] = int(
                        job["images_done"] / job["images_total"] * 100
                    )

            results = await asyncio.to_thread(
                process_folder,
                image_paths=images,
                folder_name=folder_name,
                face_analyzer=face_analyzer,
                clip_encoder=clip_encoder,
                orig_face_emb=orig_face_emb,
                orig_clip_emb=orig_clip_emb,
                face_weight=face_w,
                clip_weight=clip_w,
                min_face_score=params["min_face_score"],
                top_k=params["top_k"],
                on_progress=on_progress,
            )

            # Store results as dicts for JSON/HTML
            job["results"][folder_name] = [r.to_dict() for r in results]
            all_results.extend(results)

            job["folders_done"] = folder_idx + 1

        job["all_results"] = all_results
        job["progress"] = 100
        job["status"] = "complete"
        job["end_time"] = time.time()

        # Persist to SQLite
        store.save_job(job_id, job)

    except Exception as e:
        import traceback

        traceback.print_exc()
        job["status"] = "error"
        job["error"] = str(e)


# ---------------------------------------------------------------------------
# Routes — Pages
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------------------------------------------------------------------
# Routes — API: Session persistence
# ---------------------------------------------------------------------------


@app.get("/api/settings")
async def get_settings():
    """Return saved user settings (or defaults)."""
    return JSONResponse(store.load_settings())


@app.post("/api/settings")
async def save_settings(request: Request):
    """Save user settings (called from frontend on form changes)."""
    data = await request.json()
    store.save_settings(
        {
            "original_path": str(data.get("original_path", "data/original.png")),
            "input_path": str(data.get("input_path", "data/output")),
            "top_k": int(data.get("top_k", 5)),
            "face_weight": float(data.get("face_weight", 0.7)),
            "clip_weight": float(data.get("clip_weight", 0.3)),
            "min_face_score": float(data.get("min_face_score", 0.5)),
        }
    )
    return JSONResponse({"ok": True})


@app.get("/api/jobs/latest")
async def latest_job():
    """Return the latest completed job info for session restore."""
    latest = store.load_latest_job()
    if not latest:
        return JSONResponse({"job_id": None})

    job_id = latest["id"]
    valid = store.is_job_valid(job_id)

    # Ensure the job is loaded in memory (might have been evicted)
    if valid and job_id not in jobs:
        _restore_job_from_db(latest)

    return JSONResponse(
        {
            "job_id": job_id if valid else None,
            "valid": valid,
            "settings": latest["params"],
            "original_path": latest["original_path"],
            "input_path": latest["input_path"],
        }
    )


# ---------------------------------------------------------------------------
# Routes — API: Jobs
# ---------------------------------------------------------------------------


@app.post("/api/jobs/start", response_class=HTMLResponse)
async def start_job(request: Request):
    form = await request.form()
    original_path = str(form.get("original_path", "data/original.png"))
    input_path = str(form.get("input_path", "data/output"))
    top_k = int(form.get("top_k", 5))
    face_weight = float(form.get("face_weight", 0.7))
    clip_weight = float(form.get("clip_weight", 0.3))
    min_face_score = float(form.get("min_face_score", 0.5))

    # Normalise weights
    face_w, clip_w = normalise_weights("combined", face_weight, clip_weight)

    # Persist settings for next session
    store.save_settings(
        {
            "original_path": original_path,
            "input_path": input_path,
            "top_k": top_k,
            "face_weight": face_weight,
            "clip_weight": clip_weight,
            "min_face_score": min_face_score,
        }
    )

    # Validate paths
    if not Path(original_path).exists():
        return HTMLResponse(
            f'<div class="text-red-400 p-4">Original image not found: {original_path}</div>',
            status_code=400,
        )
    if not Path(input_path).exists():
        return HTMLResponse(
            f'<div class="text-red-400 p-4">Input folder not found: {input_path}</div>',
            status_code=400,
        )

    job_id = new_job(
        original_path,
        input_path,
        {
            "top_k": top_k,
            "face_weight": face_w,
            "clip_weight": clip_w,
            "min_face_score": min_face_score,
        },
    )

    # Launch background task
    asyncio.create_task(run_job(job_id))

    # Return HTML that connects to SSE for progress
    # NOTE: sse-close uses a separate "done" event to avoid TypeError
    # when the same event name is used for both swap and close.
    html = f"""
    <div id="sse-container"
         hx-ext="sse"
         sse-connect="/api/jobs/{job_id}/progress"
         sse-close="done">
        <div id="progress-content" sse-swap="progress">
            <div class="flex items-center gap-3 mb-2">
                <div class="animate-spin h-5 w-5 border-2 border-indigo-400 border-t-transparent rounded-full"></div>
                <span class="text-gray-300">Initializing...</span>
            </div>
            <div class="w-full bg-gray-700 rounded-full h-3">
                <div class="bg-indigo-500 h-3 rounded-full transition-all duration-300" style="width: 0%"></div>
            </div>
        </div>
        <div id="results-loader" sse-swap="complete"></div>
    </div>
    <input type="hidden" id="current-job-id" value="{job_id}" />
    """
    return HTMLResponse(html)


@app.get("/api/jobs/{job_id}/progress")
async def job_progress(job_id: str, request: Request):
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    async def event_stream():
        job = jobs[job_id]
        while True:
            if await request.is_disconnected():
                break

            status = job["status"]
            progress = job["progress"]
            current_file = job["current_file"]
            current_folder = job["current_folder"]
            folders_done = job["folders_done"]
            folders_total = job["folders_total"]
            images_done = job["images_done"]
            images_total = job["images_total"]

            if status == "loading_models":
                html = """
                <div class="flex items-center gap-3 mb-2">
                    <div class="animate-spin h-5 w-5 border-2 border-indigo-400 border-t-transparent rounded-full"></div>
                    <span class="text-gray-300">Loading ML models (this takes ~7s on first run)...</span>
                </div>
                <div class="w-full bg-gray-700 rounded-full h-3">
                    <div class="bg-indigo-500 h-3 rounded-full transition-all duration-300 animate-pulse" style="width: 15%"></div>
                </div>
                """
            elif status == "running":
                folder_info = (
                    f"Folder {folders_done + 1}/{folders_total}: {current_folder}"
                    if folders_total > 1
                    else ""
                )
                html = f"""
                <div class="mb-2">
                    <div class="flex items-center justify-between text-sm text-gray-300 mb-1">
                        <span>Processing {folder_info} &mdash; {current_file}</span>
                        <span class="font-mono text-indigo-400">{progress}% ({images_done}/{images_total})</span>
                    </div>
                    <div class="w-full bg-gray-700 rounded-full h-3">
                        <div class="bg-indigo-500 h-3 rounded-full transition-all duration-300" style="width: {progress}%"></div>
                    </div>
                </div>
                """
            elif status == "complete":
                elapsed = (job["end_time"] or time.time()) - job["start_time"]
                html = f"""
                <div class="flex items-center gap-2 text-green-400 mb-2">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                    </svg>
                    <span>Complete! Processed {images_total} images in {elapsed:.1f}s</span>
                </div>
                """
                yield f"event: progress\ndata: {_sse_encode(html)}\n\n"
                # Send complete event with auto-loading HTML
                complete_html = f'<div hx-get="/api/jobs/{job_id}/results" hx-trigger="load" hx-target="#results-section" hx-swap="innerHTML"></div>'
                yield f"event: complete\ndata: {_sse_encode(complete_html)}\n\n"
                # Send done event to close SSE connection cleanly
                yield "event: done\ndata: close\n\n"
                break
            elif status == "error":
                html = f"""
                <div class="text-red-400 flex items-center gap-2">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                    </svg>
                    <span>Error: {job.get("error", "Unknown error")}</span>
                </div>
                """
                yield f"event: progress\ndata: {_sse_encode(html)}\n\n"
                yield f"event: complete\ndata: {_sse_encode(html)}\n\n"
                yield "event: done\ndata: close\n\n"
                break
            else:
                html = """
                <div class="flex items-center gap-3">
                    <div class="animate-spin h-5 w-5 border-2 border-indigo-400 border-t-transparent rounded-full"></div>
                    <span class="text-gray-300">Queued...</span>
                </div>
                """

            yield f"event: progress\ndata: {_sse_encode(html)}\n\n"
            await asyncio.sleep(0.3)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse_encode(html: str) -> str:
    """Encode multiline HTML for SSE (replace newlines with SSE continuation)."""
    # SSE data lines: each line must start with 'data: '
    # We join all lines into one for simplicity (HTMX handles it)
    return html.replace("\n", " ").strip()


@app.get("/api/jobs/{job_id}/results", response_class=HTMLResponse)
async def job_results(job_id: str, request: Request):
    # Try loading from DB if not in memory
    if job_id not in jobs:
        saved = store.load_job(job_id)
        if saved:
            _restore_job_from_db(saved)

    if job_id not in jobs:
        return HTMLResponse("<div class='text-red-400'>Job not found</div>", 404)

    job = jobs[job_id]
    if job["status"] != "complete":
        return HTMLResponse("<div class='text-gray-400'>Job not complete yet</div>")

    results = job["results"]
    elapsed = (job["end_time"] or time.time()) - job["start_time"]
    folder_names = list(results.keys())

    # Count stats
    total_images = 0
    total_selected = 0
    total_rejected = 0
    for folder_results in results.values():
        for img in folder_results:
            total_images += 1
            if img["selected"]:
                total_selected += 1
            if img["reject_reason"]:
                total_rejected += 1

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "job_id": job_id,
            "results": results,
            "folder_names": folder_names,
            "total_images": total_images,
            "total_selected": total_selected,
            "total_rejected": total_rejected,
            "elapsed": f"{elapsed:.1f}",
            "is_recursive": job["is_recursive"],
        },
    )


@app.post("/api/jobs/{job_id}/toggle/{folder}/{filename}", response_class=HTMLResponse)
async def toggle_image(job_id: str, folder: str, filename: str, request: Request):
    if not _ensure_job_in_memory(job_id):
        return HTMLResponse("Job not found", 404)

    # Convert URL folder back to internal name
    internal_folder = "." if folder == "_flat" else folder

    job = jobs[job_id]
    folder_results = job["results"].get(internal_folder, [])

    for img in folder_results:
        if img["filename"] == filename:
            img["selected"] = not img["selected"]
            # Also update the ImageScore object in all_results
            for r in job.get("all_results", []):
                if r.path.name == filename and r.folder == internal_folder:
                    r.selected = img["selected"]
            # Persist selection change to SQLite
            store.update_selection(job_id, internal_folder, filename, img["selected"])
            return _render_image_card(job_id, internal_folder, img)

    return HTMLResponse("Image not found", 404)


def _url_folder(folder: str) -> str:
    """Convert internal folder name to URL-safe folder segment."""
    return "_flat" if folder == "." else folder


def _render_image_card(job_id: str, folder: str, img: dict) -> HTMLResponse:
    """Render a single image card HTML (must stay in sync with results.html card template)."""
    selected = img["selected"]
    border = (
        "border-indigo-500 ring-2 ring-indigo-500/50" if selected else "border-gray-700"
    )
    check = (
        """<div class="absolute top-2 right-2 bg-indigo-500 rounded-full p-1">
            <svg class="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7"></path>
            </svg>
        </div>"""
        if selected
        else ""
    )

    final_score = img["final_score"]
    face_score = img["face_score"]
    clip_score = img["clip_score"]
    reject = img["reject_reason"]

    if final_score is not None:
        score_badge = f'<span class="px-2 py-0.5 rounded-full text-xs font-bold bg-indigo-500/30 text-indigo-300">{final_score:.3f}</span>'
        details = f'<span class="text-[10px] text-gray-500">F:{face_score:.3f} C:{clip_score:.3f}</span>'
    elif reject:
        score_badge = f'<span class="px-2 py-0.5 rounded-full text-xs bg-red-500/20 text-red-400">{reject}</span>'
        details = ""
    else:
        score_badge = '<span class="px-2 py-0.5 rounded-full text-xs bg-gray-600 text-gray-400">N/A</span>'
        details = ""

    filename = img["filename"]
    url_folder = _url_folder(folder)
    safe_folder = url_folder.replace("/", "__")
    safe_filename = filename.replace(".", "_")
    card_id = f"card-{safe_folder}-{safe_filename}"

    # Toggle button: Select or Deselect
    if selected:
        toggle_btn = f"""<button hx-post="/api/jobs/{job_id}/toggle/{url_folder}/{filename}"
                                hx-target="#{card_id}" hx-swap="outerHTML"
                                class="flex-1 px-2 py-1 rounded text-[11px] font-medium transition-colors
                                       bg-red-500/20 text-red-400 hover:bg-red-500/30"
                                title="Remove from selection">Deselect</button>"""
    else:
        toggle_btn = f"""<button hx-post="/api/jobs/{job_id}/toggle/{url_folder}/{filename}"
                                hx-target="#{card_id}" hx-swap="outerHTML"
                                class="flex-1 px-2 py-1 rounded text-[11px] font-medium transition-colors
                                       bg-indigo-500/20 text-indigo-300 hover:bg-indigo-500/30"
                                title="Add to selection">Select</button>"""

    compare_btn = f"""<button @click="$dispatch('compare-image', {{
                            src: '/images/{job_id}/{url_folder}/{filename}',
                            filename: '{filename}',
                            folder: '{folder}',
                            face_score: '{face_score}',
                            clip_score: '{clip_score}',
                            final_score: '{final_score}',
                            cardId: '{card_id}'
                        }})"
                        class="px-2 py-1 rounded text-[11px] font-medium transition-colors
                               bg-gray-700 text-gray-300 hover:bg-gray-600"
                        title="Compare with original">Compare</button>"""

    # Determine status for filtering
    if selected:
        status = "selected"
    elif reject:
        status = "rejected"
    else:
        status = "scored"

    html = f"""
    <div id="{card_id}"
         class="relative rounded-lg overflow-hidden border-2 {border} bg-gray-800 transition-all duration-200 group"
         data-compare-src="/images/{job_id}/{url_folder}/{filename}"
         data-compare-filename="{filename}"
         data-compare-face="{face_score}"
         data-compare-clip="{clip_score}"
         data-compare-final="{final_score}"
         data-compare-folder="{folder}"
         data-status="{status}"
    >
        <div class="aspect-[3/4] overflow-hidden bg-gray-900">
            <img src="/images/{job_id}/{url_folder}/{filename}"
                 alt="{filename}"
                 class="w-full h-full object-cover"
                 loading="lazy" />
        </div>
        <div class="p-2 space-y-1.5">
            <div class="flex items-center justify-between gap-1">
                {score_badge}
                {details}
            </div>
            <p class="text-[10px] text-gray-500 truncate" title="{filename}">{filename}</p>
            <div class="flex items-center gap-1.5 pt-0.5">
                {toggle_btn}
                {compare_btn}
            </div>
        </div>
        {check}
    </div>
    """
    return HTMLResponse(html)


async def _do_export(job: dict) -> tuple[int, Path]:
    """Export selected images and CSV report to results/. Return (count, output_dir)."""
    all_results = job.get("all_results", [])
    output_dir = Path("results")
    is_recursive = job["is_recursive"]

    n_copied = await asyncio.to_thread(
        copy_selected, all_results, output_dir, is_recursive
    )
    await asyncio.to_thread(write_csv, all_results, output_dir / "report.csv")
    return n_copied, output_dir


@app.post("/api/jobs/{job_id}/export", response_class=HTMLResponse)
async def export_job(job_id: str):
    if not _ensure_job_in_memory(job_id):
        return HTMLResponse("Job not found", 404)

    job = jobs[job_id]
    if job["status"] != "complete":
        return HTMLResponse("Job not complete", 400)

    if not job.get("all_results"):
        return HTMLResponse(
            '<div class="flex items-center gap-2 px-4 py-2.5 bg-yellow-900/80 border border-yellow-500/30 rounded-lg text-yellow-300 text-sm shadow-lg backdrop-blur-sm animate-toast" onanimationend="this.remove()">No results to export</div>'
        )

    n_copied, output_dir = await _do_export(job)

    html = f"""
    <div class="flex items-center gap-2 px-4 py-2.5 bg-green-900/80 border border-green-500/30 rounded-lg text-green-300 text-sm shadow-lg backdrop-blur-sm animate-toast"
         onanimationend="this.remove()">
        <svg class="w-4 h-4 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
        </svg>
        <span>{n_copied} images exported to <code class="bg-gray-800 px-1 rounded">results/</code></span>
    </div>
    """
    return HTMLResponse(html)


@app.post("/api/jobs/{job_id}/export-download")
async def export_download_job(job_id: str):
    """Export selected images, then zip results/ and return as a downloadable archive."""
    if not _ensure_job_in_memory(job_id):
        return JSONResponse({"error": "Job not found"}, 404)

    job = jobs[job_id]
    if job["status"] != "complete":
        return JSONResponse({"error": "Job not complete"}, 400)

    if not job.get("all_results"):
        return JSONResponse({"error": "No results to export"}, 400)

    n_copied, output_dir = await _do_export(job)

    # Create zip archive in a temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp.close()
    zip_path = Path(tmp.name)

    await asyncio.to_thread(
        shutil.make_archive,
        str(zip_path.with_suffix("")),  # base name without .zip
        "zip",
        str(output_dir),
    )

    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=f"character-sniper-results-{job_id}.zip",
        background=None,  # cleaned up below
    )


@app.get("/api/browse")
async def browse_path(path: str = "."):
    """Return directory listing as JSON for folder picker."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return JSONResponse({"error": "Path not found", "entries": []})
    if not p.is_dir():
        return JSONResponse({"error": "Not a directory", "entries": []})

    entries = []
    try:
        for child in sorted(p.iterdir()):
            if child.name.startswith("."):
                continue
            entries.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "is_dir": child.is_dir(),
                    "is_image": child.suffix.lower() in IMAGE_EXTENSIONS
                    if child.is_file()
                    else False,
                }
            )
    except PermissionError:
        return JSONResponse({"error": "Permission denied", "entries": []})

    return JSONResponse({"path": str(p), "entries": entries})


# ---------------------------------------------------------------------------
# Image serving
# ---------------------------------------------------------------------------


def _ensure_job_in_memory(job_id: str) -> bool:
    """Load job from DB into memory if needed. Return True if available."""
    if job_id in jobs:
        return True
    saved = store.load_job(job_id)
    if saved:
        _restore_job_from_db(saved)
        return True
    return False


@app.get("/images/{job_id}/original")
async def serve_original(job_id: str):
    if not _ensure_job_in_memory(job_id):
        return JSONResponse({"error": "Job not found"}, 404)
    path = Path(jobs[job_id]["original_path"])
    if not path.exists():
        return JSONResponse({"error": "File not found"}, 404)
    return FileResponse(path)


@app.get("/images/{job_id}/_flat/{filename}")
async def serve_image_flat(job_id: str, filename: str):
    """Serve images from flat (non-recursive) input folders."""
    if not _ensure_job_in_memory(job_id):
        return JSONResponse({"error": "Job not found"}, 404)
    file_path = Path(jobs[job_id]["input_path"]) / filename
    if not file_path.exists():
        return JSONResponse({"error": "File not found"}, 404)
    return FileResponse(file_path)


@app.get("/images/{job_id}/{folder}/{filename}")
async def serve_image(job_id: str, folder: str, filename: str):
    if not _ensure_job_in_memory(job_id):
        return JSONResponse({"error": "Job not found"}, 404)

    job = jobs[job_id]
    input_path = Path(job["input_path"])
    file_path = input_path / folder / filename

    if not file_path.exists():
        return JSONResponse({"error": "File not found"}, 404)

    return FileResponse(file_path)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
