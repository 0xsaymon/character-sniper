# AGENTS.md — Character Sniper

## Project Overview

Python CLI + FastAPI web app for AI-powered image similarity scoring.
Compares generated images against a reference face using InsightFace (ArcFace) embeddings
(70% weight) and OpenCLIP (ViT-B-32) visual similarity (30% weight).

Two entry points:
- **CLI**: `character_sniper.py` — batch processing, CSV reports, top-K selection
- **Web UI**: `server.py` — FastAPI + HTMX/SSE real-time UI with gallery and compare modal

## Build / Run / Test Commands

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install onnxruntime          # macOS
# pip install onnxruntime-gpu    # NVIDIA GPU

# Run CLI
python character_sniper.py --report

# Run web server
python server.py
# or: uvicorn server:app --reload --port 8000

# Lint (ruff is used, no config file — defaults apply)
ruff check .
ruff format --check .

# Format
ruff format .

# Tests — no test suite exists yet
# If tests are added, use pytest:
#   pip install pytest
#   pytest                        # run all tests
#   pytest tests/test_foo.py      # run single file
#   pytest tests/test_foo.py::test_bar  # run single test
#   pytest -k "keyword"           # run tests matching keyword
```

## Directory Structure

```
character_sniper.py   # Core engine: FaceAnalyzer, CLIPEncoder, scoring, CLI
server.py             # FastAPI web server, SSE progress, gallery endpoints
session_store.py      # SQLite session persistence (settings, jobs, results)
requirements.txt      # Python dependencies
templates/
  base.html           # Layout: Tailwind CSS, HTMX 2.0, Alpine.js 3.15 (all via CDN)
  index.html          # Main page: settings form, SSE progress, compare modal
  results.html        # Results partial: folder sidebar + image cards grid
data/                 # User data (gitignored)
  original.png        # Reference face image
  output/             # Generated images to analyze
  sessions.db         # SQLite database for session persistence (auto-created)
results/              # Output (gitignored): selected images + report.csv
```

## Code Style

### Python

- **Python version**: 3.10+ required
- **Future annotations**: Always use `from __future__ import annotations` at top of every file
- **Formatter**: Ruff with default settings (~88-100 char line width)
- **String quotes**: Double quotes (`"text"`)
- **String formatting**: f-strings exclusively — no `.format()` or `%`
- **Path handling**: Always use `pathlib.Path` — never raw string paths

### Naming Conventions

- **Functions / variables / modules**: `snake_case`
- **Classes**: `PascalCase` (`FaceAnalyzer`, `CLIPEncoder`, `ImageScore`)
- **Constants**: `UPPER_SNAKE_CASE` (`IMAGE_EXTENSIONS`, `ProgressCallback`)
- **Internal / private**: Single underscore prefix (`_model_cache`, `_sse_encode`)
- **Type aliases**: `UPPER_SNAKE_CASE` (`ProgressCallback = Callable[[int, int, str], None]`)

### Type Hints

Type hints are used throughout. Follow these patterns:
```python
from __future__ import annotations

def process(path: Path, threshold: float = 0.5) -> list[ImageScore]:
    ...

def find_face(img: np.ndarray) -> Optional[Face]:
    """Return the highest-confidence face or None."""
    ...

ProgressCallback = Callable[[int, int, str], None]
```

- Use lowercase generics: `list[T]`, `dict[K, V]`, `tuple[T, ...]`
- Use `Optional[T]` for nullable values
- Use `Callable[[args], return]` for callback types
- Annotate all function signatures (parameters and return types)

### Imports

Order: standard library → third-party → local. Separate groups with a blank line.
```python
from __future__ import annotations

import sys
import shutil
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from character_sniper import FaceAnalyzer, CLIPEncoder
```

### Docstrings

Google-style, single-line for simple functions:
```python
def best_face(faces: list) -> Face | None:
    """Return the highest-confidence face or None."""
```

Multi-line when needed — no docstrings on trivial or obvious methods.

### Error Handling

- Raise `ValueError` for domain/validation errors
- Use `sys.exit(str)` for fatal CLI errors (prints message and exits)
- In async server code: `try/except` with `traceback.print_exc()` for background job errors
- Never silently swallow exceptions — always log or re-raise

### File Organization

Use section separators for logical grouping within a file:
```python
# ── Model Loading ─────────────────────────────────────────────
# ── Image Processing ──────────────────────────────────────────
# ── CLI Entry Point ───────────────────────────────────────────
```

### Dataclasses

Use `@dataclass` for structured data. Include serialization when needed:
```python
@dataclass
class ImageScore:
    path: Path
    face_sim: float
    clip_sim: float
    weighted: float

    def to_dict(self) -> dict[str, object]:
        ...
```

## Frontend / Templates

- **Templating**: Jinja2 with `{% extends "base.html" %}` / `{% block content %}`
- **CSS**: Tailwind CSS via CDN — dark theme (gray-900 bg, gray-100 text)
- **Interactivity**: HTMX 2.0 (`hx-post`, `hx-target`, `hx-swap`) + Alpine.js (`x-data`, `x-show`, `@click`)
- **Real-time updates**: Server-Sent Events (SSE) via `sse-starlette`
- **No build step** — all frontend dependencies are CDN-loaded

## Key Abstractions

- `FaceAnalyzer` — wraps InsightFace `buffalo_l` model; face detection, landmark validation, bbox checks
- `CLIPEncoder` / `DummyCLIP` — wraps OpenCLIP ViT-B-32; `DummyCLIP` is a no-op fallback
- `ImageScore` — dataclass for per-image similarity results
- `process_folder()` — scores all images in a folder, accepts `ProgressCallback`
- `prepare_original()` — extracts reference face + CLIP embeddings from original image
- `discover_folders()` / `detect_mode()` — auto-detects flat vs recursive input structure
- `SessionStore` — SQLite-backed persistence for web UI sessions (settings, jobs, results)

## Session Persistence

The web UI persists state to `data/sessions.db` (SQLite, WAL mode) so that page reloads
and server restarts do not lose work. Zero external dependencies — uses Python's `sqlite3`.

### What is persisted

| Data | When saved | Storage |
|------|-----------|---------|
| User settings (paths, weights, top-K) | On each job start | `settings` table (singleton row) |
| Completed jobs + all scoring results | When job finishes | `jobs` + `job_results` tables |
| Image selection changes (select/deselect) | On each toggle | `job_results.selected` column |

### Session restore flow

1. **Server startup** (`lifespan`): loads latest completed job from SQLite into in-memory `jobs` dict
2. **Page load** (`Alpine.js init()`): `GET /api/settings` fills form, `GET /api/jobs/latest` triggers HTMX load of results
3. **Invalidation**: if input folder mtime is newer than job creation time, results are considered stale and not restored

### API endpoints (session)

- `GET /api/settings` — return saved user settings (or defaults)
- `POST /api/settings` — save user settings as JSON body
- `GET /api/jobs/latest` — return latest valid job ID for auto-restore

### Schema (3 tables)

- `settings` — singleton row (id=1) with form field values
- `jobs` — completed/errored jobs with params and metadata
- `job_results` — per-image scoring data with `UNIQUE(job_id, folder, filename)`, CASCADE delete

## Scoring Formula

```
weighted_score = (face_cosine_similarity × 0.7) + (clip_crop_similarity × 0.3)
```

Face similarity is the primary signal; CLIP provides supplementary visual consistency scoring.
