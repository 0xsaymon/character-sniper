# Character Sniper

![Tests](https://img.shields.io/badge/tests-123%20passed-brightgreen) ![Coverage](https://img.shields.io/badge/coverage-84%25-green)

Automatically select AI-generated images that are most similar to an original reference photo. Built for workflows where you generate many variations from a single reference (e.g. for LoRA training datasets) and need to pick the best ones without manually reviewing thousands of images.

Available as a **CLI** for batch processing and a **Web UI** with real-time progress, filterable gallery, and side-by-side comparison.

## How it works

Each generated image is compared against the original reference using two metrics:

- **Face similarity (70% weight)** — InsightFace ArcFace embeddings. Extracts a 512-dim face identity vector and computes cosine similarity. This is the primary signal: "is this the same person?"
- **CLIP similarity (30% weight)** — OpenCLIP ViT-B-32 embeddings of the face crop region (not the full image). Captures visual style, lighting, skin texture around the face. Using crops avoids penalizing images with different body poses or camera angles.

```
score = face_similarity * face_weight + clip_similarity * clip_weight
```

Bad images are filtered automatically:
- No face detected or detection confidence below threshold
- Invalid facial landmark geometry (eyes-nose-mouth in wrong positions)
- Face bounding box too small (<64px)
- Face similarity below minimum threshold

The top-K scoring images per folder are automatically selected.

## Requirements

- Python 3.10+
- macOS (Apple Silicon) or Windows/Linux (NVIDIA GPU)
- ~2 GB disk for models (downloaded on first run)

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/character-sniper.git
cd character-sniper

python3 -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

pip install -r requirements.txt

# Pick one:
pip install onnxruntime            # macOS / CPU
pip install onnxruntime-gpu        # NVIDIA GPU
```

Models (InsightFace `buffalo_l` + OpenCLIP `ViT-B-32`) are downloaded automatically on first run (~900 MB).

## Data layout

Place your images in the `data/` directory (gitignored):

```
data/
  original.png              # your reference photo
  output/                   # generated images go here
    img_001.png
    img_002.png
    ...
```

For batch mode with multiple prompts, use subfolders:

```
data/
  original.png
  output/
    prompt_001/             # 50-100 images per prompt
      img_001.png
      ...
    prompt_002/
      img_001.png
      ...
```

The tool auto-detects whether `output/` contains images directly (flat mode) or subfolders (recursive mode).

## Usage

### Web UI

```bash
source venv/bin/activate
python server.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Configure paths and scoring parameters in the settings form, then start processing.

Features:
- **Real-time progress** via Server-Sent Events
- **Filter** results by status — all, selected, scored, rejected
- **Compare** any image side-by-side with the original, navigate with arrow keys
- **Select / deselect** images manually from the gallery or the compare modal
- **Folder sidebar** for multi-folder inputs with live selection counters
- **Export** selected images to `results/` or download as a zip archive
- **Session persistence** — settings, results, and selections survive page reloads and server restarts (SQLite)

No build step — frontend uses Tailwind CSS, HTMX, and Alpine.js via CDN.

### CLI

```bash
source venv/bin/activate
python character_sniper.py --report
```

Custom paths:

```bash
python character_sniper.py \
  --original /path/to/reference.png \
  --input /path/to/generated/ \
  --output /path/to/results/ \
  --top-k 5 \
  --report
```

All options:

| Flag | Description | Default |
|------|-------------|---------|
| `--original` | Path to reference image | `data/original.png` |
| `--input` | Folder with generated images | `data/output/` |
| `--output` | Folder for selected images | `results/` |
| `--top-k` | Best images to select per folder | `5` |
| `--method` | Scoring: `face`, `clip`, `combined` | `combined` |
| `--face-weight` | Weight for face similarity | `0.7` |
| `--clip-weight` | Weight for CLIP similarity | `0.3` |
| `--clip-mode` | CLIP compares: `crop` or `full` | `crop` |
| `--crop-expand` | Expand face bbox for CLIP crop | `1.5` |
| `--min-face-score` | Min detection confidence | `0.5` |
| `--min-similarity` | Min face cosine similarity | `0` (disabled) |
| `--report` | Write CSV report with all scores | off |
| `--clip-model` | OpenCLIP model name | `ViT-B-32` |
| `--clip-pretrained` | OpenCLIP weights | `laion2b_s34b_b79k` |

## Output

Selected images are copied to `results/`. In recursive mode, the subfolder structure is preserved.

With `--report`, a `report.csv` is generated with scores for every processed image:

| folder | filename | face_score | clip_score | final_score | det_score | reject_reason | selected |
|--------|----------|------------|------------|-------------|-----------|---------------|----------|
| prompt_001 | img_023.png | 0.87 | 0.72 | 0.83 | 0.98 | | True |
| prompt_001 | img_045.png | 0.82 | 0.68 | 0.78 | 0.95 | | False |
| prompt_001 | img_067.png | | | | | no_face_or_low_det | False |

## Typical workflow

1. **Generate** 50-100 images per prompt (200 prompts = 10,000-20,000 images)
2. **First pass** — select top 5 per prompt:
   ```bash
   python character_sniper.py --top-k 5 --report
   ```
   Result: 1,000 images (200 folders x 5)
3. **Second pass** — narrow down to top 1 per prompt:
   ```bash
   python character_sniper.py --input results --output final --top-k 1 --report
   ```
   Result: 200 images ready for LoRA training

## Performance

| Platform | Speed | 10,000 images |
|----------|-------|---------------|
| macOS M3 Max (CoreML + MPS) | ~17 img/s | ~10 min |
| Windows RTX 5080 (CUDA) | ~20+ img/s | ~8 min |

## Project structure

```
character_sniper.py    Core engine: FaceAnalyzer, CLIPEncoder, scoring, CLI
server.py              FastAPI web server, SSE progress, gallery endpoints
session_store.py       SQLite session persistence (settings, jobs, results)
requirements.txt       Python dependencies
templates/
  base.html            Layout: Tailwind CSS, HTMX 2.0, Alpine.js (CDN)
  index.html           Main page: settings, progress, compare modal
  results.html         Results partial: filter bar, folder sidebar, image grid
data/                  User data (gitignored)
  sessions.db          Auto-created SQLite database for session state
results/               Output (gitignored)
```

## License

MIT
