# Character Sniper

Automatically select AI-generated images that are most similar to an original reference photo. Built for workflows where you generate many variations from a single reference (e.g. for LoRA training datasets) and need to pick the best ones without manually reviewing thousands of images.

## How it works

The script compares each generated image against the original reference using two metrics:

- **Face similarity (70% weight)** — InsightFace ArcFace embeddings. Extracts a 512-dim face identity vector and computes cosine similarity. This is the primary signal: "is this the same person?"
- **CLIP similarity (30% weight)** — OpenCLIP ViT-B-32 embeddings of the face crop region (not the full image). Captures visual style, lighting, skin texture around the face. Using crops avoids penalizing images with different body poses or camera angles.

Bad images are filtered automatically:
- No face detected
- Face detection confidence below threshold (deformed/blurry faces)
- Invalid facial landmark geometry (eyes-nose-mouth in wrong positions)
- Face bounding box too small (<64px)

## Requirements

- Python 3.10+
- macOS (Apple Silicon — MPS + CoreML) or Windows/Linux (NVIDIA GPU — CUDA)

## Setup

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/character-sniper.git
cd character-sniper

# Create virtual environment
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# macOS — uses the standard onnxruntime (CoreML support included)
pip install onnxruntime

# Windows/Linux with NVIDIA GPU
pip install onnxruntime-gpu
```

On first run, the script downloads ~900MB of model weights (InsightFace buffalo_l + OpenCLIP ViT-B-32). These are cached for subsequent runs.

## Data layout

Place your images in the `data/` directory (gitignored):

```
character-sniper/
  data/
    original.png              # your reference photo
    output/                   # generated images go here
      img_001.png
      img_002.png
      ...
```

For batch mode with multiple prompts, use subfolders:

```
character-sniper/
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

The script auto-detects whether `output/` contains images directly (flat mode) or subfolders with images (recursive mode).

## Usage

### Basic (uses defaults: `data/original.png` and `data/output/`)

```bash
python character_sniper.py --report
```

### Custom paths

```bash
python character_sniper.py \
  --original /path/to/reference.png \
  --input /path/to/generated/ \
  --output /path/to/results/ \
  --report
```

### All options

```
--original          Path to reference image (default: data/original.png)
--input             Folder with generated images (default: data/output/)
--output            Folder for selected images (default: results/)
--top-k             Best images to select per folder (default: 5)
--method            Scoring: face | clip | combined (default: combined)
--face-weight       Weight for face similarity (default: 0.7)
--clip-weight       Weight for CLIP similarity (default: 0.3)
--clip-mode         CLIP compares: crop | full (default: crop)
--crop-expand       Expand face bbox for CLIP crop (default: 1.5x)
--min-face-score    Min detection confidence to accept (default: 0.5)
--min-similarity    Min face cosine similarity to keep (default: 0, disabled)
--report            Write CSV report with all scores
--clip-model        OpenCLIP model name (default: ViT-B-32)
--clip-pretrained   OpenCLIP weights (default: laion2b_s34b_b79k)
```

## Output

Selected images are copied to the `--output` folder (default: `results/`). In recursive mode, the subfolder structure is preserved.

With `--report`, a `report.csv` is generated containing scores for every processed image:

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

First run includes a one-time model download (~900MB).

## License

MIT
