#!/usr/bin/env python3
"""
Character Sniper — find generated images most similar to an original reference.

Uses InsightFace (ArcFace) for face identity similarity and OpenCLIP for
visual similarity of the face crop region.  Works on macOS (MPS / CoreML)
and Windows/Linux (CUDA).

Usage (default — auto-detects flat or recursive):
    python character_sniper.py --report

Custom paths:
    python character_sniper.py --original path/to/ref.png --input path/to/images --report

Web UI:
    python server.py
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Device / provider helpers
# ---------------------------------------------------------------------------


def get_torch_device() -> str:
    """Pick the best available PyTorch device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_onnx_providers() -> list[str]:
    """Pick the best available ONNX Runtime execution providers."""
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    providers: list[str] = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------


class FaceAnalyzer:
    """Thin wrapper around InsightFace FaceAnalysis."""

    def __init__(self, det_size: int = 640):
        from insightface.app import FaceAnalysis

        providers = get_onnx_providers()
        print(f"[InsightFace] providers: {providers}")
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size))

    def get_best_face(self, img_bgr: np.ndarray, min_det_score: float = 0.5):
        """Return the highest-confidence face or None."""
        faces = self.app.get(img_bgr)
        if not faces:
            return None
        best = max(faces, key=lambda f: f.det_score)
        if best.det_score < min_det_score:
            return None
        return best

    @staticmethod
    def landmarks_ok(face) -> bool:
        """Basic sanity: eyes above nose above mouth."""
        kps = face.kps
        if kps is None:
            return False
        left_eye, right_eye, nose, left_mouth, right_mouth = kps
        if not (left_eye[1] < nose[1] and right_eye[1] < nose[1]):
            return False
        if not (nose[1] < left_mouth[1] and nose[1] < right_mouth[1]):
            return False
        eye_dist = np.linalg.norm(right_eye - left_eye)
        bbox_w = face.bbox[2] - face.bbox[0]
        if bbox_w > 0 and eye_dist / bbox_w < 0.1:
            return False
        return True

    @staticmethod
    def face_bbox_big_enough(face, min_px: int = 64) -> bool:
        x1, y1, x2, y2 = face.bbox
        return (x2 - x1) >= min_px and (y2 - y1) >= min_px


class CLIPEncoder:
    """Thin wrapper around open_clip for image embeddings."""

    def __init__(
        self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"
    ):
        import open_clip

        self.device = get_torch_device()
        print(f"[CLIP] device: {self.device}, model: {model_name}/{pretrained}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def encode_image(self, pil_img: Image.Image) -> np.ndarray:
        """Return L2-normalised embedding (1-D numpy)."""
        tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        features = self.model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()


class DummyCLIP:
    """No-op CLIP encoder for face-only mode."""

    def encode_image(self, _):
        return np.zeros(512)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def list_images(folder: Path) -> list[Path]:
    """List image files in *folder* (non-recursive, sorted)."""
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_bgr(path: Path) -> Optional[np.ndarray]:
    """Read image as BGR numpy via OpenCV."""
    return cv2.imread(str(path))


def crop_face_region(img_bgr: np.ndarray, face, expand: float = 1.5) -> Image.Image:
    """Crop an expanded bounding-box around the face, return as RGB PIL Image."""
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = face.bbox
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    new_w, new_h = bw * expand, bh * expand
    nx1 = max(0, int(cx - new_w / 2))
    ny1 = max(0, int(cy - new_h / 2))
    nx2 = min(w, int(cx + new_w / 2))
    ny2 = min(h, int(cy + new_h / 2))
    crop = img_bgr[ny1:ny2, nx1:nx2]
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def pil_from_path(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ---------------------------------------------------------------------------
# Per-image result
# ---------------------------------------------------------------------------


@dataclass
class ImageScore:
    path: Path
    folder: str  # relative subfolder name (or "." for flat)
    face_score: Optional[float] = None
    clip_score: Optional[float] = None
    final_score: Optional[float] = None
    det_score: Optional[float] = None
    face_bbox: Optional[str] = None
    reject_reason: Optional[str] = None
    selected: bool = False

    def to_dict(self) -> dict:
        """Serialise for JSON / web responses."""
        return {
            "path": str(self.path),
            "filename": self.path.name,
            "folder": self.folder,
            "face_score": self.face_score,
            "clip_score": self.clip_score,
            "final_score": self.final_score,
            "det_score": self.det_score,
            "face_bbox": self.face_bbox,
            "reject_reason": self.reject_reason,
            "selected": self.selected,
        }


# ---------------------------------------------------------------------------
# Progress callback type
# ---------------------------------------------------------------------------

# on_progress(current_index, total, current_filename)
ProgressCallback = Callable[[int, int, str], None]


# ---------------------------------------------------------------------------
# Auto-detect flat vs recursive mode
# ---------------------------------------------------------------------------


def detect_mode(input_dir: Path) -> bool:
    """Return True for recursive (subfolder) mode, False for flat."""
    has_root_images = any(
        p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        for p in input_dir.iterdir()
    )
    subdirs_with_images = [
        d
        for d in input_dir.iterdir()
        if d.is_dir()
        and not d.name.startswith(".")
        and any(
            f.suffix.lower() in IMAGE_EXTENSIONS for f in d.iterdir() if f.is_file()
        )
    ]
    if subdirs_with_images:
        return True
    if has_root_images:
        return False
    raise FileNotFoundError(
        f"No images found in {input_dir} (neither at root nor in subfolders)"
    )


# ---------------------------------------------------------------------------
# Discover folders
# ---------------------------------------------------------------------------


def discover_folders(input_dir: Path) -> tuple[bool, list[Path]]:
    """Return (is_recursive, list_of_folders_to_process)."""
    recursive = detect_mode(input_dir)
    if recursive:
        subfolders = sorted(
            d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
        )
        return True, subfolders
    return False, [input_dir]


# ---------------------------------------------------------------------------
# Core processing — with progress callback
# ---------------------------------------------------------------------------


def prepare_original(
    original_path: Path,
    face_analyzer: FaceAnalyzer,
    clip_encoder,
    clip_mode: str = "crop",
    crop_expand: float = 1.5,
):
    """Extract embeddings from the original reference image.

    Returns (orig_face_emb, orig_clip_emb) or raises ValueError.
    """
    orig_bgr = load_bgr(original_path)
    if orig_bgr is None:
        raise ValueError(f"Cannot read original image: {original_path}")

    orig_face = face_analyzer.get_best_face(orig_bgr, min_det_score=0.0)
    if orig_face is None:
        raise ValueError(f"No face detected in original image: {original_path}")

    orig_face_emb = orig_face.normed_embedding

    if clip_mode == "crop":
        orig_clip_img = crop_face_region(orig_bgr, orig_face, expand=crop_expand)
    else:
        orig_clip_img = pil_from_path(original_path)
    orig_clip_emb = clip_encoder.encode_image(orig_clip_img)

    return orig_face_emb, orig_clip_emb


def score_single_image(
    img_path: Path,
    folder_name: str,
    face_analyzer: FaceAnalyzer,
    clip_encoder,
    orig_face_emb: np.ndarray,
    orig_clip_emb: np.ndarray,
    *,
    face_weight: float = 0.7,
    clip_weight: float = 0.3,
    clip_mode: str = "crop",
    crop_expand: float = 1.5,
    min_face_score: float = 0.5,
    min_similarity: float = 0.0,
) -> ImageScore:
    """Score a single image against pre-computed original embeddings."""
    score = ImageScore(path=img_path, folder=folder_name)

    img_bgr = load_bgr(img_path)
    if img_bgr is None:
        score.reject_reason = "unreadable"
        return score

    face = face_analyzer.get_best_face(img_bgr, min_det_score=min_face_score)
    if face is None:
        score.reject_reason = "no_face_or_low_det"
        return score

    score.det_score = round(float(face.det_score), 4)
    score.face_bbox = str([int(v) for v in face.bbox])

    if not face_analyzer.landmarks_ok(face):
        score.reject_reason = "bad_landmarks"
        return score

    if not face_analyzer.face_bbox_big_enough(face):
        score.reject_reason = "face_too_small"
        return score

    # face similarity
    face_sim = cosine_sim(orig_face_emb, face.normed_embedding)
    score.face_score = round(face_sim, 4)

    if face_sim < min_similarity:
        score.reject_reason = f"face_sim_below_{min_similarity}"
        return score

    # clip similarity
    if clip_weight > 0:
        if clip_mode == "crop":
            clip_img = crop_face_region(img_bgr, face, expand=crop_expand)
        else:
            clip_img = pil_from_path(img_path)
        clip_sim = cosine_sim(orig_clip_emb, clip_encoder.encode_image(clip_img))
        score.clip_score = round(clip_sim, 4)
    else:
        clip_sim = 0.0
        score.clip_score = 0.0

    # final weighted score
    score.final_score = round(face_weight * face_sim + clip_weight * clip_sim, 4)
    return score


def process_folder(
    image_paths: list[Path],
    folder_name: str,
    face_analyzer: FaceAnalyzer,
    clip_encoder,
    orig_face_emb: np.ndarray,
    orig_clip_emb: np.ndarray,
    *,
    face_weight: float = 0.7,
    clip_weight: float = 0.3,
    clip_mode: str = "crop",
    crop_expand: float = 1.5,
    min_face_score: float = 0.5,
    min_similarity: float = 0.0,
    top_k: int = 5,
    on_progress: Optional[ProgressCallback] = None,
) -> list[ImageScore]:
    """Score all images in a folder, mark top_k as selected."""
    results: list[ImageScore] = []

    for i, img_path in enumerate(image_paths):
        result = score_single_image(
            img_path,
            folder_name,
            face_analyzer,
            clip_encoder,
            orig_face_emb,
            orig_clip_emb,
            face_weight=face_weight,
            clip_weight=clip_weight,
            clip_mode=clip_mode,
            crop_expand=crop_expand,
            min_face_score=min_face_score,
            min_similarity=min_similarity,
        )
        results.append(result)

        if on_progress:
            on_progress(i + 1, len(image_paths), img_path.name)

    # select top-k
    scorable = [r for r in results if r.final_score is not None]
    scorable.sort(key=lambda r: r.final_score or 0, reverse=True)
    for r in scorable[:top_k]:
        r.selected = True

    return results


# ---------------------------------------------------------------------------
# CSV report
# ---------------------------------------------------------------------------


def write_csv(results: list[ImageScore], csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "folder",
                "filename",
                "face_score",
                "clip_score",
                "final_score",
                "det_score",
                "face_bbox",
                "reject_reason",
                "selected",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.folder,
                    r.path.name,
                    r.face_score if r.face_score is not None else "",
                    r.clip_score if r.clip_score is not None else "",
                    r.final_score if r.final_score is not None else "",
                    r.det_score if r.det_score is not None else "",
                    r.face_bbox or "",
                    r.reject_reason or "",
                    r.selected,
                ]
            )
    print(f"Report saved -> {csv_path}")


# ---------------------------------------------------------------------------
# Copy selected images
# ---------------------------------------------------------------------------


def copy_selected(results: list[ImageScore], output_dir: Path, recursive: bool):
    selected = [r for r in results if r.selected]
    if not selected:
        print("No images selected -- nothing to copy.")
        return 0
    for r in selected:
        dest_dir = output_dir / r.folder if recursive else output_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(r.path, dest_dir / r.path.name)
    print(f"Copied {len(selected)} images -> {output_dir}")
    return len(selected)


# ---------------------------------------------------------------------------
# Normalise weights helper
# ---------------------------------------------------------------------------


def normalise_weights(method: str, face_w: float, clip_w: float) -> tuple[float, float]:
    if method == "face":
        return 1.0, 0.0
    if method == "clip":
        return 0.0, 1.0
    total = face_w + clip_w
    if total <= 0:
        raise ValueError("face_weight + clip_weight must be > 0")
    return face_w / total, clip_w / total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Character Sniper -- pick generated images most similar to an original reference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--original",
        type=str,
        default="data/original.png",
        help="Path to the reference image (default: data/original.png)",
    )
    p.add_argument(
        "--input",
        type=str,
        default="data/output",
        help="Folder with generated images (default: data/output/)",
    )
    p.add_argument(
        "--output",
        type=str,
        default="results",
        help="Folder for selected images (default: results/)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many best images to select per folder (default: 5)",
    )
    p.add_argument("--method", choices=["face", "clip", "combined"], default="combined")
    p.add_argument("--face-weight", type=float, default=0.7)
    p.add_argument("--clip-weight", type=float, default=0.3)
    p.add_argument("--clip-mode", choices=["crop", "full"], default="crop")
    p.add_argument("--crop-expand", type=float, default=1.5)
    p.add_argument("--min-face-score", type=float, default=0.5)
    p.add_argument("--min-similarity", type=float, default=0.0)
    p.add_argument("--report", action="store_true", help="Write CSV report")
    p.add_argument("--clip-model", type=str, default="ViT-B-32")
    p.add_argument("--clip-pretrained", type=str, default="laion2b_s34b_b79k")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main (CLI entry point)
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    original_path = Path(args.original)
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not original_path.exists():
        sys.exit(f"Original image not found: {original_path}")
    if not input_dir.exists():
        sys.exit(f"Input folder not found: {input_dir}")

    face_w, clip_w = normalise_weights(args.method, args.face_weight, args.clip_weight)

    print("=" * 60)
    print("  Character Sniper")
    print("=" * 60)
    print(f"  Original  : {original_path}")
    print(f"  Input     : {input_dir}")
    print(f"  Output    : {output_dir}")
    print(f"  Top-K     : {args.top_k}")
    print(f"  Method    : {args.method} (face={face_w:.2f}, clip={clip_w:.2f})")
    print(f"  CLIP mode : {args.clip_mode} (expand={args.crop_expand}x)")
    print(f"  Min det   : {args.min_face_score}")
    print(f"  Min sim   : {args.min_similarity}")
    print(f"  Report    : {args.report}")
    print("=" * 60)

    t0 = time.time()
    print("\nLoading models ...")

    face_analyzer = FaceAnalyzer()
    clip_encoder = (
        CLIPEncoder(args.clip_model, args.clip_pretrained)
        if clip_w > 0
        else DummyCLIP()
    )

    print(f"Models loaded in {time.time() - t0:.1f}s\n")

    # prepare original embeddings
    orig_face_emb, orig_clip_emb = prepare_original(
        original_path,
        face_analyzer,
        clip_encoder,
        clip_mode=args.clip_mode,
        crop_expand=args.crop_expand,
    )

    # discover folders
    recursive, subfolders = discover_folders(input_dir)
    print(
        f"Mode: {'recursive' if recursive else 'flat'} ({len(subfolders)} folder(s))\n"
    )

    # process
    all_results: list[ImageScore] = []
    total_selected = 0
    total_rejected = 0
    total_processed = 0

    for folder in subfolders:
        images = list_images(folder)
        if not images:
            continue

        folder_name = folder.name if recursive else "."

        # CLI progress via tqdm
        pbar = tqdm(
            total=len(images), desc=f"  [{folder_name}]", unit="img", leave=False
        )

        def cli_progress(current, total, filename, _pbar=pbar):
            _pbar.update(1)

        results = process_folder(
            image_paths=images,
            folder_name=folder_name,
            face_analyzer=face_analyzer,
            clip_encoder=clip_encoder,
            orig_face_emb=orig_face_emb,
            orig_clip_emb=orig_clip_emb,
            face_weight=face_w,
            clip_weight=clip_w,
            clip_mode=args.clip_mode,
            crop_expand=args.crop_expand,
            min_face_score=args.min_face_score,
            min_similarity=args.min_similarity,
            top_k=args.top_k,
            on_progress=cli_progress,
        )
        pbar.close()
        all_results.extend(results)

        n_sel = sum(1 for r in results if r.selected)
        n_rej = sum(1 for r in results if r.reject_reason)
        total_selected += n_sel
        total_rejected += n_rej
        total_processed += len(images)

        top = [r for r in results if r.selected]
        top.sort(key=lambda r: r.final_score or 0, reverse=True)
        if top:
            tqdm.write(
                f"  [{folder_name}] {len(images)} imgs -> "
                f"{n_sel} selected (best={top[0].final_score}, "
                f"worst={top[-1].final_score}), {n_rej} rejected"
            )

    # copy & report
    print()
    copy_selected(all_results, output_dir, recursive)
    if args.report:
        write_csv(all_results, output_dir / "report.csv")

    print(f"\n{'=' * 60}")
    print(
        f"  Done!  {total_processed} processed, {total_selected} selected, "
        f"{total_rejected} rejected  ({time.time() - t0:.1f}s)"
    )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
