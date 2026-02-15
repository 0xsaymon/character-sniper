"""Shared fixtures for Character Sniper tests."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from session_store import SessionStore


# ── Tiny test image helpers ───────────────────────────────────


def _create_test_image(path: Path, width: int = 64, height: int = 64) -> Path:
    """Create a minimal valid PNG at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    img.save(path)
    return path


@pytest.fixture()
def test_image(tmp_path: Path) -> Path:
    """A single small PNG image in a temp directory."""
    return _create_test_image(tmp_path / "test.png")


@pytest.fixture()
def flat_image_dir(tmp_path: Path) -> Path:
    """Flat directory with 5 PNG images."""
    d = tmp_path / "flat_images"
    d.mkdir()
    for i in range(5):
        _create_test_image(d / f"img_{i:03d}.png")
    return d


@pytest.fixture()
def recursive_image_dir(tmp_path: Path) -> Path:
    """Recursive directory structure with two sub-folders, each having 3 images."""
    root = tmp_path / "recursive_images"
    root.mkdir()
    for folder_name in ("folder_a", "folder_b"):
        sub = root / folder_name
        sub.mkdir()
        for i in range(3):
            _create_test_image(sub / f"img_{i:03d}.png")
    return root


# ── SessionStore fixture ──────────────────────────────────────


@pytest.fixture()
def store(tmp_path: Path) -> SessionStore:
    """SessionStore backed by a temp-dir SQLite database."""
    return SessionStore(db_path=tmp_path / "test.db")


# ── Mock face object ─────────────────────────────────────────


class MockFace:
    """Lightweight stand-in for an InsightFace face result."""

    def __init__(
        self,
        *,
        det_score: float = 0.95,
        bbox: tuple[float, float, float, float] = (50, 50, 200, 200),
        kps: np.ndarray | None = None,
        normed_embedding: np.ndarray | None = None,
    ):
        self.det_score = det_score
        self.bbox = np.array(bbox, dtype=np.float32)
        if kps is None:
            # Standard frontal-face keypoints (eyes above nose above mouth)
            self.kps = np.array(
                [
                    [80, 80],  # left eye
                    [160, 80],  # right eye
                    [120, 130],  # nose
                    [90, 170],  # left mouth
                    [150, 170],  # right mouth
                ],
                dtype=np.float32,
            )
        else:
            self.kps = kps
        if normed_embedding is None:
            emb = np.random.default_rng(42).standard_normal(512).astype(np.float32)
            self.normed_embedding = emb / np.linalg.norm(emb)
        else:
            self.normed_embedding = normed_embedding


@pytest.fixture()
def mock_face() -> MockFace:
    return MockFace()


# ── Mock FaceAnalyzer / CLIPEncoder ──────────────────────────


@pytest.fixture()
def mock_face_analyzer(mock_face: MockFace) -> MagicMock:
    """A MagicMock that behaves like FaceAnalyzer."""
    fa = MagicMock()
    fa.get_best_face.return_value = mock_face
    fa.landmarks_ok = MagicMock(return_value=True)
    fa.face_bbox_big_enough = MagicMock(return_value=True)
    return fa


@pytest.fixture()
def mock_clip_encoder() -> MagicMock:
    """A MagicMock that behaves like CLIPEncoder."""
    ce = MagicMock()
    emb = np.random.default_rng(99).standard_normal(512).astype(np.float32)
    ce.encode_image.return_value = emb / np.linalg.norm(emb)
    return ce


# ── Sample job data for SessionStore tests ────────────────────


def make_sample_job_data(
    *,
    original_path: str = "/tmp/original.png",
    input_path: str = "/tmp/output",
    status: str = "complete",
    is_recursive: bool = False,
) -> dict[str, Any]:
    """Build a realistic job data dict for store.save_job()."""
    now = time.time()
    return {
        "status": status,
        "original_path": original_path,
        "input_path": input_path,
        "params": {
            "top_k": 5,
            "face_weight": 0.7,
            "clip_weight": 0.3,
            "min_face_score": 0.5,
        },
        "is_recursive": is_recursive,
        "start_time": now - 10,
        "end_time": now,
        "error": None,
        "results": {
            ".": [
                {
                    "path": "/tmp/output/img_000.png",
                    "filename": "img_000.png",
                    "folder": ".",
                    "face_score": 0.95,
                    "clip_score": 0.88,
                    "final_score": 0.929,
                    "det_score": 0.98,
                    "face_bbox": "[50, 50, 200, 200]",
                    "reject_reason": None,
                    "selected": True,
                },
                {
                    "path": "/tmp/output/img_001.png",
                    "filename": "img_001.png",
                    "folder": ".",
                    "face_score": 0.80,
                    "clip_score": 0.75,
                    "final_score": 0.785,
                    "det_score": 0.90,
                    "face_bbox": "[60, 60, 180, 180]",
                    "reject_reason": None,
                    "selected": False,
                },
            ]
        },
    }
