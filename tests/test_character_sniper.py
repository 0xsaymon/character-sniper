"""Tests for character_sniper.py — pure functions (no ML models required)."""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from character_sniper import (
    ImageScore,
    DummyCLIP,
    cosine_sim,
    normalise_weights,
    list_images,
    detect_mode,
    discover_folders,
    write_csv,
    copy_selected,
    FaceAnalyzer,
    load_bgr,
    pil_from_path,
    crop_face_region,
)
from tests.conftest import MockFace, _create_test_image


# ── cosine_sim ────────────────────────────────────────────────


class TestCosineSim:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_sim(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_sim(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_sim(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_near_zero_vectors(self):
        """Should not raise ZeroDivisionError due to epsilon."""
        a = np.array([1e-12, 1e-12])
        b = np.array([1e-12, 1e-12])
        result = cosine_sim(a, b)
        assert isinstance(result, float)

    def test_high_dimensional(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal(512)
        a = a / np.linalg.norm(a)
        assert cosine_sim(a, a) == pytest.approx(1.0, abs=1e-5)


# ── normalise_weights ─────────────────────────────────────────


class TestNormaliseWeights:
    def test_face_only(self):
        assert normalise_weights("face", 0.7, 0.3) == (1.0, 0.0)

    def test_clip_only(self):
        assert normalise_weights("clip", 0.7, 0.3) == (0.0, 1.0)

    def test_combined_default(self):
        fw, cw = normalise_weights("combined", 0.7, 0.3)
        assert fw == pytest.approx(0.7)
        assert cw == pytest.approx(0.3)

    def test_combined_normalises(self):
        fw, cw = normalise_weights("combined", 2.0, 1.0)
        assert fw + cw == pytest.approx(1.0)
        assert fw == pytest.approx(2 / 3)
        assert cw == pytest.approx(1 / 3)

    def test_zero_weights_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            normalise_weights("combined", 0.0, 0.0)

    def test_negative_sum_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            normalise_weights("combined", -1.0, 0.5)


# ── ImageScore ────────────────────────────────────────────────


class TestImageScore:
    def test_defaults(self):
        score = ImageScore(path=Path("test.png"), folder=".")
        assert score.face_score is None
        assert score.selected is False

    def test_to_dict_keys(self):
        score = ImageScore(path=Path("a/b.png"), folder="a")
        d = score.to_dict()
        expected = {
            "path",
            "filename",
            "folder",
            "face_score",
            "clip_score",
            "final_score",
            "det_score",
            "face_bbox",
            "reject_reason",
            "selected",
        }
        assert set(d.keys()) == expected

    def test_to_dict_values(self):
        score = ImageScore(
            path=Path("folder/img.png"),
            folder="folder",
            face_score=0.95,
            clip_score=0.8,
            final_score=0.905,
            selected=True,
        )
        d = score.to_dict()
        assert d["filename"] == "img.png"
        assert d["folder"] == "folder"
        assert d["face_score"] == 0.95
        assert d["selected"] is True

    def test_to_dict_path_is_string(self):
        score = ImageScore(path=Path("/some/path.png"), folder=".")
        d = score.to_dict()
        assert isinstance(d["path"], str)


# ── DummyCLIP ─────────────────────────────────────────────────


class TestDummyCLIP:
    def test_returns_zero_vector(self):
        clip = DummyCLIP()
        result = clip.encode_image(None)
        assert isinstance(result, np.ndarray)
        assert result.shape == (512,)
        assert np.all(result == 0.0)


# ── FaceAnalyzer static methods ──────────────────────────────


class TestFaceAnalyzerStatics:
    def test_landmarks_ok_valid(self):
        face = MockFace()
        assert FaceAnalyzer.landmarks_ok(face) is True

    def test_landmarks_ok_none_kps(self):
        face = MockFace()
        face.kps = None
        assert FaceAnalyzer.landmarks_ok(face) is False

    def test_landmarks_ok_eyes_below_nose(self):
        face = MockFace(
            kps=np.array(
                [
                    [80, 150],  # left eye BELOW nose
                    [160, 150],  # right eye BELOW nose
                    [120, 130],  # nose
                    [90, 170],  # left mouth
                    [150, 170],  # right mouth
                ],
                dtype=np.float32,
            )
        )
        assert FaceAnalyzer.landmarks_ok(face) is False

    def test_landmarks_ok_nose_below_mouth(self):
        face = MockFace(
            kps=np.array(
                [
                    [80, 80],  # left eye
                    [160, 80],  # right eye
                    [120, 180],  # nose BELOW mouth
                    [90, 170],  # left mouth
                    [150, 170],  # right mouth
                ],
                dtype=np.float32,
            )
        )
        assert FaceAnalyzer.landmarks_ok(face) is False

    def test_landmarks_ok_tiny_eye_distance(self):
        face = MockFace(
            kps=np.array(
                [
                    [119, 80],  # left eye very close to right
                    [121, 80],  # right eye
                    [120, 130],  # nose
                    [90, 170],  # left mouth
                    [150, 170],  # right mouth
                ],
                dtype=np.float32,
            )
        )
        # bbox width is 150 (200-50), eye_dist/bbox_w = 2/150 < 0.1
        assert FaceAnalyzer.landmarks_ok(face) is False

    def test_face_bbox_big_enough_true(self):
        face = MockFace(bbox=(50, 50, 200, 200))
        assert FaceAnalyzer.face_bbox_big_enough(face, min_px=64)

    def test_face_bbox_big_enough_false(self):
        face = MockFace(bbox=(0, 0, 30, 30))
        assert not FaceAnalyzer.face_bbox_big_enough(face, min_px=64)

    def test_face_bbox_exact_boundary(self):
        face = MockFace(bbox=(0, 0, 64, 64))
        assert FaceAnalyzer.face_bbox_big_enough(face, min_px=64)

    def test_face_bbox_asymmetric(self):
        """Width ok but height too small."""
        face = MockFace(bbox=(0, 0, 100, 50))
        assert not FaceAnalyzer.face_bbox_big_enough(face, min_px=64)


# ── list_images ───────────────────────────────────────────────


class TestListImages:
    def test_lists_png_files(self, flat_image_dir: Path):
        images = list_images(flat_image_dir)
        assert len(images) == 5
        assert all(p.suffix == ".png" for p in images)

    def test_sorted_order(self, flat_image_dir: Path):
        images = list_images(flat_image_dir)
        names = [p.name for p in images]
        assert names == sorted(names)

    def test_ignores_non_image_files(self, tmp_path: Path):
        (tmp_path / "data.csv").write_text("a,b")
        (tmp_path / "readme.txt").write_text("hello")
        _create_test_image(tmp_path / "photo.jpg")
        images = list_images(tmp_path)
        assert len(images) == 1
        assert images[0].name == "photo.jpg"

    def test_ignores_subdirectories(self, recursive_image_dir: Path):
        # Root of recursive dir has no images, only subdirs
        images = list_images(recursive_image_dir)
        assert len(images) == 0

    def test_multiple_extensions(self, tmp_path: Path):
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"):
            _create_test_image(tmp_path / f"test{ext}")
        images = list_images(tmp_path)
        assert len(images) == 6

    def test_empty_directory(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert list_images(empty) == []


# ── detect_mode / discover_folders ────────────────────────────


class TestDetectMode:
    def test_flat_mode(self, flat_image_dir: Path):
        assert detect_mode(flat_image_dir) is False

    def test_recursive_mode(self, recursive_image_dir: Path):
        assert detect_mode(recursive_image_dir) is True

    def test_no_images_raises(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="No images found"):
            detect_mode(empty)

    def test_hidden_dirs_ignored(self, tmp_path: Path):
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        _create_test_image(hidden / "img.png")
        with pytest.raises(FileNotFoundError):
            detect_mode(tmp_path)


class TestDiscoverFolders:
    def test_flat(self, flat_image_dir: Path):
        is_rec, folders = discover_folders(flat_image_dir)
        assert is_rec is False
        assert len(folders) == 1
        assert folders[0] == flat_image_dir

    def test_recursive(self, recursive_image_dir: Path):
        is_rec, folders = discover_folders(recursive_image_dir)
        assert is_rec is True
        assert len(folders) == 2
        names = {f.name for f in folders}
        assert names == {"folder_a", "folder_b"}


# ── write_csv ─────────────────────────────────────────────────


class TestWriteCsv:
    def test_creates_file(self, tmp_path: Path):
        results = [
            ImageScore(
                path=Path("test.png"),
                folder=".",
                face_score=0.9,
                clip_score=0.8,
                final_score=0.87,
                selected=True,
            )
        ]
        csv_path = tmp_path / "report.csv"
        write_csv(results, csv_path)
        assert csv_path.exists()

    def test_csv_content(self, tmp_path: Path):
        results = [
            ImageScore(
                path=Path("img.png"),
                folder="f1",
                face_score=0.95,
                clip_score=0.88,
                final_score=0.929,
                det_score=0.98,
                face_bbox="[50,50,200,200]",
                selected=True,
            ),
            ImageScore(
                path=Path("bad.png"),
                folder="f1",
                reject_reason="no_face_or_low_det",
            ),
        ]
        csv_path = tmp_path / "report.csv"
        write_csv(results, csv_path)

        with open(csv_path, encoding="utf-8") as f:
            reader = list(csv.reader(f))
        assert reader[0][0] == "folder"  # header
        assert len(reader) == 3  # header + 2 rows
        assert reader[1][0] == "f1"
        assert reader[1][1] == "img.png"
        assert reader[2][7] == "no_face_or_low_det"

    def test_creates_parent_dirs(self, tmp_path: Path):
        csv_path = tmp_path / "deep" / "nested" / "report.csv"
        write_csv([], csv_path)
        assert csv_path.exists()

    def test_empty_results(self, tmp_path: Path):
        csv_path = tmp_path / "empty.csv"
        write_csv([], csv_path)
        with open(csv_path, encoding="utf-8") as f:
            reader = list(csv.reader(f))
        assert len(reader) == 1  # header only


# ── copy_selected ─────────────────────────────────────────────


class TestCopySelected:
    def test_copies_selected_flat(self, flat_image_dir: Path, tmp_path: Path):
        images = list_images(flat_image_dir)
        results = [
            ImageScore(path=images[0], folder=".", selected=True),
            ImageScore(path=images[1], folder=".", selected=False),
            ImageScore(path=images[2], folder=".", selected=True),
        ]
        out = tmp_path / "out"
        count = copy_selected(results, out, recursive=False)
        assert count == 2
        assert (out / images[0].name).exists()
        assert (out / images[2].name).exists()
        assert not (out / images[1].name).exists()

    def test_copies_selected_recursive(self, recursive_image_dir: Path, tmp_path: Path):
        folder_a = recursive_image_dir / "folder_a"
        imgs = list_images(folder_a)
        results = [ImageScore(path=imgs[0], folder="folder_a", selected=True)]
        out = tmp_path / "out"
        count = copy_selected(results, out, recursive=True)
        assert count == 1
        assert (out / "folder_a" / imgs[0].name).exists()

    def test_no_selected_returns_zero(self, flat_image_dir: Path, tmp_path: Path):
        images = list_images(flat_image_dir)
        results = [ImageScore(path=images[0], folder=".", selected=False)]
        count = copy_selected(results, tmp_path / "out", recursive=False)
        assert count == 0


# ── Image helpers ─────────────────────────────────────────────


class TestImageHelpers:
    def test_load_bgr(self, test_image: Path):
        img = load_bgr(test_image)
        assert img is not None
        assert img.shape == (64, 64, 3)

    def test_load_bgr_missing_file(self, tmp_path: Path):
        result = load_bgr(tmp_path / "nope.png")
        assert result is None

    def test_pil_from_path(self, test_image: Path):
        img = pil_from_path(test_image)
        assert img.size == (64, 64)
        assert img.mode == "RGB"

    def test_crop_face_region(self, test_image: Path):
        bgr = load_bgr(test_image)
        face = MockFace(bbox=(10, 10, 50, 50))
        crop = crop_face_region(bgr, face, expand=1.2)
        assert crop.mode == "RGB"
        assert crop.size[0] > 0 and crop.size[1] > 0


# ── get_torch_device / get_onnx_providers ─────────────────────


class TestDeviceHelpers:
    def test_get_torch_device_cpu(self):
        with patch("character_sniper.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = False
            from character_sniper import get_torch_device

            # Re-import to get the patched version
            assert get_torch_device() == "cpu"

    def test_get_onnx_providers_cpu_only(self):
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            from character_sniper import get_onnx_providers

            providers = get_onnx_providers()
            assert "CPUExecutionProvider" in providers
