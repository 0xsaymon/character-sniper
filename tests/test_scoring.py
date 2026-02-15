"""Tests for character_sniper.py — scoring pipeline (with mocked ML models)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from character_sniper import (
    prepare_original,
    score_single_image,
    process_folder,
)


# ── prepare_original ──────────────────────────────────────────


class TestPrepareOriginal:
    def test_success(self, test_image: Path, mock_face_analyzer, mock_clip_encoder):
        with patch("character_sniper.load_bgr") as mock_load:
            mock_load.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
            face_emb, clip_emb = prepare_original(
                test_image, mock_face_analyzer, mock_clip_encoder
            )
        assert face_emb is not None
        assert clip_emb is not None
        assert face_emb.shape == (512,)

    def test_unreadable_image(
        self, test_image: Path, mock_face_analyzer, mock_clip_encoder
    ):
        with patch("character_sniper.load_bgr", return_value=None):
            with pytest.raises(ValueError, match="Cannot read"):
                prepare_original(test_image, mock_face_analyzer, mock_clip_encoder)

    def test_no_face_detected(
        self, test_image: Path, mock_face_analyzer, mock_clip_encoder
    ):
        mock_face_analyzer.get_best_face.return_value = None
        with patch("character_sniper.load_bgr") as mock_load:
            mock_load.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
            with pytest.raises(ValueError, match="No face detected"):
                prepare_original(test_image, mock_face_analyzer, mock_clip_encoder)

    def test_full_clip_mode(
        self, test_image: Path, mock_face_analyzer, mock_clip_encoder
    ):
        with patch("character_sniper.load_bgr") as mock_load:
            mock_load.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
            face_emb, clip_emb = prepare_original(
                test_image,
                mock_face_analyzer,
                mock_clip_encoder,
                clip_mode="full",
            )
        assert face_emb is not None
        assert clip_emb is not None


# ── score_single_image ────────────────────────────────────────


class TestScoreSingleImage:
    def _make_embeddings(self):
        rng = np.random.default_rng(42)
        face_emb = rng.standard_normal(512).astype(np.float32)
        face_emb /= np.linalg.norm(face_emb)
        clip_emb = rng.standard_normal(512).astype(np.float32)
        clip_emb /= np.linalg.norm(clip_emb)
        return face_emb, clip_emb

    def test_successful_score(
        self, test_image: Path, mock_face_analyzer, mock_clip_encoder
    ):
        face_emb, clip_emb = self._make_embeddings()
        with patch("character_sniper.load_bgr") as mock_load:
            mock_load.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
            result = score_single_image(
                test_image,
                ".",
                mock_face_analyzer,
                mock_clip_encoder,
                face_emb,
                clip_emb,
            )
        assert result.final_score is not None
        assert result.face_score is not None
        assert result.clip_score is not None
        assert result.reject_reason is None

    def test_unreadable_image(
        self, test_image: Path, mock_face_analyzer, mock_clip_encoder
    ):
        face_emb, clip_emb = self._make_embeddings()
        with patch("character_sniper.load_bgr", return_value=None):
            result = score_single_image(
                test_image,
                ".",
                mock_face_analyzer,
                mock_clip_encoder,
                face_emb,
                clip_emb,
            )
        assert result.reject_reason == "unreadable"
        assert result.final_score is None

    def test_no_face(self, test_image: Path, mock_face_analyzer, mock_clip_encoder):
        face_emb, clip_emb = self._make_embeddings()
        mock_face_analyzer.get_best_face.return_value = None
        with patch("character_sniper.load_bgr") as mock_load:
            mock_load.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
            result = score_single_image(
                test_image,
                ".",
                mock_face_analyzer,
                mock_clip_encoder,
                face_emb,
                clip_emb,
            )
        assert result.reject_reason == "no_face_or_low_det"

    def test_bad_landmarks(
        self, test_image: Path, mock_face_analyzer, mock_clip_encoder
    ):
        face_emb, clip_emb = self._make_embeddings()
        mock_face_analyzer.landmarks_ok.return_value = False
        with patch("character_sniper.load_bgr") as mock_load:
            mock_load.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
            result = score_single_image(
                test_image,
                ".",
                mock_face_analyzer,
                mock_clip_encoder,
                face_emb,
                clip_emb,
            )
        assert result.reject_reason == "bad_landmarks"

    def test_face_too_small(
        self, test_image: Path, mock_face_analyzer, mock_clip_encoder
    ):
        face_emb, clip_emb = self._make_embeddings()
        mock_face_analyzer.face_bbox_big_enough.return_value = False
        with patch("character_sniper.load_bgr") as mock_load:
            mock_load.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
            result = score_single_image(
                test_image,
                ".",
                mock_face_analyzer,
                mock_clip_encoder,
                face_emb,
                clip_emb,
            )
        assert result.reject_reason == "face_too_small"

    def test_face_only_mode(
        self, test_image: Path, mock_face_analyzer, mock_clip_encoder
    ):
        """When clip_weight=0, clip should not be called."""
        face_emb, clip_emb = self._make_embeddings()
        with patch("character_sniper.load_bgr") as mock_load:
            mock_load.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
            result = score_single_image(
                test_image,
                ".",
                mock_face_analyzer,
                mock_clip_encoder,
                face_emb,
                clip_emb,
                face_weight=1.0,
                clip_weight=0.0,
            )
        assert result.clip_score == 0.0
        mock_clip_encoder.encode_image.assert_not_called()

    def test_min_similarity_rejection(
        self, test_image: Path, mock_face_analyzer, mock_clip_encoder
    ):
        face_emb, clip_emb = self._make_embeddings()
        # Make face similarity very low by using opposite embedding
        neg_face_emb = -face_emb
        with patch("character_sniper.load_bgr") as mock_load:
            mock_load.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
            result = score_single_image(
                test_image,
                ".",
                mock_face_analyzer,
                mock_clip_encoder,
                neg_face_emb,
                clip_emb,
                min_similarity=0.9,
            )
        assert result.reject_reason is not None
        assert "face_sim_below" in result.reject_reason


# ── process_folder ────────────────────────────────────────────


class TestProcessFolder:
    def test_top_k_selection(
        self, flat_image_dir: Path, mock_face_analyzer, mock_clip_encoder
    ):
        from character_sniper import list_images

        images = list_images(flat_image_dir)
        rng = np.random.default_rng(42)
        face_emb = rng.standard_normal(512).astype(np.float32)
        face_emb /= np.linalg.norm(face_emb)
        clip_emb = rng.standard_normal(512).astype(np.float32)
        clip_emb /= np.linalg.norm(clip_emb)

        with patch("character_sniper.load_bgr") as mock_load:
            mock_load.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
            results = process_folder(
                image_paths=images,
                folder_name=".",
                face_analyzer=mock_face_analyzer,
                clip_encoder=mock_clip_encoder,
                orig_face_emb=face_emb,
                orig_clip_emb=clip_emb,
                top_k=2,
            )

        assert len(results) == 5
        selected = [r for r in results if r.selected]
        assert len(selected) == 2

    def test_progress_callback(
        self, flat_image_dir: Path, mock_face_analyzer, mock_clip_encoder
    ):
        from character_sniper import list_images

        images = list_images(flat_image_dir)
        rng = np.random.default_rng(42)
        face_emb = rng.standard_normal(512).astype(np.float32)
        face_emb /= np.linalg.norm(face_emb)
        clip_emb = rng.standard_normal(512).astype(np.float32)
        clip_emb /= np.linalg.norm(clip_emb)

        progress_calls = []

        def on_progress(current, total, filename):
            progress_calls.append((current, total, filename))

        with patch("character_sniper.load_bgr") as mock_load:
            mock_load.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
            process_folder(
                image_paths=images,
                folder_name=".",
                face_analyzer=mock_face_analyzer,
                clip_encoder=mock_clip_encoder,
                orig_face_emb=face_emb,
                orig_clip_emb=clip_emb,
                top_k=2,
                on_progress=on_progress,
            )

        assert len(progress_calls) == 5
        assert progress_calls[0][0] == 1  # first call: current=1
        assert progress_calls[-1][0] == 5  # last call: current=5
        assert progress_calls[0][1] == 5  # total always 5

    def test_all_rejected(
        self, flat_image_dir: Path, mock_face_analyzer, mock_clip_encoder
    ):
        """When all images are rejected, no images should be selected."""
        from character_sniper import list_images

        images = list_images(flat_image_dir)
        mock_face_analyzer.get_best_face.return_value = None
        rng = np.random.default_rng(42)
        face_emb = rng.standard_normal(512).astype(np.float32)
        face_emb /= np.linalg.norm(face_emb)
        clip_emb = rng.standard_normal(512).astype(np.float32)
        clip_emb /= np.linalg.norm(clip_emb)

        with patch("character_sniper.load_bgr") as mock_load:
            mock_load.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
            results = process_folder(
                image_paths=images,
                folder_name=".",
                face_analyzer=mock_face_analyzer,
                clip_encoder=mock_clip_encoder,
                orig_face_emb=face_emb,
                orig_clip_emb=clip_emb,
                top_k=3,
            )

        assert all(not r.selected for r in results)
        assert all(r.reject_reason is not None for r in results)
