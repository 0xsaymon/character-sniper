"""Tests for session_store.SessionStore."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from session_store import SessionStore, _DEFAULT_SETTINGS
from tests.conftest import make_sample_job_data


# ── Schema / init ─────────────────────────────────────────────


class TestInit:
    def test_creates_db_file(self, tmp_path: Path):
        db = tmp_path / "sub" / "test.db"
        store = SessionStore(db_path=db)
        assert db.exists()
        store.close()

    def test_wal_mode(self, store: SessionStore):
        row = store._conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0] == "wal"

    def test_foreign_keys_enabled(self, store: SessionStore):
        row = store._conn.execute("PRAGMA foreign_keys").fetchone()
        assert row[0] == 1

    def test_schema_version_stored(self, store: SessionStore):
        row = store._conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        assert row is not None
        assert row[0] == "1"

    def test_tables_exist(self, store: SessionStore):
        tables = {
            r[0]
            for r in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {"meta", "settings", "jobs", "job_results"}.issubset(tables)


# ── Settings ──────────────────────────────────────────────────


class TestSettings:
    def test_load_defaults_when_empty(self, store: SessionStore):
        settings = store.load_settings()
        assert settings == _DEFAULT_SETTINGS

    def test_save_and_load_round_trip(self, store: SessionStore):
        custom = {
            "original_path": "/custom/ref.png",
            "input_path": "/custom/output",
            "top_k": 10,
            "face_weight": 0.9,
            "clip_weight": 0.1,
            "min_face_score": 0.3,
        }
        store.save_settings(custom)
        loaded = store.load_settings()
        assert loaded == custom

    def test_upsert_overwrites(self, store: SessionStore):
        first = dict(_DEFAULT_SETTINGS, top_k=3)
        store.save_settings(first)
        second = dict(_DEFAULT_SETTINGS, top_k=99)
        store.save_settings(second)
        loaded = store.load_settings()
        assert loaded["top_k"] == 99


# ── Jobs ──────────────────────────────────────────────────────


class TestJobs:
    def test_save_and_load_job(self, store: SessionStore):
        data = make_sample_job_data()
        store.save_job("job1", data)
        loaded = store.load_job("job1")
        assert loaded is not None
        assert loaded["id"] == "job1"
        assert loaded["status"] == "complete"
        assert "." in loaded["results"]
        assert len(loaded["results"]["."]) == 2

    def test_load_missing_job_returns_none(self, store: SessionStore):
        assert store.load_job("nonexistent") is None

    def test_load_latest_job(self, store: SessionStore):
        data1 = make_sample_job_data()
        data1["start_time"] = time.time() - 100
        data1["end_time"] = time.time() - 90
        store.save_job("old_job", data1)

        data2 = make_sample_job_data()
        data2["start_time"] = time.time() - 10
        data2["end_time"] = time.time()
        store.save_job("new_job", data2)

        latest = store.load_latest_job()
        assert latest is not None
        assert latest["id"] == "new_job"

    def test_load_latest_job_empty_db(self, store: SessionStore):
        assert store.load_latest_job() is None

    def test_load_latest_ignores_errored_jobs(self, store: SessionStore):
        data = make_sample_job_data(status="error")
        store.save_job("err_job", data)
        assert store.load_latest_job() is None

    def test_list_jobs(self, store: SessionStore):
        store.save_job("j1", make_sample_job_data())
        store.save_job("j2", make_sample_job_data())
        listing = store.list_jobs()
        assert len(listing) == 2
        ids = {j["id"] for j in listing}
        assert ids == {"j1", "j2"}

    def test_delete_job_cascades(self, store: SessionStore):
        store.save_job("doomed", make_sample_job_data())
        assert store.load_job("doomed") is not None

        store.delete_job("doomed")
        assert store.load_job("doomed") is None
        # Results should be deleted too (CASCADE)
        count = store._conn.execute(
            "SELECT COUNT(*) FROM job_results WHERE job_id = 'doomed'"
        ).fetchone()[0]
        assert count == 0

    def test_job_results_structure(self, store: SessionStore):
        store.save_job("j_struct", make_sample_job_data())
        loaded = store.load_job("j_struct")
        img = loaded["results"]["."][0]
        expected_keys = {
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
        assert set(img.keys()) == expected_keys

    def test_save_job_with_elapsed(self, store: SessionStore):
        data = make_sample_job_data()
        store.save_job("j_elapsed", data)
        loaded = store.load_job("j_elapsed")
        assert loaded["elapsed"] is not None
        assert loaded["elapsed"] == pytest.approx(10.0, abs=1)


# ── Selection updates ─────────────────────────────────────────


class TestSelection:
    def test_update_selection(self, store: SessionStore):
        store.save_job("sel_job", make_sample_job_data())
        # img_001 starts as not selected
        loaded = store.load_job("sel_job")
        img1 = [i for i in loaded["results"]["."] if i["filename"] == "img_001.png"][0]
        assert img1["selected"] is False

        store.update_selection("sel_job", ".", "img_001.png", True)
        loaded = store.load_job("sel_job")
        img1 = [i for i in loaded["results"]["."] if i["filename"] == "img_001.png"][0]
        assert img1["selected"] is True

    def test_update_selection_toggle_off(self, store: SessionStore):
        store.save_job("sel_job2", make_sample_job_data())
        # img_000 starts as selected
        store.update_selection("sel_job2", ".", "img_000.png", False)
        loaded = store.load_job("sel_job2")
        img0 = [i for i in loaded["results"]["."] if i["filename"] == "img_000.png"][0]
        assert img0["selected"] is False


# ── Validation ────────────────────────────────────────────────


class TestValidation:
    def test_is_job_valid_missing_job(self, store: SessionStore):
        assert store.is_job_valid("nope") is False

    def test_is_job_valid_missing_folder(self, store: SessionStore, tmp_path: Path):
        data = make_sample_job_data(input_path=str(tmp_path / "gone"))
        store.save_job("v1", data)
        assert store.is_job_valid("v1") is False

    def test_is_job_valid_folder_not_modified(
        self, store: SessionStore, tmp_path: Path
    ):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        # Set mtime to past
        past = time.time() - 3600
        os.utime(input_dir, (past, past))

        data = make_sample_job_data(input_path=str(input_dir))
        store.save_job("v2", data)
        assert store.is_job_valid("v2") is True

    def test_is_job_valid_folder_modified_after_job(
        self, store: SessionStore, tmp_path: Path
    ):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        # Save job with timestamps in the past
        data = make_sample_job_data(input_path=str(input_dir))
        data["start_time"] = time.time() - 3600
        data["end_time"] = time.time() - 3590
        store.save_job("v3", data)

        # Touch the folder to simulate modification
        future = time.time() + 100
        os.utime(input_dir, (future, future))

        assert store.is_job_valid("v3") is False

    def test_is_job_valid_subfolder_modified(self, store: SessionStore, tmp_path: Path):
        input_dir = tmp_path / "input"
        sub = input_dir / "sub"
        sub.mkdir(parents=True)
        past = time.time() - 3600
        os.utime(input_dir, (past, past))
        os.utime(sub, (past, past))

        data = make_sample_job_data(input_path=str(input_dir))
        store.save_job("v4", data)

        # Modify subfolder
        future = time.time() + 100
        os.utime(sub, (future, future))

        assert store.is_job_valid("v4") is False


# ── Close ─────────────────────────────────────────────────────


class TestClose:
    def test_close(self, store: SessionStore):
        store.close()
        with pytest.raises(Exception):
            store.load_settings()
