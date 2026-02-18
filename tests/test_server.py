"""Tests for server.py — helpers and API endpoints."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.conftest import _create_test_image, make_sample_job_data


# ── Helper functions (pure, no app needed) ────────────────────


class TestUrlFolderFilter:
    def test_flat_folder(self):
        from server import _url_folder_filter

        assert _url_folder_filter(".") == "_flat"

    def test_named_folder(self):
        from server import _url_folder_filter

        assert _url_folder_filter("folder_a") == "folder_a"


class TestUrlFolder:
    def test_flat(self):
        from server import _url_folder

        assert _url_folder(".") == "_flat"

    def test_named(self):
        from server import _url_folder

        assert _url_folder("some_folder") == "some_folder"


class TestSseEncode:
    def test_collapses_newlines(self):
        from server import _sse_encode

        html = "<div>\n  <span>hello</span>\n</div>"
        result = _sse_encode(html)
        assert "\n" not in result
        assert "<span>hello</span>" in result

    def test_strips_whitespace(self):
        from server import _sse_encode

        assert _sse_encode("  hello  ") == "hello"


class TestNewJob:
    def test_creates_job_with_unique_id(self):
        from server import new_job, jobs

        # Clean state
        jobs.clear()
        j1 = new_job(
            "/a",
            "/b",
            {"top_k": 5, "face_weight": 0.7, "clip_weight": 0.3, "min_face_score": 0.5},
        )
        j2 = new_job(
            "/a",
            "/b",
            {"top_k": 5, "face_weight": 0.7, "clip_weight": 0.3, "min_face_score": 0.5},
        )
        assert j1 != j2
        assert j1 in jobs
        assert j2 in jobs
        jobs.clear()

    def test_job_initial_state(self):
        from server import new_job, jobs

        jobs.clear()
        jid = new_job(
            "/orig.png",
            "/input",
            {"top_k": 3, "face_weight": 0.7, "clip_weight": 0.3, "min_face_score": 0.5},
        )
        j = jobs[jid]
        assert j["status"] == "queued"
        assert j["progress"] == 0
        assert j["original_path"] == "/orig.png"
        assert j["input_path"] == "/input"
        assert j["params"]["top_k"] == 3
        assert j["results"] == {}
        assert j["error"] is None
        jobs.clear()


class TestListDirectory:
    def test_existing_directory(self, tmp_path: Path):
        from server import _list_directory

        (tmp_path / "file.txt").write_text("hi")
        (tmp_path / "sub").mkdir()
        result = _list_directory(str(tmp_path))
        assert "error" not in result
        names = {e["name"] for e in result["entries"]}
        assert "file.txt" in names
        assert "sub" in names

    def test_nonexistent_path(self):
        from server import _list_directory

        result = _list_directory("/nonexistent/path/xyz")
        assert result["error"] == "Path not found"

    def test_file_not_dir(self, tmp_path: Path):
        from server import _list_directory

        f = tmp_path / "file.txt"
        f.write_text("hi")
        result = _list_directory(str(f))
        assert result["error"] == "Not a directory"

    def test_hidden_files_excluded(self, tmp_path: Path):
        from server import _list_directory

        (tmp_path / ".hidden").write_text("secret")
        (tmp_path / "visible.txt").write_text("hello")
        result = _list_directory(str(tmp_path))
        names = {e["name"] for e in result["entries"]}
        assert ".hidden" not in names
        assert "visible.txt" in names

    def test_image_detection(self, tmp_path: Path):
        from server import _list_directory

        _create_test_image(tmp_path / "photo.png")
        (tmp_path / "data.csv").write_text("a,b")
        result = _list_directory(str(tmp_path))
        entries = {e["name"]: e for e in result["entries"]}
        assert entries["photo.png"]["is_image"] is True
        assert entries["data.csv"]["is_image"] is False


# ── API endpoints via TestClient ──────────────────────────────


@pytest.fixture()
def client(tmp_path: Path):
    """Create a TestClient with isolated SessionStore."""
    import server

    # Backup and replace global store with a temp one
    original_store = server.store
    server.store = server.SessionStore(db_path=tmp_path / "test.db")

    # Clear in-memory jobs and model cache
    original_jobs = dict(server.jobs)
    original_cache = dict(server._model_cache)
    server.jobs.clear()
    server._model_cache.clear()

    with TestClient(server.app) as c:
        yield c

    # Restore
    server.store.close()
    server.store = original_store
    server.jobs.clear()
    server.jobs.update(original_jobs)
    server._model_cache.clear()
    server._model_cache.update(original_cache)


class TestSettingsAPI:
    def test_get_default_settings(self, client):
        resp = client.get("/api/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert "original_path" in data
        assert "top_k" in data

    def test_save_and_get_settings(self, client):
        payload = {
            "original_path": "/custom/ref.png",
            "input_path": "/custom/output",
            "top_k": 10,
            "face_weight": 0.9,
            "clip_weight": 0.1,
            "min_face_score": 0.3,
            "workers": 2,
        }
        resp = client.post("/api/settings", json=payload)
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        resp = client.get("/api/settings")
        data = resp.json()
        assert data["original_path"] == "/custom/ref.png"
        assert data["top_k"] == 10
        assert data["workers"] == 2


class TestLatestJobAPI:
    def test_no_jobs(self, client):
        resp = client.get("/api/jobs/latest")
        assert resp.status_code == 200
        assert resp.json()["job_id"] is None

    def test_with_completed_job(self, client, tmp_path: Path):
        import os
        import server

        # Create input dir with mtime in the past so is_job_valid passes
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        past = time.time() - 7200
        os.utime(input_dir, (past, past))

        data = make_sample_job_data(input_path=str(input_dir))
        server.store.save_job("test_job", data)

        resp = client.get("/api/jobs/latest")
        body = resp.json()
        assert body["job_id"] == "test_job"
        assert body["valid"] is True


class TestBrowseAPI:
    def test_browse_cwd(self, client):
        resp = client.get("/api/browse", params={"path": "."})
        assert resp.status_code == 200
        data = resp.json()
        assert "entries" in data
        assert "path" in data

    def test_browse_nonexistent(self, client):
        resp = client.get("/api/browse", params={"path": "/nonexistent/xyz"})
        data = resp.json()
        assert data["error"] == "Path not found"

    def test_browse_tmp_dir(self, client, tmp_path: Path):
        (tmp_path / "hello.txt").write_text("world")
        resp = client.get("/api/browse", params={"path": str(tmp_path)})
        data = resp.json()
        names = {e["name"] for e in data["entries"]}
        # tmp_path also contains test.db from the client fixture
        assert "hello.txt" in names


class TestIndexPage:
    def test_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


class TestJobResultsAPI:
    def test_missing_job(self, client):
        resp = client.get("/api/jobs/nonexistent/results")
        assert resp.status_code == 404

    def test_completed_job(self, client):
        import server

        data = make_sample_job_data()
        server.jobs["j1"] = {
            "status": "complete",
            "progress": 100,
            "current_file": "",
            "current_folder": "",
            "folders_done": 1,
            "folders_total": 1,
            "images_done": 2,
            "images_total": 2,
            "results": data["results"],
            "all_results": [],
            "original_path": data["original_path"],
            "input_path": data["input_path"],
            "params": data["params"],
            "error": None,
            "is_recursive": False,
            "start_time": time.time() - 10,
            "end_time": time.time(),
        }
        resp = client.get("/api/jobs/j1/results")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


class TestToggleAPI:
    def test_toggle_image(self, client):
        import server

        data = make_sample_job_data()
        server.jobs["t1"] = {
            "status": "complete",
            "progress": 100,
            "current_file": "",
            "current_folder": "",
            "folders_done": 1,
            "folders_total": 1,
            "images_done": 2,
            "images_total": 2,
            "results": data["results"],
            "all_results": [],
            "original_path": data["original_path"],
            "input_path": data["input_path"],
            "params": data["params"],
            "error": None,
            "is_recursive": False,
            "start_time": time.time() - 10,
            "end_time": time.time(),
        }
        # Also save to DB so _ensure_job_in_memory + update_selection work
        server.store.save_job(
            "t1", {**data, "start_time": time.time() - 10, "end_time": time.time()}
        )

        # img_001 starts as not selected
        resp = client.post("/api/jobs/t1/toggle/_flat/img_001.png")
        assert resp.status_code == 200
        # Verify it was toggled
        img = [
            i
            for i in server.jobs["t1"]["results"]["."]
            if i["filename"] == "img_001.png"
        ][0]
        assert img["selected"] is True

    def test_toggle_missing_job(self, client):
        resp = client.post("/api/jobs/nope/toggle/_flat/img.png")
        assert resp.status_code == 404


class TestImageServing:
    def test_serve_original(self, client, tmp_path: Path):
        import server

        img_path = _create_test_image(tmp_path / "original.png")
        server.jobs["img_job"] = {
            "status": "complete",
            "original_path": str(img_path),
            "input_path": str(tmp_path),
            "params": {},
            "results": {},
            "all_results": [],
        }
        resp = client.get("/images/img_job/original")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("image/")

    def test_serve_flat_image(self, client, tmp_path: Path):
        import server

        _create_test_image(tmp_path / "photo.png")
        server.jobs["flat_job"] = {
            "status": "complete",
            "original_path": str(tmp_path / "orig.png"),
            "input_path": str(tmp_path),
            "params": {},
            "results": {},
            "all_results": [],
        }
        resp = client.get("/images/flat_job/_flat/photo.png")
        assert resp.status_code == 200

    def test_serve_recursive_image(self, client, tmp_path: Path):
        import server

        sub = tmp_path / "folder_a"
        sub.mkdir()
        _create_test_image(sub / "img.png")
        server.jobs["rec_job"] = {
            "status": "complete",
            "original_path": str(tmp_path / "orig.png"),
            "input_path": str(tmp_path),
            "params": {},
            "results": {},
            "all_results": [],
        }
        resp = client.get("/images/rec_job/folder_a/img.png")
        assert resp.status_code == 200

    def test_serve_missing_image(self, client):
        import server

        server.jobs["miss_job"] = {
            "status": "complete",
            "original_path": "/tmp/nope.png",
            "input_path": "/tmp",
            "params": {},
            "results": {},
            "all_results": [],
        }
        resp = client.get("/images/miss_job/original")
        assert resp.status_code == 404

    def test_serve_missing_job(self, client):
        resp = client.get("/images/nope/original")
        assert resp.status_code == 404


class TestStartJobAPI:
    def test_missing_original(self, client, tmp_path: Path):
        """Starting a job with non-existent original should return 400."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        resp = client.post(
            "/api/jobs/start",
            data={
                "original_path": str(tmp_path / "nope.png"),
                "input_path": str(input_dir),
                "top_k": "5",
                "face_weight": "0.7",
                "clip_weight": "0.3",
                "min_face_score": "0.5",
            },
        )
        assert resp.status_code == 400
        assert "not found" in resp.text.lower()

    def test_missing_input(self, client, tmp_path: Path):
        """Starting a job with non-existent input path should return 400."""
        img = _create_test_image(tmp_path / "original.png")
        resp = client.post(
            "/api/jobs/start",
            data={
                "original_path": str(img),
                "input_path": str(tmp_path / "nope"),
                "top_k": "5",
                "face_weight": "0.7",
                "clip_weight": "0.3",
                "min_face_score": "0.5",
            },
        )
        assert resp.status_code == 400
        assert "not found" in resp.text.lower()


class TestExportAPI:
    def test_export_missing_job(self, client):
        resp = client.post("/api/jobs/nope/export")
        assert resp.status_code == 404

    def test_export_no_results(self, client):
        import server

        server.jobs["empty_job"] = {
            "status": "complete",
            "all_results": [],
            "results": {},
            "original_path": "/tmp/orig.png",
            "input_path": "/tmp",
            "params": {},
            "is_recursive": False,
        }
        resp = client.post("/api/jobs/empty_job/export")
        assert resp.status_code == 200
        assert "No results" in resp.text
