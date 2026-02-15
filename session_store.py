"""
Session persistence for Character Sniper Web UI.

Uses SQLite (stdlib) to store user settings, completed jobs, and scoring
results so that state survives page reloads and server restarts.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ── Schema version ────────────────────────────────────────────

_SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- User settings (singleton row, id always = 1)
CREATE TABLE IF NOT EXISTS settings (
    id             INTEGER PRIMARY KEY CHECK (id = 1),
    original_path  TEXT    NOT NULL DEFAULT 'data/original.png',
    input_path     TEXT    NOT NULL DEFAULT 'data/output',
    top_k          INTEGER NOT NULL DEFAULT 5,
    face_weight    REAL    NOT NULL DEFAULT 0.70,
    clip_weight    REAL    NOT NULL DEFAULT 0.30,
    min_face_score REAL    NOT NULL DEFAULT 0.50,
    updated_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Completed (or errored) jobs
CREATE TABLE IF NOT EXISTS jobs (
    id             TEXT PRIMARY KEY,
    status         TEXT NOT NULL,
    original_path  TEXT NOT NULL,
    input_path     TEXT NOT NULL,
    top_k          INTEGER NOT NULL,
    face_weight    REAL NOT NULL,
    clip_weight    REAL NOT NULL,
    min_face_score REAL NOT NULL,
    is_recursive   INTEGER NOT NULL DEFAULT 0,
    elapsed        REAL,
    error          TEXT,
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Per-image scoring results
CREATE TABLE IF NOT EXISTS job_results (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id        TEXT    NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    folder        TEXT    NOT NULL,
    filename      TEXT    NOT NULL,
    file_path     TEXT    NOT NULL,
    face_score    REAL,
    clip_score    REAL,
    final_score   REAL,
    det_score     REAL,
    face_bbox     TEXT,
    reject_reason TEXT,
    selected      INTEGER NOT NULL DEFAULT 0,
    UNIQUE(job_id, folder, filename)
);

CREATE INDEX IF NOT EXISTS idx_job_results_job ON job_results(job_id);
"""

# Default settings (must match Alpine.js defaults in index.html)
_DEFAULT_SETTINGS: dict[str, object] = {
    "original_path": "data/original.png",
    "input_path": "data/output",
    "top_k": 5,
    "face_weight": 0.70,
    "clip_weight": 0.30,
    "min_face_score": 0.50,
}


class SessionStore:
    """SQLite-backed persistence for web UI sessions."""

    def __init__(self, db_path: Path = Path("data/sessions.db")) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()
        print(f"[session] SQLite store: {db_path}")

    # ── Schema init ───────────────────────────────────────────

    def _init_schema(self) -> None:
        """Create tables if they don't exist and run migrations."""
        self._conn.executescript(_SCHEMA_SQL)
        # Ensure meta version
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(_SCHEMA_VERSION),),
            )
        self._conn.commit()

    # ── Settings ──────────────────────────────────────────────

    def load_settings(self) -> dict[str, object]:
        """Return saved settings or defaults."""
        row = self._conn.execute("SELECT * FROM settings WHERE id = 1").fetchone()
        if row is None:
            return dict(_DEFAULT_SETTINGS)
        return {
            "original_path": row["original_path"],
            "input_path": row["input_path"],
            "top_k": row["top_k"],
            "face_weight": row["face_weight"],
            "clip_weight": row["clip_weight"],
            "min_face_score": row["min_face_score"],
        }

    def save_settings(self, settings: dict[str, object]) -> None:
        """Upsert user settings (always id=1)."""
        self._conn.execute(
            """INSERT INTO settings (id, original_path, input_path, top_k,
                                     face_weight, clip_weight, min_face_score, updated_at)
               VALUES (1, :original_path, :input_path, :top_k,
                        :face_weight, :clip_weight, :min_face_score, datetime('now'))
               ON CONFLICT(id) DO UPDATE SET
                   original_path  = excluded.original_path,
                   input_path     = excluded.input_path,
                   top_k          = excluded.top_k,
                   face_weight    = excluded.face_weight,
                   clip_weight    = excluded.clip_weight,
                   min_face_score = excluded.min_face_score,
                   updated_at     = excluded.updated_at
            """,
            settings,
        )
        self._conn.commit()

    # ── Jobs ──────────────────────────────────────────────────

    def save_job(self, job_id: str, job_data: dict) -> None:
        """Persist a completed/errored job and its results."""
        params = job_data["params"]
        elapsed = None
        if job_data.get("end_time") and job_data.get("start_time"):
            elapsed = job_data["end_time"] - job_data["start_time"]

        self._conn.execute(
            """INSERT OR REPLACE INTO jobs
               (id, status, original_path, input_path, top_k,
                face_weight, clip_weight, min_face_score,
                is_recursive, elapsed, error, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                job_data["status"],
                job_data["original_path"],
                job_data["input_path"],
                params["top_k"],
                params["face_weight"],
                params["clip_weight"],
                params["min_face_score"],
                int(job_data.get("is_recursive", False)),
                elapsed,
                job_data.get("error"),
                datetime.fromtimestamp(
                    job_data.get("start_time", time.time()), tz=timezone.utc
                ).isoformat(),
            ),
        )

        # Save results
        results = job_data.get("results", {})
        for folder_name, folder_results in results.items():
            for img in folder_results:
                self._conn.execute(
                    """INSERT OR REPLACE INTO job_results
                       (job_id, folder, filename, file_path,
                        face_score, clip_score, final_score, det_score,
                        face_bbox, reject_reason, selected)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        folder_name,
                        img["filename"],
                        img["path"],
                        img.get("face_score"),
                        img.get("clip_score"),
                        img.get("final_score"),
                        img.get("det_score"),
                        img.get("face_bbox"),
                        img.get("reject_reason"),
                        int(img.get("selected", False)),
                    ),
                )

        self._conn.commit()

    def load_job(self, job_id: str) -> Optional[dict]:
        """Load a job + its results from SQLite. Return None if not found."""
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if row is None:
            return None

        # Build results dict: folder -> [image_dict, ...]
        results: dict[str, list[dict]] = {}
        rows = self._conn.execute(
            "SELECT * FROM job_results WHERE job_id = ? ORDER BY id", (job_id,)
        ).fetchall()
        for r in rows:
            folder = r["folder"]
            if folder not in results:
                results[folder] = []
            results[folder].append(
                {
                    "path": r["file_path"],
                    "filename": r["filename"],
                    "folder": r["folder"],
                    "face_score": r["face_score"],
                    "clip_score": r["clip_score"],
                    "final_score": r["final_score"],
                    "det_score": r["det_score"],
                    "face_bbox": r["face_bbox"],
                    "reject_reason": r["reject_reason"],
                    "selected": bool(r["selected"]),
                }
            )

        return {
            "id": row["id"],
            "status": row["status"],
            "original_path": row["original_path"],
            "input_path": row["input_path"],
            "params": {
                "top_k": row["top_k"],
                "face_weight": row["face_weight"],
                "clip_weight": row["clip_weight"],
                "min_face_score": row["min_face_score"],
            },
            "is_recursive": bool(row["is_recursive"]),
            "elapsed": row["elapsed"],
            "error": row["error"],
            "created_at": row["created_at"],
            "results": results,
        }

    def load_latest_job(self) -> Optional[dict]:
        """Load the most recent completed job."""
        row = self._conn.execute(
            "SELECT id FROM jobs WHERE status = 'complete' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return self.load_job(row["id"])

    def list_jobs(self) -> list[dict]:
        """Return summary list of all jobs (without full results)."""
        rows = self._conn.execute(
            "SELECT id, status, original_path, input_path, elapsed, error, created_at "
            "FROM jobs ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_job(self, job_id: str) -> None:
        """Delete a job and its results (CASCADE)."""
        self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        self._conn.commit()

    # ── Selection updates ─────────────────────────────────────

    def update_selection(
        self, job_id: str, folder: str, filename: str, selected: bool
    ) -> None:
        """Update the selected flag for a single image."""
        self._conn.execute(
            "UPDATE job_results SET selected = ? "
            "WHERE job_id = ? AND folder = ? AND filename = ?",
            (int(selected), job_id, folder, filename),
        )
        self._conn.commit()

    # ── Invalidation ──────────────────────────────────────────

    def is_job_valid(self, job_id: str) -> bool:
        """Check if the input folder hasn't been modified since the job was created.

        A job is valid when the input directory still exists and its mtime
        is not newer than the job creation time.
        """
        row = self._conn.execute(
            "SELECT input_path, created_at FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if row is None:
            return False

        input_path = Path(row["input_path"])
        if not input_path.exists():
            return False

        # Parse ISO timestamp from DB
        created_str = row["created_at"]
        try:
            job_time = datetime.fromisoformat(created_str).timestamp()
        except ValueError:
            # Fallback: sqlite datetime format "YYYY-MM-DD HH:MM:SS"
            job_time = datetime.strptime(created_str, "%Y-%m-%d %H:%M:%S").timestamp()

        # Check mtime of the input folder itself and its immediate children
        folder_mtime = input_path.stat().st_mtime
        if folder_mtime > job_time:
            return False

        # Also check subfolder mtimes for recursive mode
        for child in input_path.iterdir():
            if child.is_dir() and not child.name.startswith("."):
                if child.stat().st_mtime > job_time:
                    return False

        return True

    # ── Cleanup ───────────────────────────────────────────────

    def close(self) -> None:
        self._conn.close()
