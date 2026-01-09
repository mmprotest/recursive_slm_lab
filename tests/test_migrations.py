from __future__ import annotations

from pathlib import Path
import sqlite3
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from recursive_slm_lab.memory import connect


def test_migrations_add_new_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            condition TEXT NOT NULL,
            prompt TEXT NOT NULL,
            candidate_code TEXT NOT NULL,
            passed INTEGER NOT NULL,
            test_log TEXT NOT NULL,
            created_at TEXT NOT NULL,
            code_hash TEXT NOT NULL
        );
        CREATE TABLE failures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            condition TEXT NOT NULL,
            prompt TEXT NOT NULL,
            candidate_code TEXT NOT NULL,
            passed INTEGER NOT NULL,
            test_log TEXT NOT NULL,
            created_at TEXT NOT NULL,
            code_hash TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()

    conn = connect(db_path)
    episode_columns = {row[1] for row in conn.execute("PRAGMA table_info(episodes)").fetchall()}
    failure_columns = {row[1] for row in conn.execute("PRAGMA table_info(failures)").fetchall()}
    for column in ["run_id", "prompt_hash", "retrieval_used", "memory_sources", "memory_top_score"]:
        assert column in episode_columns
        assert column in failure_columns
    runs_table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'"
    ).fetchone()
    assert runs_table is not None
    conn.close()
