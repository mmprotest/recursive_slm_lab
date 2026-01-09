from __future__ import annotations

import sqlite3


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row[1] == column for row in rows)


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            mode TEXT NOT NULL,
            backend TEXT NOT NULL,
            model TEXT NOT NULL,
            adapter_name TEXT,
            memory_enabled INTEGER NOT NULL,
            semantic_enabled INTEGER NOT NULL,
            learning_enabled INTEGER NOT NULL,
            k INTEGER NOT NULL,
            max_tokens INTEGER NOT NULL,
            temperature REAL NOT NULL,
            top_p REAL NOT NULL,
            top_k INTEGER NOT NULL,
            notes TEXT,
            config_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS promotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            previous_adapter_name TEXT,
            candidate_adapter_name TEXT,
            decision TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS regression_tasks (
            task_id TEXT PRIMARY KEY,
            rank INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    if _table_exists(conn, "episodes"):
        if not _column_exists(conn, "episodes", "run_id"):
            conn.execute("ALTER TABLE episodes ADD COLUMN run_id INTEGER")
        if not _column_exists(conn, "episodes", "prompt_hash"):
            conn.execute("ALTER TABLE episodes ADD COLUMN prompt_hash TEXT")
        if not _column_exists(conn, "episodes", "retrieval_used"):
            conn.execute("ALTER TABLE episodes ADD COLUMN retrieval_used INTEGER DEFAULT 0")
        if not _column_exists(conn, "episodes", "memory_sources"):
            conn.execute("ALTER TABLE episodes ADD COLUMN memory_sources TEXT")
        if not _column_exists(conn, "episodes", "memory_top_score"):
            conn.execute("ALTER TABLE episodes ADD COLUMN memory_top_score REAL")

    if _table_exists(conn, "failures"):
        if not _column_exists(conn, "failures", "run_id"):
            conn.execute("ALTER TABLE failures ADD COLUMN run_id INTEGER")
        if not _column_exists(conn, "failures", "prompt_hash"):
            conn.execute("ALTER TABLE failures ADD COLUMN prompt_hash TEXT")
        if not _column_exists(conn, "failures", "retrieval_used"):
            conn.execute("ALTER TABLE failures ADD COLUMN retrieval_used INTEGER DEFAULT 0")
        if not _column_exists(conn, "failures", "memory_sources"):
            conn.execute("ALTER TABLE failures ADD COLUMN memory_sources TEXT")
        if not _column_exists(conn, "failures", "memory_top_score"):
            conn.execute("ALTER TABLE failures ADD COLUMN memory_top_score REAL")

    conn.commit()
