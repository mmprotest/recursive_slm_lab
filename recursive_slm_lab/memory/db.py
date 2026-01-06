from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


@dataclass
class Episode:
    task_id: str
    condition: str
    prompt: str
    candidate_code: str
    passed: bool
    test_log: str
    created_at: str
    code_hash: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(db_path: str | Path) -> None:
    conn = connect(db_path)
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(schema)
    conn.commit()
    conn.close()


def insert_episode(
    conn: sqlite3.Connection,
    task_id: str,
    condition: str,
    prompt: str,
    candidate_code: str,
    passed: bool,
    test_log: str,
) -> None:
    code_hash = hashlib.sha256(candidate_code.encode("utf-8")).hexdigest()
    created_at = _utc_now()
    table = "episodes" if passed else "failures"
    conn.execute(
        f"""
        INSERT INTO {table}
        (task_id, condition, prompt, candidate_code, passed, test_log, created_at, code_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (task_id, condition, prompt, candidate_code, int(passed), test_log, created_at, code_hash),
    )
    conn.commit()


def insert_episode_many(
    conn: sqlite3.Connection,
    rows: list[tuple[str, str, str, str, bool, str]],
) -> None:
    if not rows:
        return
    episode_rows: list[tuple[str, str, str, str, int, str, str, str]] = []
    failure_rows: list[tuple[str, str, str, str, int, str, str, str]] = []
    for task_id, condition, prompt, candidate_code, passed, test_log in rows:
        code_hash = hashlib.sha256(candidate_code.encode("utf-8")).hexdigest()
        created_at = _utc_now()
        values = (task_id, condition, prompt, candidate_code, int(passed), test_log, created_at, code_hash)
        if passed:
            episode_rows.append(values)
        else:
            failure_rows.append(values)

    if episode_rows:
        conn.executemany(
            """
            INSERT INTO episodes
            (task_id, condition, prompt, candidate_code, passed, test_log, created_at, code_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            episode_rows,
        )
    if failure_rows:
        conn.executemany(
            """
            INSERT INTO failures
            (task_id, condition, prompt, candidate_code, passed, test_log, created_at, code_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            failure_rows,
        )
    conn.commit()


def fetch_passed_episodes(conn: sqlite3.Connection) -> list[Episode]:
    rows = conn.execute(
        """
        SELECT task_id, condition, prompt, candidate_code, passed, test_log, created_at, code_hash
        FROM episodes
        WHERE passed = 1
        ORDER BY created_at DESC
        """
    ).fetchall()
    return [
        Episode(
            task_id=row[0],
            condition=row[1],
            prompt=row[2],
            candidate_code=row[3],
            passed=bool(row[4]),
            test_log=row[5],
            created_at=row[6],
            code_hash=row[7],
        )
        for row in rows
    ]


def list_adapters(conn: sqlite3.Connection) -> list[tuple[str, str, int]]:
    return conn.execute(
        "SELECT name, path, active FROM adapters ORDER BY created_at DESC"
    ).fetchall()


def set_active_adapter(conn: sqlite3.Connection, name: str) -> None:
    conn.execute("UPDATE adapters SET active = 0")
    conn.execute("UPDATE adapters SET active = 1 WHERE name = ?", (name,))
    conn.commit()


def register_adapter(conn: sqlite3.Connection, name: str, path: str, notes: str | None = None) -> None:
    created_at = _utc_now()
    conn.execute(
        """
        INSERT INTO adapters (name, path, created_at, notes, active)
        VALUES (?, ?, ?, ?, 0)
        """,
        (name, path, created_at, notes),
    )
    conn.commit()


def rollback_adapter(conn: sqlite3.Connection, name: str) -> None:
    conn.execute("UPDATE adapters SET active = 0 WHERE name = ?", (name,))
    conn.commit()


def get_active_adapter(conn: sqlite3.Connection) -> tuple[str, str] | None:
    row = conn.execute("SELECT name, path FROM adapters WHERE active = 1").fetchone()
    if not row:
        return None
    return row[0], row[1]
