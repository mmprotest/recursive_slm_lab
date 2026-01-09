from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..policy import DEFAULT_POLICY, Policy
from .migrations import ensure_schema

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


@dataclass
class RunMeta:
    mode: str
    backend: str
    model: str
    adapter_name: str | None
    memory_enabled: bool
    semantic_enabled: bool
    learning_enabled: bool
    k: int
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    notes: str | None = None
    config_json: dict | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    ensure_schema(conn)
    return conn


def init_db(db_path: str | Path) -> None:
    conn = connect(db_path)
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(schema)
    conn.commit()
    _ensure_default_policy(conn)
    conn.close()


def start_run(conn: sqlite3.Connection, meta: RunMeta) -> int:
    created_at = _utc_now()
    config_json = json.dumps(meta.config_json or {})
    cursor = conn.execute(
        """
        INSERT INTO runs
        (
            created_at, mode, backend, model, adapter_name, memory_enabled, semantic_enabled,
            learning_enabled, k, max_tokens, temperature, top_p, top_k, notes, config_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            created_at,
            meta.mode,
            meta.backend,
            meta.model,
            meta.adapter_name,
            int(meta.memory_enabled),
            int(meta.semantic_enabled),
            int(meta.learning_enabled),
            meta.k,
            meta.max_tokens,
            meta.temperature,
            meta.top_p,
            meta.top_k,
            meta.notes,
            config_json,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def insert_episode(
    conn: sqlite3.Connection,
    task_id: str,
    condition: str,
    prompt: str,
    candidate_code: str,
    passed: bool,
    test_log: str,
    run_id: int | None = None,
    prompt_hash: str | None = None,
    retrieval_used: bool = False,
    memory_sources: str | None = None,
    memory_top_score: float | None = None,
) -> None:
    code_hash = hashlib.sha256(candidate_code.encode("utf-8")).hexdigest()
    created_at = _utc_now()
    table = "episodes" if passed else "failures"
    conn.execute(
        f"""
        INSERT INTO {table}
        (
            task_id, condition, prompt, candidate_code, passed, test_log, created_at, code_hash,
            run_id, prompt_hash, retrieval_used, memory_sources, memory_top_score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            task_id,
            condition,
            prompt,
            candidate_code,
            int(passed),
            test_log,
            created_at,
            code_hash,
            run_id,
            prompt_hash,
            int(retrieval_used),
            memory_sources,
            memory_top_score,
        ),
    )
    conn.commit()


def insert_episode_many(
    conn: sqlite3.Connection,
    rows: list[tuple],
) -> None:
    if not rows:
        return
    episode_rows: list[tuple] = []
    failure_rows: list[tuple] = []
    for row in rows:
        if len(row) == 6:
            (
                task_id,
                condition,
                prompt,
                candidate_code,
                passed,
                test_log,
            ) = row
            run_id = None
            prompt_hash = None
            retrieval_used = 0
            memory_sources = None
            memory_top_score = None
        else:
            (
                task_id,
                condition,
                prompt,
                candidate_code,
                passed,
                test_log,
                run_id,
                prompt_hash,
                retrieval_used,
                memory_sources,
                memory_top_score,
            ) = row
        code_hash = hashlib.sha256(candidate_code.encode("utf-8")).hexdigest()
        created_at = _utc_now()
        values = (
            task_id,
            condition,
            prompt,
            candidate_code,
            int(passed),
            test_log,
            created_at,
            code_hash,
            run_id,
            prompt_hash,
            int(retrieval_used),
            memory_sources,
            memory_top_score,
        )
        if passed:
            episode_rows.append(values)
        else:
            failure_rows.append(values)

    if episode_rows:
        conn.executemany(
            """
            INSERT INTO episodes
            (
                task_id, condition, prompt, candidate_code, passed, test_log, created_at, code_hash,
                run_id, prompt_hash, retrieval_used, memory_sources, memory_top_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            episode_rows,
        )
    if failure_rows:
        conn.executemany(
            """
            INSERT INTO failures
            (
                task_id, condition, prompt, candidate_code, passed, test_log, created_at, code_hash,
                run_id, prompt_hash, retrieval_used, memory_sources, memory_top_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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


def mark_task_seen(conn: sqlite3.Connection, task_id: str) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO train_progress (task_id, first_seen_at)
        VALUES (?, ?)
        """,
        (task_id, _utc_now()),
    )
    conn.commit()


def fetch_seen_task_ids(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT task_id FROM train_progress").fetchall()
    return {row[0] for row in rows}


def wipe_memory(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM episodes")
    conn.execute("DELETE FROM failures")
    conn.execute("DELETE FROM semantic_rules")
    conn.execute("DELETE FROM procedures")
    conn.commit()


def register_policy(
    conn: sqlite3.Connection,
    name: str,
    policy: Policy,
    parent_policy_name: str | None = None,
    notes: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO policies (created_at, name, parent_policy_name, policy_json, notes)
        VALUES (?, ?, ?, ?, ?)
        """,
        (_utc_now(), name, parent_policy_name, policy.to_json(), notes),
    )
    conn.commit()


def list_policies(conn: sqlite3.Connection) -> list[tuple[str, str, str | None]]:
    return conn.execute(
        "SELECT name, created_at, parent_policy_name FROM policies ORDER BY created_at DESC"
    ).fetchall()


def get_policy(conn: sqlite3.Connection, name: str) -> Policy:
    row = conn.execute(
        "SELECT policy_json FROM policies WHERE name = ?",
        (name,),
    ).fetchone()
    if not row:
        raise ValueError(f"Policy '{name}' not found")
    return Policy.from_json(row[0])


def get_active_policy(conn: sqlite3.Connection) -> Policy:
    row = conn.execute("SELECT policy_name FROM active_policy WHERE singleton = 1").fetchone()
    if not row:
        return DEFAULT_POLICY
    try:
        return get_policy(conn, row[0])
    except ValueError:
        return DEFAULT_POLICY


def set_active_policy(conn: sqlite3.Connection, name: str) -> None:
    conn.execute(
        """
        INSERT INTO active_policy (singleton, policy_name, updated_at)
        VALUES (1, ?, ?)
        ON CONFLICT(singleton) DO UPDATE SET
            policy_name = excluded.policy_name,
            updated_at = excluded.updated_at
        """,
        (name, _utc_now()),
    )
    conn.commit()


def _ensure_default_policy(conn: sqlite3.Connection) -> None:
    row = conn.execute("SELECT name FROM policies WHERE name = ?", ("default",)).fetchone()
    if not row:
        register_policy(conn, "default", DEFAULT_POLICY, parent_policy_name=None, notes="initial")
    row = conn.execute("SELECT policy_name FROM active_policy WHERE singleton = 1").fetchone()
    if not row:
        set_active_policy(conn, "default")
