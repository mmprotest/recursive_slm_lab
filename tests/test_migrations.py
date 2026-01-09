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
    policy_table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='policies'"
    ).fetchone()
    assert policy_table is not None
    cache_table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='verification_cache'"
    ).fetchone()
    assert cache_table is not None
    conn.close()


def test_migrations_upgrade_rules_and_procedures(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy_rules.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE semantic_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE NOT NULL,
            rule_text TEXT NOT NULL,
            evidence_count INTEGER NOT NULL,
            last_verified_at TEXT NOT NULL
        );
        CREATE TABLE procedures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern TEXT UNIQUE NOT NULL,
            recipe_text TEXT NOT NULL,
            evidence_count INTEGER NOT NULL,
            last_verified_at TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()

    conn = connect(db_path)
    rule_columns = {row[1] for row in conn.execute("PRAGMA table_info(semantic_rules)").fetchall()}
    proc_columns = {row[1] for row in conn.execute("PRAGMA table_info(procedures)").fetchall()}
    for column in [
        "created_at",
        "origin_episode_ids",
        "eval_snapshot",
        "active",
        "superseded_by",
    ]:
        assert column in rule_columns
        assert column in proc_columns
    conn.close()
