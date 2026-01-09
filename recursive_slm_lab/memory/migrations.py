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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS policies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            name TEXT NOT NULL UNIQUE,
            parent_policy_name TEXT,
            policy_json TEXT NOT NULL,
            notes TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS active_policy (
            singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
            policy_name TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS policy_promotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            previous_policy_name TEXT,
            candidate_policy_name TEXT,
            decision TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            notes TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS verification_cache (
            key TEXT PRIMARY KEY,
            passed INTEGER NOT NULL,
            log TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    _ensure_rules_schema(conn)
    _ensure_procedures_schema(conn)

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


def _ensure_rules_schema(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "semantic_rules"):
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                rule_text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                origin_episode_ids TEXT NOT NULL,
                evidence_count INTEGER NOT NULL,
                eval_snapshot TEXT,
                active INTEGER NOT NULL,
                superseded_by INTEGER,
                last_verified_at TEXT NOT NULL
            )
            """
        )
        _ensure_rules_fts(conn)
        return
    if _column_exists(conn, "semantic_rules", "created_at"):
        _ensure_rules_fts(conn)
        return
    conn.execute("ALTER TABLE semantic_rules RENAME TO semantic_rules_old")
    conn.execute(
        """
        CREATE TABLE semantic_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL,
            rule_text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            origin_episode_ids TEXT NOT NULL,
            evidence_count INTEGER NOT NULL,
            eval_snapshot TEXT,
            active INTEGER NOT NULL,
            superseded_by INTEGER,
            last_verified_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO semantic_rules (key, rule_text, created_at, origin_episode_ids, evidence_count, eval_snapshot, active, superseded_by, last_verified_at)
        SELECT key, rule_text, last_verified_at, '[]', evidence_count, NULL, 1, NULL, last_verified_at
        FROM semantic_rules_old
        """
    )
    conn.execute("DROP TABLE semantic_rules_old")
    _ensure_rules_fts(conn)


def _ensure_procedures_schema(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "procedures"):
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS procedures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT NOT NULL,
                recipe_text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                origin_episode_ids TEXT NOT NULL,
                evidence_count INTEGER NOT NULL,
                eval_snapshot TEXT,
                active INTEGER NOT NULL,
                superseded_by INTEGER,
                last_verified_at TEXT NOT NULL
            )
            """
        )
        _ensure_procedures_fts(conn)
        return
    if _column_exists(conn, "procedures", "created_at"):
        _ensure_procedures_fts(conn)
        return
    conn.execute("ALTER TABLE procedures RENAME TO procedures_old")
    conn.execute(
        """
        CREATE TABLE procedures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern TEXT NOT NULL,
            recipe_text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            origin_episode_ids TEXT NOT NULL,
            evidence_count INTEGER NOT NULL,
            eval_snapshot TEXT,
            active INTEGER NOT NULL,
            superseded_by INTEGER,
            last_verified_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO procedures (pattern, recipe_text, created_at, origin_episode_ids, evidence_count, eval_snapshot, active, superseded_by, last_verified_at)
        SELECT pattern, recipe_text, last_verified_at, '[]', evidence_count, NULL, 1, NULL, last_verified_at
        FROM procedures_old
        """
    )
    conn.execute("DROP TABLE procedures_old")
    _ensure_procedures_fts(conn)


def _ensure_rules_fts(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS rules_fts")
    conn.execute("DROP TRIGGER IF EXISTS rules_ai")
    conn.execute("DROP TRIGGER IF EXISTS rules_ad")
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS rules_fts USING fts5(
            rule_text, content='semantic_rules', content_rowid='id'
        )
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS rules_ai AFTER INSERT ON semantic_rules BEGIN
            INSERT INTO rules_fts(rowid, rule_text) VALUES (new.id, new.rule_text);
        END;
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS rules_ad AFTER DELETE ON semantic_rules BEGIN
            INSERT INTO rules_fts(rules_fts, rowid, rule_text) VALUES ('delete', old.id, old.rule_text);
        END;
        """
    )


def _ensure_procedures_fts(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS procedures_fts")
    conn.execute("DROP TRIGGER IF EXISTS procedures_ai")
    conn.execute("DROP TRIGGER IF EXISTS procedures_ad")
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS procedures_fts USING fts5(
            recipe_text, content='procedures', content_rowid='id'
        )
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS procedures_ai AFTER INSERT ON procedures BEGIN
            INSERT INTO procedures_fts(rowid, recipe_text) VALUES (new.id, new.recipe_text);
        END;
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS procedures_ad AFTER DELETE ON procedures BEGIN
            INSERT INTO procedures_fts(procedures_fts, rowid, recipe_text) VALUES ('delete', old.id, old.recipe_text);
        END;
        """
    )
