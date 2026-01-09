from __future__ import annotations

import sqlite3

from recursive_slm_lab.memory import connect


def _table_names(db_path) -> set[str]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    conn.close()
    return {row[0] for row in rows}


def test_connect_migrate_flag(tmp_path) -> None:
    db_path = tmp_path / "memory.sqlite"
    conn = connect(db_path, migrate=False)
    conn.close()
    tables = _table_names(db_path)
    assert "runs" not in tables
    assert "schema_meta" not in tables

    conn = connect(db_path, migrate=True)
    conn.close()
    tables = _table_names(db_path)
    assert "runs" in tables
    assert "schema_meta" in tables
