from __future__ import annotations

from pathlib import Path
import json
import sys
from datetime import datetime, timezone

sys.path.append(str(Path(__file__).resolve().parents[1]))

from recursive_slm_lab.memory import (
    connect,
    activate_rule,
    rollback_rules,
    activate_procedure,
    rollback_procedures,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def test_rule_activation_and_rollback(tmp_path: Path) -> None:
    db_path = tmp_path / "rules.sqlite"
    conn = connect(db_path)
    now = _now()
    conn.execute(
        """
        INSERT INTO semantic_rules
        (key, rule_text, created_at, origin_episode_ids, evidence_count, eval_snapshot, active, superseded_by, last_verified_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("rule_key", "rule v1", now, json.dumps([1]), 1, json.dumps({}), 1, None, now),
    )
    conn.execute(
        """
        INSERT INTO semantic_rules
        (key, rule_text, created_at, origin_episode_ids, evidence_count, eval_snapshot, active, superseded_by, last_verified_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("rule_key", "rule v2", now, json.dumps([2]), 1, json.dumps({}), 0, None, now),
    )
    conn.commit()
    rule_ids = [row[0] for row in conn.execute("SELECT id FROM semantic_rules ORDER BY id").fetchall()]
    activate_rule(conn, rule_ids[1])
    old_rule = conn.execute(
        "SELECT active, superseded_by FROM semantic_rules WHERE id = ?",
        (rule_ids[0],),
    ).fetchone()
    assert old_rule[0] == 0
    assert old_rule[1] == rule_ids[1]
    rollback_rules(conn, to_rule_id=rule_ids[0])
    restored = conn.execute(
        "SELECT active FROM semantic_rules WHERE id = ?",
        (rule_ids[0],),
    ).fetchone()
    assert restored[0] == 1
    conn.close()


def test_procedure_activation_and_rollback(tmp_path: Path) -> None:
    db_path = tmp_path / "procedures.sqlite"
    conn = connect(db_path)
    now = _now()
    conn.execute(
        """
        INSERT INTO procedures
        (pattern, recipe_text, created_at, origin_episode_ids, evidence_count, eval_snapshot, active, superseded_by, last_verified_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("proc_key", "proc v1", now, json.dumps([1]), 1, json.dumps({}), 1, None, now),
    )
    conn.execute(
        """
        INSERT INTO procedures
        (pattern, recipe_text, created_at, origin_episode_ids, evidence_count, eval_snapshot, active, superseded_by, last_verified_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("proc_key", "proc v2", now, json.dumps([2]), 1, json.dumps({}), 0, None, now),
    )
    conn.commit()
    proc_ids = [row[0] for row in conn.execute("SELECT id FROM procedures ORDER BY id").fetchall()]
    activate_procedure(conn, proc_ids[1])
    old_proc = conn.execute(
        "SELECT active, superseded_by FROM procedures WHERE id = ?",
        (proc_ids[0],),
    ).fetchone()
    assert old_proc[0] == 0
    assert old_proc[1] == proc_ids[1]
    rollback_procedures(conn, to_procedure_id=proc_ids[0])
    restored = conn.execute(
        "SELECT active FROM procedures WHERE id = ?",
        (proc_ids[0],),
    ).fetchone()
    assert restored[0] == 1
    conn.close()
