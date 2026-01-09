from __future__ import annotations

from pathlib import Path
import hashlib
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from recursive_slm_lab.meta.curriculum import mine_curriculum
from recursive_slm_lab.memory import connect, init_db
from recursive_slm_lab.tasks import load_tasks, validate_tasks


def _hash_task(task) -> str:
    payload = f"{task.prompt}\n{task.signature}\n{task.reference_tests}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_mine_curriculum_generates_tasks(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.sqlite"
    init_db(db_path)
    conn = connect(db_path)
    task = load_tasks()[0]
    code_hash = hashlib.sha256("print('fail')".encode("utf-8")).hexdigest()
    conn.execute(
        """
        INSERT INTO failures
        (task_id, condition, prompt, candidate_code, passed, test_log, created_at, code_hash)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'), ?)
        """,
        (task.task_id, "train", task.prompt, "print('fail')", 0, "AssertionError", code_hash),
    )
    conn.commit()
    out_path = tmp_path / "generated.jsonl"
    report = mine_curriculum(conn, out_path, max_new=5, seed=1234)
    conn.close()
    assert report["written"] > 0
    generated = load_tasks(out_path)
    validate_tasks(generated)
    hashes = {_hash_task(task) for task in generated}
    assert len(hashes) == len(generated)
