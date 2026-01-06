from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from recursive_slm_lab.memory import init_db, connect, consolidate
from recursive_slm_lab.tasks import load_tasks, split_tasks
from recursive_slm_lab.llm.mock import MockBackend
from recursive_slm_lab.loop import run_iteration
from recursive_slm_lab.eval import evaluate_conditions


def test_smoke_pipeline(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.sqlite"
    init_db(db_path)

    tasks = load_tasks()
    train_pool, heldout = split_tasks(tasks, heldout_size=10)

    conn = connect(db_path)
    backend = MockBackend()
    run_iteration(
        conn,
        tasks=train_pool[:5],
        backend=backend,
        k=3,
        max_tokens=128,
        temperature=0.1,
        memory_enabled=True,
        condition="trainpool",
    )
    consolidate(conn, min_evidence=1)
    conn.close()

    output_path = tmp_path / "results.json"
    results = evaluate_conditions(
        db_path=str(db_path),
        backend_name="mock",
        conditions=["baseline", "memory", "learning", "memory_learning"],
        k=1,
        heldout_size=10,
        output_path=str(output_path),
    )

    assert output_path.exists()
    payload = json.loads(output_path.read_text())
    assert len(payload["conditions"]) == 4
