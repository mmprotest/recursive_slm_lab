from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest

from recursive_slm_lab.cli import cli_seed_tasks
from recursive_slm_lab.memory import init_db, connect, consolidate
from recursive_slm_lab.tasks import load_tasks, split_tasks
from recursive_slm_lab.llm.mock import MockBackend
from recursive_slm_lab.loop import run_iteration
from recursive_slm_lab.eval import evaluate_conditions, plot_results


def test_smoke_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSLM_FAST_VERIFY", "1")
    tasks_path = tmp_path / "bundled_tasks.jsonl"
    cli_seed_tasks(regen=True, count=120, out=tasks_path)
    db_path = tmp_path / "memory.sqlite"
    init_db(db_path)

    tasks = load_tasks(tasks_path)
    train_pool, _ = split_tasks(tasks, heldout_size=20)

    conn = connect(db_path)
    backend = MockBackend(baseline_success_rate=0.2)
    run_iteration(
        conn,
        tasks=train_pool[:20],
        backend=backend,
        k=2,
        max_tokens=64,
        temperature=0.0,
        memory_enabled=True,
        condition="trainpool",
    )
    consolidate(conn, min_evidence=1)
    conn.close()

    output_path = tmp_path / "results.json"
    results = evaluate_conditions(
        db_path=str(db_path),
        backend_name="mock",
        conditions=["baseline", "memory", "semantic", "memory_learning"],
        k=1,
        heldout_size=20,
        task_limit=20,
        output_path=str(output_path),
    )
    _ = evaluate_conditions(
        db_path=str(db_path),
        backend_name="mock",
        conditions=["baseline", "memory", "semantic", "memory_learning"],
        k=1,
        heldout_size=20,
        task_limit=20,
        output_path=None,
    )

    assert output_path.exists()
    payload = json.loads(output_path.read_text())
    assert len(payload["conditions"]) == 4
    assert {item["condition"] for item in payload["conditions"]} == {
        "baseline",
        "memory",
        "semantic",
        "memory_learning",
    }
    scores = {item["condition"]: item["pass_at_1"] for item in results["conditions"]}
    assert all(condition in scores for condition in ["baseline", "memory", "semantic", "memory_learning"])

    pytest.importorskip("matplotlib")
    plot_path = tmp_path / "results.png"
    plot_results(str(db_path), str(plot_path))
    assert plot_path.exists()
