from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from recursive_slm_lab.llm.mock import MockBackend
from recursive_slm_lab.loop import run_iteration, IterationResult
from recursive_slm_lab.memory import connect, init_db
from recursive_slm_lab.tasks import load_tasks


def test_run_iteration_returns_iteration_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSLM_FAST_VERIFY", "1")
    monkeypatch.setenv("RSLM_VERIFY_WORKERS", "1")
    db_path = tmp_path / "memory.sqlite"
    init_db(db_path)

    tasks = load_tasks()[:2]
    conn = connect(db_path)
    backend = MockBackend(baseline_success_rate=0.2)

    results = run_iteration(
        conn,
        tasks=tasks,
        backend=backend,
        k=2,
        max_tokens=64,
        temperature=0.0,
        top_p=0.9,
        top_k=50,
        memory_enabled=False,
        condition="trainpool",
        verify_workers=1,
    )
    conn.close()

    assert isinstance(results, list)
    assert len(results) == len(tasks)
    assert all(isinstance(result, IterationResult) for result in results)
    assert all(result.attempts == 2 for result in results)
