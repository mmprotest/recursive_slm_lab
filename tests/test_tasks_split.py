from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from recursive_slm_lab.tasks import Task, split_tasks


def test_split_tasks_honors_heldout_only() -> None:
    tasks = [
        Task("t1", "p1", "f1", "()", "tests", ["assert True"], tags=["heldout_only"]),
        Task("t2", "p2", "f2", "()", "tests", ["assert True"]),
        Task("t3", "p3", "f3", "()", "tests", ["assert True"], tags=["heldout_only"]),
        Task("t4", "p4", "f4", "()", "tests", ["assert True"]),
        Task("t5", "p5", "f5", "()", "tests", ["assert True"]),
    ]

    train_pool, heldout = split_tasks(tasks, heldout_size=3, seed=1)
    heldout_ids = {task.task_id for task in heldout}
    assert {"t1", "t3"}.issubset(heldout_ids)
    assert len(heldout) == 3
    assert len(train_pool) == 2
