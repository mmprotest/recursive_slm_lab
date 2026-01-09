from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt: str
    function_name: str
    signature: str
    reference_tests: str
    assert_tests: list[str] | None = None
    tags: list[str] | None = None


BUNDLED_PATH = Path(__file__).parent / "bundled_tasks.jsonl"


def load_tasks(path: Path | str = BUNDLED_PATH) -> list[Task]:
    path = Path(path)
    tasks: list[Task] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            tasks.append(Task(**payload))
    return tasks


def validate_tasks(tasks: Iterable[Task]) -> None:
    for task in tasks:
        if not task.task_id:
            raise ValueError("task_id missing")
        if not task.prompt:
            raise ValueError(f"prompt missing for {task.task_id}")
        if not task.function_name:
            raise ValueError(f"function_name missing for {task.task_id}")
        if not task.signature.startswith("("):
            raise ValueError(f"signature invalid for {task.task_id}")
        if "def" in task.reference_tests and "pytest" not in task.reference_tests:
            raise ValueError(f"reference_tests missing pytest usage for {task.task_id}")
        if not task.assert_tests:
            raise ValueError(f"assert_tests missing for {task.task_id}")


def split_tasks(tasks: list[Task], heldout_size: int, seed: int = 1337) -> tuple[list[Task], list[Task]]:
    heldout_only = [task for task in tasks if task.tags and "heldout_only" in task.tags]
    remaining = [task for task in tasks if task not in heldout_only]
    hashed = []
    for task in remaining:
        digest = hashlib.sha256(f"{seed}:{task.task_id}".encode("utf-8")).hexdigest()
        hashed.append((digest, task))
    hashed.sort(key=lambda item: item[0])
    needed = max(heldout_size - len(heldout_only), 0)
    heldout_extra = [task for _, task in hashed[:needed]]
    heldout = heldout_only + heldout_extra
    train_pool = [task for _, task in hashed[needed:]]
    if len(heldout) < heldout_size:
        raise ValueError(
            f"Not enough tasks to fill heldout_size={heldout_size} "
            f"(only {len(heldout)} available)."
        )
    return train_pool, heldout
