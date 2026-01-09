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
    category: str
    difficulty: int
    assert_tests: list[str] | None = None
    tags: list[str] | None = None


BUNDLED_PATH = Path(__file__).parent / "bundled_tasks.jsonl"
HIDDEN_PATH = Path(__file__).parent / "hidden_tasks.jsonl"
GENERATED_PATH = Path("artifacts/generated_tasks.jsonl")


def load_tasks(
    path: Path | str = BUNDLED_PATH,
    include_generated: bool = False,
    generated_path: Path | str = GENERATED_PATH,
) -> list[Task]:
    path = Path(path)
    tasks: list[Task] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            payload.setdefault("category", "uncategorized")
            payload.setdefault("difficulty", 1)
            tasks.append(Task(**payload))
    if include_generated:
        gen_path = Path(generated_path)
        if gen_path.exists():
            tasks.extend(load_tasks(gen_path))
    return tasks


def load_hidden_tasks(path: Path | str = HIDDEN_PATH) -> list[Task]:
    path = Path(path)
    if not path.exists():
        return []
    return load_tasks(path)


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
        if not task.category:
            raise ValueError(f"category missing for {task.task_id}")
        if task.difficulty < 1:
            raise ValueError(f"difficulty invalid for {task.task_id}")


def split_tasks(
    tasks: list[Task],
    heldout_size: int,
    seed: int = 1337,
    heldout_categories: set[str] | None = None,
    hidden_tasks: list[Task] | None = None,
) -> tuple[list[Task], list[Task], list[Task]]:
    heldout_categories = heldout_categories or {"parsing"}
    heldout_only = [
        task
        for task in tasks
        if (task.tags and "heldout_only" in task.tags) or task.category in heldout_categories
    ]
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
    hidden = hidden_tasks or []
    return train_pool, heldout, hidden
