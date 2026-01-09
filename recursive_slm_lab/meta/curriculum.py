from __future__ import annotations

import hashlib
import random
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..tasks import Task, load_tasks, validate_tasks
from ..tasks.generator import generate_constant_tasks, generate_tasks, write_tasks


_ERROR_PATTERNS = {
    "assertion": re.compile(r"AssertionError", re.IGNORECASE),
    "type_error": re.compile(r"TypeError", re.IGNORECASE),
    "name_error": re.compile(r"NameError", re.IGNORECASE),
    "index_error": re.compile(r"IndexError", re.IGNORECASE),
    "key_error": re.compile(r"KeyError", re.IGNORECASE),
    "value_error": re.compile(r"ValueError", re.IGNORECASE),
    "syntax_error": re.compile(r"SyntaxError", re.IGNORECASE),
    "timeout": re.compile(r"Timeout", re.IGNORECASE),
}


@dataclass(frozen=True)
class FailureCluster:
    error_type: str
    assertion_pattern: str | None
    function_name: str
    category: str
    difficulty: int


def _extract_assertion_pattern(log: str) -> str | None:
    if not log:
        return None
    match = re.search(r"assert\s+[^\n]+", log)
    if match:
        return match.group(0)[:120]
    return None


def _error_type(log: str) -> str:
    for name, pattern in _ERROR_PATTERNS.items():
        if pattern.search(log):
            return name
    return "unknown"


def cluster_failures(conn: sqlite3.Connection, task_map: dict[str, Task]) -> dict[FailureCluster, list[str]]:
    rows = conn.execute(
        """
        SELECT task_id, test_log
        FROM failures
        ORDER BY created_at DESC
        """
    ).fetchall()
    clusters: dict[FailureCluster, list[str]] = {}
    for task_id, log in rows:
        task = task_map.get(task_id)
        if not task:
            continue
        cluster = FailureCluster(
            error_type=_error_type(log),
            assertion_pattern=_extract_assertion_pattern(log),
            function_name=task.function_name,
            category=task.category,
            difficulty=task.difficulty,
        )
        clusters.setdefault(cluster, []).append(task_id)
    return clusters


def _normalize_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.strip().lower())


def _task_key(task: Task) -> str:
    payload = f"{_normalize_prompt(task.prompt)}|{task.signature}|{task.reference_tests}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dedupe_tasks(candidates: list[Task], existing: dict[str, Task]) -> list[Task]:
    seen = {_task_key(task) for task in existing.values()}
    unique: list[Task] = []
    for task in candidates:
        key = _task_key(task)
        if key in seen:
            continue
        seen.add(key)
        unique.append(task)
    return unique


def mine_curriculum(
    conn: sqlite3.Connection,
    out_path: Path,
    max_new: int,
    seed: int = 1337,
) -> dict:
    base_tasks = load_tasks(include_generated=Config().include_generated_tasks)
    task_map = {task.task_id: task for task in base_tasks}
    clusters = cluster_failures(conn, task_map)
    rng = random.Random(seed)

    generated: list[Task] = []
    pool = [Task(**payload) for payload in generate_tasks(250)]
    by_category: dict[str, list[Task]] = {}
    for task in pool:
        by_category.setdefault(task.category, []).append(task)

    for cluster, task_ids in clusters.items():
        desired_difficulty = min(cluster.difficulty + 1, 3)
        category_pool = [task for task in by_category.get(cluster.category, []) if task.difficulty >= desired_difficulty]
        rng.shuffle(category_pool)
        take = max(1, min(5, max_new - len(generated)))
        generated.extend(category_pool[:take])
        if len(generated) >= max_new:
            break

    add_values = list(range(121, 141))
    mul_values = list(range(41, 61))
    rng.shuffle(add_values)
    rng.shuffle(mul_values)
    extras = generate_constant_tasks(add_values[:10], mul_values[:10], difficulty=2, add_range=300, mul_range=200)
    generated.extend(Task(**payload) for payload in extras)

    unique_tasks = _dedupe_tasks(generated, task_map)[:max_new]
    if not unique_tasks:
        return {"generated": 0, "written": 0, "out_path": str(out_path)}
    validate_tasks(unique_tasks)
    write_tasks([task.__dict__ for task in unique_tasks], out_path)
    return {
        "generated": len(generated),
        "written": len(unique_tasks),
        "out_path": str(out_path),
        "clusters": len(clusters),
    }
