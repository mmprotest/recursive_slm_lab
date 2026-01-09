from __future__ import annotations

import re
from collections import Counter
import random
from typing import Iterable

from ..tasks import Task


def schedule_tasks(
    conn,
    tasks: Iterable[Task],
    limit: int | None,
    seed: int,
    exploration_fraction: float = 0.2,
) -> list[Task]:
    tasks = list(tasks)
    if limit is None or limit <= 0 or limit >= len(tasks):
        return tasks
    rng = random.Random(seed)
    failure_stats, failure_tokens = _failure_signals(conn, window=200)
    scored = []
    for task in tasks:
        score = 0.0
        stats = failure_stats.get(task.task_id)
        if stats:
            score += stats["count"] * 2.0
            score += stats["recency"]
        if failure_tokens:
            score += _jaccard_similarity(_tokenize(task.prompt), failure_tokens)
        scored.append((score, task.task_id, task))
    scored.sort(key=lambda item: (-item[0], item[1]))
    explore_count = max(1, int(limit * exploration_fraction)) if limit > 1 else 0
    exploit_count = max(0, limit - explore_count)
    selected = [task for _, _, task in scored[:exploit_count]]
    remaining = [task for _, _, task in scored[exploit_count:]]
    rng.shuffle(remaining)
    selected.extend(remaining[: explore_count])
    return selected[:limit]


def _failure_signals(conn, window: int = 200) -> tuple[dict[str, dict[str, float]], set[str]]:
    try:
        rows = conn.execute(
            """
            SELECT task_id, prompt
            FROM failures
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (window,),
        ).fetchall()
    except Exception:
        return {}, set()
    if not rows:
        return {}, set()
    counts = Counter(task_id for task_id, _ in rows)
    total = len(rows)
    failure_stats = {}
    for idx, (task_id, _) in enumerate(rows):
        recency = (total - idx) / total
        if task_id not in failure_stats:
            failure_stats[task_id] = {"count": float(counts[task_id]), "recency": recency}
        else:
            failure_stats[task_id]["recency"] = max(failure_stats[task_id]["recency"], recency)
    tokens = set()
    for _, prompt in rows:
        tokens.update(_tokenize(prompt))
    return failure_stats, tokens


def _tokenize(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-zA-Z0-9]+", text.lower()) if token}


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
