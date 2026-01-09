from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from ..eval.metrics import compute_pass_rates
from ..llm.localhf import LocalHFBackend
from ..loop.sampling import generate_candidates
from ..memory import get_active_adapter
from .adapters import activate_adapter
from ..tasks import load_tasks, split_tasks, Task
from ..verify import verify_candidate
from .sft_lora import train_lora_adapter


@dataclass
class PromotionConfig:
    out_dir: str
    heldout_size: int
    heldout_limit: int | None
    regression_size: int
    k: int
    min_improvement: float
    max_regression_drop: float
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    base_model_path: str | None
    regression_seed: int = 1337


@dataclass
class PromotionResult:
    promoted: bool
    decision: str
    candidate_adapter: str | None
    previous_adapter: str | None
    metrics: dict
    message: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_regression_tasks(
    conn: sqlite3.Connection,
    train_pool: list[Task],
    regression_size: int,
    seed: int,
) -> list[Task]:
    rows = conn.execute(
        "SELECT task_id FROM regression_tasks ORDER BY rank ASC"
    ).fetchall()
    task_ids = [row[0] for row in rows]
    if len(task_ids) < regression_size:
        rng = random.Random(seed)
        remaining = [task for task in train_pool if task.task_id not in task_ids]
        rng.shuffle(remaining)
        needed = regression_size - len(task_ids)
        additions = remaining[:needed]
        rank_start = len(task_ids)
        now = _utc_now()
        for idx, task in enumerate(additions, start=rank_start):
            conn.execute(
                "INSERT OR IGNORE INTO regression_tasks (task_id, rank, created_at) VALUES (?, ?, ?)",
                (task.task_id, idx, now),
            )
        conn.commit()
        task_ids += [task.task_id for task in additions]
    task_map = {task.task_id: task for task in train_pool}
    selected = [task_map[task_id] for task_id in task_ids if task_id in task_map]
    selected = selected[:regression_size]
    if len(selected) < regression_size:
        raise ValueError(
            f"Not enough training tasks to fill regression_size={regression_size} "
            f"(only {len(selected)} available)."
        )
    return selected


def _evaluate_pass_rate(
    backend: LocalHFBackend,
    tasks: list[Task],
    k: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> float:
    outcomes: list[list[bool]] = []
    for task in tasks:
        candidates = generate_candidates(
            backend,
            task_prompt=task.prompt,
            function_name=task.function_name,
            signature=task.signature,
            memory_context=None,
            k=k,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        task_outcomes: list[bool] = []
        for candidate in candidates:
            verification = verify_candidate(candidate.code, task.reference_tests, task.assert_tests)
            task_outcomes.append(verification.passed)
        outcomes.append(task_outcomes)
    return compute_pass_rates(outcomes, k=k).pass_at_1


def train_and_maybe_promote(conn: sqlite3.Connection, cfg: PromotionConfig) -> PromotionResult:
    previous_adapter = get_active_adapter(conn)
    previous_adapter_name = previous_adapter[0] if previous_adapter else None

    training_result = train_lora_adapter(conn, cfg.out_dir, cfg.base_model_path)
    if not training_result.trained:
        payload = {
            "decision": "rejected",
            "reason": "training_failed",
            "message": training_result.message,
            "config": cfg.__dict__,
        }
        conn.execute(
            """
            INSERT INTO promotions (created_at, previous_adapter_name, candidate_adapter_name, decision, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (_utc_now(), previous_adapter_name, training_result.adapter_name, "rejected", json.dumps(payload)),
        )
        conn.commit()
        return PromotionResult(
            promoted=False,
            decision="rejected",
            candidate_adapter=None,
            previous_adapter=previous_adapter_name,
            metrics={},
            message=training_result.message,
        )

    tasks = load_tasks()
    train_pool, heldout = split_tasks(tasks, heldout_size=cfg.heldout_size)
    if cfg.heldout_limit is not None:
        heldout = heldout[:cfg.heldout_limit]

    regression_tasks = _load_regression_tasks(conn, train_pool, cfg.regression_size, cfg.regression_seed)

    baseline_backend = LocalHFBackend(cfg.base_model_path)
    candidate_backend = LocalHFBackend(cfg.base_model_path, adapter_path=training_result.adapter_path)

    baseline_heldout = _evaluate_pass_rate(
        baseline_backend, heldout, cfg.k, cfg.max_tokens, cfg.temperature, cfg.top_p, cfg.top_k
    )
    candidate_heldout = _evaluate_pass_rate(
        candidate_backend, heldout, cfg.k, cfg.max_tokens, cfg.temperature, cfg.top_p, cfg.top_k
    )

    previous_heldout = None
    previous_regression = None
    if previous_adapter:
        previous_backend = LocalHFBackend(cfg.base_model_path, adapter_path=previous_adapter[1])
        previous_heldout = _evaluate_pass_rate(
            previous_backend, heldout, cfg.k, cfg.max_tokens, cfg.temperature, cfg.top_p, cfg.top_k
        )
        previous_regression = _evaluate_pass_rate(
            previous_backend,
            regression_tasks,
            cfg.k,
            cfg.max_tokens,
            cfg.temperature,
            cfg.top_p,
            cfg.top_k,
        )
        regression_reference = previous_regression
    else:
        regression_reference = _evaluate_pass_rate(
            baseline_backend,
            regression_tasks,
            cfg.k,
            cfg.max_tokens,
            cfg.temperature,
            cfg.top_p,
            cfg.top_k,
        )

    candidate_regression = _evaluate_pass_rate(
        candidate_backend,
        regression_tasks,
        cfg.k,
        cfg.max_tokens,
        cfg.temperature,
        cfg.top_p,
        cfg.top_k,
    )

    improvement = candidate_heldout - baseline_heldout
    regression_drop = candidate_regression - regression_reference

    promoted = improvement >= cfg.min_improvement and regression_drop >= -cfg.max_regression_drop
    decision = "promoted" if promoted else "rejected"
    if promoted:
        activate_adapter(conn, training_result.adapter_name)

    metrics = {
        "heldout": {
            "baseline": baseline_heldout,
            "candidate": candidate_heldout,
            "previous": previous_heldout,
            "previous_adapter": previous_adapter_name,
        },
        "regression": {
            "reference": regression_reference,
            "candidate": candidate_regression,
            "previous": previous_regression,
        },
        "improvement": improvement,
        "regression_drop": regression_drop,
    }
    payload = {
        "decision": decision,
        "metrics": metrics,
        "config": cfg.__dict__,
        "previous_adapter": previous_adapter_name,
        "candidate_adapter": training_result.adapter_name,
        "candidate_path": training_result.adapter_path,
    }
    conn.execute(
        """
        INSERT INTO promotions (created_at, previous_adapter_name, candidate_adapter_name, decision, payload_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (_utc_now(), previous_adapter_name, training_result.adapter_name, decision, json.dumps(payload)),
    )
    conn.commit()

    message = "Adapter promoted." if promoted else "Adapter rejected."
    return PromotionResult(
        promoted=promoted,
        decision=decision,
        candidate_adapter=training_result.adapter_name,
        previous_adapter=previous_adapter_name,
        metrics=metrics,
        message=message,
    )
