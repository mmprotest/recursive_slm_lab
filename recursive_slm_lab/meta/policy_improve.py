from __future__ import annotations

import json
import re
import sqlite3
import random
from dataclasses import dataclass
from datetime import datetime, timezone

from ..eval.gating import EvalConfig, evaluate_policy_pass_rate
from ..policy import Policy, DEFAULT_POLICY
from ..memory import register_policy, set_active_policy
from ..tasks import load_tasks, split_tasks, Task


@dataclass
class PolicyProposal:
    policy: Policy
    raw_json: str
    attempts: int


@dataclass
class PolicyPromotionResult:
    decision: str
    candidate_name: str | None
    metrics: dict
    message: str


def propose_policy(
    backend,
    current_policy: Policy,
    recent_failures_summary: str,
    constraints: dict,
) -> PolicyProposal:
    prompt = (
        "You are proposing a new policy JSON for recursive self-improvement. "
        "Return STRICT JSON matching the Policy schema exactly. "
        "Do not include commentary.\n\n"
        f"Current policy JSON:\n{current_policy.to_json()}\n\n"
        f"Failure summary:\n{recent_failures_summary}\n\n"
        f"Constraints:\n{json.dumps(constraints, indent=2)}\n"
    )
    attempts = 0
    last_error: str | None = None
    while attempts < 3:
        attempts += 1
        response = backend.generate(
            [
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
        )
        raw = _extract_json_text(response.text)
        try:
            payload = json.loads(raw)
            merged = _merge_policy_dict(current_policy.to_dict(), payload)
            validated = _validate_policy_dict(merged)
            policy = Policy.from_dict(validated)
            return PolicyProposal(policy=policy, raw_json=json.dumps(validated), attempts=attempts)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            last_error = str(exc)
            continue
    raise ValueError(f"Policy proposal failed after {attempts} attempts: {last_error}")


def evaluate_and_maybe_promote_policy(
    conn: sqlite3.Connection,
    backend,
    candidate_policy: Policy,
    heldout_size: int,
    heldout_limit: int | None,
    regression_size: int,
    repeats: int,
    seed: int,
    regression_seed: int,
    deterministic: bool,
    min_delta: float,
    max_drop: float,
    notes: str | None = None,
) -> PolicyPromotionResult:
    tasks = load_tasks()
    train_pool, heldout = split_tasks(tasks, heldout_size=heldout_size)
    if heldout_limit is not None:
        heldout = heldout[:heldout_limit]
    regression_tasks = _load_regression_tasks(conn, train_pool, regression_size, regression_seed)
    current_policy = _active_policy(conn)
    previous_policy_name = _active_policy_name(conn)

    eval_config = EvalConfig(repeats=repeats, deterministic=deterministic, seed=seed)
    baseline_metrics = evaluate_policy_pass_rate(
        backend,
        heldout,
        current_policy,
        repeats=eval_config.repeats,
        seed=eval_config.seed,
        deterministic=eval_config.deterministic,
    )
    candidate_metrics = evaluate_policy_pass_rate(
        backend,
        heldout,
        candidate_policy,
        repeats=eval_config.repeats,
        seed=eval_config.seed,
        deterministic=eval_config.deterministic,
    )
    regression_baseline = evaluate_policy_pass_rate(
        backend,
        regression_tasks,
        current_policy,
        repeats=eval_config.repeats,
        seed=eval_config.seed,
        deterministic=eval_config.deterministic,
    )
    regression_candidate = evaluate_policy_pass_rate(
        backend,
        regression_tasks,
        candidate_policy,
        repeats=eval_config.repeats,
        seed=eval_config.seed,
        deterministic=eval_config.deterministic,
    )

    improvement = candidate_metrics["mean"] - baseline_metrics["mean"]
    regression_drop = regression_candidate["mean"] - regression_baseline["mean"]
    promoted = improvement >= min_delta and regression_drop >= -max_drop
    decision = "accept" if promoted else "reject"

    candidate_name = None
    if promoted:
        candidate_name = _next_policy_name(conn)
        register_policy(conn, candidate_name, candidate_policy, parent_policy_name=previous_policy_name)
        set_active_policy(conn, candidate_name)

    metrics = {
        "heldout": {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "improvement": improvement,
            "min_delta": min_delta,
        },
        "regression": {
            "baseline": regression_baseline,
            "candidate": regression_candidate,
            "drop": regression_drop,
            "max_drop": max_drop,
        },
    }
    _record_policy_promotion(
        conn,
        previous_policy_name=previous_policy_name,
        candidate_policy_name=candidate_name,
        decision=decision,
        metrics=metrics,
        notes=notes,
    )
    message = "Policy promoted." if promoted else "Policy rejected."
    return PolicyPromotionResult(decision=decision, candidate_name=candidate_name, metrics=metrics, message=message)


def _validate_policy_dict(payload: dict) -> dict:
    payload = dict(payload)
    payload["retrieval_top_n"] = int(_clamp(payload.get("retrieval_top_n", 0), 0, 10))
    payload["retrieval_extra_terms_mode"] = payload.get("retrieval_extra_terms_mode", "none")
    for key in ("sampling_train", "sampling_eval"):
        block = payload.get(key, {})
        block["k"] = int(_clamp(block.get("k", 1), 1, 16))
        block["temperature"] = float(_clamp(block.get("temperature", 0.0), 0.0, 1.5))
        block["top_p"] = float(_clamp(block.get("top_p", 1.0), 0.1, 1.0))
        block["top_k"] = int(_clamp(block.get("top_k", 0), 0, 200))
        if key == "sampling_eval":
            block["k"] = max(1, min(block["k"], 16))
        payload[key] = block
    return payload


def _merge_policy_dict(base: dict, proposed: dict) -> dict:
    merged = dict(base)
    for key, value in proposed.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _extract_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


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
        remaining = [task for task in train_pool if task.task_id not in task_ids]
        rng = random.Random(seed)
        rng.shuffle(remaining)
        needed = regression_size - len(task_ids)
        additions = remaining[:needed]
        now = _utc_now()
        for idx, task in enumerate(additions, start=len(task_ids)):
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _active_policy(conn: sqlite3.Connection) -> Policy:
    row = conn.execute("SELECT policy_name FROM active_policy WHERE singleton = 1").fetchone()
    if not row:
        return DEFAULT_POLICY
    policy_row = conn.execute(
        "SELECT policy_json FROM policies WHERE name = ?",
        (row[0],),
    ).fetchone()
    if not policy_row:
        return DEFAULT_POLICY
    return Policy.from_json(policy_row[0])


def _active_policy_name(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT policy_name FROM active_policy WHERE singleton = 1").fetchone()
    return row[0] if row else None


def _next_policy_name(conn: sqlite3.Connection) -> str:
    rows = conn.execute("SELECT name FROM policies WHERE name LIKE 'pol%'").fetchall()
    max_id = 0
    for (name,) in rows:
        match = re.match(r"pol(\d+)", name)
        if match:
            max_id = max(max_id, int(match.group(1)))
    return f"pol{max_id + 1:03d}"


def _record_policy_promotion(
    conn: sqlite3.Connection,
    previous_policy_name: str | None,
    candidate_policy_name: str | None,
    decision: str,
    metrics: dict,
    notes: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO policy_promotions
        (created_at, previous_policy_name, candidate_policy_name, decision, metrics_json, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (_utc_now(), previous_policy_name, candidate_policy_name, decision, json.dumps(metrics), notes),
    )
    conn.commit()
