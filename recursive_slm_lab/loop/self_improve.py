from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional
import random

from ..config import Config
from ..eval.robust import robust_eval_conditions
from ..llm import LocalHFBackend, MockBackend, OpenAICompatBackend
from ..memory import (
    connect,
    consolidate,
    fetch_seen_task_ids,
    get_active_adapter,
    get_active_policy,
)
from ..meta import summarize_failures, propose_policy, evaluate_and_maybe_promote_policy
from ..tasks import load_tasks, load_hidden_tasks, split_tasks
from ..training import train_and_maybe_promote, PromotionConfig
from ..util import ensure_dir
from .iteration import run_iteration


def run_self_improve(
    *,
    db_path: str,
    tasks_source: str,
    cycles: int,
    train_k: int,
    train_limit: int,
    heldout_size: int,
    backend: str,
    base_url: str | None,
    model: str | None,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    memory_enabled: bool,
    unseen_only: bool,
    train_seed: int,
    enable_policy_improve: bool,
    enable_adapter_train: bool,
    artifacts_dir: Path,
    verify_mode: str | None,
    seed: int,
    enable_self_patch: bool,
    self_patch_callback: Callable[[dict], None] | None = None,
) -> dict:
    config = Config(
        db_path=db_path,
        backend=backend,
        base_url=base_url or Config().base_url,
        model=model or Config().model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    if enable_adapter_train and config.backend != "localhf":
        raise ValueError("Adapter training requires backend 'localhf'")
    ensure_dir(artifacts_dir)

    with _override_env("RSLM_VERIFY_MODE", verify_mode or "docker"):
        conn = connect(db_path)
        tasks = _load_tasks(tasks_source, config)
        hidden_tasks = load_hidden_tasks()
        train_pool, _, _ = split_tasks(tasks, heldout_size=heldout_size, hidden_tasks=hidden_tasks)

        cycle_summaries: list[dict] = []
        best_heldout = -1.0
        best_condition = None

        for cycle in range(1, cycles + 1):
            cycle_start = _utc_now()
            backend_impl = _resolve_backend(config, conn)
            policy_before = _active_policy_name(conn)
            adapter_before = _active_adapter_name(conn)
            selected = _select_unseen_tasks(conn, train_pool, train_limit, unseen_only, train_seed + cycle)
            policy = get_active_policy(conn)
            iteration_results = run_iteration(
                conn,
                tasks=selected,
                backend=backend_impl,
                k=train_k,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                memory_enabled=memory_enabled,
                condition="trainpool",
                policy=policy,
                db_path=db_path,
            )
            passed_any = sum(1 for result in iteration_results if result.passed)
            consolidate(conn, min_evidence=3)

            policy_message = "Policy improve disabled."
            if enable_policy_improve:
                failure_summary = summarize_failures(conn, limit=50)
                constraints = {
                    "retrieval_top_n": "[0, 10]",
                    "temperature": "[0, 1.5]",
                    "k": "[1, 16]",
                    "top_p": "[0.1, 1.0]",
                    "top_k": "[0, 200]",
                }
                proposal = propose_policy(
                    backend_impl,
                    current_policy=policy,
                    recent_failures_summary=failure_summary,
                    constraints=constraints,
                )
                policy_result = evaluate_and_maybe_promote_policy(
                    conn,
                    backend_impl,
                    candidate_policy=proposal.policy,
                    heldout_size=heldout_size,
                    heldout_limit=None,
                    regression_size=25,
                    repeats=3,
                    seed=seed,
                    regression_seed=seed,
                    deterministic=True,
                    min_delta=0.01,
                    max_drop=0.0,
                    notes=failure_summary,
                )
                policy_message = policy_result.message

            adapter_message = "Adapter training disabled."
            if enable_adapter_train:
                adapter_out = artifacts_dir / "adapters" / f"cycle_{cycle:03d}"
                adapter_out.mkdir(parents=True, exist_ok=True)
                promotion_config = PromotionConfig(
                    out_dir=str(adapter_out),
                    heldout_size=heldout_size,
                    heldout_limit=None,
                    regression_size=25,
                    k=1,
                    min_improvement=0.02,
                    max_regression_drop=0.0,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    top_k=0,
                    base_model_path=config.hf_model_path,
                    repeats=3,
                    deterministic=True,
                    seed=seed,
                )
                adapter_result = train_and_maybe_promote(conn, promotion_config)
                adapter_message = adapter_result.message

            robust_payload = robust_eval_conditions(
                db_path=db_path,
                backend_name=config.backend,
                conditions=_resolve_conditions(config.backend),
                k=1,
                heldout_size=heldout_size,
                task_limit=None,
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                policy_name=None,
                repeats=3,
                deterministic=True,
                seed=seed,
            )
            eval_summary = _best_condition_summary(robust_payload.get("conditions", []))
            previous_best = best_heldout
            if eval_summary["heldout_pass_at_1"] > best_heldout:
                best_heldout = eval_summary["heldout_pass_at_1"]
                best_condition = eval_summary["condition"]

            policy_after = _active_policy_name(conn)
            adapter_after = _active_adapter_name(conn)
            cycle_end = _utc_now()
            cycle_payload = {
                "cycle": cycle,
                "started_at": cycle_start,
                "completed_at": cycle_end,
                "policy": {
                    "before": policy_before,
                    "after": policy_after,
                    "decision": policy_message,
                },
                "adapter": {
                    "before": adapter_before,
                    "after": adapter_after,
                    "decision": adapter_message,
                },
                "train": {
                    "tasks_attempted": len(iteration_results),
                    "passed_any": passed_any,
                    "attempts_per_task": [result.attempts for result in iteration_results],
                    "task_ids": [task.task_id for task in selected],
                    "k": train_k,
                },
                "eval": eval_summary,
                "robust_eval": robust_payload,
            }
            artifact_path = artifacts_dir / f"cycle_{cycle:03d}.json"
            artifact_path.write_text(json.dumps(cycle_payload, indent=2), encoding="utf-8")

            improved = eval_summary["heldout_pass_at_1"] > previous_best
            tests_failing = passed_any == 0
            if enable_self_patch and self_patch_callback and (not improved or tests_failing):
                self_patch_context = {
                    "cycle": cycle,
                    "last_cycle": cycle_payload,
                }
                self_patch_callback(self_patch_context)

            cycle_summaries.append(cycle_payload)

        conn.close()

    conn = connect(db_path)
    summary = {
        "cycles": cycles,
        "best_heldout_pass_at_1": best_heldout,
        "best_condition": best_condition,
        "active_policy": _active_policy_name(conn),
        "active_adapter": _active_adapter_name(conn),
        "artifacts_dir": str(artifacts_dir),
    }
    conn.close()
    return summary


def _resolve_backend(config: Config, conn) -> object:
    if config.backend == "mock":
        return MockBackend()
    if config.backend == "openai":
        if not config.base_url:
            raise ValueError("RSLM_BASE_URL required for openai backend")
        return OpenAICompatBackend(config.base_url, config.model, config.api_key)
    if config.backend == "localhf":
        adapter = None
        active = get_active_adapter(conn)
        if active:
            adapter = active[1]
        if not config.hf_model_path:
            raise ValueError("RSLM_HF_MODEL_ID required for localhf backend")
        return LocalHFBackend(config.hf_model_path, adapter_path=adapter, torch_dtype=config.torch_dtype)
    return MockBackend()


def _resolve_conditions(backend: str) -> list[str]:
    if backend == "mock":
        return ["baseline", "memory", "semantic", "memory_learning"]
    return ["baseline", "memory", "learning", "memory_learning"]


def _load_tasks(tasks_source: str, config: Config):
    if tasks_source == "bundled":
        return load_tasks(include_generated=config.include_generated_tasks)
    return load_tasks(Path(tasks_source))


def _select_unseen_tasks(conn, tasks, task_limit: Optional[int], unseen_only: bool, seed: int) -> list:
    if not unseen_only:
        return tasks[:task_limit] if task_limit is not None else tasks
    seen_ids = fetch_seen_task_ids(conn)
    unseen = [task for task in tasks if task.task_id not in seen_ids]
    rng = random.Random(seed)
    rng.shuffle(unseen)
    if task_limit is None:
        return unseen
    if len(unseen) >= task_limit:
        return unseen[:task_limit]
    remaining = [task for task in tasks if task.task_id in seen_ids]
    rng.shuffle(remaining)
    needed = task_limit - len(unseen)
    return unseen + remaining[:needed]


def _best_condition_summary(conditions: list[dict]) -> dict:
    best = {
        "condition": None,
        "heldout_pass_at_1": 0.0,
        "hidden_pass_at_1": 0.0,
    }
    for condition in conditions:
        heldout = condition.get("heldout", {}).get("pass_at_1", 0.0)
        hidden = condition.get("hidden", {}).get("pass_at_1", 0.0)
        if heldout > best["heldout_pass_at_1"]:
            best = {
                "condition": condition.get("condition"),
                "heldout_pass_at_1": heldout,
                "hidden_pass_at_1": hidden,
            }
    return best


def _active_policy_name(conn) -> str:
    row = conn.execute("SELECT policy_name FROM active_policy WHERE singleton = 1").fetchone()
    return row[0] if row else "default"


def _active_adapter_name(conn) -> str | None:
    active = get_active_adapter(conn)
    return active[0] if active else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _override_env(key: str, value: str):
    previous = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous
