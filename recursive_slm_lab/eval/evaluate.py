from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..config import Config
from ..llm import MockBackend, OpenAICompatBackend, LocalHFBackend
from ..memory import connect, fetch_passed_episodes, retrieve_memory, get_active_adapter
from ..tasks import load_tasks, split_tasks, Task
from ..verify import verify_candidate
from ..loop.sampling import generate_candidates
from .metrics import compute_pass_rates, MetricSummary


@dataclass
class ConditionResult:
    condition: str
    outcomes: list[list[bool]]
    summary: MetricSummary
    notes: str
    learned_unavailable: bool


def _build_backend(config: Config, condition: str, adapter_path: str | None) -> tuple[object, str]:
    if config.backend == "mock":
        return MockBackend(), "mock"
    if config.backend == "openai":
        if not config.base_url:
            raise ValueError("RSLM_BASE_URL required for openai backend")
        return OpenAICompatBackend(config.base_url, config.model, config.api_key), "openai"
    if config.backend == "localhf":
        if not config.hf_model_path:
            raise ValueError("RSLM_HF_MODEL_ID required for localhf backend")
        adapter = adapter_path if condition in {"learning", "memory_learning"} else None
        return LocalHFBackend(
            config.hf_model_path,
            adapter_path=adapter,
            torch_dtype=config.torch_dtype,
        ), "localhf"
    return MockBackend(), "mock"


def evaluate_conditions(
    db_path: str,
    backend_name: str,
    conditions: list[str],
    k: int,
    heldout_size: int,
    task_limit: int | None,
    output_path: str | None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> dict:
    config = Config(db_path=db_path, backend=backend_name)
    tasks = load_tasks()
    _, heldout = split_tasks(tasks, heldout_size=heldout_size)
    if task_limit is not None:
        heldout = heldout[:task_limit]

    conn = connect(db_path)
    results: list[ConditionResult] = []

    active_adapter = get_active_adapter(conn)

    max_tokens = max_tokens if max_tokens is not None else config.max_tokens
    temperature = temperature if temperature is not None else config.temperature
    top_p = top_p if top_p is not None else config.top_p
    top_k = top_k if top_k is not None else config.top_k

    for condition in conditions:
        notes = ""
        learned_unavailable = False

        adapter_path = active_adapter[1] if active_adapter else None
        backend, backend_label = _build_backend(config, condition, adapter_path)
        effective_condition = condition
        if backend_label == "mock" and condition == "learning":
            effective_condition = "semantic"

        memory_enabled = effective_condition in {"memory", "memory_learning"}
        semantic_enabled = effective_condition in {"semantic", "memory_learning"}
        learning_enabled = condition in {"learning", "memory_learning"}

        baseline_learning = condition == "learning" and backend_label != "localhf" and effective_condition == "learning"
        if baseline_learning:
            notes = "Learning requested but adapters unavailable; running baseline."
            learned_unavailable = True
            memory_enabled = False
            semantic_enabled = False
            learning_enabled = False
            effective_condition = "baseline"
        if learning_enabled and backend_label == "localhf" and not active_adapter:
            notes = "Learning requested but no active adapter is set; using base model."
            learned_unavailable = True

        outcomes: list[list[bool]] = []
        for task in heldout:
            memory_context = None
            if (memory_enabled or semantic_enabled) and not baseline_learning:
                extra_terms = [task.function_name.rsplit("_", 1)[0], task.function_name]
                memory_context = retrieve_memory(conn, task.prompt, extra_terms=extra_terms)
                if memory_enabled and not semantic_enabled and memory_context:
                    memory_context = memory_context.filter_sources({"episode"})
                if semantic_enabled and not memory_enabled and memory_context:
                    memory_context = memory_context.filter_sources({"rule", "procedure"})
            candidates = generate_candidates(
                backend,
                task_prompt=task.prompt,
                function_name=task.function_name,
                signature=task.signature,
                memory_context=memory_context,
                k=k,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            task_outcomes: list[bool] = []
            for cand in candidates:
                verification = verify_candidate(cand.code, task.reference_tests, task.assert_tests)
                task_outcomes.append(verification.passed)
            outcomes.append(task_outcomes)
        summary = compute_pass_rates(outcomes, k=k)
        results.append(
            ConditionResult(effective_condition, outcomes, summary, notes, learned_unavailable)
        )

    task_map = {task.task_id: task for task in tasks}
    memory_precision = _memory_precision(conn, task_map, sample_size=10)
    regression_info = _regression_check(conn, active_adapter)

    payload = {
        "conditions": [
            {
                "condition": result.condition,
                "pass_at_1": result.summary.pass_at_1,
                "pass_at_k": result.summary.pass_at_k,
                "notes": result.notes,
                "learned_unavailable": result.learned_unavailable,
            }
            for result in results
        ],
        "memory_precision": memory_precision,
        "regression_check": regression_info,
    }

    created_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO eval_runs (created_at, heldout_size, k, backend, payload_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (created_at, heldout_size, k, backend_name, json.dumps(payload)),
    )
    conn.commit()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    conn.close()
    return payload


def _memory_precision(conn, task_map: dict[str, Task], sample_size: int = 10) -> float:
    episodes = fetch_passed_episodes(conn)
    if not episodes:
        return 0.0
    sample = episodes[:sample_size]
    passed = 0
    for ep in sample:
        task = task_map.get(ep.task_id)
        if not task:
            continue
        verification = verify_candidate(ep.candidate_code, task.reference_tests, task.assert_tests)
        if verification.passed:
            passed += 1
    return passed / len(sample)


def _regression_check(conn, active_adapter: tuple[str, str] | None) -> dict:
    if not active_adapter:
        return {"status": "skipped", "reason": "No active adapter"}
    return {"status": "skipped", "reason": "Regression checks require LocalHF mode"}
