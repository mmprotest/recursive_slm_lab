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


def _build_backend(config: Config, condition: str, adapter_path: str | None) -> tuple[object, str]:
    if config.backend == "mock":
        return MockBackend(), "mock"
    if config.backend == "openai":
        if not config.base_url:
            raise ValueError("RSLM_BASE_URL required for openai backend")
        return OpenAICompatBackend(config.base_url, config.model, config.api_key), "openai"
    if config.backend == "localhf":
        if not config.hf_model_path:
            raise ValueError("RSLM_HF_MODEL_PATH required for localhf backend")
        adapter = adapter_path if condition in {"learning", "memory_learning"} else None
        return LocalHFBackend(config.hf_model_path, adapter_path=adapter), "localhf"
    return MockBackend(), "mock"


def evaluate_conditions(
    db_path: str,
    backend_name: str,
    conditions: list[str],
    k: int,
    heldout_size: int,
    output_path: str | None,
) -> dict:
    config = Config(db_path=db_path, backend=backend_name)
    tasks = load_tasks()
    _, heldout = split_tasks(tasks, heldout_size=heldout_size)

    conn = connect(db_path)
    results: list[ConditionResult] = []

    active_adapter = get_active_adapter(conn)

    for condition in conditions:
        memory_enabled = condition in {"memory", "memory_learning"}
        learning_enabled = condition in {"learning", "memory_learning"}
        notes = ""

        adapter_path = active_adapter[1] if active_adapter else None
        backend, backend_label = _build_backend(config, condition, adapter_path)
        if learning_enabled and backend_label != "localhf":
            notes = "Learning requested but backend does not support local adapters; using baseline inference."
        if learning_enabled and backend_label == "localhf" and not active_adapter:
            notes = "Learning requested but no active adapter is set; using base model."

        outcomes: list[list[bool]] = []
        for task in heldout:
            memory_context = None
            if memory_enabled or learning_enabled:
                memory_context = retrieve_memory(conn, task.prompt)
                if memory_enabled and not learning_enabled and memory_context:
                    memory_context = memory_context.filter_sources({"episode"})
                if learning_enabled and not memory_enabled and memory_context:
                    memory_context = memory_context.filter_sources({"rule", "procedure"})
            candidates = generate_candidates(
                backend,
                task_prompt=task.prompt,
                function_name=task.function_name,
                signature=task.signature,
                memory_context=memory_context,
                k=k,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )
            task_outcomes: list[bool] = []
            for cand in candidates:
                verification = verify_candidate(cand.code, task.reference_tests)
                task_outcomes.append(verification.passed)
            outcomes.append(task_outcomes)
        summary = compute_pass_rates(outcomes, k=k)
        results.append(ConditionResult(condition, outcomes, summary, notes))

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
        verification = verify_candidate(ep.candidate_code, task.reference_tests)
        if verification.passed:
            passed += 1
    return passed / len(sample)


def _regression_check(conn, active_adapter: tuple[str, str] | None) -> dict:
    if not active_adapter:
        return {"status": "skipped", "reason": "No active adapter"}
    return {"status": "skipped", "reason": "Regression checks require LocalHF mode"}
