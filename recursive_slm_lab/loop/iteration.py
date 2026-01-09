from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor

from ..llm.localhf import LocalHFBackend
from ..llm.mock import MockBackend
from ..llm.openai_compat import OpenAICompatBackend
from ..memory import insert_episode_many, retrieve_memory, mark_task_seen, RunMeta, start_run, get_active_adapter
from ..policy import DEFAULT_POLICY, Policy
from ..verify import verify_candidate
from ..tasks import Task
from ..llm.base import LLMBackend
from .sampling import generate_candidates


@dataclass
class IterationResult:
    task_id: str
    passed: bool
    attempts: int


def run_iteration(
    conn,
    tasks: list[Task],
    backend: LLMBackend,
    k: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    memory_enabled: bool,
    condition: str,
    policy: Policy | None = None,
    db_path: str | None = None,
    verify_workers: int | None = None,
    seed: int | None = None,
) -> list[IterationResult]:
    policy = policy or DEFAULT_POLICY
    verify_workers = _resolve_verify_workers(verify_workers)
    backend_name = "mock"
    model_name = "unknown"
    if isinstance(backend, MockBackend):
        backend_name = "mock"
        model_name = backend.model_name
    elif isinstance(backend, OpenAICompatBackend):
        backend_name = "openai"
        model_name = backend.model
    elif isinstance(backend, LocalHFBackend):
        backend_name = "localhf"
        model_name = backend.model_path
    adapter_name = None
    active = get_active_adapter(conn)
    if active:
        adapter_name = active[0]

    run_id = start_run(
        conn,
        RunMeta(
            mode=condition,
            backend=backend_name,
            model=model_name,
            adapter_name=adapter_name,
            memory_enabled=memory_enabled,
            semantic_enabled=False,
            learning_enabled=False,
            k=k,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            config_json={
                "condition": condition,
                "memory_enabled": memory_enabled,
                "seed": seed,
            },
        ),
    )
    iteration_results: list[IterationResult] = []
    for task_index, task in enumerate(tasks):
        mark_task_seen(conn, task.task_id)
        memory_context = None
        if memory_enabled:
            memory_context = retrieve_memory(
                conn,
                task.prompt,
                policy=policy,
                function_name=task.function_name,
            )
        candidates = generate_candidates(
            backend,
            task_prompt=task.prompt,
            function_name=task.function_name,
            signature=task.signature,
            memory_context=memory_context,
            policy=policy,
            k=k,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=(seed + task_index * 1000) if seed is not None else None,
        )
        passed_any = False
        retrieval_used = bool(memory_context and memory_context.hits)
        memory_sources = None
        memory_top_score = None
        if memory_context and memory_context.hits:
            sources = sorted({hit.source for hit in memory_context.hits})
            memory_sources = ",".join(sources)
            memory_top_score = min(hit.score for hit in memory_context.hits)
        episode_rows: list[tuple] = []
        if verify_workers and verify_workers > 1 and len(candidates) > 1:
            args_list = [
                (candidate.code, task.reference_tests, task.assert_tests, db_path)
                for candidate in candidates
            ]
            with ProcessPoolExecutor(max_workers=verify_workers) as executor:
                verifications = list(executor.map(_verify_candidate_worker, args_list))
        else:
            verifications = [
                verify_candidate(
                    candidate.code,
                    task.reference_tests,
                    task.assert_tests,
                    conn=conn,
                )
                for candidate in candidates
            ]
        for candidate, verification in zip(candidates, verifications):
            prompt_hash = hashlib.sha256(candidate.prompt.encode("utf-8")).hexdigest()
            episode_rows.append(
                (
                    task.task_id,
                    condition,
                    candidate.prompt,
                    candidate.code,
                    verification.passed,
                    verification.log,
                    run_id,
                    prompt_hash,
                    retrieval_used,
                    memory_sources,
                    memory_top_score,
                )
            )
            if verification.passed:
                passed_any = True
        insert_episode_many(conn, episode_rows)
        iteration_results.append(
            IterationResult(task_id=task.task_id, passed=passed_any, attempts=len(candidates))
        )
    return iteration_results


def _resolve_verify_workers(verify_workers: int | None) -> int:
    if verify_workers is not None:
        return max(1, verify_workers)
    env_value = os.getenv("RSLM_VERIFY_WORKERS")
    if env_value:
        try:
            return max(1, int(env_value))
        except ValueError:
            return 1
    return min(4, os.cpu_count() or 2)


def _verify_candidate_worker(args: tuple[str, str, list[str] | None, str | None]):
    code, reference_tests, assert_tests, db_path = args
    return verify_candidate(
        code,
        reference_tests,
        assert_tests,
        db_path=db_path,
    )
