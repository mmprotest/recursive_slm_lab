from __future__ import annotations

from dataclasses import dataclass

from ..memory import insert_episode_many, retrieve_memory, mark_task_seen
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
) -> list[IterationResult]:
    results: list[IterationResult] = []
    for task in tasks:
        mark_task_seen(conn, task.task_id)
        memory_context = None
        if memory_enabled:
            extra_terms = [task.function_name.rsplit("_", 1)[0], task.function_name]
            memory_context = retrieve_memory(conn, task.prompt, extra_terms=extra_terms)
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
        passed_any = False
        episode_rows: list[tuple[str, str, str, str, bool, str]] = []
        for candidate in candidates:
            verification = verify_candidate(candidate.code, task.reference_tests, task.assert_tests)
            episode_rows.append(
                (
                    task.task_id,
                    condition,
                    candidate.prompt,
                    candidate.code,
                    verification.passed,
                    verification.log,
                )
            )
            if verification.passed:
                passed_any = True
        insert_episode_many(conn, episode_rows)
        results.append(IterationResult(task_id=task.task_id, passed=passed_any, attempts=len(candidates)))
    return results
