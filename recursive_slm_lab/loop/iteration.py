from __future__ import annotations

from dataclasses import dataclass

from ..memory import insert_episode, retrieve_memory
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
    memory_enabled: bool,
    condition: str,
) -> list[IterationResult]:
    results: list[IterationResult] = []
    for task in tasks:
        memory_context = None
        if memory_enabled:
            memory_context = retrieve_memory(conn, task.prompt)
        candidates = generate_candidates(
            backend,
            task_prompt=task.prompt,
            function_name=task.function_name,
            signature=task.signature,
            memory_context=memory_context,
            k=k,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        passed_any = False
        for candidate in candidates:
            verification = verify_candidate(candidate.code, task.reference_tests)
            insert_episode(
                conn,
                task_id=task.task_id,
                condition=condition,
                prompt=candidate.prompt,
                candidate_code=candidate.code,
                passed=verification.passed,
                test_log=verification.log,
            )
            if verification.passed:
                passed_any = True
        results.append(IterationResult(task_id=task.task_id, passed=passed_any, attempts=len(candidates)))
    return results
