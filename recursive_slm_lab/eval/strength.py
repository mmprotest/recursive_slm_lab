from __future__ import annotations

import re

from ..tasks import Task
from ..verify import verify_candidate


def _parse_signature(signature: str) -> list[str]:
    cleaned = signature.strip()
    if not cleaned.startswith("(") or not cleaned.endswith(")"):
        return []
    inner = cleaned[1:-1].strip()
    if not inner:
        return []
    return [param.strip() for param in inner.split(",") if param.strip()]


def _solution_stub(function_name: str, signature: str, body: str) -> str:
    return f"def {function_name}{signature}:\n    {body}\n"


def _cheat_solutions(task: Task) -> list[str]:
    params = _parse_signature(task.signature)
    first_param = params[0] if params else None
    candidates = [
        _solution_stub(task.function_name, task.signature, "return 0"),
        _solution_stub(task.function_name, task.signature, "return None"),
    ]
    if first_param:
        candidates.append(_solution_stub(task.function_name, task.signature, f"return {first_param}"))
    candidates.append(_solution_stub(task.function_name, task.signature, "return []"))
    candidates.append(_solution_stub(task.function_name, task.signature, "return ''"))
    if first_param:
        if re.search(r"(values|nums|items|list|arr|array)", first_param):
            candidates.append(_solution_stub(task.function_name, task.signature, f"return sorted({first_param})"))
        candidates.append(_solution_stub(task.function_name, task.signature, f"return {first_param} + 1"))
    return candidates


def is_task_weak(task: Task) -> bool:
    for candidate_code in _cheat_solutions(task):
        verification = verify_candidate(candidate_code, task.reference_tests, task.assert_tests)
        if verification.passed:
            return True
    return False


def find_weak_tasks(tasks: list[Task]) -> list[str]:
    weak: list[str] = []
    for task in tasks:
        if is_task_weak(task):
            weak.append(task.task_id)
    return weak
