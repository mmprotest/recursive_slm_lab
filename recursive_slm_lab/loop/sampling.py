from __future__ import annotations

from dataclasses import dataclass

from ..llm.base import LLMBackend
from ..memory.retrieval import MemoryContext
from .prompts import build_prompt


@dataclass
class Candidate:
    prompt: str
    code: str
    model: str


def generate_candidates(
    backend: LLMBackend,
    task_prompt: str,
    function_name: str,
    signature: str,
    memory_context: MemoryContext | None,
    k: int,
    max_tokens: int,
    temperature: float,
) -> list[Candidate]:
    memory_block = memory_context.format() if memory_context else ""
    example_code = memory_context.first_code() if memory_context else None
    prompt = build_prompt(task_prompt, function_name, signature, memory_block, example_code)
    candidates: list[Candidate] = []
    for _ in range(k):
        response = backend.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        candidates.append(Candidate(prompt=prompt, code=response.text.strip(), model=response.model))
    return candidates
