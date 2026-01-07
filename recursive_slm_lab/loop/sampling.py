from __future__ import annotations

from dataclasses import dataclass

from ..llm.base import LLMBackend
from ..llm.localhf import extract_python_function_code
from ..memory.retrieval import MemoryContext
from .prompts import build_prompt


@dataclass
class Candidate:
    prompt: str
    code: str
    model: str


SYSTEM_PROMPT = "You are a code generation assistant."


def generate_candidates(
    backend: LLMBackend,
    task_prompt: str,
    function_name: str,
    signature: str,
    memory_context: MemoryContext | None,
    k: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> list[Candidate]:
    memory_block = memory_context.format() if memory_context else ""
    example_code = memory_context.first_code() if memory_context else None
    prompt = build_prompt(task_prompt, function_name, signature, memory_block, example_code)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    candidates: list[Candidate] = []
    for _ in range(k):
        response = backend.generate(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        code = extract_python_function_code(response.text.strip(), function_name)
        candidates.append(Candidate(prompt=prompt, code=code.strip(), model=response.model))
    return candidates
