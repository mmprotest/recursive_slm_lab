from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    model: str


class LLMBackend:
    def generate(self, prompt: str, max_tokens: int, temperature: float) -> LLMResponse:
        raise NotImplementedError
