from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    model: str


class LLMBackend:
    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> LLMResponse:
        raise NotImplementedError
