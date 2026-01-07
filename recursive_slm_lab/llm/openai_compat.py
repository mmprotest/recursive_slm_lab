from __future__ import annotations

from dataclasses import dataclass

from .base import LLMBackend, LLMResponse


@dataclass
class OpenAICompatBackend(LLMBackend):
    base_url: str
    model: str
    api_key: str | None = None

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> LLMResponse:
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if top_k:
            payload["top_k"] = top_k
        url = self.base_url.rstrip("/") + "/chat/completions"
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        text = data["choices"][0]["message"]["content"]
        return LLMResponse(text=text, model=self.model)
