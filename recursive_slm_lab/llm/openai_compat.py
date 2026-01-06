from __future__ import annotations

from dataclasses import dataclass

from .base import LLMBackend, LLMResponse


@dataclass
class OpenAICompatBackend(LLMBackend):
    base_url: str
    model: str
    api_key: str | None = None

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> LLMResponse:
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a code generation assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        url = self.base_url.rstrip("/") + "/chat/completions"
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        text = data["choices"][0]["message"]["content"]
        return LLMResponse(text=text, model=self.model)
