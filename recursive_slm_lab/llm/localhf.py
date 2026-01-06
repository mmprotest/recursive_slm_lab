from __future__ import annotations

from dataclasses import dataclass
import importlib.util

from .base import LLMBackend, LLMResponse


@dataclass
class LocalHFBackend(LLMBackend):
    model_path: str
    adapter_path: str | None = None

    def __post_init__(self) -> None:
        required = ["torch", "transformers", "peft"]
        missing = [name for name in required if importlib.util.find_spec(name) is None]
        if missing:
            raise RuntimeError(f"LocalHF dependencies missing: {', '.join(missing)}")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(self.model_path)
        if self.adapter_path:
            model = PeftModel.from_pretrained(model, self.adapter_path)
        self._model = model

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> LLMResponse:
        inputs = self._tokenizer(prompt, return_tensors="pt")
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
        )
        text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return LLMResponse(text=text, model=self.model_path)
