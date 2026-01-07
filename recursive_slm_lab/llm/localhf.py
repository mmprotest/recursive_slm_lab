from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import logging
import re

from packaging import version

from .base import LLMBackend, LLMResponse

MIN_TRANSFORMERS_VERSION = "4.51.0"
LOGGER = logging.getLogger(__name__)


def _strip_think(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.lstrip()


def _extract_first_python_fence(text: str) -> str | None:
    fence = re.search(r"```python\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return None


def postprocess_output(text: str) -> str:
    cleaned = _strip_think(text)
    fenced = _extract_first_python_fence(cleaned)
    if fenced:
        return fenced.strip()
    return cleaned.strip()


def extract_python_function_code(text: str, function_name: str) -> str:
    cleaned = postprocess_output(text)
    if not cleaned:
        return cleaned
    lines = cleaned.splitlines()
    def_index = None
    for idx, line in enumerate(lines):
        if re.match(rf"^\s*def\s+{re.escape(function_name)}\s*\(", line):
            def_index = idx
            break
    if def_index is None:
        return cleaned

    start_idx = def_index
    for idx in range(def_index - 1, -1, -1):
        stripped = lines[idx].strip()
        if not stripped:
            start_idx = idx
            continue
        if stripped.startswith(("import ", "from ", "def ", "class ", "@", "#")):
            start_idx = idx
            continue
        break

    def_indent = len(lines[def_index]) - len(lines[def_index].lstrip(" "))
    end_idx = len(lines)
    for idx in range(def_index + 1, len(lines)):
        line = lines[idx]
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent <= def_indent and stripped.startswith(("def ", "class ", "@")):
            end_idx = idx
            break
    code_lines = lines[start_idx:end_idx]
    return "\n".join(code_lines).strip()


@dataclass
class LocalHFBackend(LLMBackend):
    model_path: str
    adapter_path: str | None = None
    torch_dtype: str | None = None

    def __post_init__(self) -> None:
        required = ["torch", "transformers", "peft", "accelerate"]
        missing = [name for name in required if importlib.util.find_spec(name) is None]
        if missing:
            hint = "pip install -e '.[localhf]'"
            raise RuntimeError(
                f"LocalHF dependencies missing: {', '.join(missing)}. Install with {hint}."
            )

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version
        from peft import PeftModel

        if version.parse(transformers_version) < version.parse(MIN_TRANSFORMERS_VERSION):
            raise RuntimeError(
                f"Transformers>={MIN_TRANSFORMERS_VERSION} is required for LocalHF. "
                f"Detected {transformers_version}."
            )

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            try:
                torch.randn(1, device="cuda")
            except Exception:
                LOGGER.warning(
                    "CUDA present but kernels not supported (likely sm_120). Falling back to CPU."
                )
                device = "cpu"
        dtype = self._resolve_dtype(device)
        device_map: str | None = "auto" if device == "cuda" else None

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map=device_map,
        )
        if self.adapter_path:
            model = PeftModel.from_pretrained(model, self.adapter_path)
        self._model = model
        self._device = device

    def _resolve_dtype(self, device: str):
        if device == "cpu":
            return self._torch.float32
        if self.torch_dtype:
            if self.torch_dtype == "fp16":
                return self._torch.float16
            if self.torch_dtype == "bf16":
                return self._torch.bfloat16
            if self.torch_dtype == "fp32":
                return self._torch.float32
        if self._torch.cuda.is_bf16_supported():
            return self._torch.bfloat16
        return self._torch.float16

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> LLMResponse:
        self._model.eval()
        input_ids = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_len = input_ids.shape[-1]
        input_ids = input_ids.to(self._model.device)
        attention_mask = self._torch.ones_like(input_ids)
        with self._torch.no_grad():
            outputs = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        generated = outputs[0, input_len:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        return LLMResponse(text=postprocess_output(text), model=self.model_path)
