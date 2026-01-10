from __future__ import annotations

from dataclasses import dataclass
import gc
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


def _build_generate_kwargs(
    input_ids,
    attention_mask,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    model_config,
) -> dict:
    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_tokens,
    }
    do_sample = temperature > 0
    if do_sample:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        gen_kwargs["top_k"] = top_k
    else:
        from transformers import GenerationConfig

        gen_kwargs["do_sample"] = False
        for key in ("temperature", "top_p", "top_k", "typical_p", "min_p"):
            gen_kwargs.pop(key, None)
        greedy_config = GenerationConfig.from_model_config(model_config)
        greedy_config.do_sample = False
        greedy_config.temperature = 1.0
        greedy_config.top_p = 1.0
        greedy_config.top_k = 0
        greedy_config.num_beams = 1
        gen_kwargs["generation_config"] = greedy_config
    return gen_kwargs


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
        device_map = {"": 0} if device == "cuda" else None
        load_kwargs = {"device_map": device_map, "low_cpu_mem_usage": True}
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=dtype,
                **load_kwargs,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                **load_kwargs,
            )
        self._base_model = model
        self._peft_model = None
        self._active_adapter = None
        if self.adapter_path:
            self._peft_model = PeftModel.from_pretrained(self._base_model, self.adapter_path)
            model = self._peft_model
            self._active_adapter = self.adapter_path
        self._model = model
        self._device = device
        LOGGER.info(
            "Loaded LocalHF model=%s device=%s dtype=%s device_map=%s adapter=%s",
            self.model_path,
            device,
            dtype,
            device_map,
            self.adapter_path or "none",
        )

    def _stable_adapter_name(self, adapter_path: str) -> str:
        name = adapter_path.strip("/").split("/")[-1] or "adapter"
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

    def set_adapter(self, adapter_path: str | None) -> None:
        if adapter_path == self._active_adapter:
            return
        from peft import PeftModel

        previous = self._active_adapter
        if adapter_path is None:
            self._model = self._base_model
            self.adapter_path = None
            self._active_adapter = None
            LOGGER.info("LocalHF adapter cleared (previous=%s).", previous or "none")
            return

        multi_adapter = self._peft_model is not None and hasattr(self._peft_model, "load_adapter")
        if self._peft_model is None:
            self._peft_model = PeftModel.from_pretrained(self._base_model, adapter_path)
            self._model = self._peft_model
            self.adapter_path = adapter_path
            self._active_adapter = adapter_path
            LOGGER.info(
                "LocalHF adapter activated: %s (previous=%s, multi_adapter=%s)",
                adapter_path,
                previous or "none",
                False,
            )
            return

        if multi_adapter:
            adapter_name = self._stable_adapter_name(adapter_path)
            self._peft_model.load_adapter(adapter_path, adapter_name=adapter_name)
            self._peft_model.set_adapter(adapter_name)
            self._model = self._peft_model
            self.adapter_path = adapter_path
            self._active_adapter = adapter_path
            LOGGER.info(
                "LocalHF adapter activated: %s (previous=%s, multi_adapter=%s)",
                adapter_path,
                previous or "none",
                True,
            )
            return

        self._peft_model = None
        self._model = self._base_model
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        self._peft_model = PeftModel.from_pretrained(self._base_model, adapter_path)
        self._model = self._peft_model
        self.adapter_path = adapter_path
        self._active_adapter = adapter_path
        LOGGER.info(
            "LocalHF adapter activated: %s (previous=%s, multi_adapter=%s)",
            adapter_path,
            previous or "none",
            False,
        )

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
        gen_kwargs = _build_generate_kwargs(
            input_ids,
            attention_mask,
            max_tokens,
            temperature,
            top_p,
            top_k,
            self._model.config,
        )
        LOGGER.info(
            "LocalHF generate: do_sample=%s generation_config=%s kwargs_keys=%s",
            gen_kwargs.get("do_sample"),
            "generation_config" in gen_kwargs,
            sorted(gen_kwargs.keys()),
        )
        LOGGER.debug("LocalHF generate kwargs keys: %s", sorted(gen_kwargs.keys()))
        LOGGER.debug(
            "LocalHF sampling params: temperature=%s top_p=%s top_k=%s",
            temperature,
            top_p,
            top_k,
        )
        with self._torch.no_grad():
            outputs = self._model.generate(**gen_kwargs)
        generated = outputs[0, input_len:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        return LLMResponse(text=postprocess_output(text), model=self.model_path)
