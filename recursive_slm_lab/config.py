from __future__ import annotations

import os
from dataclasses import dataclass


def _get_env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _get_env_int(name: str, default: str) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Config:
    db_path: str = os.getenv("RSLM_DB", "artifacts/memory.sqlite")
    backend: str = os.getenv("RSLM_BACKEND", "mock")
    base_url: str | None = os.getenv("RSLM_BASE_URL")
    model: str = os.getenv("RSLM_MODEL", "mock-model")
    api_key: str | None = os.getenv("RSLM_API_KEY")
    hf_model_path: str | None = os.getenv("RSLM_HF_MODEL_ID") or os.getenv("RSLM_HF_MODEL_PATH")
    torch_dtype: str | None = os.getenv("RSLM_TORCH_DTYPE")
    max_tokens: int = _get_env_int("RSLM_MAX_TOKENS", "256")
    temperature: float = _get_env_float("RSLM_TEMPERATURE", "0.2")
    top_p: float = _get_env_float("RSLM_TOP_P", "0.9")
    top_k: int = _get_env_int("RSLM_TOP_K", "50")
