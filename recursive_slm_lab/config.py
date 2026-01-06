from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    db_path: str = os.getenv("RSLM_DB", "artifacts/memory.sqlite")
    backend: str = os.getenv("RSLM_BACKEND", "mock")
    base_url: str | None = os.getenv("RSLM_BASE_URL")
    model: str = os.getenv("RSLM_MODEL", "mock-model")
    api_key: str | None = os.getenv("RSLM_API_KEY")
    hf_model_path: str | None = os.getenv("RSLM_HF_MODEL_PATH")
    max_tokens: int = int(os.getenv("RSLM_MAX_TOKENS", "256"))
    temperature: float = float(os.getenv("RSLM_TEMPERATURE", "0.2"))
