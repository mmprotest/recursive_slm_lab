from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SamplingConfig:
    k: int
    temperature: float
    top_p: float
    top_k: int


@dataclass(frozen=True)
class ConsolidationConfig:
    min_evidence: int
    enable_llm_consolidation: bool
    max_rules: int
    max_procedures: int


@dataclass(frozen=True)
class Policy:
    prompt_contract: str
    prompt_template: str
    retrieval_top_n: int
    retrieval_min_score: float | None
    retrieval_extra_terms_mode: str
    sampling_train: SamplingConfig
    sampling_eval: SamplingConfig
    consolidation: ConsolidationConfig

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, payload: dict) -> Policy:
        sampling_train = SamplingConfig(**payload["sampling_train"])
        sampling_eval = SamplingConfig(**payload["sampling_eval"])
        consolidation = ConsolidationConfig(**payload["consolidation"])
        return cls(
            prompt_contract=payload["prompt_contract"],
            prompt_template=payload["prompt_template"],
            retrieval_top_n=payload["retrieval_top_n"],
            retrieval_min_score=payload.get("retrieval_min_score"),
            retrieval_extra_terms_mode=payload["retrieval_extra_terms_mode"],
            sampling_train=sampling_train,
            sampling_eval=sampling_eval,
            consolidation=consolidation,
        )

    @classmethod
    def from_json(cls, raw: str) -> Policy:
        return cls.from_dict(json.loads(raw))


DEFAULT_PROMPT_CONTRACT = (
    "Return ONLY valid Python code. Define ONLY the requested function. "
    "Do not include tests, explanations, or markdown."
)

DEFAULT_PROMPT_TEMPLATE = (
    "{prompt_contract}\n\n"
    "Task: {task_prompt}\n"
    "Function Name: {function_name}\n"
    "Signature: {signature}\n"
    "Implement the function accordingly."
    "{memory_blocks}"
)

DEFAULT_POLICY = Policy(
    prompt_contract=DEFAULT_PROMPT_CONTRACT,
    prompt_template=DEFAULT_PROMPT_TEMPLATE,
    retrieval_top_n=3,
    retrieval_min_score=None,
    retrieval_extra_terms_mode="function_prefix+name",
    sampling_train=SamplingConfig(k=8, temperature=0.2, top_p=0.9, top_k=50),
    sampling_eval=SamplingConfig(k=1, temperature=0.0, top_p=1.0, top_k=0),
    consolidation=ConsolidationConfig(
        min_evidence=3,
        enable_llm_consolidation=False,
        max_rules=20,
        max_procedures=20,
    ),
)
