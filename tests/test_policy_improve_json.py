from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest

from recursive_slm_lab.meta import policy_improve
from recursive_slm_lab.llm.base import LLMResponse
from recursive_slm_lab.policy import DEFAULT_POLICY


def _constraints() -> dict:
    return {
        "retrieval_top_n": (0, 10),
        "sampling_train": {
            "k": (1, 16),
            "temperature": (0.0, 1.5),
            "top_p": (0.1, 1.0),
            "top_k": (0, 200),
        },
        "sampling_eval": {
            "k": (1, 16),
            "temperature": (0.0, 1.5),
            "top_p": (0.1, 1.0),
            "top_k": (0, 200),
        },
    }


class FakeBackend:
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = outputs
        self._index = 0

    def generate(self, *args, **kwargs) -> LLMResponse:
        output = self._outputs[self._index]
        self._index = min(self._index + 1, len(self._outputs) - 1)
        return LLMResponse(text=output, model="fake")


def test_extract_json_text_strict() -> None:
    assert policy_improve._extract_json_text('{"a": 1}') == '{"a": 1}'
    mixed = "prefix {\"a\": 2, \"b\": 3} suffix"
    assert policy_improve._extract_json_text(mixed) == '{"a": 2, "b": 3}'
    assert policy_improve._extract_json_text("no json here") == ""


def test_validate_policy_json_requires_fields_and_clamps() -> None:
    constraints = _constraints()
    payload = {
        "retrieval_top_n": 99,
        "sampling_train": {"k": 0, "temperature": 2.5, "top_p": 0.0, "top_k": 999},
        "sampling_eval": {"k": 3, "temperature": 0.3, "top_p": 1.5, "top_k": -5},
    }
    validated = policy_improve._validate_policy_json(payload, constraints)
    assert validated["retrieval_top_n"] == 10
    assert validated["sampling_train"]["k"] == 1
    assert validated["sampling_train"]["temperature"] == 1.5
    assert validated["sampling_train"]["top_p"] == 0.1
    assert validated["sampling_train"]["top_k"] == 200
    assert validated["sampling_eval"]["top_p"] == 1.0
    assert validated["sampling_eval"]["top_k"] == 0

    with pytest.raises(ValueError, match="sampling_train.k"):
        policy_improve._validate_policy_json(
            {
                "retrieval_top_n": 1,
                "sampling_train": {"temperature": 0.2, "top_p": 0.9, "top_k": 50},
                "sampling_eval": {"k": 1, "temperature": 0.2, "top_p": 0.9, "top_k": 50},
            },
            constraints,
        )


def test_propose_policy_retries_and_succeeds() -> None:
    backend = FakeBackend(
        [
            "not json",
            '{"retrieval_top_n": 4, "sampling_train": {"k": 2, "temperature": 0.2, '
            '"top_p": 0.9, "top_k": 50}, "sampling_eval": {"k": 1, "temperature": 0.0, '
            '"top_p": 1.0, "top_k": 0}}',
        ]
    )
    proposal = policy_improve.propose_policy(
        backend,
        current_policy=DEFAULT_POLICY,
        recent_failures_summary="none",
        constraints=_constraints(),
    )
    assert proposal.attempts == 2
    assert proposal.policy.retrieval_top_n == 4


def test_propose_policy_failure_includes_attempts_and_snippet() -> None:
    backend = FakeBackend(["no json output"])
    with pytest.raises(ValueError) as excinfo:
        policy_improve.propose_policy(
            backend,
            current_policy=DEFAULT_POLICY,
            recent_failures_summary="none",
            constraints=_constraints(),
        )
    message = str(excinfo.value)
    assert "attempts" in message
    assert "Final output snippet" in message
