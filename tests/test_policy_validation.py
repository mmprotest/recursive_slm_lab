from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from recursive_slm_lab.meta import policy_improve


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


def _base_payload(retrieval_top_n) -> dict:
    return {
        "retrieval_top_n": retrieval_top_n,
        "sampling_train": {"k": 2, "temperature": 0.2, "top_p": 0.9, "top_k": 50},
        "sampling_eval": {"k": 1, "temperature": 0.0, "top_p": 1.0, "top_k": 0},
    }


@pytest.mark.parametrize("value", [3, 3.0, "3", " 3 "])
def test_retrieval_top_n_accepts_int_like_values(value) -> None:
    validated = policy_improve._validate_policy_json(_base_payload(value), _constraints())
    assert validated["retrieval_top_n"] == 3


@pytest.mark.parametrize("value", ["three", None, {}, []])
def test_retrieval_top_n_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError) as excinfo:
        policy_improve._validate_policy_json(_base_payload(value), _constraints())
    message = str(excinfo.value)
    assert "retrieval_top_n" in message
    assert "type=" in message


def test_top_k_clamps_and_reports_types() -> None:
    payload = _base_payload(3)
    payload["sampling_train"]["top_k"] = 999
    validated = policy_improve._validate_policy_json(payload, _constraints())
    assert validated["sampling_train"]["top_k"] == 200
