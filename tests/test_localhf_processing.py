from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest

from recursive_slm_lab.llm import localhf


def test_extract_python_function_code_strips_think_and_fence() -> None:
    text = """
<think>Draft solution</think>
```python
import math

def foo(x):
    return x + 1


def bar(y):
    return y - 1
```
Extra trailing commentary.
"""
    extracted = localhf.extract_python_function_code(text, "foo")
    assert "def foo" in extracted
    assert "def bar" not in extracted
    assert extracted.strip().startswith("import math")


def test_localhf_dependency_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_find_spec(name: str):
        return None

    monkeypatch.setattr(localhf.importlib.util, "find_spec", fake_find_spec)
    with pytest.raises(RuntimeError) as excinfo:
        localhf.LocalHFBackend("fake-model")
    message = str(excinfo.value)
    assert "LocalHF dependencies missing" in message
    assert "pip install -e '.[localhf]'" in message


def test_localhf_generate_kwargs_no_sampling_when_temperature_zero() -> None:
    gen_kwargs = localhf._build_generate_kwargs(
        input_ids=[1, 2, 3],
        attention_mask=[1, 1, 1],
        max_tokens=64,
        temperature=0.0,
        top_p=0.9,
        top_k=50,
    )
    assert gen_kwargs["do_sample"] is False
    assert "temperature" not in gen_kwargs
    assert "top_p" not in gen_kwargs
    assert "top_k" not in gen_kwargs
