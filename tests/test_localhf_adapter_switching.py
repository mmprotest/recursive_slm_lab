from __future__ import annotations

import sys
import types

from recursive_slm_lab.llm.localhf import LocalHFBackend


class _FakeCuda:
    def is_available(self) -> bool:
        return False

    def empty_cache(self) -> None:
        return None


class _FakeTorch:
    cuda = _FakeCuda()


def test_localhf_reuses_peft_model_with_multi_adapter(monkeypatch) -> None:
    class FakePeftModel:
        from_pretrained_calls = 0

        def __init__(self) -> None:
            self.loaded = []
            self.active = None

        @classmethod
        def from_pretrained(cls, base_model, adapter_path):
            cls.from_pretrained_calls += 1
            return cls()

        def load_adapter(self, adapter_path, adapter_name=None):
            self.loaded.append((adapter_path, adapter_name))

        def set_adapter(self, adapter_name):
            self.active = adapter_name

    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=FakePeftModel))

    backend = LocalHFBackend.__new__(LocalHFBackend)
    backend._base_model = object()
    backend._model = backend._base_model
    backend._peft_model = None
    backend._active_adapter = None
    backend.adapter_path = None
    backend._torch = _FakeTorch()

    backend.set_adapter("path/to/adapter_one")
    assert FakePeftModel.from_pretrained_calls == 1
    first_peft = backend._peft_model

    backend.set_adapter("path/to/adapter_two")
    assert FakePeftModel.from_pretrained_calls == 1
    assert backend._peft_model is first_peft
    assert backend._model is backend._peft_model
