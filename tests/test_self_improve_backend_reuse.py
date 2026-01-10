from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from recursive_slm_lab.llm import MockBackend
from recursive_slm_lab.loop import self_improve
from recursive_slm_lab.memory import init_db


def test_self_improve_reuses_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSLM_FAST_VERIFY", "1")
    monkeypatch.setenv("RSLM_VERIFY_WORKERS", "1")

    db_path = tmp_path / "memory.sqlite"
    init_db(db_path)
    artifacts_dir = tmp_path / "artifacts_self_improve"

    calls = {"count": 0}

    def fake_resolve_backend(config, conn):
        calls["count"] += 1
        return MockBackend()

    monkeypatch.setattr(self_improve, "_resolve_backend", fake_resolve_backend)

    self_improve.run_self_improve(
        db_path=str(db_path),
        tasks_source="bundled",
        cycles=2,
        train_k=1,
        train_limit=1,
        heldout_size=5,
        backend="mock",
        base_url=None,
        model=None,
        max_tokens=32,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        memory_enabled=False,
        unseen_only=True,
        train_seed=1337,
        enable_policy_improve=False,
        enable_adapter_train=False,
        artifacts_dir=artifacts_dir,
        verify_mode="local",
        seed=1337,
        enable_self_patch=False,
        self_patch_callback=None,
    )

    assert calls["count"] == 1
