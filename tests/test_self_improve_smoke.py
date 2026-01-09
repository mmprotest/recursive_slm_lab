from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from recursive_slm_lab.loop import run_self_improve
from recursive_slm_lab.memory import init_db


def test_self_improve_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSLM_FAST_VERIFY", "1")
    monkeypatch.setenv("RSLM_VERIFY_WORKERS", "1")

    db_path = tmp_path / "memory.sqlite"
    init_db(db_path)
    artifacts_dir = tmp_path / "artifacts_self_improve"

    summary = run_self_improve(
        db_path=str(db_path),
        tasks_source="bundled",
        cycles=1,
        train_k=2,
        train_limit=2,
        heldout_size=10,
        backend="mock",
        base_url=None,
        model=None,
        max_tokens=64,
        temperature=0.0,
        top_p=0.9,
        top_k=50,
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

    assert summary["cycles"] == 1
    artifact_path = artifacts_dir / "cycle_001.json"
    assert artifact_path.exists()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    for key in ["policy", "adapter", "train", "eval", "robust_eval", "started_at", "completed_at"]:
        assert key in payload
