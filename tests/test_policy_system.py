from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest

from recursive_slm_lab.memory import connect, init_db, register_policy, set_active_policy, get_active_policy
from recursive_slm_lab.policy import Policy, SamplingConfig, ConsolidationConfig, DEFAULT_POLICY
from recursive_slm_lab.loop.prompts import build_prompt
from recursive_slm_lab.memory.retrieval import retrieve_memory
from recursive_slm_lab.eval.gating import evaluate_policy_pass_rate
from recursive_slm_lab.llm.mock import MockBackend
from recursive_slm_lab.verify import runner as verify_runner


def test_policy_db_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.sqlite"
    init_db(db_path)
    conn = connect(db_path)
    policy = Policy(
        prompt_contract="contract",
        prompt_template="{prompt_contract} {task_prompt}",
        retrieval_top_n=2,
        retrieval_min_score=None,
        retrieval_extra_terms_mode="none",
        retrieval_match_mode="and",
        sampling_train=SamplingConfig(k=2, temperature=0.2, top_p=0.9, top_k=50),
        sampling_eval=SamplingConfig(k=1, temperature=0.0, top_p=1.0, top_k=0),
        consolidation=ConsolidationConfig(
            min_evidence=2,
            enable_llm_consolidation=False,
            max_rules=5,
            max_procedures=5,
        ),
    )
    register_policy(conn, "testpol", policy, parent_policy_name="default")
    set_active_policy(conn, "testpol")
    active = get_active_policy(conn)
    conn.close()
    assert active.to_dict() == policy.to_dict()


def test_prompt_uses_policy_override() -> None:
    policy = Policy(
        prompt_contract="ONLY_CODE",
        prompt_template="{prompt_contract}::{task_prompt}::{function_name}::{signature}{memory_blocks}",
        retrieval_top_n=1,
        retrieval_min_score=None,
        retrieval_extra_terms_mode="none",
        retrieval_match_mode="and",
        sampling_train=SamplingConfig(k=2, temperature=0.2, top_p=0.9, top_k=50),
        sampling_eval=SamplingConfig(k=1, temperature=0.0, top_p=1.0, top_k=0),
        consolidation=ConsolidationConfig(
            min_evidence=2,
            enable_llm_consolidation=False,
            max_rules=5,
            max_procedures=5,
        ),
    )
    prompt = build_prompt("do thing", "do_thing", "()", None, None, policy=policy)
    assert "ONLY_CODE::do thing::do_thing::()" in prompt


def test_retrieval_policy_top_n_and_terms(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.sqlite"
    init_db(db_path)
    conn = connect(db_path)
    conn.execute(
        """
        INSERT INTO episodes (task_id, condition, prompt, candidate_code, passed, test_log, created_at, code_hash)
        VALUES ('t1', 'train', 'prompt', 'def foo():\\n    return 1', 1, 'ok', datetime('now'), 'hash')
        """
    )
    conn.commit()
    policy = DEFAULT_POLICY
    result = retrieve_memory(
        conn,
        "",
        policy=policy,
        function_name="foo",
    )
    conn.close()
    assert len(result.hits) <= policy.retrieval_top_n
    assert any("def foo" in hit.text for hit in result.hits)


def test_deterministic_gating_repeats_match() -> None:
    tasks = []
    for idx in range(3):
        tasks.append(
            type("TaskStub", (), {
                "prompt": "Return n plus 1.",
                "function_name": f"add_const_{idx}",
                "signature": "(n)",
                "reference_tests": "import pytest\nfrom solution import *\n\ndef test_example():\n    assert True\n",
                "assert_tests": ["assert True"],
            })()
        )
    backend = MockBackend()
    metrics = evaluate_policy_pass_rate(
        backend,
        tasks,
        DEFAULT_POLICY,
        repeats=3,
        deterministic=True,
        seed=42,
    )
    assert metrics["per_repeat"][0] == metrics["per_repeat"][1] == metrics["per_repeat"][2]


def test_verification_cache_hit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path = tmp_path / "memory.sqlite"
    init_db(db_path)
    calls = {"count": 0}

    def fake_run_in_sandbox(solution_code, test_code, assert_tests=None):
        calls["count"] += 1
        return type(
            "Result",
            (),
            {"passed": True, "stdout": "ok", "stderr": ""},
        )()

    monkeypatch.setattr(verify_runner, "run_in_sandbox", fake_run_in_sandbox)
    verify_runner.verify_candidate("def foo():\n    return 1", "tests", ["assert True"], db_path=str(db_path))
    verify_runner.verify_candidate("def foo():\n    return 1", "tests", ["assert True"], db_path=str(db_path))
    assert calls["count"] == 1
