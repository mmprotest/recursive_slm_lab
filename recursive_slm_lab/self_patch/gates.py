from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .models import PatchGateResult, PatchGateStep


_OUTPUT_LIMIT = 4000


def run_patch_gates(
    workdir: Path,
    db_path: str,
    heldout_size: int,
    heldout_limit: int | None,
    regression_size: int,
    seed: int,
) -> PatchGateResult:
    steps: list[PatchGateStep] = []

    def run_step(name: str, command: list[str]) -> PatchGateStep:
        result = subprocess.run(
            command,
            cwd=workdir,
            capture_output=True,
            text=True,
        )
        step = PatchGateStep(
            name=name,
            command=command,
            returncode=result.returncode,
            stdout=_truncate(result.stdout),
            stderr=_truncate(result.stderr),
            passed=result.returncode == 0,
        )
        steps.append(step)
        return step

    compile_step = run_step(
        "compileall",
        [sys.executable, "-m", "compileall", "recursive_slm_lab"],
    )
    if not compile_step.passed:
        return PatchGateResult(passed=False, steps=steps)

    pytest_step = run_step("pytest", [sys.executable, "-m", "pytest", "-q"])
    if not pytest_step.passed:
        return PatchGateResult(passed=False, steps=steps)

    eval_payload: dict | None = None
    eval_step = run_step(
        "robust_eval",
        [
            sys.executable,
            "-c",
            _robust_eval_script(
                db_path=db_path,
                heldout_size=heldout_size,
                heldout_limit=heldout_limit,
                regression_size=regression_size,
                seed=seed,
            ),
        ],
    )
    if eval_step.passed:
        try:
            eval_payload = json.loads(eval_step.stdout)
        except json.JSONDecodeError:
            eval_payload = None
    return PatchGateResult(passed=all(step.passed for step in steps), steps=steps, eval_payload=eval_payload)


def _robust_eval_script(
    db_path: str,
    heldout_size: int,
    heldout_limit: int | None,
    regression_size: int,
    seed: int,
) -> str:
    heldout_limit_value = "None" if heldout_limit is None else str(heldout_limit)
    regression_value = str(regression_size)
    return (
        "import json\n"
        "from recursive_slm_lab.eval.robust import split_for_gating, evaluate_tasks\n"
        "from recursive_slm_lab.llm.mock import MockBackend\n"
        "from recursive_slm_lab.memory import connect, get_active_policy\n"
        f"heldout_size={heldout_size}\n"
        f"heldout_limit={heldout_limit_value}\n"
        f"regression_size={regression_value}\n"
        f"seed={seed}\n"
        f"db_path={json.dumps(db_path)}\n"
        "_, heldout, hidden, _ = split_for_gating(heldout_size)\n"
        "if heldout_limit is not None:\n"
        "    heldout = heldout[:heldout_limit]\n"
        "if regression_size:\n"
        "    hidden = hidden[:regression_size]\n"
        "conn = connect(db_path)\n"
        "policy = get_active_policy(conn)\n"
        "backend = MockBackend()\n"
        "heldout_result = evaluate_tasks(\n"
        "    backend,\n"
        "    heldout,\n"
        "    policy,\n"
        "    memory_enabled=False,\n"
        "    semantic_enabled=False,\n"
        "    k=1,\n"
        "    max_tokens=64,\n"
        "    temperature=0.0,\n"
        "    top_p=1.0,\n"
        "    top_k=0,\n"
        "    deterministic=True,\n"
        "    seed=seed,\n"
        "    repeats=3,\n"
        "    conn=conn,\n"
        ")\n"
        "hidden_result = evaluate_tasks(\n"
        "    backend,\n"
        "    hidden,\n"
        "    policy,\n"
        "    memory_enabled=False,\n"
        "    semantic_enabled=False,\n"
        "    k=1,\n"
        "    max_tokens=64,\n"
        "    temperature=0.0,\n"
        "    top_p=1.0,\n"
        "    top_k=0,\n"
        "    deterministic=True,\n"
        "    seed=seed,\n"
        "    repeats=3,\n"
        "    conn=conn,\n"
        ")\n"
        "conn.close()\n"
        "payload = {\n"
        "    'heldout': {'pass_at_1': heldout_result.pass_at_1, 'pass_at_k': heldout_result.pass_at_k},\n"
        "    'hidden': {'pass_at_1': hidden_result.pass_at_1, 'pass_at_k': hidden_result.pass_at_k},\n"
        "}\n"
        "print(json.dumps(payload))\n"
    )


def _truncate(value: str) -> str:
    if value is None:
        return ""
    if len(value) <= _OUTPUT_LIMIT:
        return value
    return value[:_OUTPUT_LIMIT] + "\n...[truncated]..."
