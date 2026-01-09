from __future__ import annotations

from dataclasses import dataclass
import math
import random

from ..loop.sampling import generate_candidates
from ..policy import Policy, SamplingConfig
from ..verify import verify_candidate
from ..eval.metrics import compute_pass_rates


@dataclass(frozen=True)
class EvalConfig:
    repeats: int = 3
    deterministic: bool = True
    seed: int = 1337


def evaluate_policy_pass_rate(
    backend,
    tasks,
    policy: Policy,
    repeats: int = 3,
    seed: int = 1337,
    deterministic: bool = True,
    max_tokens: int = 256,
    db_path: str | None = None,
) -> dict:
    repeats = max(1, repeats)
    sampling = _effective_sampling(policy.sampling_eval, deterministic)
    per_repeat: list[float] = []
    for rep in range(repeats):
        _seed_backend(backend, seed, rep, deterministic)
        outcomes: list[list[bool]] = []
        for task in tasks:
            candidates = generate_candidates(
                backend,
                task_prompt=task.prompt,
                function_name=task.function_name,
                signature=task.signature,
                memory_context=None,
                policy=policy,
                k=sampling.k,
                max_tokens=max_tokens,
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                top_k=sampling.top_k,
            )
            task_outcomes: list[bool] = []
            for candidate in candidates:
                verification = verify_candidate(
                    candidate.code,
                    task.reference_tests,
                    task.assert_tests,
                    db_path=db_path,
                )
                task_outcomes.append(verification.passed)
            outcomes.append(task_outcomes)
        pass_rate = compute_pass_rates(outcomes, k=sampling.k).pass_at_1
        per_repeat.append(pass_rate)
    mean = sum(per_repeat) / len(per_repeat)
    stderr, ci95 = _mean_stderr_ci(per_repeat)
    return {
        "mean": mean,
        "per_repeat": per_repeat,
        "stderr": stderr,
        "ci95": ci95,
    }


def _effective_sampling(sampling: SamplingConfig, deterministic: bool) -> SamplingConfig:
    if not deterministic:
        return sampling
    return SamplingConfig(
        k=1,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
    )


def _mean_stderr_ci(values: list[float]) -> tuple[float, float]:
    if len(values) <= 1:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    stderr = math.sqrt(variance / len(values))
    return stderr, 1.96 * stderr


def _seed_backend(backend, seed: int, repeat: int, deterministic: bool) -> None:
    if hasattr(backend, "seed"):
        backend.seed = seed if deterministic else seed + repeat
    random.seed(seed if deterministic else seed + repeat)
