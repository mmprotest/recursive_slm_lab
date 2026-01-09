from __future__ import annotations

from dataclasses import dataclass
import random

from ..config import Config
from ..llm import MockBackend, OpenAICompatBackend, LocalHFBackend
from ..loop.sampling import generate_candidates
from ..memory import connect, retrieve_memory, get_active_adapter, get_active_policy, get_policy
from ..policy import Policy, SamplingConfig, DEFAULT_POLICY
from ..tasks import Task, load_tasks, load_hidden_tasks, split_tasks
from ..verify import verify_candidate
from .metrics import compute_pass_rates
from .strength import find_weak_tasks


@dataclass(frozen=True)
class SplitResult:
    pass_at_1: float
    pass_at_k: float
    per_task_pass_at_1: list[int]
    per_task_pass_at_k: list[int]


@dataclass(frozen=True)
class DeltaCI:
    mean: float
    ci_low: float
    ci_high: float


def _build_backend(config: Config, condition: str, adapter_path: str | None) -> tuple[object, str]:
    if config.backend == "mock":
        return MockBackend(), "mock"
    if config.backend == "openai":
        if not config.base_url:
            raise ValueError("RSLM_BASE_URL required for openai backend")
        return OpenAICompatBackend(config.base_url, config.model, config.api_key), "openai"
    if config.backend == "localhf":
        if not config.hf_model_path:
            raise ValueError("RSLM_HF_MODEL_ID required for localhf backend")
        adapter = adapter_path if condition in {"learning", "memory_learning"} else None
        return LocalHFBackend(
            config.hf_model_path,
            adapter_path=adapter,
            torch_dtype=config.torch_dtype,
        ), "localhf"
    return MockBackend(), "mock"


def _resolve_sampling(
    policy: Policy,
    k: int,
    temperature: float,
    top_p: float,
    top_k: int,
    deterministic: bool,
) -> SamplingConfig:
    if deterministic:
        return policy.sampling_eval
    return SamplingConfig(k=k, temperature=temperature, top_p=top_p, top_k=top_k)


def _seed_backend(backend, seed: int, repeat: int, deterministic: bool) -> None:
    if hasattr(backend, "seed"):
        backend.seed = seed if deterministic else seed + repeat
    random.seed(seed if deterministic else seed + repeat)


def _per_task_pass_vectors(outcomes: list[list[bool]], k: int) -> SplitResult:
    pass_at_1_vec = [1 if outcomes[i] and outcomes[i][0] else 0 for i in range(len(outcomes))]
    pass_at_k_vec = [1 if any(outcomes[i][:k]) else 0 for i in range(len(outcomes))]
    pass_at_1 = sum(pass_at_1_vec) / len(pass_at_1_vec) if pass_at_1_vec else 0.0
    pass_at_k = sum(pass_at_k_vec) / len(pass_at_k_vec) if pass_at_k_vec else 0.0
    return SplitResult(
        pass_at_1=pass_at_1,
        pass_at_k=pass_at_k,
        per_task_pass_at_1=pass_at_1_vec,
        per_task_pass_at_k=pass_at_k_vec,
    )


def paired_bootstrap_ci(
    baseline: list[int],
    candidate: list[int],
    num_samples: int = 1000,
    seed: int = 1337,
) -> DeltaCI:
    if len(baseline) != len(candidate):
        raise ValueError("Baseline and candidate lists must have equal length")
    if not baseline:
        return DeltaCI(mean=0.0, ci_low=0.0, ci_high=0.0)
    rng = random.Random(seed)
    deltas = []
    indices = list(range(len(baseline)))
    for _ in range(num_samples):
        sample_indices = [rng.choice(indices) for _ in indices]
        sample_delta = sum(candidate[i] - baseline[i] for i in sample_indices) / len(indices)
        deltas.append(sample_delta)
    deltas.sort()
    mean_delta = sum(candidate) / len(candidate) - sum(baseline) / len(baseline)
    low_idx = int(0.025 * (len(deltas) - 1))
    high_idx = int(0.975 * (len(deltas) - 1))
    return DeltaCI(mean=mean_delta, ci_low=deltas[low_idx], ci_high=deltas[high_idx])


def evaluate_tasks(
    backend,
    tasks: list[Task],
    policy: Policy,
    memory_enabled: bool,
    semantic_enabled: bool,
    k: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    deterministic: bool,
    seed: int,
    repeats: int,
    conn,
) -> SplitResult:
    sampling = _resolve_sampling(policy, k, temperature, top_p, top_k, deterministic)
    outcomes: list[list[bool]] = []
    repeats = max(1, repeats)
    for repeat in range(repeats):
        _seed_backend(backend, seed, repeat, deterministic)
        for task in tasks:
            memory_context = None
            if memory_enabled or semantic_enabled:
                memory_context = retrieve_memory(
                    conn,
                    task.prompt,
                    policy=policy,
                    function_name=task.function_name,
                )
                if memory_enabled and not semantic_enabled and memory_context:
                    memory_context = memory_context.filter_sources({"episode"})
                if semantic_enabled and not memory_enabled and memory_context:
                    memory_context = memory_context.filter_sources({"rule", "procedure"})
            candidates = generate_candidates(
                backend,
                task_prompt=task.prompt,
                function_name=task.function_name,
                signature=task.signature,
                memory_context=memory_context,
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
                    conn=conn,
                )
                task_outcomes.append(verification.passed)
            outcomes.append(task_outcomes)
    return _per_task_pass_vectors(outcomes, k=sampling.k)


def robust_eval_conditions(
    db_path: str,
    backend_name: str,
    conditions: list[str],
    k: int,
    heldout_size: int,
    task_limit: int | None,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    policy_name: str | None,
    repeats: int,
    deterministic: bool,
    seed: int,
) -> dict:
    config = Config(db_path=db_path, backend=backend_name)
    tasks = load_tasks(include_generated=config.include_generated_tasks)
    hidden_tasks = load_hidden_tasks()
    _, heldout, hidden = split_tasks(
        tasks,
        heldout_size=heldout_size,
        hidden_tasks=hidden_tasks,
        seed=seed,
    )
    if task_limit is not None:
        heldout = heldout[:task_limit]
        hidden = hidden[:task_limit]
    conn = connect(db_path)
    policy = get_policy(conn, policy_name) if policy_name else get_active_policy(conn)
    active_adapter = get_active_adapter(conn)
    weak_hidden_ids = set(find_weak_tasks(hidden))
    hidden_strong = [task for task in hidden if task.task_id not in weak_hidden_ids]

    results = []
    for condition in conditions:
        adapter_path = active_adapter[1] if active_adapter else None
        backend, backend_label = _build_backend(config, condition, adapter_path)
        effective_condition = condition
        if backend_label == "mock" and condition == "learning":
            effective_condition = "semantic"
        memory_enabled = effective_condition in {"memory", "memory_learning"}
        semantic_enabled = effective_condition in {"semantic", "memory_learning"}
        learning_enabled = condition in {"learning", "memory_learning"}
        if learning_enabled and backend_label != "localhf":
            memory_enabled = False
            semantic_enabled = False
        heldout_result = evaluate_tasks(
            backend,
            heldout,
            policy,
            memory_enabled,
            semantic_enabled,
            k,
            max_tokens,
            temperature,
            top_p,
            top_k,
            deterministic,
            seed,
            repeats,
            conn,
        )
        hidden_result = evaluate_tasks(
            backend,
            hidden_strong,
            policy,
            memory_enabled,
            semantic_enabled,
            k,
            max_tokens,
            temperature,
            top_p,
            top_k,
            deterministic,
            seed,
            repeats,
            conn,
        )
        results.append(
            {
                "condition": effective_condition,
                "heldout": {
                    "pass_at_1": heldout_result.pass_at_1,
                    "pass_at_k": heldout_result.pass_at_k,
                },
                "hidden": {
                    "pass_at_1": hidden_result.pass_at_1,
                    "pass_at_k": hidden_result.pass_at_k,
                },
            }
        )
    conn.close()
    return {
        "conditions": results,
        "weak_hidden_task_ids": sorted(weak_hidden_ids),
        "splits": {
            "heldout": [task.task_id for task in heldout],
            "hidden": [task.task_id for task in hidden_strong],
        },
    }


def gate_delta(
    baseline: SplitResult,
    candidate: SplitResult,
    min_gain: float,
    max_regress: float,
    seed: int = 1337,
) -> dict:
    delta = paired_bootstrap_ci(
        baseline.per_task_pass_at_1,
        candidate.per_task_pass_at_1,
        seed=seed,
    )
    promote = delta.mean >= min_gain and delta.ci_low >= 0.0 and delta.mean >= -max_regress
    return {
        "delta": delta,
        "min_gain": min_gain,
        "max_regress": max_regress,
        "promote": promote,
    }


def evaluate_policy_pair(
    backend,
    tasks: list[Task],
    baseline: Policy,
    candidate: Policy,
    max_tokens: int,
    deterministic: bool,
    seed: int,
    repeats: int,
    conn,
) -> tuple[SplitResult, SplitResult]:
    base_result = evaluate_tasks(
        backend,
        tasks,
        baseline,
        memory_enabled=False,
        semantic_enabled=False,
        k=baseline.sampling_eval.k,
        max_tokens=max_tokens,
        temperature=baseline.sampling_eval.temperature,
        top_p=baseline.sampling_eval.top_p,
        top_k=baseline.sampling_eval.top_k,
        deterministic=deterministic,
        seed=seed,
        repeats=repeats,
        conn=conn,
    )
    cand_result = evaluate_tasks(
        backend,
        tasks,
        candidate,
        memory_enabled=False,
        semantic_enabled=False,
        k=candidate.sampling_eval.k,
        max_tokens=max_tokens,
        temperature=candidate.sampling_eval.temperature,
        top_p=candidate.sampling_eval.top_p,
        top_k=candidate.sampling_eval.top_k,
        deterministic=deterministic,
        seed=seed,
        repeats=repeats,
        conn=conn,
    )
    return base_result, cand_result


def evaluate_adapter_pair(
    baseline_backend,
    candidate_backend,
    tasks: list[Task],
    k: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    deterministic: bool,
    seed: int,
    repeats: int,
    conn,
) -> tuple[SplitResult, SplitResult]:
    policy_payload = DEFAULT_POLICY.to_dict()
    policy_payload["retrieval_top_n"] = 0
    policy_payload["sampling_train"] = {"k": k, "temperature": temperature, "top_p": top_p, "top_k": top_k}
    policy_payload["sampling_eval"] = {"k": k, "temperature": temperature, "top_p": top_p, "top_k": top_k}
    dummy_policy = Policy.from_dict(policy_payload)
    base_result = evaluate_tasks(
        baseline_backend,
        tasks,
        dummy_policy,
        memory_enabled=False,
        semantic_enabled=False,
        k=k,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        deterministic=deterministic,
        seed=seed,
        repeats=repeats,
        conn=conn,
    )
    cand_result = evaluate_tasks(
        candidate_backend,
        tasks,
        dummy_policy,
        memory_enabled=False,
        semantic_enabled=False,
        k=k,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        deterministic=deterministic,
        seed=seed,
        repeats=repeats,
        conn=conn,
    )
    return base_result, cand_result


def split_for_gating(heldout_size: int) -> tuple[list[Task], list[Task], list[Task], list[str]]:
    tasks = load_tasks(include_generated=Config().include_generated_tasks)
    hidden_tasks = load_hidden_tasks()
    train_pool, heldout, hidden = split_tasks(tasks, heldout_size=heldout_size, hidden_tasks=hidden_tasks)
    weak_hidden_ids = find_weak_tasks(hidden)
    hidden_strong = [task for task in hidden if task.task_id not in set(weak_hidden_ids)]
    return train_pool, heldout, hidden_strong, weak_hidden_ids
