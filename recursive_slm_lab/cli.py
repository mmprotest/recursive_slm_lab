from __future__ import annotations

from pathlib import Path
from typing import Optional
import random

import typer
from rich import print

from .config import Config
from .llm import MockBackend, OpenAICompatBackend, LocalHFBackend
from .memory import (
    init_db,
    connect,
    consolidate,
    consolidate_llm,
    get_active_adapter,
    list_policies,
    get_policy,
    get_active_policy,
    set_active_policy,
    fetch_seen_task_ids,
    wipe_memory,
)
from .tasks import load_tasks, validate_tasks, split_tasks
from .loop import run_iteration
from .eval import evaluate_conditions, plot_results
from .meta import summarize_failures, propose_policy, evaluate_and_maybe_promote_policy
from .training import (
    train_lora_adapter,
    get_adapters,
    activate_adapter,
    deactivate_adapter,
    train_and_maybe_promote,
    PromotionConfig,
)

app = typer.Typer(help="Recursive SLM Lab CLI")


def _resolve_backend(config: Config) -> object:
    if config.backend == "mock":
        return MockBackend()
    if config.backend == "openai":
        if not config.base_url:
            raise typer.BadParameter("RSLM_BASE_URL is required for openai backend")
        return OpenAICompatBackend(config.base_url, config.model, config.api_key)
    if config.backend == "localhf":
        adapter = None
        conn = connect(config.db_path)
        active = get_active_adapter(conn)
        if active:
            adapter = active[1]
        conn.close()
        if not config.hf_model_path:
            raise typer.BadParameter("RSLM_HF_MODEL_ID is required for localhf backend")
        try:
            return LocalHFBackend(
                config.hf_model_path,
                adapter_path=adapter,
                torch_dtype=config.torch_dtype,
            )
        except RuntimeError as exc:
            print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1)
    return MockBackend()


def _select_unseen_tasks(
    conn,
    tasks,
    task_limit: Optional[int],
    unseen_only: bool,
    seed: int,
) -> list:
    if not unseen_only:
        return tasks[:task_limit] if task_limit is not None else tasks
    seen_ids = fetch_seen_task_ids(conn)
    unseen = [task for task in tasks if task.task_id not in seen_ids]
    rng = random.Random(seed)
    rng.shuffle(unseen)
    if task_limit is None:
        return unseen
    if len(unseen) >= task_limit:
        return unseen[:task_limit]
    print("[yellow]Warning: ran out of unseen tasks, wrapping around.[/yellow]")
    remaining = [task for task in tasks if task.task_id in seen_ids]
    rng.shuffle(remaining)
    needed = task_limit - len(unseen)
    return unseen + remaining[:needed]


@app.command("init-db")
def cli_init_db(db: str = typer.Option(..., help="Path to SQLite DB")) -> None:
    init_db(db)
    print(f"Initialized DB at {db}")


@app.command("seed-tasks")
def cli_seed_tasks(
    regen: bool = typer.Option(False, help="Regenerate the bundled tasks"),
    regen_families: bool = typer.Option(
        False, help="Regenerate tasks using the expanded family generator"
    ),
    count: int = typer.Option(200, help="Number of tasks to generate"),
    out: Optional[Path] = typer.Option(None, help="Optional output path for JSONL"),
) -> None:
    if count < 120:
        raise typer.BadParameter("--count must be at least 120")
    output_path = out or (Path(__file__).parent / "tasks" / "bundled_tasks.jsonl")
    if regen or regen_families or not output_path.exists():
        from .tasks.generator import generate_tasks, write_tasks

        tasks_payload = generate_tasks(count)
        write_tasks(tasks_payload, output_path)
    tasks = load_tasks(output_path)
    validate_tasks(tasks)
    print(f"Seeded and validated {len(tasks)} tasks at {output_path}")


@app.command("run-iteration")
def cli_run_iteration(
    db: str = typer.Option(..., help="Path to SQLite DB"),
    tasks: str = typer.Option("bundled", help="Task source"),
    k: int = typer.Option(8, help="Candidates per task"),
    mode: str = typer.Option("trainpool", help="trainpool or heldout"),
    backend: Optional[str] = typer.Option(None, help="mock|openai|localhf"),
    base_url: Optional[str] = typer.Option(None, help="OpenAI base URL"),
    model: Optional[str] = typer.Option(None, help="Model name"),
    temperature: float = typer.Option(0.2, help="Sampling temperature"),
    max_tokens: int = typer.Option(256, help="Max tokens"),
    top_p: float = typer.Option(0.9, help="Top-p nucleus sampling"),
    top_k: int = typer.Option(50, help="Top-k sampling"),
    memory_enabled: bool = typer.Option(
        False, "--memory-enabled/--no-memory", help="Enable memory retrieval"
    ),
    heldout_size: int = typer.Option(40, help="Heldout size for splitting"),
    task_limit: Optional[int] = typer.Option(None, help="Limit number of tasks"),
    unseen_only: bool = typer.Option(
        True,
        "--unseen-only/--no-unseen-only",
        help="Select unseen train tasks only.",
    ),
    train_seed: int = typer.Option(1337, help="Shuffle seed for train selection"),
    verify_workers: Optional[int] = typer.Option(
        None, help="Parallel verification workers (defaults to RSLM_VERIFY_WORKERS)"
    ),
) -> None:
    config = Config(
        db_path=db,
        backend=backend or Config().backend,
        base_url=base_url or Config().base_url,
        model=model or Config().model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    all_tasks = load_tasks()
    train_pool, heldout = split_tasks(all_tasks, heldout_size=heldout_size)
    selected = train_pool if mode == "trainpool" else heldout

    conn = connect(db)
    policy = get_active_policy(conn)
    if mode == "trainpool":
        selected = _select_unseen_tasks(conn, selected, task_limit, unseen_only, train_seed)
    elif task_limit is not None:
        selected = selected[:task_limit]

    backend_impl = _resolve_backend(config)
    results = run_iteration(
        conn,
        tasks=selected,
        backend=backend_impl,
        k=k,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        memory_enabled=memory_enabled,
        condition=mode,
        policy=policy,
        db_path=db,
        verify_workers=verify_workers,
    )
    conn.close()
    passed = sum(1 for r in results if r.passed)
    print(f"Iteration complete. Passed {passed}/{len(results)} tasks.")


@app.command("consolidate")
def cli_consolidate(
    db: str = typer.Option(..., help="Path to SQLite DB"),
    min_evidence: int = typer.Option(3, help="Minimum evidence count"),
) -> None:
    conn = connect(db)
    consolidate(conn, min_evidence=min_evidence)
    conn.close()
    print("Consolidation complete.")


@app.command("consolidate-llm")
def cli_consolidate_llm(
    db: str = typer.Option(..., help="Path to SQLite DB"),
    backend: str = typer.Option("mock", help="mock|openai|localhf"),
    heldout_size: int = typer.Option(40, help="Heldout size"),
    task_limit: Optional[int] = typer.Option(None, help="Limit number of heldout tasks"),
    sample_episodes: int = typer.Option(80, help="Number of recent passed episodes to sample"),
    max_rules: int = typer.Option(20, help="Maximum combined rules/procedures"),
    min_gain: float = typer.Option(0.01, help="Minimum pass@1 gain to accept"),
) -> None:
    config = Config(db_path=db, backend=backend)
    conn = connect(db)
    tasks = load_tasks()
    _, heldout = split_tasks(tasks, heldout_size=heldout_size)
    if task_limit is not None:
        heldout = heldout[:task_limit]
    backend_impl = _resolve_backend(config)
    report = consolidate_llm(
        conn,
        backend_impl,
        heldout_tasks=heldout,
        sample_episodes=sample_episodes,
        max_rules=max_rules,
        min_gain=min_gain,
        k=1,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
    )
    conn.close()
    if backend == "mock":
        print("[yellow]Warning: mock backend may not generate useful rules.[/yellow]")
    print(report)


@app.command("eval")
def cli_eval(
    db: str = typer.Option(..., help="Path to SQLite DB"),
    backend: str = typer.Option("mock", help="mock|openai|localhf"),
    conditions: str = typer.Option("all", help="all or comma-separated list"),
    k: int = typer.Option(1, help="pass@k"),
    heldout_size: int = typer.Option(40, help="Heldout size"),
    task_limit: Optional[int] = typer.Option(None, help="Limit number of heldout tasks"),
    output: Optional[str] = typer.Option(None, help="Optional output JSON"),
    max_tokens: int = typer.Option(256, help="Max tokens"),
    temperature: float = typer.Option(0.2, help="Sampling temperature"),
    top_p: float = typer.Option(0.9, help="Top-p nucleus sampling"),
    top_k: int = typer.Option(50, help="Top-k sampling"),
    policy_name: Optional[str] = typer.Option(None, help="Evaluate a specific policy by name"),
    deterministic: bool = typer.Option(True, "--deterministic/--stochastic", help="Deterministic eval"),
    repeats: int = typer.Option(1, help="Repeat evaluations for stability"),
    seed: int = typer.Option(1337, help="Seed for deterministic eval"),
) -> None:
    if conditions == "all":
        if backend == "mock":
            condition_list = ["baseline", "memory", "semantic", "memory_learning"]
        else:
            condition_list = ["baseline", "memory", "learning", "memory_learning"]
    else:
        condition_list = [c.strip() for c in conditions.split(",") if c.strip()]
    payload = evaluate_conditions(
        db_path=db,
        backend_name=backend,
        conditions=condition_list,
        k=k,
        heldout_size=heldout_size,
        task_limit=task_limit,
        output_path=output,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        policy_name=policy_name,
        repeats=repeats,
        deterministic=deterministic,
        seed=seed,
    )
    if output:
        print(f"Saved results to {output}")
    print(payload)


@app.command("plot")
def cli_plot(
    input: str = typer.Option(..., help="Input results JSON"),
    output: str = typer.Option("artifacts/results.png", help="Output PNG"),
) -> None:
    plot_results(input, output)
    print(f"Saved plot to {output}")


@app.command("train-lora")
def cli_train_lora(
    db: str = typer.Option(..., help="Path to SQLite DB"),
    out: str = typer.Option(..., help="Output adapter directory"),
) -> None:
    config = Config(db_path=db)
    conn = connect(db)
    result = train_lora_adapter(conn, out, config.hf_model_path)
    conn.close()
    if result.trained:
        print(f"Adapter saved at {result.adapter_path}")
    else:
        print(result.message)
        raise typer.Exit(code=1)


@app.command("train-and-promote")
def cli_train_and_promote(
    db: str = typer.Option(..., help="Path to SQLite DB"),
    out: str = typer.Option(..., help="Output adapter directory"),
    heldout_size: int = typer.Option(40, help="Heldout size"),
    heldout_limit: Optional[int] = typer.Option(None, help="Limit heldout tasks"),
    regression_size: int = typer.Option(25, help="Regression task count"),
    k: int = typer.Option(1, help="pass@k"),
    min_improvement: float = typer.Option(0.02, help="Absolute pass@1 gain required"),
    max_regression_drop: float = typer.Option(0.0, help="Allowed regression drop"),
    backend: Optional[str] = typer.Option(None, help="Backend (must be localhf)"),
    max_tokens: int = typer.Option(256, help="Max tokens"),
    temperature: float = typer.Option(0.0, help="Sampling temperature"),
    top_p: float = typer.Option(0.9, help="Top-p nucleus sampling"),
    top_k: int = typer.Option(50, help="Top-k sampling"),
    deterministic: bool = typer.Option(True, "--deterministic/--stochastic", help="Deterministic gating"),
    repeats: int = typer.Option(3, help="Repeat evaluations for gating"),
    seed: int = typer.Option(1337, help="Seed for deterministic gating"),
) -> None:
    config = Config(
        db_path=db,
        backend=backend or Config().backend,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    if config.backend != "localhf":
        raise typer.BadParameter("train-and-promote requires --backend localhf")
    conn = connect(db)
    result = train_and_maybe_promote(
        conn,
        PromotionConfig(
            out_dir=out,
            heldout_size=heldout_size,
            heldout_limit=heldout_limit,
            regression_size=regression_size,
            k=k,
            min_improvement=min_improvement,
            max_regression_drop=max_regression_drop,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            base_model_path=config.hf_model_path,
            repeats=repeats,
            deterministic=deterministic,
            seed=seed,
        ),
    )
    conn.close()
    print(result.message)
    if not result.promoted and result.decision == "rejected":
        print(result.metrics)


@app.command("policy-list")
def cli_policy_list(db: str = typer.Option(..., help="Path to SQLite DB")) -> None:
    conn = connect(db)
    policies = list_policies(conn)
    conn.close()
    for name, created_at, parent in policies:
        parent_label = parent or "none"
        print(f"{name} (created {created_at}, parent {parent_label})")


@app.command("policy-show")
def cli_policy_show(
    db: str = typer.Option(..., help="Path to SQLite DB"),
    name: str = typer.Option(..., help="Policy name"),
) -> None:
    conn = connect(db)
    policy = get_policy(conn, name)
    conn.close()
    print(policy.to_json())


@app.command("policy-set-active")
def cli_policy_set_active(
    db: str = typer.Option(..., help="Path to SQLite DB"),
    name: str = typer.Option(..., help="Policy name"),
) -> None:
    conn = connect(db)
    set_active_policy(conn, name)
    conn.close()
    print(f"Activated policy {name}")


@app.command("policy-run-meta-iteration")
def cli_policy_run_meta_iteration(
    db: str = typer.Option(..., help="Path to SQLite DB"),
    backend: str = typer.Option("mock", help="mock|openai|localhf"),
    model: Optional[str] = typer.Option(None, help="Model name"),
    base_url: Optional[str] = typer.Option(None, help="OpenAI base URL"),
    seed: int = typer.Option(1337, help="Seed"),
    heldout_size: int = typer.Option(40, help="Heldout size for split"),
    heldout_limit: Optional[int] = typer.Option(None, help="Limit heldout tasks"),
    regression_size: int = typer.Option(25, help="Regression task count"),
    regression_seed: int = typer.Option(1337, help="Regression seed"),
    repeats: int = typer.Option(3, help="Repeat evaluations"),
    min_delta: float = typer.Option(0.01, help="Minimum heldout improvement"),
    max_drop: float = typer.Option(0.0, help="Maximum regression drop"),
    failures: int = typer.Option(50, help="Failure history size to summarize"),
    deterministic: bool = typer.Option(True, "--deterministic/--stochastic", help="Deterministic gating"),
) -> None:
    config = Config(
        db_path=db,
        backend=backend,
        base_url=base_url or Config().base_url,
        model=model or Config().model,
    )
    conn = connect(db)
    backend_impl = _resolve_backend(config)
    current_policy = get_active_policy(conn)
    failure_summary = summarize_failures(conn, limit=failures)
    constraints = {
        "retrieval_top_n": "[0, 10]",
        "temperature": "[0, 1.5]",
        "k": "[1, 16]",
        "top_p": "[0.1, 1.0]",
        "top_k": "[0, 200]",
    }
    proposal = propose_policy(
        backend_impl,
        current_policy=current_policy,
        recent_failures_summary=failure_summary,
        constraints=constraints,
    )
    result = evaluate_and_maybe_promote_policy(
        conn,
        backend_impl,
        candidate_policy=proposal.policy,
        heldout_size=heldout_size,
        heldout_limit=heldout_limit,
        regression_size=regression_size,
        repeats=repeats,
        seed=seed,
        regression_seed=regression_seed,
        deterministic=deterministic,
        min_delta=min_delta,
        max_drop=max_drop,
        notes=failure_summary,
    )
    conn.close()
    print(result.message)
    print(
        {
            "decision": result.decision,
            "candidate": result.candidate_name,
            "metrics": result.metrics,
        }
    )


@app.command("list-adapters")
def cli_list_adapters(db: str = typer.Option(..., help="Path to SQLite DB")) -> None:
    conn = connect(db)
    adapters = get_adapters(conn)
    conn.close()
    for adapter in adapters:
        status = "active" if adapter.active else "inactive"
        print(f"{adapter.name}: {adapter.path} ({status})")


@app.command("set-active-adapter")
def cli_set_active_adapter(
    db: str = typer.Option(..., help="Path to SQLite DB"),
    name: str = typer.Option(..., help="Adapter name"),
) -> None:
    conn = connect(db)
    activate_adapter(conn, name)
    conn.close()
    print(f"Activated adapter {name}")


@app.command("rollback-adapter")
def cli_rollback_adapter(
    db: str = typer.Option(..., help="Path to SQLite DB"),
    name: str = typer.Option(..., help="Adapter name"),
) -> None:
    conn = connect(db)
    deactivate_adapter(conn, name)
    conn.close()
    print(f"Rolled back adapter {name}")


@app.command("wipe-memory")
def cli_wipe_memory(db: str = typer.Option(..., help="Path to SQLite DB")) -> None:
    conn = connect(db)
    wipe_memory(conn)
    conn.close()
    print("Wiped memory tables (episodes, failures, rules, procedures).")
