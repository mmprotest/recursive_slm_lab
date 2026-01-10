from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

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
    list_rules,
    list_procedures,
    activate_rule,
    deactivate_rule,
    rollback_rules,
    activate_procedure,
    deactivate_procedure,
    rollback_procedures,
)
from .tasks import load_tasks, load_hidden_tasks, validate_tasks, split_tasks
from .loop import run_iteration, run_self_improve
from .loop.scheduler import schedule_tasks
from .eval import evaluate_conditions, plot_results, find_weak_tasks
from .eval.robust import robust_eval_conditions
from .meta import summarize_failures, propose_policy, evaluate_and_maybe_promote_policy
from .meta.curriculum import mine_curriculum
from .training import (
    train_lora_adapter,
    get_adapters,
    activate_adapter,
    deactivate_adapter,
    train_and_maybe_promote,
    PromotionConfig,
)
from .self_patch import run_self_patch
from .util import git_commit_hash, write_manifest, setup_logging

app = typer.Typer(help="Recursive SLM Lab CLI")
rules_app = typer.Typer(help="Manage semantic rules")
procedures_app = typer.Typer(help="Manage procedures")
app.add_typer(rules_app, name="rules")
app.add_typer(procedures_app, name="procedures")


@app.callback()
def main(
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Log level (DEBUG, INFO, WARNING, ERROR)",
        case_sensitive=False,
    ),
) -> None:
    setup_logging(log_level)


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
        return schedule_tasks(conn, tasks, task_limit, seed)
    seen_ids = fetch_seen_task_ids(conn)
    unseen = [task for task in tasks if task.task_id not in seen_ids]
    if task_limit is None:
        return schedule_tasks(conn, unseen, None, seed)
    if len(unseen) >= task_limit:
        return schedule_tasks(conn, unseen, task_limit, seed)
    print("[yellow]Warning: ran out of unseen tasks, wrapping around.[/yellow]")
    remaining = [task for task in tasks if task.task_id in seen_ids]
    needed = task_limit - len(unseen)
    scheduled_remaining = schedule_tasks(conn, remaining, needed, seed)
    return unseen + scheduled_remaining


def _load_latest_cycle_artifact(artifacts_dir: Path) -> dict | None:
    if not artifacts_dir.exists():
        return None
    candidates = sorted(artifacts_dir.glob("cycle_*.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        return None
    latest = candidates[-1]
    try:
        return json.loads(latest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


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


@app.command("seed-hidden-tasks")
def cli_seed_hidden_tasks(
    count: int = typer.Option(40, help="Number of hidden tasks to generate"),
    out: Optional[Path] = typer.Option(None, help="Optional output path for JSONL"),
) -> None:
    if count < 12:
        raise typer.BadParameter("--count must be at least 12")
    output_path = out or (Path(__file__).parent / "tasks" / "hidden_tasks.jsonl")
    from .tasks.generator import generate_constant_tasks, write_tasks

    add_count = count // 2
    mul_count = count - add_count
    add_values = list(range(120, 120 + add_count))
    mul_values = list(range(40, 40 + mul_count))
    tasks_payload = generate_constant_tasks(
        add_values,
        mul_values,
        difficulty=2,
        add_range=300,
        mul_range=200,
    )
    write_tasks(tasks_payload, output_path)
    tasks = load_tasks(output_path)
    validate_tasks(tasks)
    print(f"Seeded and validated {len(tasks)} hidden tasks at {output_path}")


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
    memory_enabled: Optional[bool] = typer.Option(
        None,
        "--memory-enabled",
        flag_value=True,
        help="Enable memory retrieval",
    ),
    no_memory: bool = typer.Option(
        False,
        "--no-memory",
        help="Disable memory retrieval",
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
    seed: int = typer.Option(1337, help="Seed for candidate sampling"),
) -> None:
    resolved_memory = False
    if no_memory:
        resolved_memory = False
    elif memory_enabled is not None:
        resolved_memory = memory_enabled

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

    all_tasks = load_tasks(include_generated=config.include_generated_tasks)
    hidden_tasks = load_hidden_tasks()
    train_pool, heldout, _ = split_tasks(
        all_tasks,
        heldout_size=heldout_size,
        hidden_tasks=hidden_tasks,
    )
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
        memory_enabled=resolved_memory,
        condition=mode,
        policy=policy,
        db_path=db,
        verify_workers=verify_workers,
        seed=seed,
    )
    conn.close()
    passed = sum(1 for r in results if r.passed)
    print(f"Iteration complete. Passed {passed}/{len(results)} tasks.")
    manifest_payload = {
        "command": "run-iteration",
        "config": {
            "backend": config.backend,
            "model": config.model,
            "base_url": config.base_url,
            "memory_enabled": resolved_memory,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
        },
        "git_commit": git_commit_hash(),
        "task_ids": [task.task_id for task in selected],
        "results": {"passed": passed, "total": len(results)},
    }
    write_manifest("run-iteration", manifest_payload)


@app.command("self-improve")
def cli_self_improve(
    db: str = typer.Option(..., help="Path to SQLite DB"),
    tasks: str = typer.Option("bundled", help="Task source"),
    cycles: int = typer.Option(3, help="Number of improvement cycles"),
    train_k: int = typer.Option(2, help="Candidates per task"),
    train_limit: int = typer.Option(25, help="Train tasks per cycle"),
    heldout_size: int = typer.Option(40, help="Heldout size"),
    backend: Optional[str] = typer.Option(None, help="mock|openai|localhf"),
    base_url: Optional[str] = typer.Option(None, help="OpenAI base URL"),
    model: Optional[str] = typer.Option(None, help="Model name"),
    max_tokens: int = typer.Option(256, help="Max tokens"),
    temperature: float = typer.Option(0.2, help="Sampling temperature"),
    top_p: float = typer.Option(0.9, help="Top-p nucleus sampling"),
    top_k: int = typer.Option(50, help="Top-k sampling"),
    memory_enabled: Optional[bool] = typer.Option(
        None,
        "--memory-enabled",
        flag_value=True,
        help="Enable memory retrieval",
    ),
    no_memory: bool = typer.Option(False, "--no-memory", help="Disable memory retrieval"),
    unseen_only: bool = typer.Option(
        True,
        "--unseen-only/--no-unseen-only",
        help="Select unseen train tasks only.",
    ),
    enable_policy_improve: Optional[bool] = typer.Option(
        None,
        "--enable-policy-improve/--disable-policy-improve",
        help="Enable policy improvement during the cycle",
    ),
    enable_adapter_train: Optional[bool] = typer.Option(
        None,
        "--enable-adapter-train/--disable-adapter-train",
        help="Enable adapter training + promotion (localhf only)",
    ),
    enable_self_patch: Optional[bool] = typer.Option(
        None,
        "--enable-self-patch/--disable-self-patch",
        help="Enable self-patching at the end of a cycle",
    ),
    artifacts_dir: str = typer.Option("artifacts_self_improve", help="Artifact output directory"),
    verify_mode: Optional[str] = typer.Option(None, help="Override verification mode (local|docker)"),
    seed: int = typer.Option(1337, help="Seed"),
) -> None:
    resolved_memory = False
    if no_memory:
        resolved_memory = False
    elif memory_enabled is not None:
        resolved_memory = memory_enabled

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

    resolved_policy_improve = enable_policy_improve
    if resolved_policy_improve is None:
        resolved_policy_improve = config.backend != "mock"

    resolved_adapter_train = enable_adapter_train
    if resolved_adapter_train is None:
        resolved_adapter_train = config.backend == "localhf" and bool(config.hf_model_path)
    if resolved_adapter_train and config.backend != "localhf":
        raise typer.BadParameter("--enable-adapter-train requires --backend localhf")

    resolved_self_patch = enable_self_patch
    if resolved_self_patch is None:
        resolved_self_patch = False
    if resolved_self_patch and config.backend == "mock":
        raise typer.BadParameter("--enable-self-patch requires a non-mock backend")

    backend_impl = _resolve_backend(config)

    def _run_self_patch(context: dict) -> None:
        run_self_patch(
            backend=backend_impl,
            repo_root=Path.cwd(),
            db_path=db,
            context=context,
            artifacts_dir=Path("artifacts_self_patch"),
            heldout_size=heldout_size,
            heldout_limit=None,
            regression_size=25,
            seed=seed,
            apply_patch=True,
        )

    summary = run_self_improve(
        db_path=db,
        tasks_source=tasks,
        cycles=cycles,
        train_k=train_k,
        train_limit=train_limit,
        heldout_size=heldout_size,
        backend=config.backend,
        base_url=config.base_url,
        model=config.model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        memory_enabled=resolved_memory,
        unseen_only=unseen_only,
        train_seed=seed,
        enable_policy_improve=resolved_policy_improve,
        enable_adapter_train=resolved_adapter_train,
        artifacts_dir=Path(artifacts_dir),
        verify_mode=verify_mode,
        seed=seed,
        enable_self_patch=resolved_self_patch,
        self_patch_callback=_run_self_patch if resolved_self_patch else None,
    )
    print(summary)


@app.command("self-patch")
def cli_self_patch(
    db: str = typer.Option(..., help="Path to SQLite DB"),
    backend: str = typer.Option("openai", help="mock|openai|localhf"),
    base_url: Optional[str] = typer.Option(None, help="OpenAI base URL"),
    model: Optional[str] = typer.Option(None, help="Model name"),
    cycles: int = typer.Option(1, help="Number of patch attempts"),
    context_source: str = typer.Option(
        "latest_cycle_artifact", help="Context source (latest_cycle_artifact)"
    ),
    heldout_size: int = typer.Option(40, help="Heldout size"),
    heldout_limit: Optional[int] = typer.Option(None, help="Limit heldout tasks"),
    regression_size: int = typer.Option(25, help="Regression task count"),
    seed: int = typer.Option(1337, help="Seed"),
    apply_patch: bool = typer.Option(True, "--apply/--no-apply", help="Apply patch on pass"),
) -> None:
    if backend == "mock":
        raise typer.BadParameter("self-patch requires a non-mock backend")
    config = Config(
        db_path=db,
        backend=backend,
        base_url=base_url or Config().base_url,
        model=model or Config().model,
    )
    backend_impl = _resolve_backend(config)

    context: dict = {"context_source": context_source}
    if context_source == "latest_cycle_artifact":
        artifact = _load_latest_cycle_artifact(Path("artifacts_self_improve"))
        if artifact:
            context["latest_cycle_artifact"] = artifact

    conn = connect(db)
    context["failure_summary"] = summarize_failures(conn, limit=50)
    conn.close()

    for _ in range(cycles):
        result = run_self_patch(
            backend=backend_impl,
            repo_root=Path.cwd(),
            db_path=db,
            context=context,
            artifacts_dir=Path("artifacts_self_patch"),
            heldout_size=heldout_size,
            heldout_limit=heldout_limit,
            regression_size=regression_size,
            seed=seed,
            apply_patch=apply_patch,
        )
        print(result.message)


@app.command("consolidate")
def cli_consolidate(
    db: str = typer.Option(..., help="Path to SQLite DB"),
    min_evidence: int = typer.Option(3, help="Minimum evidence count"),
) -> None:
    conn = connect(db)
    eval_snapshot = {
        "source": "cli",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    consolidate(conn, min_evidence=min_evidence, eval_snapshot=eval_snapshot)
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
    max_hidden_regress: float = typer.Option(0.0, help="Max hidden regression tolerance"),
) -> None:
    config = Config(db_path=db, backend=backend)
    conn = connect(db)
    tasks = load_tasks(include_generated=config.include_generated_tasks)
    hidden_tasks = load_hidden_tasks()
    _, heldout, hidden = split_tasks(
        tasks,
        heldout_size=heldout_size,
        hidden_tasks=hidden_tasks,
    )
    if task_limit is not None:
        heldout = heldout[:task_limit]
    hidden = hidden[:task_limit] if task_limit is not None else hidden
    weak_hidden_ids = set(find_weak_tasks(hidden))
    hidden = [task for task in hidden if task.task_id not in weak_hidden_ids]
    backend_impl = _resolve_backend(config)
    report = consolidate_llm(
        conn,
        backend_impl,
        heldout_tasks=heldout,
        hidden_tasks=hidden,
        sample_episodes=sample_episodes,
        max_rules=max_rules,
        min_gain=min_gain,
        max_hidden_regress=max_hidden_regress,
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


@app.command("mine-curriculum")
def cli_mine_curriculum(
    db: str = typer.Option(..., help="Path to SQLite DB"),
    out: Path = typer.Option("artifacts/generated_tasks.jsonl", help="Output JSONL path"),
    max_new: int = typer.Option(50, help="Maximum new tasks to generate"),
    seed: int = typer.Option(1337, help="Seed for sampling"),
) -> None:
    conn = connect(db)
    report = mine_curriculum(conn, out, max_new=max_new, seed=seed)
    conn.close()
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


@app.command("eval-robust")
def cli_eval_robust(
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
    payload = robust_eval_conditions(
        db_path=db,
        backend_name=backend,
        conditions=condition_list,
        k=k,
        heldout_size=heldout_size,
        task_limit=task_limit,
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
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved results to {output}")
    write_manifest(
        "eval-robust",
        {
            "command": "eval-robust",
            "config": {
                "backend": backend,
                "model": Config().model,
                "policy": policy_name or "active",
                "deterministic": deterministic,
                "repeats": repeats,
            },
            "git_commit": git_commit_hash(),
            "splits": payload.get("splits", {}),
            "metrics": payload.get("conditions", []),
            "weak_hidden_task_ids": payload.get("weak_hidden_task_ids", []),
        },
    )
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
    noise_band: float = typer.Option(0.005, help="Noise band for heldout improvements"),
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
            noise_band=noise_band,
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
    noise_band: float = typer.Option(0.005, help="Noise band for heldout improvements"),
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
        noise_band=noise_band,
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


@rules_app.command("list")
def cli_rules_list(
    db: str = typer.Option(..., help="Path to SQLite DB"),
    all: bool = typer.Option(False, "--all", help="Include inactive rules"),
) -> None:
    conn = connect(db)
    rules = list_rules(conn, include_inactive=all)
    conn.close()
    for rule in rules:
        status = "active" if rule.active else "inactive"
        print(f"{rule.rule_id} {rule.key} ({status}) {rule.text}")


@rules_app.command("activate")
def cli_rules_activate(
    db: str = typer.Option(..., help="Path to SQLite DB"),
    rule_id: int = typer.Option(..., help="Rule id to activate"),
) -> None:
    conn = connect(db)
    activate_rule(conn, rule_id)
    conn.close()
    print(f"Activated rule {rule_id}")


@rules_app.command("deactivate")
def cli_rules_deactivate(
    db: str = typer.Option(..., help="Path to SQLite DB"),
    rule_id: int = typer.Option(..., help="Rule id to deactivate"),
) -> None:
    conn = connect(db)
    deactivate_rule(conn, rule_id)
    conn.close()
    print(f"Deactivated rule {rule_id}")


@rules_app.command("rollback")
def cli_rules_rollback(
    db: str = typer.Option(..., help="Path to SQLite DB"),
    to_timestamp: Optional[str] = typer.Option(None, help="Rollback to timestamp (ISO format)"),
    to_rule_id: Optional[int] = typer.Option(None, help="Rollback to rule id"),
) -> None:
    conn = connect(db)
    rollback_rules(conn, to_timestamp=to_timestamp, to_rule_id=to_rule_id)
    conn.close()
    print("Rolled back rules")


@procedures_app.command("list")
def cli_procedures_list(
    db: str = typer.Option(..., help="Path to SQLite DB"),
    all: bool = typer.Option(False, "--all", help="Include inactive procedures"),
) -> None:
    conn = connect(db)
    procedures = list_procedures(conn, include_inactive=all)
    conn.close()
    for proc in procedures:
        status = "active" if proc.active else "inactive"
        print(f"{proc.procedure_id} {proc.pattern} ({status}) {proc.text}")


@procedures_app.command("activate")
def cli_procedures_activate(
    db: str = typer.Option(..., help="Path to SQLite DB"),
    procedure_id: int = typer.Option(..., help="Procedure id to activate"),
) -> None:
    conn = connect(db)
    activate_procedure(conn, procedure_id)
    conn.close()
    print(f"Activated procedure {procedure_id}")


@procedures_app.command("deactivate")
def cli_procedures_deactivate(
    db: str = typer.Option(..., help="Path to SQLite DB"),
    procedure_id: int = typer.Option(..., help="Procedure id to deactivate"),
) -> None:
    conn = connect(db)
    deactivate_procedure(conn, procedure_id)
    conn.close()
    print(f"Deactivated procedure {procedure_id}")


@procedures_app.command("rollback")
def cli_procedures_rollback(
    db: str = typer.Option(..., help="Path to SQLite DB"),
    to_timestamp: Optional[str] = typer.Option(None, help="Rollback to timestamp (ISO format)"),
    to_procedure_id: Optional[int] = typer.Option(None, help="Rollback to procedure id"),
) -> None:
    conn = connect(db)
    rollback_procedures(conn, to_timestamp=to_timestamp, to_procedure_id=to_procedure_id)
    conn.close()
    print("Rolled back procedures")


@app.command("wipe-memory")
def cli_wipe_memory(db: str = typer.Option(..., help="Path to SQLite DB")) -> None:
    conn = connect(db)
    wipe_memory(conn)
    conn.close()
    print("Wiped memory tables (episodes, failures, rules, procedures).")
