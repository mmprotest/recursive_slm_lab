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
    get_active_adapter,
    fetch_seen_task_ids,
    wipe_memory,
)
from .tasks import load_tasks, validate_tasks, split_tasks
from .loop import run_iteration
from .eval import evaluate_conditions, plot_results
from .training import train_lora_adapter, get_adapters, activate_adapter, deactivate_adapter

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
    count: int = typer.Option(200, help="Number of tasks to generate"),
    out: Optional[Path] = typer.Option(None, help="Optional output path for JSONL"),
) -> None:
    if count < 120:
        raise typer.BadParameter("--count must be at least 120")
    output_path = out or (Path(__file__).parent / "tasks" / "bundled_tasks.jsonl")
    if regen or not output_path.exists():
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
