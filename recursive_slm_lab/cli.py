from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print

from .config import Config
from .llm import MockBackend, OpenAICompatBackend, LocalHFBackend
from .memory import init_db, connect, consolidate, get_active_adapter
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
            raise typer.BadParameter("RSLM_HF_MODEL_PATH is required for localhf backend")
        return LocalHFBackend(config.hf_model_path, adapter_path=adapter)
    return MockBackend()


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
    memory_enabled: bool = typer.Option(
        False, "--memory-enabled/--no-memory", help="Enable memory retrieval"
    ),
    heldout_size: int = typer.Option(40, help="Heldout size for splitting"),
    task_limit: Optional[int] = typer.Option(None, help="Limit number of tasks"),
) -> None:
    config = Config(
        db_path=db,
        backend=backend or Config().backend,
        base_url=base_url or Config().base_url,
        model=model or Config().model,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    all_tasks = load_tasks()
    train_pool, heldout = split_tasks(all_tasks, heldout_size=heldout_size)
    selected = train_pool if mode == "trainpool" else heldout
    if task_limit is not None:
        selected = selected[:task_limit]

    conn = connect(db)
    backend_impl = _resolve_backend(config)
    results = run_iteration(
        conn,
        tasks=selected,
        backend=backend_impl,
        k=k,
        max_tokens=max_tokens,
        temperature=temperature,
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
    backend: str = typer.Option("mock", help="mock|openai"),
) -> None:
    _ = backend
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
