from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def plot_results(input_path: str, output_path: str) -> None:
    import matplotlib.pyplot as plt

    db_runs = _load_eval_runs(input_path)
    if db_runs is not None:
        if not db_runs:
            raise ValueError("No eval_runs entries found for plotting.")
        if len(db_runs) > 1:
            condition_names = _collect_conditions(db_runs)
            fig, ax = plt.subplots(figsize=(8, 4))
            x_values = list(range(1, len(db_runs) + 1))
            for condition in condition_names:
                series = [
                    _extract_pass_at_1(payload, condition) for payload in db_runs
                ]
                ax.plot(x_values, series, marker="o", label=condition)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Evaluation run")
            ax.set_ylabel("pass@1")
            ax.set_title("Evaluation Results Over Iterations")
            ax.legend()
        else:
            data = db_runs[0]
            fig, ax = plt.subplots(figsize=(8, 4))
            names, scores = _extract_bar_data(data)
            ax.bar(names, scores)
            ax.set_ylim(0, 1)
            ax.set_ylabel("pass@1")
            ax.set_title("Evaluation Results")
            for idx, score in enumerate(scores):
                ax.text(idx, score + 0.02, f"{score:.2f}", ha="center")
    else:
        data = json.loads(Path(input_path).read_text(encoding="utf-8"))
        names, scores = _extract_bar_data(data)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(names, scores)
        ax.set_ylim(0, 1)
        ax.set_ylabel("pass@1")
        ax.set_title("Evaluation Results")
        for idx, score in enumerate(scores):
            ax.text(idx, score + 0.02, f"{score:.2f}", ha="center")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)


def _load_eval_runs(input_path: str) -> list[dict] | None:
    path = Path(input_path)
    if not path.exists() or path.suffix.lower() not in {".sqlite", ".db", ".sqlite3"}:
        return None
    try:
        conn = sqlite3.connect(path)
        rows = conn.execute(
            "SELECT payload_json FROM eval_runs ORDER BY created_at ASC"
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        if "conn" in locals():
            conn.close()
    if not rows:
        return []
    return [json.loads(row[0]) for row in rows]


def _collect_conditions(payloads: list[dict]) -> list[str]:
    seen: list[str] = []
    for payload in payloads:
        for item in payload.get("conditions", []):
            name = item.get("condition")
            if name and name not in seen:
                seen.append(name)
    return seen


def _extract_pass_at_1(payload: dict, condition: str) -> float:
    for item in payload.get("conditions", []):
        if item.get("condition") == condition:
            return item.get("pass_at_1", 0.0)
    return 0.0


def _extract_bar_data(payload: dict) -> tuple[list[str], list[float]]:
    conditions = payload.get("conditions", [])
    names = [item["condition"] for item in conditions]
    scores = [item["pass_at_1"] for item in conditions]
    return names, scores
