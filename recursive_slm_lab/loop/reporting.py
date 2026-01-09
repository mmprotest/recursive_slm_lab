from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..util import ensure_dir


def write_iteration_report(
    *,
    conn,
    iteration: int,
    artifacts_dir: Path,
    config_snapshot: dict,
    train_metrics: dict,
    heldout_metrics: dict,
    regression_metrics: dict,
    ablations: dict,
    promotions: dict,
    cycle_window: dict,
) -> dict:
    ensure_dir(artifacts_dir)
    report_dir = artifacts_dir / f"iteration_{iteration:03d}"
    ensure_dir(report_dir)
    memory_updates = _memory_updates(conn, cycle_window["started_at"])
    report = {
        "iteration": iteration,
        "generated_at": _utc_now(),
        "config": config_snapshot,
        "trainpool": train_metrics,
        "heldout": heldout_metrics,
        "regression": regression_metrics,
        "ablations": ablations,
        "promotions": promotions,
        "memory_updates": memory_updates,
        "cycle_window": cycle_window,
    }
    (report_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (report_dir / "report.md").write_text(_render_markdown(report), encoding="utf-8")
    return report


def _memory_updates(conn, since_iso: str) -> dict:
    rules = conn.execute(
        """
        SELECT id, key, rule_text, created_at, origin_episode_ids, evidence_count, eval_snapshot
        FROM semantic_rules
        WHERE created_at >= ?
        ORDER BY created_at ASC
        """,
        (since_iso,),
    ).fetchall()
    procedures = conn.execute(
        """
        SELECT id, pattern, recipe_text, created_at, origin_episode_ids, evidence_count, eval_snapshot
        FROM procedures
        WHERE created_at >= ?
        ORDER BY created_at ASC
        """,
        (since_iso,),
    ).fetchall()
    return {
        "rules": [_format_memory_row(row, key_name="key", text_name="rule_text") for row in rules],
        "procedures": [
            _format_memory_row(row, key_name="pattern", text_name="recipe_text") for row in procedures
        ],
    }


def _format_memory_row(row, key_name: str, text_name: str) -> dict:
    origin_ids = json.loads(row[4]) if row[4] else []
    eval_snapshot = json.loads(row[6]) if row[6] else None
    return {
        "id": int(row[0]),
        key_name: row[1],
        text_name: row[2],
        "created_at": row[3],
        "origin_episode_ids": origin_ids,
        "evidence_count": int(row[5]),
        "eval_snapshot": eval_snapshot,
    }


def _render_markdown(report: dict) -> str:
    train = report["trainpool"]
    heldout = report["heldout"]
    regression = report["regression"]
    ablations = report["ablations"]
    promotions = report["promotions"]
    memory_updates = report["memory_updates"]
    return "\n".join(
        [
            f"# Iteration {report['iteration']:03d} Report",
            "",
            f"- Generated at: {report['generated_at']}",
            "",
            "## Metrics",
            f"- Trainpool pass-rate: {train['pass_rate']:.2f} ({train['passed']}/{train['total']})",
            f"- Heldout pass-rate: {heldout['pass_rate']:.2f}",
            f"- Regression pass-rate: {regression['pass_rate']:.2f}",
            "",
            "## Ablations",
            _render_ablation_block("Memory off vs on", ablations.get("memory", {})),
            _render_ablation_block("Learning off vs on", ablations.get("learning", {})),
            "",
            "## Promotions",
            f"- Policy: {promotions.get('policy', {}).get('decision', 'n/a')}",
            f"- Adapter: {promotions.get('adapter', {}).get('decision', 'n/a')}",
            "",
            "## Memory Updates",
            f"- Rules added: {len(memory_updates.get('rules', []))}",
            f"- Procedures added: {len(memory_updates.get('procedures', []))}",
        ]
    )


def _render_ablation_block(title: str, payload: dict) -> str:
    if not payload:
        return f"- {title}: n/a"
    baseline = payload.get("off", {})
    enabled = payload.get("on", {})
    note = payload.get("note")
    line = (
        f"- {title}: off={baseline.get('heldout_pass_at_1', 0.0):.2f}, "
        f"on={enabled.get('heldout_pass_at_1', 0.0):.2f}"
    )
    if note:
        line += f" ({note})"
    return line


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
