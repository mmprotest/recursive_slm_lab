from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from ..eval.robust import evaluate_tasks, paired_bootstrap_ci, split_for_gating
from ..llm.localhf import LocalHFBackend
from ..memory import get_active_adapter
from ..policy import DEFAULT_POLICY, Policy
from .adapters import activate_adapter
from .sft_lora import train_lora_adapter


@dataclass
class PromotionConfig:
    out_dir: str
    heldout_size: int
    heldout_limit: int | None
    regression_size: int
    k: int
    min_improvement: float
    max_regression_drop: float
    noise_band: float
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    base_model_path: str | None
    regression_seed: int = 1337
    repeats: int = 3
    deterministic: bool = True
    seed: int = 1337


@dataclass
class PromotionResult:
    promoted: bool
    decision: str
    candidate_adapter: str | None
    previous_adapter: str | None
    adapter_path: str | None
    metrics: dict
    rationale: dict
    message: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def train_and_maybe_promote(conn: sqlite3.Connection, cfg: PromotionConfig) -> PromotionResult:
    previous_adapter = get_active_adapter(conn)
    previous_adapter_name = previous_adapter[0] if previous_adapter else None

    training_result = train_lora_adapter(conn, cfg.out_dir, cfg.base_model_path)
    if not training_result.trained:
        payload = {
            "decision": "rejected",
            "reason": "training_failed",
            "message": training_result.message,
            "config": cfg.__dict__,
        }
        conn.execute(
            """
            INSERT INTO promotions (created_at, previous_adapter_name, candidate_adapter_name, decision, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (_utc_now(), previous_adapter_name, training_result.adapter_name, "rejected", json.dumps(payload)),
        )
        conn.commit()
        return PromotionResult(
            promoted=False,
            decision="rejected",
            candidate_adapter=None,
            previous_adapter=previous_adapter_name,
            adapter_path=None,
            metrics={},
            rationale={"reasons": ["training_failed"]},
            message=training_result.message,
        )

    _, heldout, hidden, weak_hidden_ids = split_for_gating(cfg.heldout_size)
    if cfg.heldout_limit is not None:
        heldout = heldout[:cfg.heldout_limit]
    if cfg.regression_size:
        hidden = hidden[:cfg.regression_size]

    baseline_backend = LocalHFBackend(cfg.base_model_path)
    dummy_policy = _dummy_policy(cfg.k, cfg.temperature, cfg.top_p, cfg.top_k)

    baseline_heldout = _evaluate_with_adapter(
        baseline_backend,
        None,
        heldout,
        dummy_policy,
        cfg.k,
        cfg.max_tokens,
        cfg.temperature,
        cfg.top_p,
        cfg.top_k,
        cfg.deterministic,
        cfg.seed,
        cfg.repeats,
        conn,
    )
    candidate_heldout = _evaluate_with_adapter(
        baseline_backend,
        training_result.adapter_path,
        heldout,
        dummy_policy,
        cfg.k,
        cfg.max_tokens,
        cfg.temperature,
        cfg.top_p,
        cfg.top_k,
        cfg.deterministic,
        cfg.seed,
        cfg.repeats,
        conn,
    )
    baseline_hidden = _evaluate_with_adapter(
        baseline_backend,
        None,
        hidden,
        dummy_policy,
        cfg.k,
        cfg.max_tokens,
        cfg.temperature,
        cfg.top_p,
        cfg.top_k,
        cfg.deterministic,
        cfg.seed,
        cfg.repeats,
        conn,
    )
    candidate_hidden = _evaluate_with_adapter(
        baseline_backend,
        training_result.adapter_path,
        hidden,
        dummy_policy,
        cfg.k,
        cfg.max_tokens,
        cfg.temperature,
        cfg.top_p,
        cfg.top_k,
        cfg.deterministic,
        cfg.seed,
        cfg.repeats,
        conn,
    )
    heldout_delta = paired_bootstrap_ci(
        baseline_heldout.per_task_pass_at_1,
        candidate_heldout.per_task_pass_at_1,
        seed=cfg.seed,
    )
    hidden_delta = paired_bootstrap_ci(
        baseline_hidden.per_task_pass_at_1,
        candidate_hidden.per_task_pass_at_1,
        seed=cfg.seed,
    )

    noise_reject = abs(heldout_delta.mean) < cfg.noise_band and heldout_delta.ci_low <= 0.0 <= heldout_delta.ci_high
    promoted = (
        heldout_delta.mean >= cfg.min_improvement
        and heldout_delta.ci_low >= 0.0
        and hidden_delta.mean >= -cfg.max_regression_drop
        and hidden_delta.ci_low >= -cfg.max_regression_drop
        and not noise_reject
    )
    rationale = _promotion_rationale(
        heldout_delta,
        hidden_delta,
        cfg.min_improvement,
        cfg.max_regression_drop,
        cfg.noise_band,
        noise_reject,
    )
    decision = "promoted" if promoted else "rejected"
    if promoted:
        activate_adapter(conn, training_result.adapter_name)

    metrics = {
        "heldout": {
            "baseline": {
                "pass_at_1": baseline_heldout.pass_at_1,
                "pass_at_k": baseline_heldout.pass_at_k,
            },
            "candidate": {
                "pass_at_1": candidate_heldout.pass_at_1,
                "pass_at_k": candidate_heldout.pass_at_k,
            },
            "delta": heldout_delta.__dict__,
            "min_improvement": cfg.min_improvement,
        },
        "hidden": {
            "baseline": {
                "pass_at_1": baseline_hidden.pass_at_1,
                "pass_at_k": baseline_hidden.pass_at_k,
            },
            "candidate": {
                "pass_at_1": candidate_hidden.pass_at_1,
                "pass_at_k": candidate_hidden.pass_at_k,
            },
            "delta": hidden_delta.__dict__,
            "max_regression_drop": cfg.max_regression_drop,
            "weak_task_ids": weak_hidden_ids,
        },
        "gating": {
            "noise_band": cfg.noise_band,
            "noise_reject": noise_reject,
            "rationale": rationale,
        },
    }
    payload = {
        "decision": decision,
        "metrics": metrics,
        "decision_rationale": rationale,
        "config": cfg.__dict__,
        "previous_adapter": previous_adapter_name,
        "candidate_adapter": training_result.adapter_name,
        "candidate_path": training_result.adapter_path,
    }
    conn.execute(
        """
        INSERT INTO promotions (created_at, previous_adapter_name, candidate_adapter_name, decision, payload_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (_utc_now(), previous_adapter_name, training_result.adapter_name, decision, json.dumps(payload)),
    )
    conn.commit()

    message = "Adapter promoted." if promoted else "Adapter rejected."
    return PromotionResult(
        promoted=promoted,
        decision=decision,
        candidate_adapter=training_result.adapter_name,
        previous_adapter=previous_adapter_name,
        adapter_path=training_result.adapter_path if promoted else None,
        metrics=metrics,
        rationale=rationale,
        message=message,
    )


def _promotion_rationale(
    heldout_delta,
    hidden_delta,
    min_improvement: float,
    max_regression_drop: float,
    noise_band: float,
    noise_reject: bool,
) -> dict:
    reasons: list[str] = []
    if heldout_delta.mean < min_improvement:
        reasons.append("heldout_mean_below_threshold")
    if heldout_delta.ci_low < 0.0:
        reasons.append("heldout_ci_low_negative")
    if noise_reject:
        reasons.append("within_noise_band")
    if hidden_delta.mean < -max_regression_drop or hidden_delta.ci_low < -max_regression_drop:
        reasons.append("regression_detected")
    return {
        "reasons": reasons,
        "min_improvement": min_improvement,
        "max_regression_drop": max_regression_drop,
        "noise_band": noise_band,
    }


def _dummy_policy(k: int, temperature: float, top_p: float, top_k: int) -> Policy:
    payload = DEFAULT_POLICY.to_dict()
    payload["retrieval_top_n"] = 0
    payload["sampling_train"] = {"k": k, "temperature": temperature, "top_p": top_p, "top_k": top_k}
    payload["sampling_eval"] = {"k": k, "temperature": temperature, "top_p": top_p, "top_k": top_k}
    return Policy.from_dict(payload)


def _evaluate_with_adapter(
    backend,
    adapter_path: str | None,
    tasks: list,
    policy: Policy,
    k: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    deterministic: bool,
    seed: int,
    repeats: int,
    conn,
):
    if adapter_path is None:
        if hasattr(backend, "set_adapter"):
            backend.set_adapter(None)
    else:
        if hasattr(backend, "set_adapter"):
            backend.set_adapter(adapter_path)
        elif hasattr(backend, "clone_with_adapter"):
            backend = backend.clone_with_adapter(adapter_path)
        else:
            raise ValueError("Backend does not support adapter switching")
    return evaluate_tasks(
        backend,
        tasks,
        policy,
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
