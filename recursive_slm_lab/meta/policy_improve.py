from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from ..eval.robust import evaluate_policy_pair, paired_bootstrap_ci, split_for_gating
from ..policy import Policy, DEFAULT_POLICY
from ..memory import register_policy, set_active_policy


@dataclass
class PolicyProposal:
    policy: Policy
    raw_json: str
    attempts: int


@dataclass
class PolicyPromotionResult:
    decision: str
    candidate_name: str | None
    metrics: dict
    rationale: dict
    message: str


def propose_policy(
    backend,
    current_policy: Policy,
    recent_failures_summary: str,
    constraints: dict,
) -> PolicyProposal:
    prompt = (
        "You are proposing a new policy JSON for recursive self-improvement. "
        "Return STRICT JSON matching the Policy schema exactly. "
        "Do not include commentary.\n\n"
        f"Current policy JSON:\n{current_policy.to_json()}\n\n"
        f"Failure summary:\n{recent_failures_summary}\n\n"
        f"Constraints:\n{json.dumps(constraints, indent=2)}\n"
    )
    attempts = 0
    last_error: str | None = None
    while attempts < 3:
        attempts += 1
        response = backend.generate(
            [
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
        )
        raw = _extract_json_text(response.text)
        try:
            payload = json.loads(raw)
            merged = _merge_policy_dict(current_policy.to_dict(), payload)
            validated = _validate_policy_dict(merged)
            policy = Policy.from_dict(validated)
            return PolicyProposal(policy=policy, raw_json=json.dumps(validated), attempts=attempts)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            last_error = str(exc)
            continue
    raise ValueError(f"Policy proposal failed after {attempts} attempts: {last_error}")


def evaluate_and_maybe_promote_policy(
    conn: sqlite3.Connection,
    backend,
    candidate_policy: Policy,
    heldout_size: int,
    heldout_limit: int | None,
    regression_size: int,
    repeats: int,
    seed: int,
    regression_seed: int,
    deterministic: bool,
    min_delta: float,
    max_drop: float,
    noise_band: float,
    notes: str | None = None,
) -> PolicyPromotionResult:
    train_pool, heldout, hidden, weak_hidden_ids = split_for_gating(heldout_size)
    if heldout_limit is not None:
        heldout = heldout[:heldout_limit]
    if regression_size:
        hidden = hidden[:regression_size]
    current_policy = _active_policy(conn)
    previous_policy_name = _active_policy_name(conn)

    baseline_heldout, candidate_heldout = evaluate_policy_pair(
        backend,
        heldout,
        current_policy,
        candidate_policy,
        max_tokens=256,
        deterministic=deterministic,
        seed=seed,
        repeats=repeats,
        conn=conn,
    )
    baseline_hidden, candidate_hidden = evaluate_policy_pair(
        backend,
        hidden,
        current_policy,
        candidate_policy,
        max_tokens=256,
        deterministic=deterministic,
        seed=seed,
        repeats=repeats,
        conn=conn,
    )
    heldout_delta = paired_bootstrap_ci(
        baseline_heldout.per_task_pass_at_1,
        candidate_heldout.per_task_pass_at_1,
        seed=seed,
    )
    hidden_delta = paired_bootstrap_ci(
        baseline_hidden.per_task_pass_at_1,
        candidate_hidden.per_task_pass_at_1,
        seed=seed,
    )
    noise_reject = abs(heldout_delta.mean) < noise_band and heldout_delta.ci_low <= 0.0 <= heldout_delta.ci_high
    promoted = (
        heldout_delta.mean >= min_delta
        and heldout_delta.ci_low >= 0.0
        and hidden_delta.mean >= -max_drop
        and hidden_delta.ci_low >= -max_drop
        and not noise_reject
    )
    decision = "accept" if promoted else "reject"
    rationale = _promotion_rationale(
        heldout_delta,
        hidden_delta,
        min_delta,
        max_drop,
        noise_band,
        noise_reject,
    )

    candidate_name = None
    if promoted:
        candidate_name = _next_policy_name(conn)
        register_policy(conn, candidate_name, candidate_policy, parent_policy_name=previous_policy_name)
        set_active_policy(conn, candidate_name)

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
            "min_delta": min_delta,
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
            "max_drop": max_drop,
            "weak_task_ids": weak_hidden_ids,
        },
        "gating": {
            "noise_band": noise_band,
            "noise_reject": noise_reject,
            "rationale": rationale,
        },
    }
    _record_policy_promotion(
        conn,
        previous_policy_name=previous_policy_name,
        candidate_policy_name=candidate_name,
        decision=decision,
        metrics=metrics,
        notes=notes,
    )
    message = "Policy promoted." if promoted else "Policy rejected."
    return PolicyPromotionResult(
        decision=decision,
        candidate_name=candidate_name,
        metrics=metrics,
        rationale=rationale,
        message=message,
    )


def _validate_policy_dict(payload: dict) -> dict:
    payload = dict(payload)
    payload["retrieval_top_n"] = int(_clamp(payload.get("retrieval_top_n", 0), 0, 10))
    payload["retrieval_extra_terms_mode"] = payload.get("retrieval_extra_terms_mode", "none")
    payload["retrieval_match_mode"] = payload.get("retrieval_match_mode", "and")
    for key in ("sampling_train", "sampling_eval"):
        block = payload.get(key, {})
        block["k"] = int(_clamp(block.get("k", 1), 1, 16))
        block["temperature"] = float(_clamp(block.get("temperature", 0.0), 0.0, 1.5))
        block["top_p"] = float(_clamp(block.get("top_p", 1.0), 0.1, 1.0))
        block["top_k"] = int(_clamp(block.get("top_k", 0), 0, 200))
        if key == "sampling_eval":
            block["k"] = max(1, min(block["k"], 16))
        payload[key] = block
    return payload


def _promotion_rationale(
    heldout_delta,
    hidden_delta,
    min_delta: float,
    max_drop: float,
    noise_band: float,
    noise_reject: bool,
) -> dict:
    reasons: list[str] = []
    if heldout_delta.mean < min_delta:
        reasons.append("heldout_mean_below_threshold")
    if heldout_delta.ci_low < 0.0:
        reasons.append("heldout_ci_low_negative")
    if noise_reject:
        reasons.append("within_noise_band")
    if hidden_delta.mean < -max_drop or hidden_delta.ci_low < -max_drop:
        reasons.append("regression_detected")
    return {
        "reasons": reasons,
        "min_delta": min_delta,
        "max_drop": max_drop,
        "noise_band": noise_band,
    }


def _merge_policy_dict(base: dict, proposed: dict) -> dict:
    merged = dict(base)
    for key, value in proposed.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _extract_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _active_policy(conn: sqlite3.Connection) -> Policy:
    row = conn.execute("SELECT policy_name FROM active_policy WHERE singleton = 1").fetchone()
    if not row:
        return DEFAULT_POLICY
    policy_row = conn.execute(
        "SELECT policy_json FROM policies WHERE name = ?",
        (row[0],),
    ).fetchone()
    if not policy_row:
        return DEFAULT_POLICY
    return Policy.from_json(policy_row[0])


def _active_policy_name(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT policy_name FROM active_policy WHERE singleton = 1").fetchone()
    return row[0] if row else None


def _next_policy_name(conn: sqlite3.Connection) -> str:
    rows = conn.execute("SELECT name FROM policies WHERE name LIKE 'pol%'").fetchall()
    max_id = 0
    for (name,) in rows:
        match = re.match(r"pol(\d+)", name)
        if match:
            max_id = max(max_id, int(match.group(1)))
    return f"pol{max_id + 1:03d}"


def _record_policy_promotion(
    conn: sqlite3.Connection,
    previous_policy_name: str | None,
    candidate_policy_name: str | None,
    decision: str,
    metrics: dict,
    notes: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO policy_promotions
        (created_at, previous_policy_name, candidate_policy_name, decision, metrics_json, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (_utc_now(), previous_policy_name, candidate_policy_name, decision, json.dumps(metrics), notes),
    )
    conn.commit()
