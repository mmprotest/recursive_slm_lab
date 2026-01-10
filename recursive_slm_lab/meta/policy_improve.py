from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
import logging

from ..eval.robust import evaluate_policy_pair, paired_bootstrap_ci, split_for_gating
from ..policy import Policy, DEFAULT_POLICY
from ..memory import register_policy, set_active_policy

LOGGER = logging.getLogger(__name__)

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
    schema_example = {
        "retrieval_top_n": 3,
        "sampling_train": {"k": 2, "temperature": 0.2, "top_p": 0.9, "top_k": 50},
        "sampling_eval": {"k": 1, "temperature": 0.0, "top_p": 1.0, "top_k": 0},
    }
    base_prompt = (
        "You are proposing a new policy JSON for recursive self-improvement.\n"
        "Constraints:\n"
        f"{json.dumps(constraints, indent=2)}\n\n"
        "Current policy JSON:\n"
        f"{current_policy.to_json()}\n\n"
        "Required JSON schema example:\n"
        f"{json.dumps(schema_example, indent=2)}\n\n"
        "Failure summary:\n"
        f"{recent_failures_summary}\n\n"
        "Output ONLY the JSON object."
    )
    attempts = 0
    last_error: str | None = None
    last_output: str = ""
    last_raw: str | None = None
    prompt = base_prompt
    while attempts < 8:
        attempts += 1
        LOGGER.info("Policy proposal attempt %d", attempts)
        response = backend.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "You MUST output exactly one JSON object and nothing else. "
                        "No prose, no Markdown, no code fences. Start with '{' and end with '}'."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.2,
            top_p=0.9,
            top_k=50,
        )
        last_output = response.text
        raw = _extract_json_text(response.text)
        last_raw = raw or None
        try:
            if raw == "":
                raise ValueError("No JSON object found in model output")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON decode error: {exc}") from exc
            validated_payload = _validate_policy_json(payload, constraints)
            merged = _merge_policy_dict(current_policy.to_dict(), validated_payload)
            policy = Policy.from_dict(merged)
            return PolicyProposal(
                policy=policy,
                raw_json=json.dumps(merged),
                attempts=attempts,
            )
        except ValueError as exc:
            last_error = str(exc)
            LOGGER.info("Policy proposal validation failed: %s", _compact_error_message(last_error))
            LOGGER.debug("Policy proposal validation detail: %s", last_error)
            LOGGER.debug("Policy proposal raw output: %s", response.text)
        prompt = _build_retry_prompt(base_prompt, last_error)
    snippet = last_output[:400]
    raw_snippet = (last_raw or "")[:400]
    message = (
        "Policy proposal failed after "
        f"{attempts} attempts. Last error: {last_error}. "
        f"Final output snippet: {snippet}. "
        f"Extracted JSON snippet: {raw_snippet}"
    )
    LOGGER.error(message)
    raise ValueError(message)


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

    greedy_baseline = _force_greedy_eval(current_policy)
    greedy_candidate = _force_greedy_eval(candidate_policy)
    baseline_heldout, candidate_heldout = evaluate_policy_pair(
        backend,
        heldout,
        greedy_baseline,
        greedy_candidate,
        max_tokens=256,
        deterministic=deterministic,
        seed=seed,
        repeats=repeats,
        conn=conn,
    )
    baseline_hidden, candidate_hidden = evaluate_policy_pair(
        backend,
        hidden,
        greedy_baseline,
        greedy_candidate,
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


def _validate_policy_json(obj: dict, constraints: dict) -> dict:
    if not isinstance(obj, dict):
        raise ValueError("Policy JSON must be an object")
    payload = dict(obj)
    if "retrieval_top_n" not in payload:
        raise ValueError("Missing required field: retrieval_top_n")
    retrieval_range = constraints["retrieval_top_n"]
    payload["retrieval_top_n"] = _coerce_int(
        "retrieval_top_n",
        payload["retrieval_top_n"],
        *retrieval_range,
    )

    for key in ("sampling_train", "sampling_eval"):
        if key not in payload or not isinstance(payload[key], dict):
            raise ValueError(f"Missing required field: {key}")
        block = dict(payload[key])
        for field in ("k", "temperature", "top_p", "top_k"):
            if field not in block:
                raise ValueError(f"Missing required field: {key}.{field}")
        ranges = constraints[key]
        block["k"] = _coerce_int(f"{key}.k", block["k"], *ranges["k"])
        block["temperature"] = _coerce_float(
            f"{key}.temperature",
            block["temperature"],
            *ranges["temperature"],
        )
        block["top_p"] = _coerce_float(f"{key}.top_p", block["top_p"], *ranges["top_p"])
        block["top_k"] = _coerce_int(f"{key}.top_k", block["top_k"], *ranges["top_k"])
        payload[key] = block
    return payload


def _coerce_int(field: str, value, low: float, high: float) -> int:
    if isinstance(value, bool):
        raise ValueError(_invalid_field_message(field, value))
    if isinstance(value, int):
        coerced = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError(_invalid_field_message(field, value))
        coerced = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            raise ValueError(_invalid_field_message(field, value))
        try:
            parsed = float(stripped)
        except ValueError as exc:
            raise ValueError(_invalid_field_message(field, value)) from exc
        if not parsed.is_integer():
            raise ValueError(_invalid_field_message(field, value))
        coerced = int(parsed)
    else:
        raise ValueError(_invalid_field_message(field, value))
    return int(_clamp(coerced, low, high))


def _coerce_float(field: str, value, low: float, high: float) -> float:
    if isinstance(value, bool):
        raise ValueError(_invalid_field_message(field, value))
    if isinstance(value, (int, float)):
        coerced = float(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            raise ValueError(_invalid_field_message(field, value))
        try:
            coerced = float(stripped)
        except ValueError as exc:
            raise ValueError(_invalid_field_message(field, value)) from exc
    else:
        raise ValueError(_invalid_field_message(field, value))
    return float(_clamp(coerced, low, high))


def _invalid_field_message(field: str, value) -> str:
    return f"Invalid field: {field} (value={value!r}, type={type(value).__name__})"


def _compact_error_message(error: str) -> str:
    if " (" in error:
        return error.split(" (", 1)[0]
    return error


def _build_retry_prompt(base_prompt: str, error: str | None) -> str:
    if not error:
        return base_prompt + "\n\nYour output was invalid. Output ONLY the JSON object. No commentary."
    field = _extract_error_field(error)
    example = _field_example(field) if field else None
    extra = f"\n\nValidation error: {error}"
    if example:
        extra += f"\n{example}"
    extra += "\nOutput ONLY the JSON object. No commentary."
    return base_prompt + extra


def _extract_error_field(error: str) -> str | None:
    match = re.search(r"Invalid field: ([^\s(]+)", error)
    if match:
        return match.group(1)
    match = re.search(r"Missing required field: ([^\s]+)", error)
    if match:
        return match.group(1)
    return None


def _field_example(field: str | None) -> str | None:
    if not field:
        return None
    examples = {
        "retrieval_top_n": 'Example: "retrieval_top_n": 3  (integer, not a string)',
        "sampling_train": 'Example: "sampling_train": {"k": 2, "temperature": 0.2, "top_p": 0.9, "top_k": 50}',
        "sampling_eval": 'Example: "sampling_eval": {"k": 1, "temperature": 0.0, "top_p": 1.0, "top_k": 0}',
        "sampling_train.k": 'Example: "sampling_train": {"k": 2}  (integer, not a string)',
        "sampling_eval.k": 'Example: "sampling_eval": {"k": 1}  (integer, not a string)',
        "sampling_train.top_k": 'Example: "sampling_train": {"top_k": 50}  (integer)',
        "sampling_eval.top_k": 'Example: "sampling_eval": {"top_k": 0}  (integer)',
        "sampling_train.temperature": 'Example: "sampling_train": {"temperature": 0.2}  (number)',
        "sampling_eval.temperature": 'Example: "sampling_eval": {"temperature": 0.0}  (number)',
        "sampling_train.top_p": 'Example: "sampling_train": {"top_p": 0.9}  (number)',
        "sampling_eval.top_p": 'Example: "sampling_eval": {"top_p": 1.0}  (number)',
    }
    return examples.get(field)


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


def _force_greedy_eval(policy: Policy) -> Policy:
    payload = policy.to_dict()
    eval_block = dict(payload.get("sampling_eval", {}))
    eval_block["temperature"] = 0.0
    eval_block["top_p"] = 1.0
    eval_block["top_k"] = 0
    payload["sampling_eval"] = eval_block
    return Policy.from_dict(payload)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _extract_json_text(text: str) -> str:
    text = text.strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return text
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return text[idx : idx + end]
    return ""


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
