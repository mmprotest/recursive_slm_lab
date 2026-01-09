from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from ..eval.robust import paired_bootstrap_ci, SplitResult
from ..llm.base import LLMBackend
from ..loop.sampling import generate_candidates
from ..tasks import Task
from ..verify import verify_candidate
from .db import fetch_passed_episodes
from .retrieval import MemoryContext, MemoryHit



@dataclass
class CandidateRule:
    kind: str
    key: str
    text: str
    keywords: list[str]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _select_validation_tasks(tasks: list[Task], keywords: list[str], size: int, seed: int) -> list[Task]:
    lowered = [kw.lower() for kw in keywords if kw]
    matches = [
        task for task in tasks if any(kw in task.prompt.lower() for kw in lowered)
    ]
    rng = random.Random(seed)
    if not matches:
        candidates = tasks[:]
        rng.shuffle(candidates)
        return candidates[:size]
    rng.shuffle(matches)
    return matches[:size]


def _evaluate_tasks(
    backend: LLMBackend,
    tasks: list[Task],
    memory_context: MemoryContext | None,
    k: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> SplitResult:
    outcomes: list[list[bool]] = []
    for task in tasks:
        candidates = generate_candidates(
            backend,
            task_prompt=task.prompt,
            function_name=task.function_name,
            signature=task.signature,
            memory_context=memory_context,
            policy=None,
            k=k,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        task_outcomes: list[bool] = []
        for candidate in candidates:
            verification = verify_candidate(candidate.code, task.reference_tests, task.assert_tests)
            task_outcomes.append(verification.passed)
        outcomes.append(task_outcomes)
    pass_at_1_vec = [1 if row and row[0] else 0 for row in outcomes]
    pass_at_k_vec = [1 if any(row[:k]) else 0 for row in outcomes]
    pass_at_1 = sum(pass_at_1_vec) / len(pass_at_1_vec) if pass_at_1_vec else 0.0
    pass_at_k = sum(pass_at_k_vec) / len(pass_at_k_vec) if pass_at_k_vec else 0.0
    return SplitResult(
        pass_at_1=pass_at_1,
        pass_at_k=pass_at_k,
        per_task_pass_at_1=pass_at_1_vec,
        per_task_pass_at_k=pass_at_k_vec,
    )


def consolidate_llm(
    conn: sqlite3.Connection,
    backend: LLMBackend,
    heldout_tasks: list[Task],
    hidden_tasks: list[Task],
    sample_episodes: int = 80,
    max_rules: int = 20,
    min_gain: float = 0.01,
    max_hidden_regress: float = 0.0,
    k: int = 1,
    max_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
    top_k: int = 50,
) -> dict:
    episodes = fetch_passed_episodes(conn)[:sample_episodes]
    if not episodes:
        return {"error": "No passed episodes available for consolidation."}

    examples = [
        {"prompt": ep.prompt, "candidate_code": ep.candidate_code}
        for ep in episodes
    ]
    origin_episode_ids = [ep.episode_id for ep in episodes]
    prompt_payload = json.dumps({"examples": examples}, indent=2)
    system_prompt = (
        "You are a strict JSON generator. Do not include commentary."
    )
    user_prompt = (
        "Analyze the following successful (prompt, code) examples and propose reusable "
        "semantic rules and step-by-step procedures for solving similar tasks. "
        "Return strict JSON with fields:\n"
        "rules: [{key, text, keywords}]\n"
        "procedures: [{pattern, text, keywords}]\n"
        f"Limit total combined items to {max_rules}.\n"
        "Examples:\n"
        f"{prompt_payload}"
    )
    response = backend.generate(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    try:
        parsed = _extract_json(response.text)
    except json.JSONDecodeError as exc:
        return {"error": f"Failed to parse JSON from backend: {exc}"}

    candidates: list[CandidateRule] = []
    for rule in parsed.get("rules", []):
        if not all(key in rule for key in ("key", "text", "keywords")):
            continue
        candidates.append(
            CandidateRule(
                kind="rule",
                key=str(rule["key"]),
                text=str(rule["text"]),
                keywords=list(rule["keywords"]) if rule["keywords"] else [],
            )
        )
    for proc in parsed.get("procedures", []):
        if not all(key in proc for key in ("pattern", "text", "keywords")):
            continue
        candidates.append(
            CandidateRule(
                kind="procedure",
                key=str(proc["pattern"]),
                text=str(proc["text"]),
                keywords=list(proc["keywords"]) if proc["keywords"] else [],
            )
        )

    report = {"proposed": len(candidates), "accepted": [], "rejected": []}
    validation_size = min(6, len(heldout_tasks))
    if validation_size == 0:
        report["error"] = "Heldout tasks unavailable for consolidation validation."
        return report
    if not hidden_tasks:
        report["error"] = "Hidden tasks unavailable for consolidation validation."
        return report

    for candidate in candidates:
        memory_context = MemoryContext(
            hits=[MemoryHit(source=candidate.kind, text=candidate.text, score=0.0)]
        )
        subset = _select_validation_tasks(heldout_tasks, candidate.keywords, validation_size, seed=1337)
        baseline = _evaluate_tasks(
            backend, subset, None, k, max_tokens, temperature, top_p, top_k
        )
        with_rule = _evaluate_tasks(
            backend, subset, memory_context, k, max_tokens, temperature, top_p, top_k
        )
        delta_heldout = paired_bootstrap_ci(
            baseline.per_task_pass_at_1,
            with_rule.per_task_pass_at_1,
            seed=1337,
        )

        hidden_subset = _select_validation_tasks(hidden_tasks, candidate.keywords, validation_size, seed=2024)
        baseline_fallback = _evaluate_tasks(
            backend, hidden_subset, None, k, max_tokens, temperature, top_p, top_k
        )
        with_rule_fallback = _evaluate_tasks(
            backend, hidden_subset, memory_context, k, max_tokens, temperature, top_p, top_k
        )
        delta_hidden = paired_bootstrap_ci(
            baseline_fallback.per_task_pass_at_1,
            with_rule_fallback.per_task_pass_at_1,
            seed=2024,
        )

        accepted = (
            delta_heldout.mean >= min_gain
            and delta_heldout.ci_low >= 0.0
            and delta_hidden.mean >= -max_hidden_regress
            and delta_hidden.ci_low >= -max_hidden_regress
        )

        if accepted:
            evidence_count = len(subset)
            now = _utc_now()
            eval_snapshot = json.dumps(
                {
                    "heldout": {
                        "pass_at_1": with_rule.pass_at_1,
                        "pass_at_k": with_rule.pass_at_k,
                        "delta": delta_heldout.__dict__,
                    },
                    "hidden": {
                        "pass_at_1": with_rule_fallback.pass_at_1,
                        "pass_at_k": with_rule_fallback.pass_at_k,
                        "delta": delta_hidden.__dict__,
                    },
                }
            )
            if candidate.kind == "rule":
                cursor = conn.execute(
                    """
                    INSERT INTO semantic_rules
                    (key, rule_text, created_at, origin_episode_ids, evidence_count, eval_snapshot, active, superseded_by, last_verified_at)
                    VALUES (?, ?, ?, ?, ?, ?, 1, NULL, ?)
                    """,
                    (
                        candidate.key,
                        candidate.text,
                        now,
                        json.dumps(origin_episode_ids),
                        evidence_count,
                        eval_snapshot,
                        now,
                    ),
                )
                new_id = int(cursor.lastrowid)
                conn.execute(
                    """
                    UPDATE semantic_rules SET active = 0, superseded_by = ?
                    WHERE key = ? AND id != ? AND active = 1
                    """,
                    (new_id, candidate.key, new_id),
                )
            else:
                cursor = conn.execute(
                    """
                    INSERT INTO procedures
                    (pattern, recipe_text, created_at, origin_episode_ids, evidence_count, eval_snapshot, active, superseded_by, last_verified_at)
                    VALUES (?, ?, ?, ?, ?, ?, 1, NULL, ?)
                    """,
                    (
                        candidate.key,
                        candidate.text,
                        now,
                        json.dumps(origin_episode_ids),
                        evidence_count,
                        eval_snapshot,
                        now,
                    ),
                )
                new_id = int(cursor.lastrowid)
                conn.execute(
                    """
                    UPDATE procedures SET active = 0, superseded_by = ?
                    WHERE pattern = ? AND id != ? AND active = 1
                    """,
                    (new_id, candidate.key, new_id),
                )
            report["accepted"].append(
                {
                    "kind": candidate.kind,
                    "key": candidate.key,
                    "delta_heldout": delta_heldout.__dict__,
                    "delta_hidden": delta_hidden.__dict__,
                    "evidence_count": evidence_count,
                }
            )
        else:
            reason = "gain_below_threshold"
            if delta_hidden.mean < -max_hidden_regress:
                reason = "hidden_regression"
            report["rejected"].append(
                {
                    "kind": candidate.kind,
                    "key": candidate.key,
                    "delta_heldout": delta_heldout.__dict__,
                    "delta_hidden": delta_hidden.__dict__,
                    "reason": reason,
                }
            )

    conn.commit()
    return report
