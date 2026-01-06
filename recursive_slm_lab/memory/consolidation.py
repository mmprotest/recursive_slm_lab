from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_rules(prompt: str) -> list[tuple[str, str]]:
    rules: list[tuple[str, str]] = []
    lowered = prompt.lower()
    if "palindrome" in lowered:
        rules.append((
            "palindrome_clean",
            "If checking palindromes, normalize case and remove non-alphanumeric characters before comparing.",
        ))
    if "vowel" in lowered:
        rules.append((
            "vowel_count",
            "For vowel counting, lowercase the string and match against aeiou.",
        ))
    if "fibonacci" in lowered:
        rules.append((
            "fib_iterative",
            "For Fibonacci numbers, use iterative accumulation with (a, b) to avoid recursion overhead.",
        ))
    if "factorial" in lowered:
        rules.append((
            "factorial_loop",
            "Factorial can be computed with a simple loop multiplying from 2..n.",
        ))
    if "prime" in lowered:
        rules.append((
            "prime_trial_div",
            "Check primes by trial division up to sqrt(n) and handle n < 2 as not prime.",
        ))
    if "median" in lowered:
        rules.append((
            "median_sort",
            "Median can be found by sorting and taking middle elements for even length.",
        ))
    if "mode" in lowered:
        rules.append((
            "mode_counter",
            "Mode can be computed with a frequency counter and choose smallest on ties.",
        ))
    if "unique" in lowered:
        rules.append((
            "unique_stable",
            "Preserve order by tracking seen items in a list and skipping duplicates.",
        ))
    return rules


def _extract_procedures(prompt: str) -> list[tuple[str, str]]:
    procedures: list[tuple[str, str]] = []
    lowered = prompt.lower()
    if "split" in lowered or "parse" in lowered:
        procedures.append((
            "parse_numbers",
            "To parse numeric input, split the string and map int over tokens.",
        ))
    if "list" in lowered and "sum" in lowered:
        procedures.append((
            "list_sum",
            "Use Python built-ins like sum(values) for list aggregation tasks.",
        ))
    if "reverse" in lowered:
        procedures.append((
            "reverse_sequence",
            "Use slicing [::-1] or reversed() to reverse strings or lists.",
        ))
    return procedures


def consolidate(conn: sqlite3.Connection, min_evidence: int = 3) -> None:
    rows = conn.execute(
        "SELECT DISTINCT task_id, prompt FROM episodes WHERE passed = 1"
    ).fetchall()
    now = _utc_now()

    rule_evidence: dict[str, set[str]] = {}
    proc_evidence: dict[str, set[str]] = {}
    rule_text: dict[str, str] = {}
    proc_text: dict[str, str] = {}

    for task_id, prompt in rows:
        for key, text in _extract_rules(prompt):
            rule_evidence.setdefault(key, set()).add(task_id)
            rule_text[key] = text
        for key, text in _extract_procedures(prompt):
            proc_evidence.setdefault(key, set()).add(task_id)
            proc_text[key] = text

    for key, task_ids in rule_evidence.items():
        evidence_count = len(task_ids)
        if evidence_count < min_evidence:
            continue
        conn.execute(
            """
            INSERT INTO semantic_rules (key, rule_text, evidence_count, last_verified_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                rule_text=excluded.rule_text,
                evidence_count=excluded.evidence_count,
                last_verified_at=excluded.last_verified_at
            """,
            (key, rule_text[key], evidence_count, now),
        )

    for key, task_ids in proc_evidence.items():
        evidence_count = len(task_ids)
        if evidence_count < min_evidence:
            continue
        conn.execute(
            """
            INSERT INTO procedures (pattern, recipe_text, evidence_count, last_verified_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(pattern) DO UPDATE SET
                recipe_text=excluded.recipe_text,
                evidence_count=excluded.evidence_count,
                last_verified_at=excluded.last_verified_at
            """,
            (key, proc_text[key], evidence_count, now),
        )

    conn.commit()
