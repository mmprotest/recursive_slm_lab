from __future__ import annotations

import re
import sqlite3
from collections import Counter


_ERROR_PATTERNS = {
    "assertion": re.compile(r"AssertionError", re.IGNORECASE),
    "type_error": re.compile(r"TypeError", re.IGNORECASE),
    "name_error": re.compile(r"NameError", re.IGNORECASE),
    "index_error": re.compile(r"IndexError", re.IGNORECASE),
    "key_error": re.compile(r"KeyError", re.IGNORECASE),
    "value_error": re.compile(r"ValueError", re.IGNORECASE),
    "syntax_error": re.compile(r"SyntaxError", re.IGNORECASE),
    "timeout": re.compile(r"Timeout", re.IGNORECASE),
}


def summarize_failures(
    conn: sqlite3.Connection,
    limit: int = 50,
) -> str:
    rows = conn.execute(
        """
        SELECT task_id, prompt, test_log
        FROM failures
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    if not rows:
        return "No recent failures recorded."

    error_counts: Counter[str] = Counter()
    examples: list[str] = []
    for task_id, prompt, test_log in rows:
        for name, pattern in _ERROR_PATTERNS.items():
            if pattern.search(test_log):
                error_counts[name] += 1
        if len(examples) < 5:
            examples.append(f"- {task_id}: {prompt.strip()[:120]}")

    common_errors = "\n".join(
        f"- {name}: {count}" for name, count in error_counts.most_common()
    ) or "- none detected"
    example_text = "\n".join(examples) or "- none"
    return (
        "Recent failure summary:\n"
        "Common error types:\n"
        f"{common_errors}\n"
        "Example failing tasks:\n"
        f"{example_text}\n"
        "Patterns: check edge cases, formatting, and missing imports."
    )
