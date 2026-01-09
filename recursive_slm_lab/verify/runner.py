from __future__ import annotations

from dataclasses import dataclass
import hashlib
import sqlite3

from ..config import Config
from ..memory import connect
from .sandbox import run_in_sandbox


@dataclass
class VerificationResult:
    passed: bool
    log: str


def verify_candidate(
    solution_code: str,
    test_code: str,
    assert_tests: list[str] | None = None,
    conn: sqlite3.Connection | None = None,
    db_path: str | None = None,
) -> VerificationResult:
    cache_key = _cache_key(solution_code, test_code, assert_tests)
    should_close = False
    if conn is None and db_path is not None:
        conn = connect(db_path, migrate=False)
        should_close = True
    if conn is None and db_path is None:
        db_path = Config().db_path
        conn = connect(db_path, migrate=True)
        should_close = True
    if conn is not None:
        cached = _fetch_cache(conn, cache_key)
        if cached is not None:
            if should_close:
                conn.close()
            return cached
    result = run_in_sandbox(solution_code, test_code, assert_tests=assert_tests)
    log = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    verification = VerificationResult(passed=result.passed, log=log)
    if conn is not None:
        _store_cache(conn, cache_key, verification)
        if should_close:
            conn.close()
    return verification


def _cache_key(
    solution_code: str,
    test_code: str,
    assert_tests: list[str] | None,
) -> str:
    payload = solution_code + "\n" + test_code + "\n" + "\n".join(assert_tests or [])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fetch_cache(conn: sqlite3.Connection, cache_key: str) -> VerificationResult | None:
    try:
        row = conn.execute(
            "SELECT passed, log FROM verification_cache WHERE key = ?",
            (cache_key,),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    if not row:
        return None
    return VerificationResult(passed=bool(row[0]), log=row[1])


def _store_cache(conn: sqlite3.Connection, cache_key: str, result: VerificationResult) -> None:
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO verification_cache (key, passed, log, created_at)
            VALUES (?, ?, ?, datetime('now'))
            """,
            (cache_key, int(result.passed), result.log),
        )
        conn.commit()
    except sqlite3.OperationalError:
        return
