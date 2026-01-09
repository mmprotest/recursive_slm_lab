from __future__ import annotations

import multiprocessing as mp

from recursive_slm_lab.memory import init_db
from recursive_slm_lab.verify import verify_candidate


def _worker(db_path: str, iterations: int) -> bool:
    solution = "def add(a, b):\n    return a + b\n"
    tests = (
        "from solution import *\n"
        "def test_add():\n"
        "    assert add(1, 2) == 3\n"
    )
    for _ in range(iterations):
        result = verify_candidate(solution, tests, db_path=db_path)
        if not result.passed:
            return False
    return True


def test_parallel_verify_does_not_break_fts(tmp_path) -> None:
    db_path = tmp_path / "memory.sqlite"
    init_db(db_path)

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=4) as pool:
        results = pool.starmap(_worker, [(str(db_path), 5) for _ in range(8)])

    assert all(results)

    # Ensure DB remains queryable after multiprocess access.
    import sqlite3

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='rules_fts'"
    ).fetchall()
    conn.close()
    assert rows
