from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from recursive_slm_lab.memory import connect, init_db, insert_episode, retrieve_memory


def _setup_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "memory.sqlite"
    init_db(db_path)
    return db_path


def test_bm25_filter_direction(tmp_path: Path) -> None:
    db_path = _setup_db(tmp_path)
    conn = connect(db_path)
    insert_episode(
        conn,
        task_id="t1",
        condition="train",
        prompt="alpha beta gamma",
        candidate_code="def foo():\n    return 1\n",
        passed=True,
        test_log="ok",
    )
    insert_episode(
        conn,
        task_id="t2",
        condition="train",
        prompt="alpha",
        candidate_code="def bar():\n    return 2\n",
        passed=True,
        test_log="ok",
    )
    context = retrieve_memory(conn, "alpha beta", top_n=5, match_mode="or")
    scores = [hit.score for hit in context.hits]
    assert len(scores) >= 2
    threshold = (min(scores) + max(scores)) / 2
    filtered = retrieve_memory(conn, "alpha beta", top_n=5, min_score=threshold, match_mode="or")
    assert filtered.hits
    assert all(hit.score <= threshold for hit in filtered.hits)
    assert len(filtered.hits) < len(context.hits)
    conn.close()


def test_or_mode_returns_superset(tmp_path: Path) -> None:
    db_path = _setup_db(tmp_path)
    conn = connect(db_path)
    insert_episode(
        conn,
        task_id="t1",
        condition="train",
        prompt="alpha",
        candidate_code="def foo():\n    return 1\n",
        passed=True,
        test_log="ok",
    )
    insert_episode(
        conn,
        task_id="t2",
        condition="train",
        prompt="beta",
        candidate_code="def bar():\n    return 2\n",
        passed=True,
        test_log="ok",
    )
    and_context = retrieve_memory(conn, "alpha beta", top_n=5, match_mode="and")
    or_context = retrieve_memory(conn, "alpha beta", top_n=5, match_mode="or")
    assert len(or_context.hits) >= len(and_context.hits)
    conn.close()
