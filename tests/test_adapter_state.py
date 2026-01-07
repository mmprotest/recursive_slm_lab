from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from recursive_slm_lab.memory import (
    connect,
    get_active_adapter,
    init_db,
    register_adapter,
    rollback_adapter,
    set_active_adapter,
)


def test_adapter_rollback_clears_active_adapter(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.sqlite"
    init_db(db_path)

    conn = connect(db_path)
    register_adapter(conn, "iter001", "/tmp/iter001", notes="test")
    register_adapter(conn, "iter002", "/tmp/iter002", notes="test")
    set_active_adapter(conn, "iter001")

    assert get_active_adapter(conn) == ("iter001", "/tmp/iter001")

    rollback_adapter(conn, "iter001")

    assert get_active_adapter(conn) is None
    conn.close()
