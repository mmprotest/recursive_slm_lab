from __future__ import annotations

from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from recursive_slm_lab.util import write_manifest


def test_manifest_is_deterministic(tmp_path: Path) -> None:
    payload = {"alpha": 1, "beta": 2}
    path = write_manifest(
        "run-iteration",
        payload,
        out_dir=tmp_path,
        timestamp="20250101T000000Z",
    )
    assert path.name == "20250101T000000Z_run-iteration.json"
    saved = json.loads(path.read_text())
    assert saved["alpha"] == 1
