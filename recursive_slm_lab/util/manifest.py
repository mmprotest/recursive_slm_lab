from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def stable_timestamp(timestamp: str | None = None) -> str:
    if timestamp is not None:
        return timestamp
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def write_manifest(
    command: str,
    payload: dict[str, Any],
    out_dir: str | Path = "artifacts/runs",
    timestamp: str | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = stable_timestamp(timestamp)
    path = out_dir / f"{ts}_{command}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
