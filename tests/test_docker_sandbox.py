from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from recursive_slm_lab.verify import docker as docker_mod


class _Result:
    def __init__(self) -> None:
        self.returncode = 0
        self.stdout = ""
        self.stderr = ""


def test_docker_command_construction(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _Result()

    monkeypatch.setattr(docker_mod, "ensure_docker_image", lambda: None)
    monkeypatch.setattr(docker_mod.subprocess, "run", fake_run)

    result = docker_mod.run_in_docker_sandbox(
        "def foo():\n    return 1\n",
        "import pytest\nfrom solution import *\n\ndef test_ok():\n    assert foo() == 1\n",
    )
    cmd = captured["cmd"]
    assert "--network" in cmd and "none" in cmd
    assert "--read-only" in cmd
    assert "--memory" in cmd
    assert "--cpus" in cmd
    assert "--pids-limit" in cmd
    assert "--tmpfs" in cmd
    assert "-v" in cmd
    assert "/work:ro" in " ".join(cmd)
    assert "pytest" in cmd
    assert result.passed
