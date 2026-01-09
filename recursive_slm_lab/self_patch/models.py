from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PatchProposal:
    diff_text: str
    raw_response: str
    model: str | None = None


@dataclass
class PatchGateStep:
    name: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    passed: bool


@dataclass
class PatchGateResult:
    passed: bool
    steps: list[PatchGateStep]
    eval_payload: dict | None = None
    error: str | None = None


@dataclass
class PatchPromotionResult:
    proposal: PatchProposal | None
    gate_result: PatchGateResult
    applied: bool
    committed: bool
    message: str
    artifact_path: Path | None = None
