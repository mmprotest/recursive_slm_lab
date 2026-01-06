from __future__ import annotations

from dataclasses import dataclass

from .sandbox import run_in_sandbox


@dataclass
class VerificationResult:
    passed: bool
    log: str


def verify_candidate(solution_code: str, test_code: str) -> VerificationResult:
    result = run_in_sandbox(solution_code, test_code)
    log = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    return VerificationResult(passed=result.passed, log=log)
