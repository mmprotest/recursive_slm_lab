from __future__ import annotations

import json
import re
from pathlib import Path

from ..llm.base import LLMBackend
from .models import PatchProposal


_DIFF_START = re.compile(r"^(diff --git|--- )", re.MULTILINE)


def propose_patch(backend: LLMBackend, repo_root: Path, context: dict) -> PatchProposal:
    prompt = (
        "You are proposing a code patch as a unified diff. "
        "Return ONLY a unified diff with no commentary or extra text.\n\n"
        "Guardrails:\n"
        "- Do not modify evaluation or verifier code.\n"
        "- Do not weaken gates or tests.\n"
        "- Keep the patch minimal and focused.\n"
        "- Do not touch recursive_slm_lab/loop/self_improve.py.\n\n"
        f"Repository root: {repo_root}\n"
        f"Context JSON:\n{json.dumps(context, indent=2)}\n"
    )
    response = backend.generate(
        [
            {"role": "system", "content": "Return only a unified diff."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1600,
        temperature=0.2,
        top_p=1.0,
        top_k=0,
    )
    diff_text = _extract_diff(response.text)
    return PatchProposal(diff_text=diff_text, raw_response=response.text, model=response.model)


def _extract_diff(text: str) -> str:
    text = text.strip()
    match = _DIFF_START.search(text)
    if not match:
        return text
    return text[match.start() :].strip()
