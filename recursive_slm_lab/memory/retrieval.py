from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from ..policy import Policy


@dataclass
class MemoryHit:
    source: str
    text: str
    score: float


@dataclass
class MemoryContext:
    hits: list[MemoryHit]

    def format(self) -> str:
        if not self.hits:
            return ""
        lines = ["Memory Context:"]
        for hit in self.hits:
            lines.append(f"- [{hit.source}] score={hit.score:.2f}")
            lines.append(hit.text)
        return "\n".join(lines)

    def first_code(self) -> str | None:
        for hit in self.hits:
            if hit.source == "episode" and hit.text.strip().startswith("def"):
                return hit.text.strip()
        return None

    def filter_sources(self, allowed: set[str]) -> MemoryContext:
        return MemoryContext(hits=[hit for hit in self.hits if hit.source in allowed])


def normalize_query(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower())
    tokens = [token for token in cleaned.split() if not token.isdigit()]
    return " ".join(tokens)


def retrieve_memory(
    conn: sqlite3.Connection,
    query: str,
    policy: Policy | None = None,
    top_n: int = 3,
    extra_terms: list[str] | None = None,
    min_score: float | None = None,
    extra_terms_mode: str | None = None,
    function_name: str | None = None,
) -> MemoryContext:
    if policy:
        top_n = policy.retrieval_top_n
        min_score = policy.retrieval_min_score
        extra_terms_mode = policy.retrieval_extra_terms_mode
    if top_n <= 0:
        return MemoryContext(hits=[])
    hits: list[MemoryHit] = []
    normalized = normalize_query(query)
    query_tokens = normalized.split() if normalized else []
    extra_tokens: list[str] = []
    if extra_terms_mode and not extra_terms:
        extra_terms = _extra_terms_from_mode(extra_terms_mode, function_name)
    if extra_terms:
        for term in extra_terms:
            extra_tokens.extend(normalize_query(term).split())
    if not query_tokens:
        query_tokens = extra_tokens
    else:
        query_tokens.extend(extra_tokens)
    safe_query = " ".join(query_tokens)
    if not safe_query:
        return MemoryContext(hits=hits)
    episode_rows = conn.execute(
        """
        SELECT episodes.candidate_code, bm25(episodes_fts) as score
        FROM episodes_fts
        JOIN episodes ON episodes_fts.rowid = episodes.id
        WHERE episodes_fts MATCH ? AND episodes.passed = 1
        ORDER BY score LIMIT ?
        """,
        (safe_query, top_n),
    ).fetchall()

    for code, score in episode_rows:
        hits.append(MemoryHit(source="episode", text=code, score=score))

    rule_rows = conn.execute(
        """
        SELECT semantic_rules.rule_text, bm25(rules_fts) as score
        FROM rules_fts
        JOIN semantic_rules ON rules_fts.rowid = semantic_rules.id
        WHERE rules_fts MATCH ?
        ORDER BY score LIMIT ?
        """,
        (safe_query, top_n),
    ).fetchall()

    for text, score in rule_rows:
        hits.append(MemoryHit(source="rule", text=text, score=score))

    proc_rows = conn.execute(
        """
        SELECT procedures.recipe_text, bm25(procedures_fts) as score
        FROM procedures_fts
        JOIN procedures ON procedures_fts.rowid = procedures.id
        WHERE procedures_fts MATCH ?
        ORDER BY score LIMIT ?
        """,
        (safe_query, top_n),
    ).fetchall()

    for text, score in proc_rows:
        hits.append(MemoryHit(source="procedure", text=text, score=score))

    if min_score is not None:
        hits = [hit for hit in hits if hit.score >= min_score]
    return MemoryContext(hits=hits)


def _extra_terms_from_mode(mode: str, function_name: str | None) -> list[str]:
    if not function_name:
        return []
    if mode == "none":
        return []
    if mode == "name_only":
        return [function_name]
    if mode == "function_prefix+name":
        prefix = function_name.rsplit("_", 1)[0]
        if prefix != function_name:
            return [prefix, function_name]
        return [function_name]
    return [function_name]
