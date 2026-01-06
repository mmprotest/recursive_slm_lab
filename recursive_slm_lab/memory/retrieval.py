from __future__ import annotations

import sqlite3
from dataclasses import dataclass


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


def retrieve_memory(conn: sqlite3.Connection, query: str, top_n: int = 3) -> MemoryContext:
    hits: list[MemoryHit] = []
    safe_query = " ".join("".join(ch if ch.isalnum() else " " for ch in query).split())
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

    return MemoryContext(hits=hits)
