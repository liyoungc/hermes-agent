"""Read path for the sqlite_vec memory plugin.

Spec: docs/superpowers/specs/2026-05-02-hermes-memory-design.md §4

Two-step retrieval:
  1. vec0 prefilter: top k=50 by cosine distance on int8 embeddings
  2. SQL CTE rerank: score = (1 - distance) * 0.7 + exp(-age_days/90) * 0.3
     filter active state + valid_to NULL or future, ORDER BY score DESC LIMIT k

`hits` bumping happens fire-and-forget after the reply is sent (caller's
responsibility to schedule). Errors are swallowed with a warning.

p95 query latency is logged to ~/.hermes/logs/memory.log. The log path is
overridable via the constructor for testing.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from .embed import voyage_embed

logger = logging.getLogger(__name__)

DEFAULT_K = 8
PREFILTER_K = 50
DEFAULT_LOG_PATH = Path.home() / ".hermes" / "logs" / "memory.log"

# Spec §4 — SQL is locked. Do not edit weights without updating the spec
# and re-running the B1 worked example.
RETRIEVE_SQL = """
WITH knn AS (
    SELECT fact_id, distance
    FROM vec_facts
    WHERE embedding MATCH vec_int8(?) AND k = {prefilter_k}
)
SELECT sf.id, sf.fact, sf.entity, sf.created_at, sf.importance,
       (1 - knn.distance)                                              AS sim,
       (julianday('now') - julianday(sf.created_at))                   AS age_days,
       (1 - knn.distance) * 0.7
         + exp(-(julianday('now') - julianday(sf.created_at)) / 90.0) * 0.3 AS score
FROM knn
JOIN semantic_facts sf ON sf.id = knn.fact_id
WHERE sf.state = 'active'
  AND (sf.valid_to IS NULL OR sf.valid_to > date('now'))
ORDER BY score DESC
LIMIT ?;
"""


@dataclass
class Fact:
    """A retrieved fact with score breakdown for prompt-injection or /memdebug."""

    id: int
    fact: str
    entity: Optional[str]
    created_at: str
    importance: int
    sim: float
    age_days: float
    score: float


def _append_log(log_path: Path, payload: dict) -> None:
    """Append one JSON line to memory.log; never raise into the read path."""
    import json
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("memory.log write failed: %s", exc)


async def read_memory(
    query: str,
    conn: sqlite3.Connection,
    *,
    k: int = DEFAULT_K,
    log_path: Path = DEFAULT_LOG_PATH,
) -> List[Fact]:
    """Embed `query`, retrieve top-`k` facts, log latency, return Fact list."""
    [qvec] = await voyage_embed([query])

    sql = RETRIEVE_SQL.format(prefilter_k=PREFILTER_K)
    t0 = time.perf_counter()
    rows = conn.execute(sql, (qvec, k)).fetchall()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    facts = [
        Fact(
            id=row["id"],
            fact=row["fact"],
            entity=row["entity"],
            created_at=row["created_at"],
            importance=row["importance"],
            sim=float(row["sim"]),
            age_days=float(row["age_days"]),
            score=float(row["score"]),
        )
        for row in rows
    ]

    _append_log(
        log_path,
        {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "q": query,
            "k": k,
            "n": len(facts),
            "sql_ms": round(elapsed_ms, 2),
        },
    )
    return facts


async def bump_hits(fact_ids: Iterable[int], conn: sqlite3.Connection) -> None:
    """Fire-and-forget UPDATE; swallow errors with a warning log.

    Caller must wrap with ``asyncio.create_task()`` to avoid blocking the
    reply. Per spec §4 hits-bump runs AFTER discord_send, so we keep this
    cheap (single UPDATE … IN (…)) and never propagate errors.
    """
    ids = list(fact_ids)
    if not ids:
        return
    placeholders = ",".join("?" * len(ids))
    try:
        conn.execute(
            f"UPDATE semantic_facts SET hits = hits + 1, "
            f"last_seen = datetime('now') WHERE id IN ({placeholders})",
            ids,
        )
        conn.commit()
    except sqlite3.Error as exc:
        logger.warning("bump_hits swallowed error for %d ids: %s", len(ids), exc)


def format_facts_for_prompt(facts: List[Fact]) -> str:
    """Render top-k facts as a markdown bullet list for system-prompt injection.

    Used by SqliteVecMemoryProvider.prefetch() in W2-3. Compact, no header —
    the surrounding prompt template owns the section title.
    """
    if not facts:
        return ""
    lines = []
    for f in facts:
        prefix = f"[{f.entity}] " if f.entity else ""
        lines.append(f"- {prefix}{f.fact}")
    return "\n".join(lines)
