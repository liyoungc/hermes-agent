"""Per-turn write-back into the sqlite_vec memory store (W3-2).

Spec: docs/superpowers/specs/2026-05-02-hermes-memory-design.md §5.1.

Hot-path flow per Discord turn:

  1. PHI gate — if ``channel`` is in PHI_BLACKLIST_CHANNELS, raw episode
     rows still land but extraction is skipped (no PHI to the cloud LLM).
  2. Extract — kimi_extract() returns 0..N ExtractedFacts.
  3. Embed — voyage_embed([user_msg, reply, *fact_texts]) in one batch.
  4. INSERT 2 episode rows (user, assistant) with
     ``ON CONFLICT(channel, external_id) DO NOTHING`` for idempotency
     under Discord redelivery / cron retries / container restarts.
  5. Fast-track facts whose ``valid_to_hint`` parses to ≤ today + 30d
     directly into ``semantic_facts`` (the trigger mirrors them into
     ``vec_facts``). Longer-lived / undated facts are JSON-stashed in
     ``episodes.metadata.stashed_facts`` for W3-3 weekly_promotion.
  6. Any exception → append a JSONL line to
     ``~/.hermes/logs/memory_write_failures.jsonl`` and swallow.
     The reply was already sent before this fired; we never propagate.

The function is fire-and-forget: the caller schedules it via
``asyncio.create_task`` (or in our case, a worker thread the provider
spawns) AFTER ``discord_send`` so write latency cannot stall the user.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .extract import (
    PHI_BLACKLIST_CHANNELS,
    ExtractedFact,
    kimi_extract,
)

logger = logging.getLogger(__name__)

# Spec §5.3 — fast-track threshold (raised from 7d to 30d): facts that
# expire within ~1 month land directly in semantic_facts so they're
# usable on the next turn instead of waiting up to 7 days for the
# weekly review.
FAST_TRACK_DAYS = 30


def _resolve_hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        return Path.home() / ".hermes"


def _failure_log_path() -> Path:
    return _resolve_hermes_home() / "logs" / "memory_write_failures.jsonl"


def _append_failure(payload: Dict[str, Any], log_path: Optional[Path] = None) -> None:
    log_path = log_path or _failure_log_path()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:
        logger.warning("memory_write_failures.jsonl write failed: %s", exc)


def _parse_valid_to_hint(hint: Optional[str]) -> Optional[date]:
    """Parse 'YYYY-MM-DD' tolerantly. Return None on bad / missing input."""
    if not hint:
        return None
    try:
        return datetime.strptime(hint.strip(), "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _fact_should_fast_track(fact: ExtractedFact, today: date) -> bool:
    """True iff fact has a valid_to_hint within FAST_TRACK_DAYS of today."""
    expiry = _parse_valid_to_hint(fact.valid_to_hint)
    if not expiry:
        return False
    return expiry <= today + timedelta(days=FAST_TRACK_DAYS)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def write_episode(
    user_msg: str,
    reply: str,
    channel: str,
    msg_id: str,
    ts: str,
    conn: sqlite3.Connection,
    *,
    embed_fn: Optional[Callable] = None,
    extract_fn: Optional[Callable] = None,
    failure_log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Persist one Discord turn to the memory store.

    Returns a summary dict for caller logging:
      {episodes: 0|1|2, fast_tracked: N, stashed: N, skipped_extract: bool}

    Never raises. Errors land in ``memory_write_failures.jsonl``.
    """
    summary: Dict[str, Any] = {
        "episodes": 0,
        "fast_tracked": 0,
        "stashed": 0,
        "skipped_extract": False,
    }
    skip_extract = channel in PHI_BLACKLIST_CHANNELS
    summary["skipped_extract"] = skip_extract

    try:
        # ---- 1. extract (skip on PHI channel)
        if skip_extract or not (extract_fn or kimi_extract):
            facts: List[ExtractedFact] = []
        else:
            extractor = extract_fn or kimi_extract
            try:
                facts = await extractor(user_msg, reply, channel, ts)
            except Exception as exc:
                # Extract failure is non-fatal — we still record the
                # raw episode so weekly_promotion can re-extract later.
                logger.warning("kimi_extract failed; continuing without facts: %s", exc)
                facts = []

        # ---- 2. embed (raw turn + each fact text in one call)
        embed = embed_fn
        if embed is None:
            from .embed import voyage_embed
            embed = voyage_embed

        texts_to_embed = [user_msg, reply] + [f.text for f in facts]
        # Filter empty strings — Voyage rejects them.
        non_empty = [(i, t) for i, t in enumerate(texts_to_embed) if t and t.strip()]
        if non_empty:
            indices, texts = zip(*non_empty)
            blobs_dense = await embed(list(texts))
            # Re-densify back to original positions; missing slots get None.
            blobs: List[Optional[bytes]] = [None] * len(texts_to_embed)
            for slot, blob in zip(indices, blobs_dense):
                blobs[slot] = blob
        else:
            blobs = [None] * len(texts_to_embed)

        user_blob, reply_blob = blobs[0], blobs[1]
        fact_blobs = blobs[2:]

        # ---- 3. partition facts into fast-track vs stash BEFORE INSERT
        today = date.today()
        fast_track: List[tuple] = []  # [(fact, blob), ...]
        stashed: List[Dict[str, Any]] = []  # JSON-serialisable dicts
        for f, blob in zip(facts, fact_blobs):
            if _fact_should_fast_track(f, today):
                if blob is not None:
                    fast_track.append((f, blob))
                else:
                    # No embedding for this fact → can't insert into
                    # semantic_facts (embedding is NOT NULL).  Demote to stash.
                    stashed.append(f.raw or _fact_to_dict(f))
            else:
                stashed.append(f.raw or _fact_to_dict(f))

        metadata = {"stashed_facts": stashed} if stashed else {}
        metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None

        # ---- 4. INSERT episodes (atomic with fast-track inserts)
        try:
            conn.execute("BEGIN")
            ep_rows = [
                (ts, channel, msg_id + ":user", "user", user_msg, 0, user_blob, metadata_json),
                (ts, channel, msg_id + ":asst", "assistant", reply, 0, reply_blob, metadata_json),
            ]
            for row in ep_rows:
                cur = conn.execute(
                    """
                    INSERT INTO episodes
                        (ts, channel, external_id, role, text, synthetic, embedding, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(channel, external_id) DO NOTHING
                    """,
                    row,
                )
                if cur.rowcount:
                    summary["episodes"] += 1

            # ---- 5. fast-track facts → semantic_facts (trigger mirrors to vec_facts)
            for f, blob in fast_track:
                conn.execute(
                    """
                    INSERT INTO semantic_facts
                        (entity, fact, embedding, importance, valid_from, valid_to)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f.entity,
                        f.text,
                        blob,
                        f.importance,
                        today.isoformat(),
                        f.valid_to_hint,
                    ),
                )
                summary["fast_tracked"] += 1

            summary["stashed"] = len(stashed)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        return summary

    except Exception as exc:
        logger.warning("write_episode failed for msg_id=%s: %s", msg_id, exc)
        _append_failure(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "channel": channel,
                "msg_id": msg_id,
                "user": user_msg,
                "reply": reply,
                "error": str(exc),
                "summary_so_far": summary,
            },
            log_path=failure_log_path,
        )
        return summary


def _fact_to_dict(f: ExtractedFact) -> Dict[str, Any]:
    """Serialise an ExtractedFact for stashing in episodes.metadata."""
    return {
        "type": f.type,
        "text": f.text,
        "entity": f.entity,
        "importance": f.importance,
        "valid_to_hint": f.valid_to_hint,
    }
