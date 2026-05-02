"""Hermes V3 memory plugin — sqlite-vec store with two-tier (hot/cold) design.

Activate via $HERMES_HOME/config.yaml:

    memory:
      provider: sqlite_vec

Read path (W2-3): on each turn, ``prefetch(query)`` runs
``read_memory()`` in a worker thread (the gateway already owns the main
asyncio loop, so we can't ``asyncio.run`` inline) and returns a markdown
block prefixed with ``## Recent relevant memories``. The retrieved fact
IDs are cached per session and bumped via ``sync_turn()`` after the
reply is sent, per spec §4 hits accounting.

Write path (W3-2): ``sync_turn`` now also fires ``write_episode`` —
records the raw turn into ``episodes``, runs Kimi extract, fast-tracks
short-lived facts directly into ``semantic_facts`` (≤ today + 30d),
stashes longer-lived facts into ``episodes.metadata.stashed_facts``
for W3-3 weekly_promotion. Errors land in
``~/.hermes/logs/memory_write_failures.jsonl`` and never propagate.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

from .read import (
    DEFAULT_K,
    Fact,
    bump_hits,
    format_facts_for_prompt,
    read_memory,
)
from .store import init_db
from .write import write_episode

logger = logging.getLogger(__name__)

PREFETCH_TIMEOUT_S = 5.0  # Voyage typical 200-400ms; 5s is the kill-switch.
# Write path: extract (~1-3s) + embed batch (~300ms) + INSERT (~5ms).
# 30s gives Kimi room to think while still bounding worst-case latency.
WRITE_TIMEOUT_S = 30.0
RECALL_HEADER = "## Recent relevant memories"


def _mem_off_active() -> bool:
    """True iff the global /mem off kill switch sentinel is present.

    Late import to avoid circular plugin loading: plugins.memreview can
    import provider symbols indirectly via the slash-command surface.
    """
    try:
        from plugins.memreview import mem_off_active
        return mem_off_active()
    except Exception:
        return False


def _default_db_path(hermes_home: str) -> Path:
    return Path(hermes_home).expanduser() / "memories" / "memory.db"


def _run_coro_in_thread(coro_factory, timeout: float):
    """Run an async coroutine in a worker thread with its own event loop.

    The hermes gateway runs its own asyncio loop, so ``asyncio.run`` from
    this synchronous ABC method would raise "cannot be called from a
    running event loop". We sidestep by spawning a dedicated thread with a
    fresh loop, joining with a timeout. ``coro_factory`` is a zero-arg
    callable that builds the coroutine inside the worker so the coroutine
    is bound to the worker's loop.
    """
    box: Dict[str, Any] = {}

    def runner():
        loop = asyncio.new_event_loop()
        try:
            box["result"] = loop.run_until_complete(coro_factory())
        except BaseException as exc:
            box["error"] = exc
        finally:
            loop.close()

    t = threading.Thread(target=runner, daemon=True, name="sqlite-vec-worker")
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"sqlite_vec worker exceeded {timeout}s")
    if "error" in box:
        raise box["error"]
    return box.get("result")


def _synth_msg_id(session_id: str, user: str, asst: str, ts: str) -> str:
    """Stable per-turn external_id for ON CONFLICT idempotency.

    We don't have the real Discord message ID at sync_turn time (the
    ABC hook only exposes user/assistant content + session_id), so we
    hash the turn into a 12-hex-char id. Bucketing ts to the minute
    means a Discord redelivery within the same minute collapses; a
    legitimate retry after >1 min would create a new row, which is
    acceptable for episode-level forensics.
    """
    raw = (session_id, user, asst, ts[:16])
    return "h" + hex(abs(hash(raw)) & 0xFFFFFFFFFFFF)[2:]


class SqliteVecMemoryProvider(MemoryProvider):
    """Hermes V3 long-term memory provider (W2-3 read + W3-2 write)."""

    def __init__(self) -> None:
        self._conn = None
        self._db_path: Optional[Path] = None
        self._last_fact_ids: Dict[str, List[int]] = {}
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "sqlite_vec"

    def is_available(self) -> bool:
        try:
            import sqlite_vec  # noqa: F401
        except ImportError:
            return False
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home")
        if not hermes_home:
            from hermes_constants import get_hermes_home
            hermes_home = str(get_hermes_home())
        self._db_path = _default_db_path(hermes_home)
        self._conn = init_db(self._db_path, check_same_thread=False)
        logger.info("sqlite_vec memory ready at %s", self._db_path)

    def system_prompt_block(self) -> str:
        # Persona stays in flat files (SOUL.md, USER.md, life-dimensions.md);
        # the recall block is emitted from prefetch() per turn.
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Embed query, fetch top-k facts, format as a markdown block.

        Returns "" on empty/trivial query, missing connection, or any
        error (Voyage outage, rate limit, etc.) so the gateway never
        blocks a reply on memory recall. Retrieved fact IDs are stashed
        for the matching ``sync_turn()`` call to bump hits.
        """
        if not self._conn or not query or not query.strip():
            return ""

        conn = self._conn
        db_lock = self._lock

        async def _do() -> List[Fact]:
            with db_lock:
                return await read_memory(query, conn, k=DEFAULT_K)

        try:
            facts = _run_coro_in_thread(_do, timeout=PREFETCH_TIMEOUT_S)
        except Exception as exc:
            logger.warning("sqlite_vec prefetch error: %s", exc)
            return ""

        if not facts:
            return ""

        with self._lock:
            self._last_fact_ids[session_id] = [f.id for f in facts]

        body = format_facts_for_prompt(facts, with_meta=True)
        return f"{RECALL_HEADER}\n{body}"

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
    ) -> None:
        """Bump hits on retrieved facts and persist the turn.

        Spec §4 + §5.1 — both happen AFTER the reply is delivered, so
        this must never raise. ``bump_hits`` swallows its own DB errors;
        ``write_episode`` swallows everything and writes failures to
        ~/.hermes/logs/memory_write_failures.jsonl.
        """
        if not self._conn:
            return
        conn = self._conn
        db_lock = self._lock

        with self._lock:
            ids = self._last_fact_ids.pop(session_id, [])

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        msg_id = _synth_msg_id(session_id, user_content, assistant_content, ts)
        channel = session_id or "unknown"

        async def _do_bump() -> None:
            if ids:
                with db_lock:
                    await bump_hits(ids, conn)

        async def _do_write() -> None:
            with db_lock:
                await write_episode(
                    user_msg=user_content,
                    reply=assistant_content,
                    channel=channel,
                    msg_id=msg_id,
                    ts=ts,
                    conn=conn,
                )

        try:
            _run_coro_in_thread(_do_bump, timeout=PREFETCH_TIMEOUT_S)
        except Exception as exc:
            logger.warning("sqlite_vec bump_hits worker error: %s", exc)

        if user_content or assistant_content:
            # /mem off kill switch: skip write_episode entirely. The hot path
            # bump_hits ran above (read-side accounting), but no new
            # episodes / facts are persisted. Read remains unaffected.
            if _mem_off_active():
                logger.info("sqlite_vec write_episode skipped (/mem off)")
            else:
                try:
                    _run_coro_in_thread(_do_write, timeout=WRITE_TIMEOUT_S)
                except Exception as exc:
                    logger.warning("sqlite_vec write_episode worker error: %s", exc)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Any:
        from tools.registry import tool_error
        return tool_error(f"sqlite_vec exposes no tools (got {tool_name!r})")

    def shutdown(self) -> None:
        if getattr(self, "_conn", None):
            self._conn.close()
            self._conn = None
