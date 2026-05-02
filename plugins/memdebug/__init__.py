"""``/memdebug`` Discord slash command — read-only retrieval diagnostic (W2-4).

Spec: docs/superpowers/specs/2026-05-02-hermes-memory-design.md §7.2.

Usage in chat:

    /memdebug <query>            -> top-8 from semantic_facts (curated)
    /memdebug rawsearch <query>  -> top-8 from episodes (raw turns, forensics)

The handler intentionally returns plain markdown text (not a Discord
embed): hermes-agent's ``register_command()`` surface is platform-neutral
and dispatches the same string to CLI / gateway / Slack.

The ``rich-embed + 👍/👎 reaction buttons`` mode is open spec §8 work — we
ship the read-only diagnostic now so the F2 monitoring path (% of
top-1 hits judged useful) is unblocked. For v1, encourage the user
to react with 👍/👎 emoji on this message; a future cron will scrape
those reactions from the channel.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

def _resolve_hermes_home() -> Path:
    """Use HERMES_HOME (set by hermes_constants) when available; else ~/.hermes."""
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        return Path.home() / ".hermes"


_HERMES_HOME = _resolve_hermes_home()
DEFAULT_DB = _HERMES_HOME / "memories" / "memory.db"
DEFAULT_K = 8
LOG_PATH = _HERMES_HOME / "logs" / "memory.log"


def _format_facts_block(facts) -> str:
    lines = ["**🧠 /memdebug** — top {} from `semantic_facts`\n".format(len(facts))]
    for i, f in enumerate(facts, start=1):
        recency = max(0.0, 1.0 - f.age_days / 365.0)  # display-only;rerank weight uses 90-day half-life
        lines.append(
            f"`{i}.` **[{f.entity or '—'}]** {_truncate(f.fact, 90)}\n"
            f"     score=`{f.score:.3f}` sim=`{f.sim:.3f}` "
            f"age=`{int(f.age_days)}d` importance=`{f.importance}`"
        )
    lines.append("\n_React 👍/👎 to flag this retrieval._")
    return "\n".join(lines)


def _truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _format_episodes_block(rows: List[sqlite3.Row]) -> str:
    if not rows:
        return (
            "**🧠 /memdebug rawsearch** — `episodes` table is empty.\n\n"
            "Episodes are written by W3 (per-turn write-back). After W3 "
            "ships, this command will surface the raw conversation turns "
            "behind any retrieval."
        )
    lines = ["**🧠 /memdebug rawsearch** — top {} from `episodes`\n".format(len(rows))]
    for i, r in enumerate(rows, start=1):
        lines.append(
            f"`{i}.` `[{r['ts']}]` `{r['channel']}/{r['role']}` "
            f"{_truncate(r['text'], 120)}"
        )
    return "\n".join(lines)


def _append_log(payload: dict) -> None:
    """Append a /memdebug invocation to ~/.hermes/logs/memory.log."""
    import json
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("memory.log write failed: %s", exc)


def _open_memory_db(path: Optional[Path] = None) -> Optional[sqlite3.Connection]:
    """Open the sqlite_vec memory.db. Returns None if it doesn't exist yet."""
    path = path or DEFAULT_DB
    if not path.exists():
        return None
    from plugins.memory.sqlite_vec.store import open_db
    return open_db(path, check_same_thread=False)


async def _do_semantic(query: str) -> str:
    from plugins.memory.sqlite_vec.read import read_memory

    conn = _open_memory_db()
    if not conn:
        return (
            "**🧠 /memdebug** — memory database not yet initialised.\n\n"
            f"Expected at `{DEFAULT_DB}`. Run `scripts/import_md.py --commit` "
            "or wait for the first agent turn after W2-3 cutover."
        )
    try:
        facts = await read_memory(query, conn, k=DEFAULT_K)
    finally:
        conn.close()
    if not facts:
        return f"**🧠 /memdebug** — no facts matched `{_truncate(query, 60)}`."
    _append_log({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cmd": "memdebug",
        "q": query,
        "n": len(facts),
        "ids": [f.id for f in facts],
    })
    return _format_facts_block(facts)


async def _do_rawsearch(query: str) -> str:
    """Substring scan of episodes.text. No vector query — this is forensics
    mode for 'did this conversation happen', not semantic recall."""
    conn = _open_memory_db()
    if not conn:
        return (
            "**🧠 /memdebug rawsearch** — memory database not yet initialised."
        )
    try:
        like = f"%{query}%"
        rows = conn.execute(
            "SELECT ts, channel, role, text FROM episodes "
            "WHERE text LIKE ? ORDER BY ts DESC LIMIT ?",
            (like, DEFAULT_K),
        ).fetchall()
    finally:
        conn.close()
    _append_log({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cmd": "memdebug-raw",
        "q": query,
        "n": len(rows),
    })
    return _format_episodes_block(rows)


HELP_TEXT = (
    "**/memdebug** — inspect what `read_memory` would return.\n"
    "Usage:\n"
    "  `/memdebug <query>` — top-8 from `semantic_facts` (curated)\n"
    "  `/memdebug rawsearch <query>` — substring scan of `episodes` (forensics)\n"
)


async def _handle_async(raw_args: str) -> str:
    args = (raw_args or "").strip()
    if not args:
        return HELP_TEXT
    if args.lower().startswith("rawsearch"):
        rest = args[len("rawsearch"):].strip()
        if not rest:
            return HELP_TEXT
        try:
            return await _do_rawsearch(rest)
        except Exception as exc:
            logger.exception("memdebug rawsearch failed")
            return f"**/memdebug rawsearch** error: `{exc}`"
    try:
        return await _do_semantic(args)
    except Exception as exc:
        logger.exception("memdebug semantic failed")
        return f"**/memdebug** error: `{exc}`"


def _handle_memdebug(raw_args: str) -> str:
    """Sync entry point. PluginContext.register_command supports async
    handlers natively, but ours is dispatched on either pathway, so we
    bridge via asyncio.run when no loop is running."""
    coro = _handle_async(raw_args)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is None:
        return asyncio.run(coro)
    # Already in a running loop — schedule and wait via a worker thread.
    import threading
    import concurrent.futures
    box = {}

    def runner():
        try:
            box["r"] = asyncio.run(coro)
        except BaseException as exc:
            box["e"] = exc

    t = threading.Thread(target=runner, daemon=True, name="memdebug-handler")
    t.start()
    t.join(timeout=15.0)
    if t.is_alive():
        return "**/memdebug** timed out (>15s)."
    if "e" in box:
        return f"**/memdebug** error: `{box['e']}`"
    return box.get("r", HELP_TEXT)


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    ctx.register_command(
        "memdebug",
        handler=_handle_memdebug,
        description="Inspect Hermes long-term memory retrieval (top-8 + scores).",
        args_hint="<query> | rawsearch <query>",
    )
