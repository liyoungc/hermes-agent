#!/usr/bin/env python3
"""Seed `semantic_facts` from a flat ``MEMORY.md`` (W2-2).

Spec: docs/superpowers/specs/2026-05-02-hermes-memory-design.md §6.1.

Format expected in ``~/.hermes/memories/MEMORY.md``::

    Topic: content
    §
    Topic: another content
    §

Each entry becomes one row in ``semantic_facts``:

    entity      = "禮揚." + slug(topic)   # "Working style"           -> "禮揚.working_style"
                                          # "Tools & Access > Proton" -> "禮揚.tools_access.proton"
    fact        = content (verbatim)
    importance  = 2
    valid_from  = '2026-05-10'
    valid_to    = NULL

Idempotent: re-running with the same input does not duplicate rows. The
natural key is ``(entity, fact)`` and is enforced by a pre-INSERT lookup.

Embeddings come from Voyage 3.5-lite via ``plugins.memory.sqlite_vec.embed``.
The trigger ``sf_after_insert`` keeps ``vec_facts`` synced automatically, so
this script writes only to ``semantic_facts``.

Usage::

    docker exec -w /opt/hermes hermes /opt/hermes/.venv/bin/python3 \
        scripts/import_md.py --dry-run
    docker exec -w /opt/hermes hermes /opt/hermes/.venv/bin/python3 \
        scripts/import_md.py --commit
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_MD = Path.home() / ".hermes" / "memories" / "MEMORY.md"
DEFAULT_DB = Path("/opt/data") / "memories" / "memory.db"
DEFAULT_VALID_FROM = "2026-05-10"  # spec §6.1
DEFAULT_IMPORTANCE = 2
DEFAULT_BATCH = 128
ENTITY_PREFIX = "禮揚"
ENTRY_SEPARATOR = re.compile(r"^§\s*$", re.MULTILINE)


@dataclass
class Entry:
    """One parsed MEMORY.md entry."""

    topic: str
    fact: str

    @property
    def entity(self) -> str:
        return f"{ENTITY_PREFIX}.{slugify_topic(self.topic)}"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def slugify_topic(topic: str) -> str:
    """Convert a human topic label to a stable entity-suffix slug.

    - Hierarchy markers ``>`` become ``.`` so prefix queries still work.
    - Lowercase, ASCII alphanum kept; runs of other chars collapse to ``_``.
    - CJK / unicode is preserved unchanged so 中文 topics stay readable.

    Examples:
        "Working style"                  -> "working_style"
        "Tools & Access > ProtonMail"   -> "tools_access.protonmail"
        "禮揚.家庭"                       -> "禮揚.家庭"  (already a slug, untouched)
    """
    parts = [p.strip() for p in topic.split(">")]
    out_parts = []
    for part in parts:
        s = part.strip().lower()
        # Collapse non-alphanum (including '&', spaces, punctuation) to underscore.
        # CJK characters are unicode word chars in Python regex with re.UNICODE
        # (default for str patterns), so [^\w] excludes them = preserved.
        s = re.sub(r"[^\w]+", "_", s, flags=re.UNICODE)
        s = s.strip("_")
        if s:
            out_parts.append(s)
    return ".".join(out_parts) if out_parts else "unknown"


def parse_memory_md(text: str) -> List[Entry]:
    """Split MEMORY.md into Entry objects.

    Skips empty blocks and blocks with no ``Topic: content`` colon. Keeps
    multi-line content (rare today but possible if a future entry wraps).
    """
    entries: List[Entry] = []
    for raw_block in ENTRY_SEPARATOR.split(text):
        block = raw_block.strip()
        if not block:
            continue
        if ":" not in block:
            logger.warning("skipping malformed block (no colon): %r", block[:60])
            continue
        topic, _, content = block.partition(":")
        topic = topic.strip()
        content = content.strip()
        if not topic or not content:
            logger.warning("skipping empty topic or content: %r", block[:60])
            continue
        entries.append(Entry(topic=topic, fact=content))
    return entries


# ---------------------------------------------------------------------------
# DB ops
# ---------------------------------------------------------------------------


def existing_keys(conn: sqlite3.Connection) -> set[Tuple[str, str]]:
    """Return the (entity, fact) pairs already present, for idempotency."""
    rows = conn.execute("SELECT entity, fact FROM semantic_facts").fetchall()
    return {(r[0], r[1]) for r in rows}


def insert_batch(
    conn: sqlite3.Connection,
    rows: List[Tuple[Entry, bytes]],
    *,
    valid_from: str,
    importance: int,
) -> int:
    """Insert one batch of (entry, embedding) pairs. Returns count inserted."""
    cur = conn.executemany(
        """
        INSERT INTO semantic_facts(entity, fact, embedding,
                                   importance, valid_from, valid_to)
        VALUES (?, ?, ?, ?, ?, NULL)
        """,
        [
            (e.entity, e.fact, blob, importance, valid_from)
            for e, blob in rows
        ],
    )
    return cur.rowcount


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def import_memory_md(
    *,
    md_path: Path,
    db_path: Path,
    dry_run: bool,
    valid_from: str = DEFAULT_VALID_FROM,
    importance: int = DEFAULT_IMPORTANCE,
    batch_size: int = DEFAULT_BATCH,
    embed_fn=None,  # injectable for tests
) -> dict:
    """Run the full import.

    Returns a summary dict: {parsed, new, skipped_dup, batches, dry_run}.
    Does not return embeddings.
    """
    text = md_path.read_text(encoding="utf-8")
    entries = parse_memory_md(text)

    # Open DB and bootstrap if needed (idempotent — store.init_db handles that).
    from plugins.memory.sqlite_vec.store import init_db
    conn = init_db(db_path)

    have = existing_keys(conn)
    new_entries = [e for e in entries if (e.entity, e.fact) not in have]
    skipped = len(entries) - len(new_entries)

    if dry_run:
        print(f"[dry-run] parsed={len(entries)} new={len(new_entries)} "
              f"already_present={skipped}")
        for e in new_entries[:10]:
            print(f"  + ({e.entity}) {e.fact[:80]!r}")
        if len(new_entries) > 10:
            print(f"  … and {len(new_entries) - 10} more")
        return {
            "parsed": len(entries),
            "new": len(new_entries),
            "skipped_dup": skipped,
            "batches": 0,
            "dry_run": True,
        }

    if not new_entries:
        print(f"nothing to import (parsed={len(entries)}, all present)")
        return {
            "parsed": len(entries),
            "new": 0,
            "skipped_dup": skipped,
            "batches": 0,
            "dry_run": False,
        }

    # Embed in batches; default uses real Voyage, tests inject a stub.
    if embed_fn is None:
        from plugins.memory.sqlite_vec.embed import voyage_embed
        embed_fn = voyage_embed

    inserted = 0
    batches = 0
    try:
        conn.execute("BEGIN")
        for i in range(0, len(new_entries), batch_size):
            chunk = new_entries[i : i + batch_size]
            blobs = await embed_fn([e.fact for e in chunk])
            if len(blobs) != len(chunk):
                raise RuntimeError(
                    f"embed returned {len(blobs)} for {len(chunk)} inputs"
                )
            inserted += insert_batch(
                conn,
                list(zip(chunk, blobs)),
                valid_from=valid_from,
                importance=importance,
            )
            batches += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    print(
        f"imported {inserted} entries in {batches} "
        f"batch{'es' if batches != 1 else ''} "
        f"(parsed={len(entries)}, skipped_dup={skipped})"
    )
    return {
        "parsed": len(entries),
        "new": inserted,
        "skipped_dup": skipped,
        "batches": batches,
        "dry_run": False,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--memory-md",
        type=Path,
        default=DEFAULT_MEMORY_MD,
        help="Path to MEMORY.md (default: ~/.hermes/memories/MEMORY.md)",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="Path to memory.db (default: /opt/data/memories/memory.db inside container)",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="Show plan, do not write")
    g.add_argument("--commit", action="store_true", help="Actually import")
    p.add_argument("--valid-from", default=DEFAULT_VALID_FROM)
    p.add_argument("--importance", type=int, default=DEFAULT_IMPORTANCE)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _build_arg_parser().parse_args(argv)

    # Live import path: ensure VOYAGE_API_KEY is loaded from ~/.hermes/.env.
    if args.commit:
        try:
            from hermes_cli.env_loader import load_hermes_dotenv
            load_hermes_dotenv(hermes_home="/opt/data", project_env=None)
        except ImportError:
            pass  # tests / non-container contexts handle env themselves

    summary = asyncio.run(
        import_memory_md(
            md_path=args.memory_md,
            db_path=args.db,
            dry_run=args.dry_run,
            valid_from=args.valid_from,
            importance=args.importance,
        )
    )
    return 0 if summary["new"] >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
