"""Tests for ``scripts/import_md.py`` (W2-2 — MEMORY.md → semantic_facts).

Uses a stub embed_fn so no network is hit; live integration is exercised
end-to-end on chococlaw via the post-test ``--commit`` smoke run.
"""

from __future__ import annotations

import asyncio
import struct
from pathlib import Path

import pytest

from plugins.memory.sqlite_vec.store import VEC_DIM, init_db
from scripts.import_md import (
    Entry,
    import_memory_md,
    parse_memory_md,
    slugify_topic,
)


def _vec(seed: int) -> bytes:
    vals = [max(-128, min(127, seed + (i % 7) - 3)) for i in range(VEC_DIM)]
    return struct.pack(f"{VEC_DIM}b", *vals)


# ---------------------------------------------------------------------------
# Slugify
# ---------------------------------------------------------------------------


def test_slugify_simple():
    assert slugify_topic("People") == "people"
    assert slugify_topic("Working style") == "working_style"
    assert slugify_topic("Privacy constraints") == "privacy_constraints"


def test_slugify_hierarchy_uses_dot():
    assert (
        slugify_topic("Tools & Access > ProtonMail Access")
        == "tools_access.protonmail_access"
    )


def test_slugify_preserves_cjk():
    # CJK characters survive the punct->underscore collapse; only > is hierarchy.
    assert slugify_topic("醫院 > 新樓") == "醫院.新樓"
    assert slugify_topic("家庭 生活") == "家庭_生活"


def test_slugify_handles_empty_or_punct_only():
    assert slugify_topic("") == "unknown"
    assert slugify_topic("!!!") == "unknown"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


SAMPLE_MD = """People: 禮揚 — physician
§
Working style: digital twin model
§
Privacy constraints: never include real PHI
§
Tools & Access > ProtonMail: D4303@sinlau.org.tw
§
"""


def test_parse_memory_md_basic():
    entries = parse_memory_md(SAMPLE_MD)
    assert len(entries) == 4
    assert entries[0].topic == "People"
    assert entries[0].fact == "禮揚 — physician"
    assert entries[0].entity == "禮揚.people"
    assert entries[3].entity == "禮揚.tools_access.protonmail"


def test_parse_skips_blocks_without_colon():
    md = "first entry: ok\n§\n\nno colon here\n§\nsecond: also ok\n§\n"
    entries = parse_memory_md(md)
    assert [e.topic for e in entries] == ["first entry", "second"]


def test_parse_handles_no_trailing_separator():
    md = "topic: content"
    entries = parse_memory_md(md)
    assert len(entries) == 1
    assert entries[0].fact == "content"


# ---------------------------------------------------------------------------
# import_memory_md (with stub embed)
# ---------------------------------------------------------------------------


def _make_stub_embed():
    counter = {"n": 0}

    async def stub(texts):
        counter["n"] += 1
        return [_vec(i + 1) for i, _ in enumerate(texts)]

    return stub, counter


def test_dry_run_does_not_write(tmp_path):
    md = tmp_path / "MEMORY.md"
    md.write_text(SAMPLE_MD, encoding="utf-8")
    db = tmp_path / "m.db"

    summary = asyncio.run(
        import_memory_md(md_path=md, db_path=db, dry_run=True)
    )
    assert summary == {
        "parsed": 4, "new": 4, "skipped_dup": 0,
        "batches": 0, "dry_run": True,
    }
    # DB still empty (init_db ran but no inserts).
    conn = init_db(db)
    [(count,)] = conn.execute("SELECT count(*) FROM semantic_facts").fetchall()
    assert count == 0


def test_commit_inserts_and_populates_vec_facts(tmp_path):
    md = tmp_path / "MEMORY.md"
    md.write_text(SAMPLE_MD, encoding="utf-8")
    db = tmp_path / "m.db"
    stub, counter = _make_stub_embed()

    summary = asyncio.run(
        import_memory_md(md_path=md, db_path=db, dry_run=False, embed_fn=stub)
    )
    assert summary["new"] == 4
    assert summary["batches"] == 1
    assert counter["n"] == 1  # one Voyage call for 4 entries

    conn = init_db(db)
    rows = conn.execute(
        "SELECT entity, fact, importance, valid_from, valid_to FROM semantic_facts ORDER BY id"
    ).fetchall()
    assert len(rows) == 4
    assert rows[0]["entity"] == "禮揚.people"
    assert rows[0]["importance"] == 2
    assert rows[0]["valid_from"] == "2026-05-10"
    assert rows[0]["valid_to"] is None

    # Trigger sf_after_insert mirrored every row into vec_facts.
    [(vec_count,)] = conn.execute("SELECT count(*) FROM vec_facts").fetchall()
    assert vec_count == 4


def test_idempotent_rerun_inserts_nothing_new(tmp_path):
    md = tmp_path / "MEMORY.md"
    md.write_text(SAMPLE_MD, encoding="utf-8")
    db = tmp_path / "m.db"
    stub, counter = _make_stub_embed()

    asyncio.run(import_memory_md(md_path=md, db_path=db, dry_run=False, embed_fn=stub))
    assert counter["n"] == 1

    summary2 = asyncio.run(
        import_memory_md(md_path=md, db_path=db, dry_run=False, embed_fn=stub)
    )
    assert summary2["new"] == 0
    assert summary2["skipped_dup"] == 4
    assert counter["n"] == 1  # second run made zero embed calls (no new rows)

    conn = init_db(db)
    [(count,)] = conn.execute("SELECT count(*) FROM semantic_facts").fetchall()
    assert count == 4


def test_partial_update_only_embeds_new(tmp_path):
    md = tmp_path / "MEMORY.md"
    md.write_text(SAMPLE_MD, encoding="utf-8")
    db = tmp_path / "m.db"
    stub, counter = _make_stub_embed()

    asyncio.run(import_memory_md(md_path=md, db_path=db, dry_run=False, embed_fn=stub))
    assert counter["n"] == 1

    md.write_text(SAMPLE_MD + "\nNew topic: brand new fact\n§\n", encoding="utf-8")
    summary = asyncio.run(
        import_memory_md(md_path=md, db_path=db, dry_run=False, embed_fn=stub)
    )
    assert summary["new"] == 1
    assert summary["skipped_dup"] == 4
    assert counter["n"] == 2  # one extra call for the one new entry


def test_rollback_on_embed_failure_leaves_db_unchanged(tmp_path):
    md = tmp_path / "MEMORY.md"
    md.write_text(SAMPLE_MD, encoding="utf-8")
    db = tmp_path / "m.db"

    async def failing(texts):
        raise RuntimeError("voyage exploded")

    with pytest.raises(RuntimeError, match="voyage exploded"):
        asyncio.run(
            import_memory_md(md_path=md, db_path=db, dry_run=False, embed_fn=failing)
        )
    conn = init_db(db)
    [(count,)] = conn.execute("SELECT count(*) FROM semantic_facts").fetchall()
    assert count == 0  # transaction rolled back
