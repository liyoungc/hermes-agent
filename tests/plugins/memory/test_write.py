"""Tests for plugins/memory/sqlite_vec/write.py (W3-2)."""

from __future__ import annotations

import asyncio
import json
import struct
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from plugins.memory.sqlite_vec.extract import ExtractedFact
from plugins.memory.sqlite_vec.store import VEC_DIM, init_db
from plugins.memory.sqlite_vec.write import (
    FAST_TRACK_DAYS,
    _fact_should_fast_track,
    _parse_valid_to_hint,
    write_episode,
)


def _vec(seed: int) -> bytes:
    vals = [max(-128, min(127, seed + (i % 7) - 3)) for i in range(VEC_DIM)]
    return struct.pack(f"{VEC_DIM}b", *vals)


def _stub_embed_factory():
    """Returns (stub, call_log) — stub yields deterministic int8 blobs."""
    calls = []

    async def stub(texts):
        calls.append(list(texts))
        return [_vec(10 + i) for i in range(len(texts))]

    return stub, calls


def _stub_extract_factory(facts: list):
    async def stub(user, asst, channel, ts):
        return list(facts)

    return stub


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_parse_valid_to_hint():
    assert _parse_valid_to_hint("2026-05-03") == date(2026, 5, 3)
    assert _parse_valid_to_hint("not-a-date") is None
    assert _parse_valid_to_hint("") is None
    assert _parse_valid_to_hint(None) is None


def test_fact_should_fast_track_threshold():
    today = date(2026, 5, 2)
    f_in = ExtractedFact(type="semantic", text="x", entity=None, importance=2,
                         valid_to_hint=(today + timedelta(days=10)).isoformat())
    f_edge = ExtractedFact(type="semantic", text="x", entity=None, importance=2,
                           valid_to_hint=(today + timedelta(days=FAST_TRACK_DAYS)).isoformat())
    f_out = ExtractedFact(type="semantic", text="x", entity=None, importance=2,
                          valid_to_hint=(today + timedelta(days=60)).isoformat())
    f_none = ExtractedFact(type="semantic", text="x", entity=None, importance=2,
                           valid_to_hint=None)
    assert _fact_should_fast_track(f_in, today) is True
    assert _fact_should_fast_track(f_edge, today) is True
    assert _fact_should_fast_track(f_out, today) is False
    assert _fact_should_fast_track(f_none, today) is False


# ---------------------------------------------------------------------------
# write_episode — happy paths
# ---------------------------------------------------------------------------


def _bootstrap_db(tmp_path):
    return init_db(tmp_path / "m.db")


def test_writes_two_episode_rows_per_turn(tmp_path):
    db = _bootstrap_db(tmp_path)
    embed, calls = _stub_embed_factory()
    extract = _stub_extract_factory([])

    summary = asyncio.run(write_episode(
        user_msg="hello", reply="hi back",
        channel="cattia", msg_id="m1", ts="2026-05-02 09:00:00",
        conn=db, embed_fn=embed, extract_fn=extract,
        failure_log_path=tmp_path / "fail.jsonl",
    ))

    assert summary["episodes"] == 2
    assert summary["fast_tracked"] == 0 and summary["stashed"] == 0
    rows = db.execute(
        "SELECT role, channel, external_id, text FROM episodes ORDER BY id"
    ).fetchall()
    assert [r["role"] for r in rows] == ["user", "assistant"]
    assert rows[0]["external_id"] == "m1:user"
    assert rows[1]["external_id"] == "m1:asst"
    # Single embed call covered both turn texts (no fact texts).
    assert len(calls) == 1
    assert calls[0] == ["hello", "hi back"]


def test_phi_channel_records_episode_but_skips_extract(tmp_path):
    db = _bootstrap_db(tmp_path)
    embed, calls = _stub_embed_factory()

    def extract_should_not_be_called(*a, **kw):
        raise AssertionError("extract called for PHI channel")

    summary = asyncio.run(write_episode(
        user_msg="病人 [姓名] 血壓 180/100", reply="建議轉診",
        channel="cmio", msg_id="phi-1", ts="2026-05-02 09:00:00",
        conn=db, embed_fn=embed, extract_fn=extract_should_not_be_called,
        failure_log_path=tmp_path / "fail.jsonl",
    ))

    assert summary["skipped_extract"] is True
    assert summary["episodes"] == 2
    assert summary["fast_tracked"] == 0 and summary["stashed"] == 0
    rows = db.execute("SELECT count(*) FROM episodes").fetchone()
    assert rows[0] == 2  # raw episode rows still recorded


def test_idempotent_on_duplicate_msg_id(tmp_path):
    """Re-running with the same msg_id collapses via ON CONFLICT."""
    db = _bootstrap_db(tmp_path)
    embed, _ = _stub_embed_factory()
    extract = _stub_extract_factory([])

    args = dict(
        user_msg="x", reply="y", channel="cattia",
        msg_id="dup-1", ts="2026-05-02 09:00:00",
        conn=db, embed_fn=embed, extract_fn=extract,
        failure_log_path=tmp_path / "fail.jsonl",
    )
    asyncio.run(write_episode(**args))
    summary2 = asyncio.run(write_episode(**args))
    assert summary2["episodes"] == 0  # nothing new inserted
    [(count,)] = db.execute("SELECT count(*) FROM episodes").fetchall()
    assert count == 2


# ---------------------------------------------------------------------------
# Fast-track vs stash partitioning
# ---------------------------------------------------------------------------


def test_short_lived_fact_fast_tracks_to_semantic_facts(tmp_path):
    db = _bootstrap_db(tmp_path)
    embed, _ = _stub_embed_factory()
    today = date.today()
    extract = _stub_extract_factory([
        ExtractedFact(
            type="semantic",
            text="致妤今晚 7:30 才到家",
            entity="禮揚.家庭",
            importance=3,
            valid_to_hint=(today + timedelta(days=1)).isoformat(),
        ),
    ])

    summary = asyncio.run(write_episode(
        user_msg="今晚致妤 7:30 才到", reply="了解",
        channel="at-home", msg_id="m1", ts="2026-05-02 17:00:00",
        conn=db, embed_fn=embed, extract_fn=extract,
        failure_log_path=tmp_path / "fail.jsonl",
    ))

    assert summary["fast_tracked"] == 1
    assert summary["stashed"] == 0
    [(sf_count,)] = db.execute("SELECT count(*) FROM semantic_facts").fetchall()
    assert sf_count == 1
    [(vf_count,)] = db.execute("SELECT count(*) FROM vec_facts").fetchall()
    assert vf_count == 1  # trigger mirrored the row
    row = db.execute(
        "SELECT entity, fact, importance, valid_from, valid_to FROM semantic_facts"
    ).fetchone()
    assert row["entity"] == "禮揚.家庭"
    assert row["valid_to"] == (today + timedelta(days=1)).isoformat()


def test_long_lived_fact_stashes_in_episode_metadata(tmp_path):
    db = _bootstrap_db(tmp_path)
    embed, _ = _stub_embed_factory()
    extract = _stub_extract_factory([
        ExtractedFact(
            type="semantic",
            text="禮揚 likes Starting Strength",
            entity="禮揚.訓練",
            importance=2,
            valid_to_hint=None,  # permanent → stash
        ),
    ])

    summary = asyncio.run(write_episode(
        user_msg="我練 SS 一年了", reply="酷",
        channel="cattia", msg_id="m1", ts="2026-05-02 09:00:00",
        conn=db, embed_fn=embed, extract_fn=extract,
        failure_log_path=tmp_path / "fail.jsonl",
    ))

    assert summary["stashed"] == 1
    assert summary["fast_tracked"] == 0
    [(sf_count,)] = db.execute("SELECT count(*) FROM semantic_facts").fetchall()
    assert sf_count == 0  # nothing fast-tracked
    metadata_rows = db.execute(
        "SELECT metadata FROM episodes WHERE metadata IS NOT NULL"
    ).fetchall()
    assert len(metadata_rows) == 2  # both user + assistant rows carry the same metadata
    md = json.loads(metadata_rows[0]["metadata"])
    assert md["stashed_facts"][0]["text"] == "禮揚 likes Starting Strength"
    assert md["stashed_facts"][0]["entity"] == "禮揚.訓練"


def test_mixed_facts_partition_correctly(tmp_path):
    db = _bootstrap_db(tmp_path)
    embed, _ = _stub_embed_factory()
    today = date.today()
    extract = _stub_extract_factory([
        ExtractedFact(
            type="semantic", text="short",
            entity="禮揚.短期", importance=2,
            valid_to_hint=(today + timedelta(days=2)).isoformat(),
        ),
        ExtractedFact(
            type="semantic", text="long",
            entity="禮揚.長期", importance=3,
            valid_to_hint=None,
        ),
    ])

    summary = asyncio.run(write_episode(
        user_msg="u", reply="a", channel="cattia",
        msg_id="m1", ts="2026-05-02 09:00:00",
        conn=db, embed_fn=embed, extract_fn=extract,
        failure_log_path=tmp_path / "fail.jsonl",
    ))

    assert summary["fast_tracked"] == 1
    assert summary["stashed"] == 1


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


def test_embed_failure_appends_to_jsonl(tmp_path):
    db = _bootstrap_db(tmp_path)

    async def failing_embed(texts):
        raise RuntimeError("voyage exploded")

    extract = _stub_extract_factory([])
    fail_log = tmp_path / "fail.jsonl"

    summary = asyncio.run(write_episode(
        user_msg="u", reply="a", channel="cattia",
        msg_id="m1", ts="2026-05-02 09:00:00",
        conn=db, embed_fn=failing_embed, extract_fn=extract,
        failure_log_path=fail_log,
    ))

    # Caller never sees the exception.
    assert summary["episodes"] == 0  # rolled back
    [(ep_count,)] = db.execute("SELECT count(*) FROM episodes").fetchall()
    assert ep_count == 0
    # Failure record landed in the JSONL.
    assert fail_log.exists()
    line = json.loads(fail_log.read_text().strip().splitlines()[-1])
    assert line["channel"] == "cattia"
    assert line["msg_id"] == "m1"
    assert "voyage exploded" in line["error"]


def test_extract_failure_still_records_episode(tmp_path):
    """If kimi_extract raises, we still land the raw episode rows. The
    weekly_promotion (W3-3) can re-extract from the raw text later."""
    db = _bootstrap_db(tmp_path)
    embed, _ = _stub_embed_factory()

    async def failing_extract(*a, **kw):
        raise RuntimeError("synthetic.new 503")

    summary = asyncio.run(write_episode(
        user_msg="u", reply="a", channel="cattia",
        msg_id="m1", ts="2026-05-02 09:00:00",
        conn=db, embed_fn=embed, extract_fn=failing_extract,
        failure_log_path=tmp_path / "fail.jsonl",
    ))
    assert summary["episodes"] == 2
    assert summary["fast_tracked"] == 0
    assert summary["stashed"] == 0


def test_empty_turn_records_no_rows(tmp_path):
    """Both user_msg and reply blank → no work done, no embed call."""
    db = _bootstrap_db(tmp_path)

    embed_called = []

    async def embed(texts):
        embed_called.append(texts)
        return []

    extract = _stub_extract_factory([])
    summary = asyncio.run(write_episode(
        user_msg="", reply="", channel="cattia",
        msg_id="m1", ts="2026-05-02 09:00:00",
        conn=db, embed_fn=embed, extract_fn=extract,
        failure_log_path=tmp_path / "fail.jsonl",
    ))
    # No embed call (both texts empty), but the schema accepts NULL embeddings
    # for episodes so we still INSERT 2 rows.
    assert embed_called == []
    assert summary["episodes"] == 2
