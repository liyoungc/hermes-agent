"""Tests for the sqlite_vec memory provider plugin (W1 scope: schema only).

Covers:
  • bootstrap_schema is idempotent (re-running does not error or duplicate)
  • all 3 tables + 4 indexes + 1 virtual table + 3 triggers exist
  • semantic_facts defaults work (created_at, valid_from, importance)
  • vec0 virtual table answers MATCH queries with k=N prefilter
  • triggers keep vec_facts synced with semantic_facts (insert/update/delete)
  • SqliteVecMemoryProvider.is_available() / initialize() / shutdown() round-trip
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from plugins.memory.sqlite_vec import SqliteVecMemoryProvider
from plugins.memory.sqlite_vec.store import (
    VEC_DIM,
    bootstrap_schema,
    init_db,
    open_db,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vec(seed: int) -> bytes:
    """Make a deterministic 512-d int8 BLOB for testing.

    int8 matches the locked decision in spec §1.4 (Voyage 3.5-lite, 512-dim, int8).
    seed is the base value (clamped to int8 range) with a small per-dim offset
    so different seeds produce different vectors but the same seed reproduces.
    """
    vals = [max(-128, min(127, seed + (i % 7) - 3)) for i in range(VEC_DIM)]
    return struct.pack(f"{VEC_DIM}b", *vals)


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_creates_all_objects(tmp_path):
    db = init_db(tmp_path / "memory.db")

    table_names = {
        row[0]
        for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    assert "episodes" in table_names
    assert "semantic_facts" in table_names

    index_names = {
        row[0]
        for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
        )
    }
    assert "idx_episodes_ts" in index_names
    assert "idx_episodes_promoted_pending" in index_names
    assert "idx_facts_entity" in index_names
    assert "idx_facts_active" in index_names

    trigger_names = {
        row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
    }
    assert "sf_after_insert" in trigger_names
    assert "sf_after_update_embedding" in trigger_names
    assert "sf_after_delete" in trigger_names

    # vec0 virtual table is registered as a regular table internally
    [(vec_count,)] = db.execute(
        "SELECT count(*) FROM sqlite_master WHERE name='vec_facts'"
    ).fetchall()
    assert vec_count >= 1


def test_bootstrap_is_idempotent(tmp_path):
    path = tmp_path / "memory.db"
    db = init_db(path)
    bootstrap_schema(db)  # second time
    bootstrap_schema(db)  # third time
    # If we got here without error and tables still query, idempotency holds.
    db.execute("SELECT count(*) FROM episodes").fetchone()
    db.execute("SELECT count(*) FROM semantic_facts").fetchone()


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_semantic_facts_defaults_are_populated(tmp_path):
    db = init_db(tmp_path / "memory.db")
    db.execute(
        "INSERT INTO semantic_facts(fact, embedding) VALUES (?, ?)",
        ("禮揚 likes Starting Strength method", _vec(10)),
    )
    db.commit()

    row = db.execute(
        "SELECT importance, state, valid_from, valid_to, created_at FROM semantic_facts"
    ).fetchone()
    assert row["importance"] == 2
    assert row["state"] == "active"
    assert row["valid_from"] is not None  # default = date('now')
    assert row["valid_to"] is None
    assert row["created_at"] is not None


def test_role_check_constraint(tmp_path):
    db = init_db(tmp_path / "memory.db")
    with pytest.raises(Exception):
        db.execute(
            "INSERT INTO episodes(ts, channel, external_id, role, text) "
            "VALUES (datetime('now'), 'cattia', 'msg-1', 'system', 'hi')"
        )


# ---------------------------------------------------------------------------
# Trigger sync between semantic_facts and vec_facts
# ---------------------------------------------------------------------------


def test_triggers_sync_insert_update_delete(tmp_path):
    db = init_db(tmp_path / "memory.db")

    # INSERT
    db.execute(
        "INSERT INTO semantic_facts(fact, embedding) VALUES (?, ?)",
        ("fact A", _vec(50)),
    )
    db.commit()
    [(count_after_insert,)] = db.execute("SELECT count(*) FROM vec_facts").fetchall()
    assert count_after_insert == 1

    # UPDATE embedding
    [fact_id] = db.execute("SELECT id FROM semantic_facts").fetchone()
    new_vec = _vec(90)
    db.execute("UPDATE semantic_facts SET embedding=? WHERE id=?", (new_vec, fact_id))
    db.commit()
    [(after_update,)] = db.execute(
        "SELECT count(*) FROM vec_facts WHERE fact_id=?", (fact_id,)
    ).fetchall()
    assert after_update == 1

    # DELETE
    db.execute("DELETE FROM semantic_facts WHERE id=?", (fact_id,))
    db.commit()
    [(count_after_delete,)] = db.execute("SELECT count(*) FROM vec_facts").fetchall()
    assert count_after_delete == 0


# ---------------------------------------------------------------------------
# vec0 retrieval
# ---------------------------------------------------------------------------


def test_vec0_match_returns_nearest(tmp_path):
    db = init_db(tmp_path / "memory.db")
    for seed, fact in [(10, "alpha"), (50, "beta"), (90, "gamma")]:
        db.execute(
            "INSERT INTO semantic_facts(fact, embedding) VALUES (?, ?)",
            (fact, _vec(seed)),
        )
    db.commit()

    query = _vec(51)
    rows = db.execute(
        "SELECT fact_id, distance FROM vec_facts WHERE embedding MATCH vec_int8(?) AND k = 2",
        (query,),
    ).fetchall()
    assert len(rows) == 2
    # Closest must be the seed=0.5 row (beta)
    closest_fact_id = rows[0]["fact_id"]
    closest_fact = db.execute(
        "SELECT fact FROM semantic_facts WHERE id=?", (closest_fact_id,)
    ).fetchone()["fact"]
    assert closest_fact == "beta"


# ---------------------------------------------------------------------------
# MemoryProvider lifecycle
# ---------------------------------------------------------------------------


def test_provider_lifecycle(tmp_path):
    p = SqliteVecMemoryProvider()
    assert p.name == "sqlite_vec"
    assert p.is_available() is True
    p.initialize(session_id="t1", hermes_home=str(tmp_path))
    assert (tmp_path / "memories" / "memory.db").exists()
    assert p.prefetch("test query") == ""  # W1: no-op
    assert p.sync_turn("hi", "hello") is None  # W1: no-op
    assert p.get_tool_schemas() == []
    p.shutdown()



# ===========================================================================
# W2-1: voyage_embed (mocked) + read_memory + bump_hits + format_facts
# ===========================================================================

import asyncio
import sqlite3
from unittest.mock import patch

import httpx
import pytest

from plugins.memory.sqlite_vec.embed import (
    VOYAGE_BATCH,
    VOYAGE_DIM,
    VoyageError,
    voyage_embed,
)
from plugins.memory.sqlite_vec.read import (
    Fact,
    bump_hits,
    format_facts_for_prompt,
    read_memory,
)


def _fake_voyage_response(texts):
    """Build a fake Voyage JSON body where each embedding is dim=512 of zeros
    except the first cell which carries the input index. Lets us round-trip
    the input ordering through _to_int8_blob."""
    return {
        "data": [
            {"index": i, "embedding": [(i % 200) - 100] + [0] * (VOYAGE_DIM - 1)}
            for i, _ in enumerate(texts)
        ]
    }


class _MockTransport(httpx.MockTransport):
    """httpx mock that records call count and returns programmable responses."""

    def __init__(self, responses):
        self.calls = []
        self._responses = list(responses)
        super().__init__(self._handler)

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        status, body = self._responses.pop(0)
        if isinstance(body, dict):
            return httpx.Response(status, json=body)
        return httpx.Response(status, text=body)


# ---------------------------------------------------------------------------
# voyage_embed
# ---------------------------------------------------------------------------


def test_voyage_embed_success(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    texts = ["hello", "world", "禮揚"]
    transport = _MockTransport([(200, _fake_voyage_response(texts))])
    client = httpx.AsyncClient(transport=transport)

    blobs = asyncio.run(voyage_embed(texts, client=client))

    assert len(blobs) == len(texts)
    for b in blobs:
        assert len(b) == VOYAGE_DIM
    # First byte encodes the (signed) index value we baked into the fake response.
    assert blobs[0][0] == (-100) & 0xFF  # input index 0 -> -100 -> unsigned 156
    assert blobs[1][0] == (-99) & 0xFF
    assert len(transport.calls) == 1


def test_voyage_embed_batches_at_128(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    texts = [f"t{i}" for i in range(200)]  # > VOYAGE_BATCH=128
    # 2 calls: first 128, then 72.
    transport = _MockTransport(
        [
            (200, _fake_voyage_response(texts[:VOYAGE_BATCH])),
            (200, _fake_voyage_response(texts[VOYAGE_BATCH:])),
        ]
    )
    client = httpx.AsyncClient(transport=transport)

    blobs = asyncio.run(voyage_embed(texts, client=client))
    assert len(blobs) == 200
    assert len(transport.calls) == 2


def test_voyage_embed_retries_on_5xx(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    texts = ["only"]
    transport = _MockTransport(
        [
            (502, "bad gateway"),
            (503, "still bad"),
            (200, _fake_voyage_response(texts)),
        ]
    )
    client = httpx.AsyncClient(transport=transport)

    # Patch sleep to avoid real backoff delay.
    with patch("plugins.memory.sqlite_vec.embed.asyncio.sleep", return_value=None):
        blobs = asyncio.run(voyage_embed(texts, client=client))

    assert len(blobs) == 1
    assert len(transport.calls) == 3


def test_voyage_embed_4xx_raises(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    transport = _MockTransport([(401, "unauthorized")])
    client = httpx.AsyncClient(transport=transport)
    with pytest.raises(VoyageError):
        asyncio.run(voyage_embed(["x"], client=client))


def test_voyage_embed_missing_key(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(VoyageError, match="VOYAGE_API_KEY"):
        asyncio.run(voyage_embed(["x"]))


def test_voyage_embed_empty_input_no_call(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    # No transport responses queued; if we make a call the test will explode.
    transport = _MockTransport([])
    client = httpx.AsyncClient(transport=transport)
    blobs = asyncio.run(voyage_embed([], client=client))
    assert blobs == []
    assert len(transport.calls) == 0


# ---------------------------------------------------------------------------
# read_memory + bump_hits
# ---------------------------------------------------------------------------


def _seed_facts(db: sqlite3.Connection):
    """Insert 3 facts at known created_at + int8 vectors that put 'beta' nearest to seed=51."""
    rows = [
        # fact text,   entity,         created_at,             vec seed
        ("alpha",      "禮揚.工作",     "2026-04-01 09:00:00",   10),
        ("beta",       "禮揚.家庭",     "2026-05-02 09:00:00",   50),
        ("gamma",      None,           "2025-12-01 09:00:00",   90),
        ("expired",    "禮揚.短期",     "2026-05-01 09:00:00",   50),
    ]
    for fact, entity, created_at, seed in rows:
        db.execute(
            "INSERT INTO semantic_facts(fact, entity, embedding, created_at, valid_to) "
            "VALUES (?, ?, ?, ?, ?)",
            (fact, entity, _vec(seed), created_at,
             "2026-01-01" if fact == "expired" else None),
        )
    db.commit()


def test_read_memory_orders_by_score(tmp_path, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    db = init_db(tmp_path / "memory.db")
    _seed_facts(db)

    # Stub voyage_embed to return a fixed query vector close to seed=51.
    async def fake_embed(texts, **kw):
        assert len(texts) == 1
        return [_vec(51)]

    log_file = tmp_path / "memory.log"
    with patch("plugins.memory.sqlite_vec.read.voyage_embed", fake_embed):
        facts = asyncio.run(read_memory("test query", db, k=8, log_path=log_file))

    fact_texts = [f.fact for f in facts]
    # 'expired' must be filtered (valid_to in past).
    assert "expired" not in fact_texts
    # 'beta' should rank first (closest vec, recent).
    assert fact_texts[0] == "beta"
    # All Fact fields populated.
    assert all(isinstance(f, Fact) for f in facts)
    assert all(f.score is not None and f.sim is not None for f in facts)
    # Latency was logged.
    assert log_file.exists()
    log_line = log_file.read_text().strip().splitlines()[-1]
    assert '"sql_ms"' in log_line and '"q": "test query"' in log_line


def test_bump_hits_increments_and_swallows(tmp_path, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    db = init_db(tmp_path / "memory.db")
    _seed_facts(db)
    ids = [r["id"] for r in db.execute("SELECT id FROM semantic_facts ORDER BY id").fetchall()]

    asyncio.run(bump_hits(ids[:2], db))
    rows = db.execute(
        "SELECT id, hits, last_seen FROM semantic_facts ORDER BY id"
    ).fetchall()
    assert rows[0]["hits"] == 1 and rows[1]["hits"] == 1
    assert rows[2]["hits"] == 0  # untouched
    assert rows[0]["last_seen"] is not None

    # Closed connection -> bump_hits must swallow the sqlite3.Error.
    db.close()
    asyncio.run(bump_hits(ids[:1], db))  # should not raise


def test_bump_hits_empty_is_noop(tmp_path):
    db = init_db(tmp_path / "memory.db")
    # Should return immediately without touching the connection.
    asyncio.run(bump_hits([], db))


def test_format_facts_for_prompt_shape():
    facts = [
        Fact(id=1, fact="禮揚 likes 5x5", entity="禮揚.訓練",
             created_at="2026-05-01", importance=2, sim=0.8, age_days=1.0, score=0.9),
        Fact(id=2, fact="致妤生日 3/19", entity=None,
             created_at="2026-04-01", importance=3, sim=0.7, age_days=30.0, score=0.6),
    ]
    out = format_facts_for_prompt(facts)
    assert "[禮揚.訓練] 禮揚 likes 5x5" in out
    assert "- 致妤生日 3/19" in out  # no entity prefix when None
    assert format_facts_for_prompt([]) == ""
