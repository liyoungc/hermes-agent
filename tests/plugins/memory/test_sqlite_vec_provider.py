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


def _vec(seed: float) -> bytes:
    """Make a deterministic 512-d float32 BLOB for testing.

    seed is broadcast across all dimensions and then perturbed slightly so
    different seeds produce different vectors but the same seed always
    yields the same bytes.
    """
    return struct.pack(f"{VEC_DIM}f", *[seed + i * 1e-4 for i in range(VEC_DIM)])


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
        ("禮揚 likes Starting Strength method", _vec(0.1)),
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
        ("fact A", _vec(0.5)),
    )
    db.commit()
    [(count_after_insert,)] = db.execute("SELECT count(*) FROM vec_facts").fetchall()
    assert count_after_insert == 1

    # UPDATE embedding
    [fact_id] = db.execute("SELECT id FROM semantic_facts").fetchone()
    new_vec = _vec(0.9)
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
    for seed, fact in [(0.1, "alpha"), (0.5, "beta"), (0.9, "gamma")]:
        db.execute(
            "INSERT INTO semantic_facts(fact, embedding) VALUES (?, ?)",
            (fact, _vec(seed)),
        )
    db.commit()

    query = _vec(0.51)
    rows = db.execute(
        "SELECT fact_id, distance FROM vec_facts WHERE embedding MATCH ? AND k = 2",
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
