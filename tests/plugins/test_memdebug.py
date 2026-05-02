"""Tests for plugins/memdebug/ — /memdebug slash command (W2-4)."""

from __future__ import annotations

import asyncio
import struct
from pathlib import Path
from unittest.mock import patch

import pytest

from plugins.memory.sqlite_vec.store import VEC_DIM, init_db
from plugins.memdebug import (
    HELP_TEXT,
    _do_rawsearch,
    _do_semantic,
    _format_facts_block,
    _handle_async,
    _handle_memdebug,
    _truncate,
)


def _vec(seed: int) -> bytes:
    vals = [max(-128, min(127, seed + (i % 7) - 3)) for i in range(VEC_DIM)]
    return struct.pack(f"{VEC_DIM}b", *vals)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_truncate_short_string_unchanged():
    assert _truncate("abc", 10) == "abc"


def test_truncate_long_string_ellipsis():
    out = _truncate("a" * 100, 10)
    assert out.endswith("…") and len(out) == 10


# ---------------------------------------------------------------------------
# Help / empty / unknown args
# ---------------------------------------------------------------------------


def test_handle_empty_returns_help():
    assert _handle_memdebug("") == HELP_TEXT
    assert _handle_memdebug("   ") == HELP_TEXT


def test_handle_rawsearch_empty_returns_help():
    assert _handle_memdebug("rawsearch") == HELP_TEXT
    assert _handle_memdebug("rawsearch   ") == HELP_TEXT


# ---------------------------------------------------------------------------
# Semantic / rawsearch via direct async helpers (so we control DB path)
# ---------------------------------------------------------------------------


def _seed_db(tmp_path):
    """Seed a fixture memory.db on tmp_path and return its path."""
    db_path = tmp_path / "memories" / "memory.db"
    conn = init_db(db_path)
    conn.execute(
        "INSERT INTO semantic_facts(fact, entity, embedding, created_at) VALUES (?,?,?,?)",
        ("致妤生日 3/19", "禮揚.家庭", _vec(50), "2026-05-02 09:00:00"),
    )
    conn.execute(
        "INSERT INTO semantic_facts(fact, entity, embedding, created_at) VALUES (?,?,?,?)",
        ("AI as digital twin", "禮揚.工作", _vec(60), "2026-05-01 09:00:00"),
    )
    conn.execute(
        "INSERT INTO episodes(ts, channel, external_id, role, text) "
        "VALUES (?, ?, ?, ?, ?)",
        ("2026-05-02 17:00:00", "cattia", "msg-1", "user", "晚餐幾點開"),
    )
    conn.commit()
    conn.close()
    return db_path


def test_do_semantic_returns_score_breakdown(tmp_path, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    db_path = _seed_db(tmp_path)

    async def fake_embed(texts, **kw):
        return [_vec(51) for _ in texts]

    with patch("plugins.memdebug.DEFAULT_DB", db_path), \
         patch("plugins.memdebug.LOG_PATH", tmp_path / "memory.log"), \
         patch("plugins.memory.sqlite_vec.read.voyage_embed", fake_embed):
        out = asyncio.run(_do_semantic("when does my wife get home"))

    assert "/memdebug" in out
    assert "致妤生日 3/19" in out  # closest fact
    # Score breakdown labels present.
    assert "score=" in out and "sim=" in out and "age=" in out
    # Reaction prompt present (until rich-embed UX lands).
    assert "👍" in out and "👎" in out
    # Log line written.
    log_path = tmp_path / "memory.log"
    assert log_path.exists()
    last_line = log_path.read_text().strip().splitlines()[-1]
    assert '"cmd": "memdebug"' in last_line


def test_do_semantic_db_missing_returns_friendly_message(tmp_path, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    missing = tmp_path / "absent.db"
    with patch("plugins.memdebug.DEFAULT_DB", missing):
        out = asyncio.run(_do_semantic("anything"))
    assert "not yet initialised" in out


def test_do_rawsearch_finds_substring(tmp_path):
    db_path = _seed_db(tmp_path)
    with patch("plugins.memdebug.DEFAULT_DB", db_path), \
         patch("plugins.memdebug.LOG_PATH", tmp_path / "memory.log"):
        out = asyncio.run(_do_rawsearch("晚餐"))
    assert "rawsearch" in out
    assert "晚餐幾點開" in out
    assert "cattia/user" in out


def test_do_rawsearch_empty_episodes_message(tmp_path):
    db_path = tmp_path / "memories" / "memory.db"
    init_db(db_path).close()  # bootstrap schema, no rows
    with patch("plugins.memdebug.DEFAULT_DB", db_path), \
         patch("plugins.memdebug.LOG_PATH", tmp_path / "memory.log"):
        out = asyncio.run(_do_rawsearch("anything"))
    assert "rawsearch" in out
    assert "Episodes are written by W3" in out


# ---------------------------------------------------------------------------
# Sync entry point + register()
# ---------------------------------------------------------------------------


def test_handle_memdebug_sync_dispatches_semantic(tmp_path, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    db_path = _seed_db(tmp_path)

    async def fake_embed(texts, **kw):
        return [_vec(51) for _ in texts]

    with patch("plugins.memdebug.DEFAULT_DB", db_path), \
         patch("plugins.memdebug.LOG_PATH", tmp_path / "memory.log"), \
         patch("plugins.memory.sqlite_vec.read.voyage_embed", fake_embed):
        out = _handle_memdebug("when does my wife get home")
    assert "致妤生日" in out


def test_register_calls_register_command():
    """register(ctx) must call ctx.register_command with the right name."""
    from plugins.memdebug import register

    captured = {}

    class FakeCtx:
        def register_command(self, name, handler, description="", args_hint=""):
            captured["name"] = name
            captured["handler"] = handler
            captured["args_hint"] = args_hint
            captured["description"] = description

    register(FakeCtx())
    assert captured["name"] == "memdebug"
    assert captured["args_hint"] == "<query> | rawsearch <query>"
    assert callable(captured["handler"])
    # The handler must accept a single positional argument (raw_args).
    assert captured["handler"].__code__.co_argcount == 1
