"""Tests for plugins/memreview/ — /memreview reject + /mem kill switch (W3-4)."""

from __future__ import annotations

import asyncio
import json
import struct
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from plugins.memory.sqlite_vec.store import VEC_DIM, init_db
from plugins.memreview import (
    _MEMREVIEW_HELP,
    _MEM_HELP,
    _handle_mem,
    _handle_memreview,
    mem_off_active,
    mem_off_path,
    register,
)


def _vec(seed: int) -> bytes:
    vals = [max(-128, min(127, seed + (i % 7) - 3)) for i in range(VEC_DIM)]
    return struct.pack(f"{VEC_DIM}b", *vals)


# ---------------------------------------------------------------------------
# /memreview help / pending
# ---------------------------------------------------------------------------


def test_memreview_empty_returns_help():
    assert _handle_memreview("") == _MEMREVIEW_HELP
    assert _handle_memreview("   ") == _MEMREVIEW_HELP


def test_memreview_pending_no_diffs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "plugins.memreview._resolve_hermes_home", lambda: tmp_path
    )
    out = _handle_memreview("pending")
    assert "no pending diffs" in out


def test_memreview_pending_lists_diffs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "plugins.memreview._resolve_hermes_home", lambda: tmp_path
    )
    pdir = tmp_path / "memories" / "pending_diffs"
    pdir.mkdir(parents=True)
    (pdir / "wk-2026-05-02.json").write_text("{}")
    (pdir / "wk-2026-05-09.json").write_text("{}")
    (pdir / "wk-2026-05-09.rejected").write_text("rejected")

    out = _handle_memreview("pending")
    assert "wk-2026-05-02" in out
    assert "wk-2026-05-09" in out
    # Rejected one carries a flag.
    assert "(rejected — will be archived Mon)" in out


# ---------------------------------------------------------------------------
# /memreview reject
# ---------------------------------------------------------------------------


def test_memreview_reject_invalid_digest_id(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "plugins.memreview._resolve_hermes_home", lambda: tmp_path
    )
    out = _handle_memreview("reject not-a-digest")
    assert "must look like" in out


def test_memreview_reject_unknown_digest(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "plugins.memreview._resolve_hermes_home", lambda: tmp_path
    )
    out = _handle_memreview("reject wk-2026-05-02")
    assert "no pending diff" in out


def test_memreview_reject_writes_sentinel(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "plugins.memreview._resolve_hermes_home", lambda: tmp_path
    )
    pdir = tmp_path / "memories" / "pending_diffs"
    pdir.mkdir(parents=True)
    diff_path = pdir / "wk-2026-05-02.json"
    diff_path.write_text("{}")

    out = _handle_memreview("reject wk-2026-05-02")
    assert "Rejected." in out
    sentinel = pdir / "wk-2026-05-02.rejected"
    assert sentinel.exists()
    assert "rejected" in sentinel.read_text().lower()


# ---------------------------------------------------------------------------
# /mem off / on / status
# ---------------------------------------------------------------------------


def test_mem_off_creates_sentinel(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "plugins.memreview._resolve_hermes_home", lambda: tmp_path
    )
    out = _handle_mem("off")
    assert "disabled" in out
    assert mem_off_path().exists()
    assert mem_off_active() is True


def test_mem_on_removes_sentinel(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "plugins.memreview._resolve_hermes_home", lambda: tmp_path
    )
    mem_off_path().write_text("set", encoding="utf-8")
    out = _handle_mem("on")
    assert "enabled" in out
    assert not mem_off_path().exists()


def test_mem_on_when_already_on_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "plugins.memreview._resolve_hermes_home", lambda: tmp_path
    )
    out = _handle_mem("on")
    assert "already enabled" in out


def test_mem_status_off(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "plugins.memreview._resolve_hermes_home", lambda: tmp_path
    )
    out = _handle_mem("status")
    assert "🔊 ON" in out  # default state
    assert "(absent)" in out


def test_mem_status_on_with_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "plugins.memreview._resolve_hermes_home", lambda: tmp_path
    )
    mem_off_path().write_text("set")
    pdir = tmp_path / "memories" / "pending_diffs"
    pdir.mkdir(parents=True)
    (pdir / "wk-2026-05-02.json").write_text("{}")

    out = _handle_mem("status")
    assert "🔇 OFF" in out
    assert "(present)" in out
    assert "wk-2026-05-02" in out


def test_mem_help_on_unknown_subcommand(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "plugins.memreview._resolve_hermes_home", lambda: tmp_path
    )
    out = _handle_mem("frobnicate")
    assert "/mem off" in out and "/mem on" in out


# ---------------------------------------------------------------------------
# register() wires both commands
# ---------------------------------------------------------------------------


def test_register_registers_both_commands():
    captured = []

    class FakeCtx:
        def register_command(self, name, handler, description="", args_hint=""):
            captured.append((name, args_hint))

    register(FakeCtx())
    names = [c[0] for c in captured]
    assert "memreview" in names
    assert "mem" in names


# ---------------------------------------------------------------------------
# End-to-end: /memreview reject then weekly_apply archives as rejected
# ---------------------------------------------------------------------------


def test_reject_then_apply_archives_as_rejected(tmp_path, monkeypatch):
    """Full flow: write pending diff -> /memreview reject -> weekly_apply
    sees the sentinel and archives the diff with status=rejected."""
    monkeypatch.setattr(
        "plugins.memreview._resolve_hermes_home", lambda: tmp_path
    )
    monkeypatch.setattr(
        "plugins.memory.sqlite_vec.promotion._resolve_hermes_home",
        lambda: tmp_path,
    )

    db = init_db(tmp_path / "m.db")
    digest_id = "wk-2026-05-02"
    pdir = tmp_path / "memories" / "pending_diffs"
    pdir.mkdir(parents=True)
    diff_payload = {
        "digest_id": digest_id, "candidate_episode_ids": [],
        "promote": [{"entity": "禮揚.x", "fact": "f", "importance": 2,
                     "valid_from": "2026-05-02", "valid_to": None,
                     "source_episode_ids": []}],
        "dedup_hits": [], "expire": [], "drop_as_noise": [],
    }
    (pdir / f"{digest_id}.json").write_text(json.dumps(diff_payload))

    # User runs /memreview reject.
    reply = _handle_memreview(f"reject {digest_id}")
    assert "Rejected." in reply

    # Apply step picks up the sentinel.
    from plugins.memory.sqlite_vec.promotion import weekly_apply
    summary = asyncio.run(weekly_apply(db, today=date(2026, 5, 2)))
    assert summary["applied"] is False
    assert summary["reason"] == "rejected"

    # No new semantic_facts row (the promote was discarded).
    [(sf,)] = db.execute("SELECT count(*) FROM semantic_facts").fetchall()
    assert sf == 0

    # Archive carries the .rejected suffix.
    archived = list((tmp_path / "memories" / "diff_archive").glob("*.rejected.json"))
    assert len(archived) == 1


def test_mem_off_short_circuits_weekly_promotion(tmp_path, monkeypatch):
    """Kill switch: /mem off must stop weekly_promotion from running its
    Kimi call (which would otherwise burn tokens and write a diff)."""
    monkeypatch.setattr(
        "plugins.memreview._resolve_hermes_home", lambda: tmp_path
    )
    monkeypatch.setattr(
        "plugins.memory.sqlite_vec.promotion._resolve_hermes_home",
        lambda: tmp_path,
    )
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")

    db = init_db(tmp_path / "m.db")
    db.execute(
        "INSERT INTO episodes(ts, channel, external_id, role, text, metadata) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("2026-05-02 09:00", "cattia", "x", "user", "hi",
         json.dumps({"stashed_facts": [{"text": "禮揚 likes X",
                                        "entity": "禮揚.x",
                                        "importance": 2}]})),
    )
    db.commit()

    # Activate kill switch.
    _handle_mem("off")
    assert mem_off_active() is True

    kimi_called = []

    async def kimi_should_not_be_called(prompt):
        kimi_called.append(prompt)
        return {}

    from plugins.memory.sqlite_vec.promotion import weekly_promotion
    summary = asyncio.run(weekly_promotion(db, kimi_fn=kimi_should_not_be_called))
    assert summary["candidates"] == 0
    assert summary["skipped"] == "/mem off active"
    # Kimi must not have been called.
    assert kimi_called == []
