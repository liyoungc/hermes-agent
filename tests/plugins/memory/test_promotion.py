"""Tests for plugins/memory/sqlite_vec/promotion.py (W3-3)."""

from __future__ import annotations

import asyncio
import json
import struct
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from plugins.memory.sqlite_vec.promotion import (
    PENDING_DIFF_TTL_DAYS,
    PROMOTION_PROMPT,
    WeekDigest,
    _apply_diff_atomic,
    _format_candidates_block,
    _format_neighbors_block,
    _purge_old_pending,
    digest_id_for,
    pending_path,
    rejection_sentinel,
    render_digest_markdown,
    weekly_apply,
    weekly_promotion,
)
from plugins.memory.sqlite_vec.store import VEC_DIM, init_db


def _vec(seed: int) -> bytes:
    vals = [max(-128, min(127, seed + (i % 7) - 3)) for i in range(VEC_DIM)]
    return struct.pack(f"{VEC_DIM}b", *vals)


# ---------------------------------------------------------------------------
# Prompt + format helpers
# ---------------------------------------------------------------------------


def test_prompt_has_required_placeholders():
    """The prompt is .format()'d with these keys; missing any breaks promotion."""
    for key in ("{digest_id}", "{today}", "{week_label}",
                "{candidates_block}", "{neighbors_block}"):
        assert key in PROMOTION_PROMPT, f"missing placeholder: {key}"


def test_prompt_carries_hard_rules():
    assert "病歷號" in PROMOTION_PROMPT
    assert "DROP_AS_NOISE" in PROMOTION_PROMPT
    assert "PROMOTE" in PROMOTION_PROMPT
    assert "DEDUP_HIT" in PROMOTION_PROMPT
    assert "EXPIRE" in PROMOTION_PROMPT


def test_format_candidates_block_marks_synthetic():
    cands = [
        {"id": 1, "ts": "2026-05-02 09:00", "channel": "cattia",
         "role": "user", "synthetic": False, "text": "hello",
         "stashed_facts": [{"text": "禮揚 likes X", "entity": "禮揚.訓練",
                            "importance": 2, "valid_to_hint": None}]},
        {"id": 2, "ts": "2026-05-02 09:00", "channel": "cron",
         "role": "assistant", "synthetic": True, "text": "cron output",
         "stashed_facts": []},
    ]
    out = _format_candidates_block(cands)
    assert "👤" in out and "🤖" in out
    assert "↳ stashed:" in out


def test_format_neighbors_block_truncates_to_top_5():
    neighbors = {
        "topic": [
            {"id": i, "fact": f"fact {i}", "entity": "x", "sim": 0.9 - i * 0.01}
            for i in range(10)
        ]
    }
    out = _format_neighbors_block(neighbors)
    # Only 5 should appear.
    assert out.count("#") == 5


# ---------------------------------------------------------------------------
# digest_id + path helpers
# ---------------------------------------------------------------------------


def test_digest_id_format():
    assert digest_id_for(date(2026, 5, 11)) == "wk-2026-05-11"


# ---------------------------------------------------------------------------
# WeekDigest
# ---------------------------------------------------------------------------


def test_week_digest_round_trip():
    raw = {
        "digest_id": "wk-2026-05-10",
        "candidate_episode_ids": [1, 2, 3],
        "promote": [{"entity": "禮揚.家庭", "fact": "x", "importance": 3}],
        "dedup_hits": [{"existing_fact_id": 5, "action": "bump_hits"}],
        "expire": [{"existing_fact_id": 7, "valid_to": "2026-05-10"}],
        "drop_as_noise": [{"episode_ids": [4], "reason": "pleasantry"}],
    }
    d = WeekDigest.from_dict(raw)
    assert d.digest_id == "wk-2026-05-10"
    assert d.to_dict()["candidate_episode_ids"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# render_digest_markdown
# ---------------------------------------------------------------------------


def test_render_digest_markdown_full_shape():
    candidates = [
        {"id": 1, "ts": "x", "channel": "c", "role": "user",
         "synthetic": False, "text": "u", "stashed_facts": []},
        {"id": 2, "ts": "x", "channel": "cron", "role": "user",
         "synthetic": True, "text": "u", "stashed_facts": []},
    ]
    d = WeekDigest.from_dict({
        "digest_id": "wk-2026-05-10",
        "candidate_episode_ids": [1, 2],
        "promote": [{"entity": "禮揚.家庭", "fact": "致妤生日 3/19",
                     "importance": 5, "valid_to": None,
                     "source_episode_ids": [1]}],
        "dedup_hits": [{"existing_fact_id": 5, "action": "bump_hits",
                        "source_episode_ids": [2]}],
        "expire": [{"existing_fact_id": 7, "valid_to": "2026-05-10",
                    "reason": "stale"}],
        "drop_as_noise": [{"episode_ids": [3], "reason": "好的"}],
    })
    md = render_digest_markdown(d, candidates)
    assert "Weekly Memory Review — 2026-05-10" in md
    assert "(1 user/assistant + 1 cron-synthetic)" in md
    assert "/memreview reject wk-2026-05-10" in md
    assert "⬆️ Promote to permanent (1)" in md
    assert "🔁 Dedup confirmations (1)" in md
    assert "🪦 Expiring (1)" in md
    assert "🗑️ Skipped as noise (1)" in md
    assert "致妤生日 3/19" in md
    assert "valid_to: 永久" in md  # null valid_to


def test_render_digest_empty_sections_collapse():
    d = WeekDigest.from_dict({"digest_id": "wk-2026-05-10",
                              "candidate_episode_ids": []})
    md = render_digest_markdown(d, [])
    assert "_No actions this week._" in md


# ---------------------------------------------------------------------------
# weekly_promotion (mocked Kimi)
# ---------------------------------------------------------------------------


def _seed_episodes(conn, today_iso: str = "2026-05-02 12:00:00"):
    """Add 2 fixture episodes with stashed_facts."""
    conn.execute(
        "INSERT INTO episodes(ts, channel, external_id, role, text, synthetic, metadata) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (today_iso, "cattia", "m1:user", "user", "我下週要去日本", 0,
         json.dumps({"stashed_facts": [
             {"type": "semantic", "text": "禮揚下週去日本", "entity": "禮揚.家庭",
              "importance": 3, "valid_to_hint": "2026-05-11"}]})),
    )
    conn.execute(
        "INSERT INTO episodes(ts, channel, external_id, role, text) "
        "VALUES (?, ?, ?, ?, ?)",
        (today_iso, "cattia", "m1:asst", "assistant", "好的", ),
    )
    conn.commit()


def test_weekly_promotion_no_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "plugins.memory.sqlite_vec.promotion._resolve_hermes_home", lambda: tmp_path
    )
    db = init_db(tmp_path / "m.db")
    summary = asyncio.run(weekly_promotion(db))
    assert summary["candidates"] == 0
    assert "skipped" in summary


def test_weekly_promotion_dry_run_returns_markdown(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "plugins.memory.sqlite_vec.promotion._resolve_hermes_home", lambda: tmp_path
    )
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    db = init_db(tmp_path / "m.db")
    _seed_episodes(db)

    async def fake_kimi(prompt):
        # Sanity: prompt was actually formatted, not left with placeholders.
        assert "{digest_id}" not in prompt
        return {
            "promote": [{"entity": "禮揚.家庭", "fact": "下週去日本",
                         "importance": 3, "valid_to": "2026-05-11",
                         "source_episode_ids": [1]}],
            "dedup_hits": [], "expire": [], "drop_as_noise": [],
        }

    async def fake_embed(texts):
        return [_vec(50) for _ in texts]

    summary = asyncio.run(weekly_promotion(
        db, dry_run=True, kimi_fn=fake_kimi,
        embed_fn=fake_embed,
    ))
    assert summary["candidates"] == 2
    assert summary["promote"] == 1
    assert summary["dry_run"] is True
    assert "markdown_preview" in summary
    assert "下週去日本" in summary["markdown_preview"]
    # Dry-run MUST NOT persist a pending diff or post to Discord.
    assert not (tmp_path / "memories" / "pending_diffs").exists() or \
           not list((tmp_path / "memories" / "pending_diffs").glob("*.json"))


def test_weekly_promotion_persists_diff_on_real_run(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "plugins.memory.sqlite_vec.promotion._resolve_hermes_home", lambda: tmp_path
    )
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    db = init_db(tmp_path / "m.db")
    _seed_episodes(db)

    async def fake_kimi(prompt):
        return {
            "promote": [], "dedup_hits": [], "expire": [],
            "drop_as_noise": [{"episode_ids": [1, 2], "reason": "no signal"}],
        }

    summary = asyncio.run(weekly_promotion(
        db, dry_run=False, kimi_fn=fake_kimi,
    ))
    # Diff was written, even with no Discord channel configured.
    files = list((tmp_path / "memories" / "pending_diffs").glob("*.json"))
    assert len(files) == 1
    diff = json.loads(files[0].read_text())
    assert diff["candidate_episode_ids"] == [1, 2]


# ---------------------------------------------------------------------------
# weekly_apply
# ---------------------------------------------------------------------------


def test_weekly_apply_no_pending_diff(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "plugins.memory.sqlite_vec.promotion._resolve_hermes_home", lambda: tmp_path
    )
    db = init_db(tmp_path / "m.db")
    summary = asyncio.run(weekly_apply(db))
    assert summary["applied"] is False
    assert "no pending diff" in summary.get("reason", "")


def test_weekly_apply_rejection_sentinel_archives_without_apply(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "plugins.memory.sqlite_vec.promotion._resolve_hermes_home", lambda: tmp_path
    )
    db = init_db(tmp_path / "m.db")

    digest_id = "wk-2026-05-02"
    pending_path(digest_id).write_text(json.dumps({
        "digest_id": digest_id, "candidate_episode_ids": [],
        "promote": [], "dedup_hits": [], "expire": [], "drop_as_noise": [],
    }))
    rejection_sentinel(digest_id).write_text("rejected", encoding="utf-8")

    summary = asyncio.run(weekly_apply(db))
    assert summary["applied"] is False
    assert summary["reason"] == "rejected"
    # Diff moved to archive_dir, sentinel removed.
    assert not pending_path(digest_id).exists()
    assert not rejection_sentinel(digest_id).exists()
    archive = list((tmp_path / "memories" / "diff_archive").glob("*.rejected.json"))
    assert len(archive) == 1


def test_weekly_apply_promotes_inserts_and_stamps(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "plugins.memory.sqlite_vec.promotion._resolve_hermes_home", lambda: tmp_path
    )
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    db = init_db(tmp_path / "m.db")
    _seed_episodes(db)

    digest_id = "wk-2026-05-02"
    pending_path(digest_id).write_text(json.dumps({
        "digest_id": digest_id,
        "candidate_episode_ids": [1, 2],
        "promote": [{"entity": "禮揚.家庭", "fact": "下週去日本",
                     "importance": 3, "valid_from": "2026-05-02",
                     "valid_to": "2026-05-11", "source_episode_ids": [1]}],
        "dedup_hits": [], "expire": [], "drop_as_noise": [],
    }))

    async def fake_embed(texts):
        return [_vec(50) for _ in texts]

    summary = asyncio.run(weekly_apply(db, embed_fn=fake_embed))
    assert summary["applied"] is True
    assert summary["promoted"] == 1
    assert summary["stamped"] == 2
    # New row in semantic_facts.
    [(sf,)] = db.execute("SELECT count(*) FROM semantic_facts").fetchall()
    assert sf == 1
    # Trigger mirrored into vec_facts.
    [(vf,)] = db.execute("SELECT count(*) FROM vec_facts").fetchall()
    assert vf == 1
    # Episodes stamped.
    rows = db.execute("SELECT id, promoted_at FROM episodes ORDER BY id").fetchall()
    assert all(r["promoted_at"] is not None for r in rows)
    # Diff moved to archive.
    archive = list((tmp_path / "memories" / "diff_archive").glob("*.applied.json"))
    assert len(archive) == 1


def test_weekly_apply_dedup_bump_increments_hits(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "plugins.memory.sqlite_vec.promotion._resolve_hermes_home", lambda: tmp_path
    )
    db = init_db(tmp_path / "m.db")
    db.execute(
        "INSERT INTO semantic_facts(fact, embedding, hits) VALUES (?, ?, ?)",
        ("禮揚 likes X", _vec(10), 0),
    )
    db.commit()

    digest_id = "wk-2026-05-02"
    pending_path(digest_id).write_text(json.dumps({
        "digest_id": digest_id, "candidate_episode_ids": [],
        "promote": [], "dedup_hits": [
            {"existing_fact_id": 1, "action": "bump_hits",
             "source_episode_ids": []}
        ], "expire": [], "drop_as_noise": [],
    }))

    summary = asyncio.run(weekly_apply(db))
    assert summary["dedup_bumped"] == 1
    [(hits,)] = db.execute("SELECT hits FROM semantic_facts WHERE id=1").fetchall()
    assert hits == 1


def test_weekly_apply_expire_sets_valid_to(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "plugins.memory.sqlite_vec.promotion._resolve_hermes_home", lambda: tmp_path
    )
    db = init_db(tmp_path / "m.db")
    db.execute(
        "INSERT INTO semantic_facts(fact, embedding) VALUES (?, ?)",
        ("禮揚 watches paper X", _vec(10)),
    )
    db.commit()

    digest_id = "wk-2026-05-02"
    pending_path(digest_id).write_text(json.dumps({
        "digest_id": digest_id, "candidate_episode_ids": [],
        "promote": [], "dedup_hits": [],
        "expire": [{"existing_fact_id": 1, "valid_to": "2026-05-02",
                    "reason": "stale"}],
        "drop_as_noise": [],
    }))

    summary = asyncio.run(weekly_apply(db, today=date(2026, 5, 2)))
    assert summary["expired"] == 1
    [(vt,)] = db.execute("SELECT valid_to FROM semantic_facts WHERE id=1").fetchall()
    assert vt == "2026-05-02"


def test_weekly_apply_purges_old_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "plugins.memory.sqlite_vec.promotion._resolve_hermes_home", lambda: tmp_path
    )
    db = init_db(tmp_path / "m.db")

    today = date(2026, 5, 2)
    old = today - timedelta(days=PENDING_DIFF_TTL_DAYS + 5)
    fresh = today - timedelta(days=2)

    pending_path(f"wk-{old.isoformat()}").write_text("{}")
    pending_path(f"wk-{fresh.isoformat()}").write_text(json.dumps({
        "digest_id": f"wk-{fresh.isoformat()}", "candidate_episode_ids": [],
        "promote": [], "dedup_hits": [], "expire": [], "drop_as_noise": [],
    }))

    summary = asyncio.run(weekly_apply(db, today=today))
    assert summary["purged"] == 1
    # Old gone, fresh applied + archived.
    assert not pending_path(f"wk-{old.isoformat()}").exists()
    archive = list((tmp_path / "memories" / "diff_archive").glob("*.applied.json"))
    assert len(archive) == 1
