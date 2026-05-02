"""Weekly promotion + apply core logic (W3-3).

Spec: docs/superpowers/specs/2026-05-02-hermes-memory-design.md §5.3 + §5.4.

Two entry points, both invoked from cron-driven thin wrappers in
``~/.hermes/scripts/`` (so they sit inside HERMES_HOME/scripts, the only
location the hermes scheduler will exec):

  weekly_promotion()  - reads 7 days of pending episodes, runs one
                        Kimi-thinking call to produce a promotion diff,
                        saves it to pending_diffs/<digest_id>.json,
                        renders + posts the digest to #memory-review.
                        Does NOT stamp episodes.promoted_at.

  weekly_apply()      - purges pending_diffs older than 14 days, loads
                        the latest, checks for the rejection sentinel
                        file, and either archives-as-rejected or
                        applies the diff atomically (promote / dedup /
                        expire) and stamps episodes.promoted_at.

The split lets the user reject Sunday's diff with /memreview reject
<digest_id> any time before Monday's apply fires.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import struct
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .embed import voyage_embed
from .extract import (
    EXTRACT_TIMEOUT,
    PHI_BLACKLIST_CHANNELS,
    SYNTHETIC_URL,
    _read_synthetic_api_key,
)

logger = logging.getLogger(__name__)

PROMOTION_MODEL = "hf:moonshotai/Kimi-K2-Thinking"
PROMOTION_FALLBACK_MODEL = "hf:moonshotai/Kimi-K2.5"
PROMOTION_TEMPERATURE = 0.2
PROMOTION_MAX_TOKENS = 8192  # diff JSON can be substantial across 7 days
PROMOTION_TIMEOUT = 120.0  # thinking-mode + 100+ episodes

PROMOTION_NEIGHBOR_K = 20  # spec §5.3: per-candidate vec_search k=20
PROMOTION_LOOKBACK_DAYS = 7
PENDING_DIFF_TTL_DAYS = 14

DISCORD_API = "https://discord.com/api/v10/channels/{channel_id}/messages"


# ---------------------------------------------------------------------------
# Prompt — designed to match spec §5.3 schema verbatim
# ---------------------------------------------------------------------------

PROMOTION_PROMPT = """You are running the weekly memory promotion review for 禮揚's personal AI.

Below is one week of conversation episodes that have not yet been reviewed.
Each candidate carries any 'stashed_facts' that the per-turn extractor
recorded in its metadata. You also see, per candidate, the top-20 existing
semantic_facts that are nearest by embedding distance — use these to decide
whether a candidate fact duplicates something already known.

HARD RULES — these override everything else:
1. NEVER promote: hospital data, patient names, 病歷號, 身分證字號, lab results,
   diagnoses about real people, hospital policy specifics, hospital colleague names.
2. Pleasantries (好的/收到/早安/明白/thanks) → drop_as_noise.
3. Synthetic episodes (synthetic=true) — promote ONLY if they contain a NEW
   commitment by 禮揚 (a meeting scheduled, a habit declared, a decision made).
4. If a candidate stashed_fact is semantically captured by an existing fact
   (sim ≥ 0.92), prefer dedup_hits over creating a new row.
5. Conservative importance: most facts are 2; only use 4-5 for permanent
   identity / family / strong commitments.

For each candidate, decide one of four actions:

  A. PROMOTE — new fact worth keeping. Emit into "promote".
       valid_to: ISO date or null (null = permanent).
       importance: 1-5 (default 2).
       source_episode_ids: which candidate episodes contributed.

  B. DEDUP_HIT — candidate fact reaffirms an existing fact. Emit into
       "dedup_hits" with the existing fact id and action="bump_hits"
       (just touch the timestamp) or "refine_text" (mild rephrasing
       worth applying).

  C. EXPIRE — an existing fact is contradicted or has gone stale.
       Emit into "expire" with existing_fact_id, valid_to=today, reason.

  D. DROP_AS_NOISE — pleasantry, low signal, or duplicates within the
       week. Emit into "drop_as_noise" with the episode ids and reason.

Every candidate episode_id must appear under exactly one action above
(in promote.source_episode_ids OR dedup_hits.source_episode_ids OR
drop_as_noise.episode_ids). The "expire" section can reference NEW
existing_fact_ids that are independent of this week's candidates —
that's fine.

Output ONE JSON object with this exact schema:

{{
  "digest_id": "{digest_id}",
  "candidate_episode_ids": [<all candidate ids you saw>],
  "promote": [
    {{
      "entity": "禮揚.<namespace>",
      "fact": "single-sentence statement",
      "importance": 1..5,
      "valid_from": "{today}",
      "valid_to": "YYYY-MM-DD" | null,
      "source_episode_ids": [int, ...]
    }}
  ],
  "dedup_hits": [
    {{
      "existing_fact_id": int,
      "action": "bump_hits" | "refine_text",
      "refined_text": "string only if action=refine_text",
      "source_episode_ids": [int, ...]
    }}
  ],
  "expire": [
    {{
      "existing_fact_id": int,
      "valid_to": "{today}",
      "reason": "short reason"
    }}
  ],
  "drop_as_noise": [
    {{
      "episode_ids": [int, ...],
      "reason": "short reason"
    }}
  ]
}}

CANDIDATES (week of {week_label}):
{candidates_block}

NEAREST-NEIGHBOR EXISTING FACTS (one block per candidate stashed_fact):
{neighbors_block}
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class WeekDigest:
    """Loaded form of pending_diffs/<digest_id>.json."""

    digest_id: str
    candidate_episode_ids: List[int]
    promote: List[Dict[str, Any]] = field(default_factory=list)
    dedup_hits: List[Dict[str, Any]] = field(default_factory=list)
    expire: List[Dict[str, Any]] = field(default_factory=list)
    drop_as_noise: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeekDigest":
        return cls(
            digest_id=data.get("digest_id", ""),
            candidate_episode_ids=list(data.get("candidate_episode_ids") or []),
            promote=list(data.get("promote") or []),
            dedup_hits=list(data.get("dedup_hits") or []),
            expire=list(data.get("expire") or []),
            drop_as_noise=list(data.get("drop_as_noise") or []),
            raw=data,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "digest_id": self.digest_id,
            "candidate_episode_ids": self.candidate_episode_ids,
            "promote": self.promote,
            "dedup_hits": self.dedup_hits,
            "expire": self.expire,
            "drop_as_noise": self.drop_as_noise,
        }


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _resolve_hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        return Path.home() / ".hermes"


def pending_dir() -> Path:
    p = _resolve_hermes_home() / "memories" / "pending_diffs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def archive_dir() -> Path:
    p = _resolve_hermes_home() / "memories" / "diff_archive"
    p.mkdir(parents=True, exist_ok=True)
    return p


def memory_log_path() -> Path:
    return _resolve_hermes_home() / "logs" / "memory.log"


def db_path() -> Path:
    return _resolve_hermes_home() / "memories" / "memory.db"


def digest_id_for(today: Optional[date] = None) -> str:
    """ISO date based digest id: wk-YYYY-MM-DD."""
    today = today or date.today()
    return f"wk-{today.isoformat()}"


def rejection_sentinel(digest_id: str) -> Path:
    return pending_dir() / f"{digest_id}.rejected"


def pending_path(digest_id: str) -> Path:
    return pending_dir() / f"{digest_id}.json"


# ---------------------------------------------------------------------------
# Shared logging
# ---------------------------------------------------------------------------


def _log_event(payload: Dict[str, Any]) -> None:
    p = memory_log_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:
        logger.warning("memory.log write failed: %s", exc)


# ---------------------------------------------------------------------------
# Promotion: candidate gathering + neighbor search
# ---------------------------------------------------------------------------


def _read_pending_episodes(conn: sqlite3.Connection, days: int = PROMOTION_LOOKBACK_DAYS) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id, ts, channel, role, text, metadata, synthetic
        FROM episodes
        WHERE promoted_at IS NULL
          AND ts > datetime('now', ?)
        ORDER BY ts
        """,
        (f"-{days} days",),
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        meta = {}
        if r["metadata"]:
            try:
                meta = json.loads(r["metadata"])
            except json.JSONDecodeError:
                meta = {}
        out.append({
            "id": r["id"],
            "ts": r["ts"],
            "channel": r["channel"],
            "role": r["role"],
            "text": r["text"],
            "synthetic": bool(r["synthetic"]),
            "stashed_facts": meta.get("stashed_facts") or [],
        })
    return out


async def _vec_search(conn: sqlite3.Connection, query: str, k: int = PROMOTION_NEIGHBOR_K) -> List[Dict[str, Any]]:
    """Find k nearest existing semantic_facts to ``query`` text.

    Returns rows like {id, fact, entity, importance, sim}.
    """
    [qvec] = await voyage_embed([query])
    rows = conn.execute(
        """
        WITH knn AS (
            SELECT fact_id, distance
            FROM vec_facts
            WHERE embedding MATCH vec_int8(?) AND k = ?
        )
        SELECT sf.id, sf.fact, sf.entity, sf.importance,
               (1 - knn.distance) AS sim
        FROM knn
        JOIN semantic_facts sf ON sf.id = knn.fact_id
        WHERE sf.state = 'active'
          AND (sf.valid_to IS NULL OR sf.valid_to > date('now'))
        ORDER BY sim DESC
        """,
        (qvec, k),
    ).fetchall()
    return [dict(r) for r in rows]


def _format_candidates_block(candidates: List[Dict[str, Any]]) -> str:
    """Render candidate episodes as a compact block for the prompt."""
    lines = []
    for c in candidates:
        marker = "🤖" if c["synthetic"] else "👤"
        text = c["text"].replace("\n", " ")
        if len(text) > 200:
            text = text[:200] + "..."
        line = f"#{c['id']} [{c['ts']}] {marker} {c['channel']}/{c['role']}: {text}"
        lines.append(line)
        for sf in c["stashed_facts"]:
            sf_text = sf.get("text", "")
            sf_entity = sf.get("entity") or "?"
            sf_vth = sf.get("valid_to_hint") or "permanent"
            lines.append(
                f"   ↳ stashed: [{sf_entity}] {sf_text[:120]} "
                f"(importance={sf.get('importance', 2)}, valid_to_hint={sf_vth})"
            )
    return "\n".join(lines) if lines else "(no candidates)"


def _format_neighbors_block(neighbors_by_fact: Dict[str, List[Dict[str, Any]]]) -> str:
    """One section per candidate stashed_fact, listing its k nearest existing facts."""
    if not neighbors_by_fact:
        return "(no candidate stashed_facts to compare against)"
    sections = []
    for stashed_text, rows in neighbors_by_fact.items():
        header = f"--- nearest to: {stashed_text[:120]} ---"
        body_lines = [
            f"  #{r['id']} sim={r['sim']:.3f} [{r['entity'] or '—'}] {r['fact'][:120]}"
            for r in rows[:5]  # top 5 per stashed fact keeps prompt short
        ]
        sections.append(header + "\n" + "\n".join(body_lines))
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Kimi thinking call
# ---------------------------------------------------------------------------


class PromotionError(RuntimeError):
    pass


async def _call_kimi_thinking(prompt: str, *, client: Optional[httpx.AsyncClient] = None) -> Dict[str, Any]:
    """Single Kimi call producing the promotion diff JSON object.

    Tries Kimi-K2-Thinking first; falls back to Kimi-K2.5 on 4xx model-not-found.
    """
    api_key = _read_synthetic_api_key()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": PROMOTION_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": PROMOTION_TEMPERATURE,
        "max_tokens": PROMOTION_MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }

    owns = client is None
    client = client or httpx.AsyncClient()
    try:
        try:
            r = await client.post(SYNTHETIC_URL, headers=headers, json=payload, timeout=PROMOTION_TIMEOUT)
        except httpx.RequestError as exc:
            raise PromotionError(f"synthetic.new network: {exc}") from exc
        if r.status_code == 404 or (r.status_code == 400 and "model" in r.text.lower()):
            logger.warning("Kimi-Thinking unavailable; falling back to %s", PROMOTION_FALLBACK_MODEL)
            payload["model"] = PROMOTION_FALLBACK_MODEL
            r = await client.post(SYNTHETIC_URL, headers=headers, json=payload, timeout=PROMOTION_TIMEOUT)
        if r.status_code >= 400:
            raise PromotionError(f"synthetic.new {r.status_code}: {r.text[:300]}")
        body = r.json()
    finally:
        if owns:
            await client.aclose()

    content = ((body.get("choices") or [{}])[0].get("message") or {}).get("content", "")
    try:
        diff = json.loads(content)
    except json.JSONDecodeError as exc:
        raise PromotionError(f"Kimi returned non-JSON: {exc}: {content[:200]}") from exc
    if not isinstance(diff, dict):
        raise PromotionError(f"Kimi returned non-object: {type(diff).__name__}")

    usage = body.get("usage") or {}
    _log_event({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cmd": "weekly_promotion_kimi",
        "model": payload["model"],
        "tokens_in": usage.get("prompt_tokens"),
        "tokens_out": usage.get("completion_tokens"),
    })
    return diff


# ---------------------------------------------------------------------------
# Digest rendering (spec §5.4)
# ---------------------------------------------------------------------------


def render_digest_markdown(diff: WeekDigest, candidates: List[Dict[str, Any]]) -> str:
    n_user = sum(1 for c in candidates if not c["synthetic"])
    n_synth = sum(1 for c in candidates if c["synthetic"])
    header = (
        f"# 📚 Weekly Memory Review — {diff.digest_id.removeprefix('wk-')}\n"
        f"{len(candidates)} episodes scanned this week "
        f"({n_user} user/assistant + {n_synth} cron-synthetic).\n"
        f"24 h to reject via `/memreview reject {diff.digest_id}`; default approve.\n"
    )

    sections = []

    if diff.promote:
        lines = [f"## ⬆️ Promote to permanent ({len(diff.promote)})"]
        for p in diff.promote:
            entity = p.get("entity", "?")
            fact = p.get("fact", "")
            importance = p.get("importance", 2)
            valid_to = p.get("valid_to") or "永久"
            srcs = p.get("source_episode_ids") or []
            src_str = (
                ", ".join(f"#{i}" for i in srcs[:5])
                + (f" +{len(srcs)-5}" if len(srcs) > 5 else "")
            )
            lines.append(f"- 🆕 **{entity}**: \"{fact}\"")
            lines.append(f"   evidence: {src_str} | importance {importance} | valid_to: {valid_to}")
        sections.append("\n".join(lines))

    if diff.dedup_hits:
        lines = [f"## 🔁 Dedup confirmations ({len(diff.dedup_hits)})"]
        for d in diff.dedup_hits:
            srcs = d.get("source_episode_ids") or []
            action = d.get("action", "bump_hits")
            lines.append(
                f"- existing #{d.get('existing_fact_id')} ← {len(srcs)} reaffirmation(s), action={action}"
            )
            if action == "refine_text" and d.get("refined_text"):
                lines.append(f"   refined → \"{d['refined_text']}\"")
        sections.append("\n".join(lines))

    if diff.expire:
        lines = [f"## 🪦 Expiring ({len(diff.expire)})"]
        for e in diff.expire:
            lines.append(
                f"- existing #{e.get('existing_fact_id')} → valid_to={e.get('valid_to')} "
                f"({e.get('reason', '—')})"
            )
        sections.append("\n".join(lines))

    if diff.drop_as_noise:
        lines = [f"## 🗑️ Skipped as noise ({len(diff.drop_as_noise)})"]
        for n in diff.drop_as_noise:
            ids = n.get("episode_ids") or []
            lines.append(f"- {len(ids)} episode(s): {n.get('reason', '—')}")
        sections.append("\n".join(lines))

    if not sections:
        sections.append("_No actions this week._")

    return header + "\n" + "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Discord posting
# ---------------------------------------------------------------------------


def discord_post(content: str, channel_id: str, *, bot_token: Optional[str] = None) -> bool:
    """POST a message to a Discord channel. Returns True on success."""
    bot_token = bot_token or os.environ.get("DISCORD_BOT_TOKEN")
    if not bot_token or not channel_id:
        logger.warning("discord_post missing bot_token or channel_id")
        return False
    # Discord rejects messages over 2000 chars; chunk if needed.
    chunks: List[str] = []
    remaining = content
    while remaining:
        if len(remaining) <= 1990:
            chunks.append(remaining)
            break
        # Split on the last newline before 1990 chars to avoid mid-line breaks.
        cut = remaining.rfind("\n", 0, 1990)
        if cut <= 0:
            cut = 1990
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")

    headers = {
        "Authorization": f"Bot {bot_token}",
        "Content-Type": "application/json",
    }
    url = DISCORD_API.format(channel_id=channel_id)
    ok = True
    with httpx.Client(timeout=20.0) as c:
        for chunk in chunks:
            r = c.post(url, headers=headers, json={"content": chunk})
            if r.status_code >= 400:
                logger.warning("discord_post failed: %s %s", r.status_code, r.text[:200])
                ok = False
                break
    return ok


def memory_review_channel_id() -> Optional[str]:
    """Resolve the Discord #memory-review channel id.

    Priority:
      1. MEMORY_REVIEW_CHANNEL_ID env var (test override)
      2. ~/.hermes/channel_directory.json -> platforms.discord (list)
         -> first entry whose name == "memory-review"
      3. Legacy flat layouts (defensive — older installs)
    """
    env = os.environ.get("MEMORY_REVIEW_CHANNEL_ID")
    if env:
        return env
    cdir = _resolve_hermes_home() / "channel_directory.json"
    if not cdir.exists():
        return None
    try:
        data = json.loads(cdir.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    # Canonical layout: platforms.discord is a list of channel dicts.
    plats = (data.get("platforms") or {})
    discord_chans = plats.get("discord")
    if isinstance(discord_chans, list):
        for c in discord_chans:
            if isinstance(c, dict) and c.get("name") == "memory-review":
                return c.get("id")

    # Defensive fallbacks for older / hand-edited layouts.
    if isinstance(data.get("memory-review"), str):
        return data["memory-review"]
    chans = data.get("channels") or {}
    m = chans.get("memory-review") if isinstance(chans, dict) else None
    if isinstance(m, str):
        return m
    if isinstance(m, dict):
        return m.get("id") or m.get("channel_id")
    return None


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


async def weekly_promotion(
    conn: sqlite3.Connection,
    *,
    today: Optional[date] = None,
    dry_run: bool = False,
    discord_channel_id: Optional[str] = None,
    kimi_fn=None,  # injectable for tests
    embed_fn=None,
) -> Dict[str, Any]:
    """Run one weekly promotion cycle. Returns a summary dict."""
    today = today or date.today()
    digest_id = digest_id_for(today)

    # /mem off kill switch — skip the entire weekly cycle.
    try:
        from plugins.memreview import mem_off_active
        if mem_off_active():
            return {
                "digest_id": digest_id,
                "candidates": 0,
                "skipped": "/mem off active",
            }
    except Exception:
        pass

    candidates = _read_pending_episodes(conn)
    if not candidates:
        return {"digest_id": digest_id, "candidates": 0, "skipped": "no candidates"}

    # Build neighbor map per stashed_fact across the week.
    neighbors_by_fact: Dict[str, List[Dict[str, Any]]] = {}
    for c in candidates:
        for sf in c["stashed_facts"]:
            text = (sf or {}).get("text") or ""
            if not text or text in neighbors_by_fact:
                continue
            try:
                neighbors_by_fact[text] = await _vec_search(conn, text)
            except Exception as exc:
                logger.warning("vec_search failed for stashed fact: %s", exc)
                neighbors_by_fact[text] = []

    prompt = PROMOTION_PROMPT.format(
        digest_id=digest_id,
        today=today.isoformat(),
        week_label=today.isoformat(),
        candidates_block=_format_candidates_block(candidates),
        neighbors_block=_format_neighbors_block(neighbors_by_fact),
    )

    kimi = kimi_fn or _call_kimi_thinking
    try:
        diff_dict = await kimi(prompt)
    except Exception as exc:
        logger.exception("Kimi promotion call failed")
        return {"digest_id": digest_id, "candidates": len(candidates), "error": str(exc)}

    # Trust-but-verify: ensure digest_id matches and required keys exist.
    diff_dict.setdefault("digest_id", digest_id)
    diff_dict.setdefault("candidate_episode_ids", [c["id"] for c in candidates])
    for k in ("promote", "dedup_hits", "expire", "drop_as_noise"):
        diff_dict.setdefault(k, [])

    digest = WeekDigest.from_dict(diff_dict)
    markdown = render_digest_markdown(digest, candidates)

    summary = {
        "digest_id": digest_id,
        "candidates": len(candidates),
        "promote": len(digest.promote),
        "dedup_hits": len(digest.dedup_hits),
        "expire": len(digest.expire),
        "drop_as_noise": len(digest.drop_as_noise),
        "dry_run": dry_run,
    }

    if dry_run:
        summary["markdown_preview"] = markdown
        return summary

    # Persist diff before posting so a Discord outage doesn't lose the work.
    pending_path(digest_id).write_text(
        json.dumps(digest.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    posted = False
    if discord_channel_id:
        posted = discord_post(markdown, discord_channel_id)
    summary["discord_posted"] = posted

    _log_event({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cmd": "weekly_promotion",
        "digest_id": digest_id,
        "summary": summary,
    })
    return summary


def _purge_old_pending(today: date) -> int:
    """Delete pending diffs older than PENDING_DIFF_TTL_DAYS."""
    cutoff = today - timedelta(days=PENDING_DIFF_TTL_DAYS)
    n = 0
    for f in pending_dir().glob("*.json"):
        try:
            stem = f.stem.removeprefix("wk-")
            d = datetime.strptime(stem, "%Y-%m-%d").date()
        except ValueError:
            continue
        if d < cutoff:
            try:
                f.unlink()
                # Also remove associated rejection sentinel if any.
                rs = f.with_suffix(".rejected")
                if rs.exists():
                    rs.unlink()
                n += 1
            except OSError:
                pass
    return n


def _latest_pending_diff() -> Optional[Path]:
    files = sorted(pending_dir().glob("wk-*.json"))
    return files[-1] if files else None


def _archive_diff(diff_path: Path, status: str) -> None:
    target = archive_dir() / f"{diff_path.stem}.{status}.json"
    diff_path.replace(target)


async def _apply_diff_atomic(
    conn: sqlite3.Connection,
    digest: WeekDigest,
    today: date,
    *,
    embed_fn=None,
) -> Dict[str, int]:
    """Apply promote / dedup / expire in one transaction; stamp promoted_at.

    Embeddings for promoted facts are computed BEFORE the transaction
    opens, so the writer lock is held only for the duration of the
    SQL statements themselves (~ms). Holding it across the Voyage HTTP
    round-trip would block concurrent writes from the hot path.

    Returns counts of each action performed.
    """
    counts = {"promoted": 0, "dedup_bumped": 0, "dedup_refined": 0, "expired": 0, "stamped": 0}

    # Pre-embed all promote texts (outside transaction).
    embed = embed_fn or voyage_embed
    promote_blobs: List[Optional[bytes]] = []
    promote_texts = [p.get("fact", "") for p in digest.promote]
    non_empty = [t for t in promote_texts if t]
    if non_empty:
        embeddings = await embed(non_empty)
        # Map back to original positions (None for empty fact strings).
        emb_iter = iter(embeddings)
        promote_blobs = [next(emb_iter) if t else None for t in promote_texts]
    else:
        promote_blobs = [None] * len(promote_texts)

    try:
        conn.execute("BEGIN")

        # 1. promote — INSERT new semantic_facts. Trigger sf_after_insert
        # mirrors each row into vec_facts automatically.
        for p, blob in zip(digest.promote, promote_blobs):
            fact = p.get("fact", "")
            if not fact or blob is None:
                continue
            conn.execute(
                """
                INSERT INTO semantic_facts
                    (entity, fact, embedding, importance, valid_from, valid_to,
                     source_episode_ids)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    p.get("entity"),
                    fact,
                    blob,
                    int(p.get("importance", 2) or 2),
                    p.get("valid_from") or today.isoformat(),
                    p.get("valid_to"),
                    json.dumps(p.get("source_episode_ids") or []),
                ),
            )
            counts["promoted"] += 1

        # 2. dedup_hits — bump the existing fact's hits + last_seen, optional refine.
        for d in digest.dedup_hits:
            fid = d.get("existing_fact_id")
            if fid is None:
                continue
            if d.get("action") == "refine_text" and d.get("refined_text"):
                conn.execute(
                    "UPDATE semantic_facts SET fact = ?, last_seen = datetime('now'), "
                    "hits = hits + 1 WHERE id = ?",
                    (d["refined_text"], fid),
                )
                counts["dedup_refined"] += 1
            else:
                conn.execute(
                    "UPDATE semantic_facts SET last_seen = datetime('now'), "
                    "hits = hits + 1 WHERE id = ?",
                    (fid,),
                )
                counts["dedup_bumped"] += 1

        # 3. expire — set valid_to (caller chose date).
        for e in digest.expire:
            fid = e.get("existing_fact_id")
            if fid is None:
                continue
            conn.execute(
                "UPDATE semantic_facts SET valid_to = ? WHERE id = ?",
                (e.get("valid_to") or today.isoformat(), fid),
            )
            counts["expired"] += 1

        # 4. stamp promoted_at on every candidate episode.
        if digest.candidate_episode_ids:
            placeholders = ",".join("?" * len(digest.candidate_episode_ids))
            conn.execute(
                f"UPDATE episodes SET promoted_at = date('now') WHERE id IN ({placeholders})",
                digest.candidate_episode_ids,
            )
            counts["stamped"] = len(digest.candidate_episode_ids)

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return counts


async def weekly_apply(
    conn: sqlite3.Connection,
    *,
    today: Optional[date] = None,
    embed_fn=None,
) -> Dict[str, Any]:
    """Apply the latest pending diff (or archive-as-rejected). Returns summary."""
    today = today or date.today()

    purged = _purge_old_pending(today)
    diff_path = _latest_pending_diff()

    if not diff_path:
        return {"purged": purged, "applied": False, "reason": "no pending diff"}

    digest_id = diff_path.stem
    sentinel = rejection_sentinel(digest_id)
    if sentinel.exists():
        _archive_diff(diff_path, "rejected")
        try:
            sentinel.unlink()
        except OSError:
            pass
        _log_event({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "cmd": "weekly_apply",
            "digest_id": digest_id,
            "result": "rejected",
        })
        return {"purged": purged, "applied": False, "digest_id": digest_id, "reason": "rejected"}

    try:
        diff_dict = json.loads(diff_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"purged": purged, "applied": False, "error": f"diff load: {exc}"}

    digest = WeekDigest.from_dict(diff_dict)
    counts = await _apply_diff_atomic(conn, digest, today, embed_fn=embed_fn)
    _archive_diff(diff_path, "applied")

    summary = {
        "purged": purged,
        "applied": True,
        "digest_id": digest_id,
        **counts,
    }
    _log_event({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cmd": "weekly_apply",
        **summary,
    })
    return summary
