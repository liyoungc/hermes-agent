"""``/memreview`` and ``/mem`` slash commands — admin / kill-switch (W3-4).

Spec: docs/superpowers/specs/2026-05-02-hermes-memory-design.md §7.1.

Two commands:

  /memreview reject <digest_id>   - per-digest opt-out. Writes a sentinel
                                    file ``pending_diffs/<digest_id>.rejected``
                                    that ``weekly_apply`` reads on Monday
                                    morning and archives the diff without
                                    applying.

  /mem off                        - global kill switch. Writes ``MEM_OFF``
                                    in HERMES_HOME. Both ``write_episode``
                                    (hot path) and ``weekly_promotion``
                                    (cold path) check for this file at the
                                    top of each call and short-circuit to
                                    a no-op + warning log.

  /mem on                         - reverses the kill switch by deleting
                                    ``MEM_OFF`` (companion to /mem off).

  /mem status                     - prints whether the kill switch is set
                                    and lists pending diffs awaiting apply.

Why slash commands and not Discord reactions: spec §7.1 explicitly chose
slash because reactions don't reliably trigger webhook events across all
bot adapters (silent kill-switch failure mode that's worse than no
switch).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def _resolve_hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        return Path.home() / ".hermes"


def _pending_dir() -> Path:
    p = _resolve_hermes_home() / "memories" / "pending_diffs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _archive_dir() -> Path:
    return _resolve_hermes_home() / "memories" / "diff_archive"


def mem_off_path() -> Path:
    """The global kill-switch sentinel."""
    return _resolve_hermes_home() / "MEM_OFF"


def mem_off_active() -> bool:
    """Public predicate consumed by promotion.py + provider.sync_turn."""
    return mem_off_path().exists()


# ---------------------------------------------------------------------------
# /memreview <subcommand>
# ---------------------------------------------------------------------------


_MEMREVIEW_HELP = (
    "**/memreview** — review or reject the weekly memory promotion digest.\n"
    "Usage:\n"
    "  `/memreview reject <digest_id>` — write the rejection sentinel; "
    "Monday's apply will archive the diff without applying it.\n"
    "  `/memreview pending` — list digests currently awaiting apply.\n"
    "  `/memreview status` — same as `pending`."
)


_DIGEST_ID_RE = re.compile(r"^wk-\d{4}-\d{2}-\d{2}$")


def _list_pending_diffs() -> List[str]:
    out = []
    for f in sorted(_pending_dir().glob("wk-*.json")):
        rejected = f.with_suffix(".rejected").exists()
        flag = " (rejected — will be archived Mon)" if rejected else ""
        out.append(f"- `{f.stem}`{flag}")
    return out


def _handle_memreview(raw_args: str) -> str:
    args = (raw_args or "").strip()
    if not args:
        return _MEMREVIEW_HELP

    parts = args.split(maxsplit=1)
    sub = parts[0].lower()

    if sub in ("pending", "status", "list"):
        items = _list_pending_diffs()
        if not items:
            return "**/memreview** — no pending diffs."
        return "**/memreview** — pending diffs:\n" + "\n".join(items)

    if sub == "reject":
        rest = parts[1].strip() if len(parts) > 1 else ""
        if not _DIGEST_ID_RE.match(rest):
            return (
                f"**/memreview reject** — digest_id must look like "
                f"`wk-YYYY-MM-DD`. Got: `{rest!r}`"
            )
        diff_path = _pending_dir() / f"{rest}.json"
        if not diff_path.exists():
            return (
                f"**/memreview reject** — no pending diff named `{rest}`. "
                f"Use `/memreview pending` to list available digest_ids."
            )
        sentinel = _pending_dir() / f"{rest}.rejected"
        try:
            sentinel.write_text(
                f"rejected via /memreview at {asyncio.get_event_loop().time()}",
                encoding="utf-8",
            )
        except (OSError, RuntimeError):
            # No running loop in some sync entry paths — write a static marker.
            try:
                sentinel.write_text("rejected", encoding="utf-8")
            except OSError as exc:
                return f"**/memreview reject** error: cannot write sentinel: `{exc}`"
        return (
            f"**Rejected.** Pending diff `{rest}` will be archived without "
            f"applying. Episodes stay pending for next Sunday's review."
        )

    return _MEMREVIEW_HELP


# ---------------------------------------------------------------------------
# /mem <subcommand>
# ---------------------------------------------------------------------------


_MEM_HELP = (
    "**/mem** — global memory write-back kill switch.\n"
    "Usage:\n"
    "  `/mem off`    — disable per-turn write-back AND weekly promotion.\n"
    "  `/mem on`     — re-enable.\n"
    "  `/mem status` — show whether the kill switch is currently set."
)


def _handle_mem(raw_args: str) -> str:
    args = (raw_args or "").strip().lower()
    if not args:
        return _MEM_HELP

    sub = args.split()[0]

    if sub == "off":
        try:
            mem_off_path().write_text(
                "set via /mem off\n", encoding="utf-8"
            )
        except OSError as exc:
            return f"**/mem off** error: `{exc}`"
        return (
            "**🔇 Memory write-back disabled.**\n"
            "Per-turn `write_episode` and weekly promotion will short-circuit "
            "until you run `/mem on`. Read path is unaffected — Cattia still "
            "retrieves from existing facts."
        )

    if sub == "on":
        p = mem_off_path()
        if not p.exists():
            return "**/mem on** — write-back was already enabled."
        try:
            p.unlink()
        except OSError as exc:
            return f"**/mem on** error: `{exc}`"
        return "**🔊 Memory write-back enabled.** Hot + cold paths resume."

    if sub == "status":
        active = mem_off_active()
        pending = _list_pending_diffs()
        lines = [
            "**/mem status**",
            f"  write-back: {'🔇 OFF' if active else '🔊 ON'}",
            f"  MEM_OFF sentinel: `{mem_off_path()}`"
            f" {'(present)' if active else '(absent)'}",
        ]
        if pending:
            lines.append("  pending diffs:")
            lines.extend("    " + p for p in pending)
        else:
            lines.append("  pending diffs: (none)")
        return "\n".join(lines)

    return _MEM_HELP


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    ctx.register_command(
        "memreview",
        handler=_handle_memreview,
        description="Review or reject the weekly Hermes memory promotion digest.",
        args_hint="reject <digest_id> | pending | status",
    )
    ctx.register_command(
        "mem",
        handler=_handle_mem,
        description="Hermes memory kill switch (off / on / status).",
        args_hint="off | on | status",
    )
