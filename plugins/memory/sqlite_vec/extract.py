"""Kimi-driven extraction from a single Discord turn (W3-1).

Spec: docs/superpowers/specs/2026-05-02-hermes-memory-design.md §5.2.

The ``EXTRACT_PROMPT`` constant is **verbatim** from the spec — do not
paraphrase. Drift here directly compromises the F2 monitoring path
(downstream weekly review will see noise).

Two-stage flow:

    1. Caller calls ``kimi_extract(user, assistant, channel, ts)``.
    2. We short-circuit to ``[]`` if ``channel`` is in
       ``PHI_BLACKLIST_CHANNELS`` — never round-trip hospital data
       through the cloud LLM.
    3. Otherwise we POST to synthetic.new'\\''s OpenAI-compatible
       chat-completions endpoint with ``temperature=0.1`` and
       ``response_format=json_object`` (Kimi K2.5 supports the OpenAI
       structured-output flag).
    4. Parse the JSON list, validate the per-item shape, return
       ``list[ExtractedFact]``. Bad rows are dropped, not fatal.

Token cost is logged to ``memory.log`` so weekly review can spot a
runaway extract budget.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)

# Spec §1.4 lock — Kimi K2.5 via synthetic.new.
SYNTHETIC_URL = "https://api.synthetic.new/v1/chat/completions"
EXTRACT_MODEL = "hf:moonshotai/Kimi-K2.5"
EXTRACT_TEMPERATURE = 0.1
EXTRACT_TIMEOUT = 30.0
EXTRACT_MAX_TOKENS = 1024  # extract output is a small JSON list

# Spec §5.1 — channels whose content never leaves the host as PHI.
PHI_BLACKLIST_CHANNELS = frozenset({"cmio", "cbme", "medicine"})

# Spec §5.2 EXTRACT_PROMPT — copy verbatim. The {placeholders} are
# substituted at call time.
EXTRACT_PROMPT = """You extract durable memories about 禮揚 from this Discord turn.
Output a JSON list. Empty list [] if nothing memorable.

HARD RULES — these override everything else:
1. NEVER extract: hospital data, patient names, 病歷號, 身分證字號, lab results,
   diagnoses about real people, hospital policy specifics, hospital colleague names.
2. NEVER extract pleasantries (好的/收到/早安/明白/thanks). Return [] if turn is just this.
3. If turn metadata says synthetic=true (cron-produced), return [] UNLESS content
   contains a NEW commitment by 禮揚 (e.g. "排了 5/22 跟 Y 開會").
4. If unsure whether content violates rule 1, ERR ON THE SIDE OF NOT EXTRACTING.

Each item:
  type: "episodic" | "semantic"
  text: short statement, zh-TW or English (match source language)
  entity: nullable. Use ".家庭", ".工作", ".研究興趣", ".健康", etc. namespacing under "禮揚."
  importance: 1-5
  valid_to_hint: ISO date if turn implies expiry. "今晚"→tomorrow, "這週"→Sunday, "這個月"→end-of-month.

Skip facts that duplicate something said in the last 5 turns.

TURN:
[{ts}] [{channel}] user: {user}
[{ts}] [{channel}] assistant: {assistant}
"""


@dataclass
class ExtractedFact:
    """One fact extracted from a turn. Distinct from the read-side ``Fact``."""

    type: str  # "episodic" | "semantic"
    text: str
    entity: Optional[str]
    importance: int
    valid_to_hint: Optional[str] = None
    raw: dict = field(default_factory=dict)  # original Kimi output for forensics


class ExtractError(RuntimeError):
    """Raised when synthetic.new is unreachable or returns malformed payload."""


def _resolve_hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        return Path.home() / ".hermes"


def _default_log_path() -> Path:
    return _resolve_hermes_home() / "logs" / "memory.log"


def _read_synthetic_api_key() -> str:
    """Resolve the synthetic.new API key.

    Priority:
      1. ``SYNTHETIC_API_KEY`` env var (test-friendly override).
      2. ``auth.json`` ``custom:synthetic`` pool, first non-expired token.

    Raises ``ExtractError`` if no key is found — the caller decides
    whether that should bubble up (W3-2 wraps and falls back).
    """
    env = os.environ.get("SYNTHETIC_API_KEY")
    if env:
        return env

    auth_path = _resolve_hermes_home() / "auth.json"
    if auth_path.exists():
        try:
            data = json.loads(auth_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ExtractError(f"auth.json parse: {exc}") from exc
        # The real auth.json uses "credential_pool" (singular). Older or
        # alternate layouts may use the plural form or top-level keys, so we
        # check all three for resilience across hermes-agent versions.
        pool = (
            (data.get("credential_pool") or {}).get("custom:synthetic")
            or (data.get("credential_pools") or {}).get("custom:synthetic")
            or data.get("custom:synthetic")
            or []
        )
        for entry in pool:
            tok = entry.get("access_token")
            if tok:
                return tok

    raise ExtractError(
        "synthetic.new API key not found. Set SYNTHETIC_API_KEY or "
        "ensure auth.json has a custom:synthetic credential."
    )


def _append_log(payload: dict, log_path: Optional[Path] = None) -> None:
    log_path = log_path or _default_log_path()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("memory.log write failed: %s", exc)


def _coerce_fact(raw: dict) -> Optional[ExtractedFact]:
    """Validate one Kimi-emitted fact dict; return None on shape errors."""
    t = raw.get("type")
    text = raw.get("text")
    if t not in ("episodic", "semantic"):
        return None
    if not isinstance(text, str) or not text.strip():
        return None
    importance = raw.get("importance", 2)
    try:
        importance = int(importance)
    except (TypeError, ValueError):
        importance = 2
    importance = max(1, min(5, importance))
    entity = raw.get("entity")
    if entity is not None and not isinstance(entity, str):
        entity = None
    valid_to_hint = raw.get("valid_to_hint")
    if valid_to_hint is not None and not isinstance(valid_to_hint, str):
        valid_to_hint = None
    return ExtractedFact(
        type=t,
        text=text.strip(),
        entity=entity,
        importance=importance,
        valid_to_hint=valid_to_hint,
        raw=raw,
    )


async def kimi_extract(
    user: str,
    assistant: str,
    channel: str,
    ts: str,
    *,
    client: Optional[httpx.AsyncClient] = None,
    log_path: Optional[Path] = None,
) -> List[ExtractedFact]:
    """Extract durable memories from one Discord turn.

    Returns ``[]`` (no API call) when ``channel`` is PHI-blacklisted, when
    both ``user`` and ``assistant`` are empty, or when Kimi returns
    malformed JSON. Otherwise raises ``ExtractError`` on transport
    failure or non-2xx response — caller (W3-2) is responsible for
    fallback bookkeeping (failure JSONL log).
    """
    if channel in PHI_BLACKLIST_CHANNELS:
        return []
    if not (user or "").strip() and not (assistant or "").strip():
        return []

    api_key = _read_synthetic_api_key()
    prompt = EXTRACT_PROMPT.format(ts=ts, channel=channel, user=user, assistant=assistant)

    payload = {
        "model": EXTRACT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": EXTRACT_TEMPERATURE,
        "max_tokens": EXTRACT_MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    owns_client = client is None
    client = client or httpx.AsyncClient()
    t0 = time.perf_counter()
    try:
        try:
            r = await client.post(
                SYNTHETIC_URL, headers=headers, json=payload, timeout=EXTRACT_TIMEOUT
            )
        except httpx.RequestError as exc:
            raise ExtractError(f"synthetic.new network error: {exc}") from exc
        if r.status_code >= 400:
            raise ExtractError(f"synthetic.new {r.status_code}: {r.text[:200]}")
        body = r.json()
    finally:
        if owns_client:
            await client.aclose()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    choice = (body.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content", "")
    usage = body.get("usage") or {}

    parsed = _parse_json_list(content)
    facts = [f for f in (_coerce_fact(item) for item in parsed) if f is not None]

    _append_log(
        {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "cmd": "kimi_extract",
            "channel": channel,
            "ms": round(elapsed_ms, 2),
            "n_raw": len(parsed),
            "n_kept": len(facts),
            "tokens_in": usage.get("prompt_tokens"),
            "tokens_out": usage.get("completion_tokens"),
        },
        log_path=log_path,
    )
    return facts


def _parse_json_list(content: str) -> list:
    """Tolerantly extract a JSON list from Kimi's ``content`` field.

    The prompt asks for a JSON list, but Kimi may wrap it in an object
    (when response_format=json_object) like ``{"facts": [...]}`` or
    return ``{}`` for empty. We accept any of:
      - bare ``[...]``
      - ``{"facts": [...]}`` / ``{"items": [...]}`` / ``{"results": [...]}``
      - ``{}`` (treated as empty list)
    """
    if not content:
        return []
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Kimi K2.5 with response_format=json_object often wraps the
        # answer in a dict like {"analysis": ..., "extracted_memories": [...]}.
        # Try the canonical key names first, then fall back to the first list-valued field.
        for key in ("facts", "items", "results", "memories", "extracted_memories", "data"):
            v = data.get(key)
            if isinstance(v, list):
                return v
        # Last-ditch fallback: any top-level list value wins.
        for v in data.values():
            if isinstance(v, list):
                return v
        # Kimi sometimes returns a single fact as a flat dict (no list wrapper).
        # Detect by the presence of the canonical fact keys.
        if "type" in data and "text" in data:
            return [data]
        return []
    return []
