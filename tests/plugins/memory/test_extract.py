"""Tests for plugins/memory/sqlite_vec/extract.py (W3-1)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from plugins.memory.sqlite_vec.extract import (
    EXTRACT_MODEL,
    EXTRACT_PROMPT,
    PHI_BLACKLIST_CHANNELS,
    ExtractError,
    ExtractedFact,
    _coerce_fact,
    _parse_json_list,
    kimi_extract,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_extract_prompt_is_verbatim_spec_5_2():
    """Spec §5.2 marks EXTRACT_PROMPT as a behavioural contract — verbatim."""
    # Anchors that uniquely identify the spec's exact wording.
    assert "You extract durable memories about 禮揚 from this Discord turn." in EXTRACT_PROMPT
    assert "HARD RULES — these override everything else:" in EXTRACT_PROMPT
    assert "ERR ON THE SIDE OF NOT EXTRACTING" in EXTRACT_PROMPT
    assert "Skip facts that duplicate something said in the last 5 turns." in EXTRACT_PROMPT
    # Placeholders must be preserved.
    assert "{ts}" in EXTRACT_PROMPT and "{channel}" in EXTRACT_PROMPT
    assert "{user}" in EXTRACT_PROMPT and "{assistant}" in EXTRACT_PROMPT


def test_phi_blacklist_matches_spec_5_1():
    assert PHI_BLACKLIST_CHANNELS == frozenset({"cmio", "cbme", "medicine"})


def test_parse_json_list_bare_array():
    assert _parse_json_list('[{"type":"semantic","text":"a"}]') == [
        {"type": "semantic", "text": "a"}
    ]


def test_parse_json_list_wrapped_object():
    assert _parse_json_list('{"facts": [{"type":"semantic","text":"a"}]}') == [
        {"type": "semantic", "text": "a"}
    ]
    assert _parse_json_list('{"items": [{"type":"semantic","text":"b"}]}') == [
        {"type": "semantic", "text": "b"}
    ]


def test_parse_json_list_empty_object_returns_empty_list():
    assert _parse_json_list("{}") == []
    assert _parse_json_list("") == []
    assert _parse_json_list("not even json") == []


def test_coerce_fact_drops_invalid_type():
    assert _coerce_fact({"type": "garbage", "text": "a"}) is None
    assert _coerce_fact({"type": "semantic"}) is None  # missing text
    assert _coerce_fact({"type": "semantic", "text": "  "}) is None  # blank text


def test_coerce_fact_clamps_importance():
    f = _coerce_fact({"type": "semantic", "text": "a", "importance": 99})
    assert f.importance == 5
    f = _coerce_fact({"type": "semantic", "text": "a", "importance": -3})
    assert f.importance == 1
    f = _coerce_fact({"type": "semantic", "text": "a", "importance": "not-int"})
    assert f.importance == 2  # default fallback


def test_coerce_fact_round_trip_full_shape():
    raw = {
        "type": "semantic",
        "text": "致妤 7:30 才到家",
        "entity": "禮揚.家庭",
        "importance": 3,
        "valid_to_hint": "2026-05-03",
    }
    f = _coerce_fact(raw)
    assert isinstance(f, ExtractedFact)
    assert f.text == "致妤 7:30 才到家"
    assert f.entity == "禮揚.家庭"
    assert f.importance == 3
    assert f.valid_to_hint == "2026-05-03"


# ---------------------------------------------------------------------------
# kimi_extract — short-circuits (no httpx call)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channel", ["cmio", "cbme", "medicine"])
def test_kimi_extract_phi_channel_returns_empty_no_call(channel, monkeypatch, tmp_path):
    """Even with no API key, PHI channels never hit the network."""
    monkeypatch.delenv("SYNTHETIC_API_KEY", raising=False)
    # Point auth.json lookup at an empty tmp dir so any leak would raise.
    monkeypatch.setattr(
        "plugins.memory.sqlite_vec.extract._resolve_hermes_home", lambda: tmp_path
    )
    out = asyncio.run(
        kimi_extract(
            "病人的血壓 180/100",
            "我建議轉診",
            channel=channel,
            ts="2026-05-02 09:00:00",
        )
    )
    assert out == []


def test_kimi_extract_empty_turn_returns_empty(monkeypatch, tmp_path):
    monkeypatch.delenv("SYNTHETIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "plugins.memory.sqlite_vec.extract._resolve_hermes_home", lambda: tmp_path
    )
    out = asyncio.run(
        kimi_extract("", "", channel="cattia", ts="2026-05-02 09:00:00")
    )
    assert out == []


# ---------------------------------------------------------------------------
# kimi_extract — mocked synthetic.new responses
# ---------------------------------------------------------------------------


def _mock_synthetic_response(facts: list, *, status: int = 200):
    """Build a synthetic.new chat-completions JSON body wrapping `facts`."""
    body = {
        "id": "test",
        "choices": [
            {
                "message": {"role": "assistant", "content": json.dumps(facts)},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 200, "completion_tokens": 80},
    }
    return status, body


class _FakeTransport(httpx.MockTransport):
    def __init__(self, status, body):
        self.calls = []
        self._status = status
        self._body = body
        super().__init__(self._h)

    def _h(self, request: httpx.Request):
        self.calls.append(request)
        return httpx.Response(self._status, json=self._body)


def test_kimi_extract_pleasantry_returns_empty_after_call(monkeypatch, tmp_path):
    monkeypatch.setenv("SYNTHETIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "plugins.memory.sqlite_vec.extract._resolve_hermes_home", lambda: tmp_path
    )
    status, body = _mock_synthetic_response([])  # Kimi correctly returns []
    transport = _FakeTransport(status, body)
    client = httpx.AsyncClient(transport=transport)

    out = asyncio.run(
        kimi_extract(
            "好的", "收到", channel="cattia", ts="2026-05-02 09:00:00",
            client=client, log_path=tmp_path / "memory.log",
        )
    )
    assert out == []
    assert len(transport.calls) == 1
    log_line = (tmp_path / "memory.log").read_text().strip()
    assert '"cmd": "kimi_extract"' in log_line
    assert '"n_kept": 0' in log_line


def test_kimi_extract_short_lived_fact_with_valid_to_hint(monkeypatch, tmp_path):
    monkeypatch.setenv("SYNTHETIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "plugins.memory.sqlite_vec.extract._resolve_hermes_home", lambda: tmp_path
    )
    facts = [
        {
            "type": "semantic",
            "text": "致妤今晚 (2026-05-02) 預計 7:30 才到家",
            "entity": "禮揚.家庭/今晚",
            "importance": 3,
            "valid_to_hint": "2026-05-03",
        }
    ]
    transport = _FakeTransport(*_mock_synthetic_response(facts))
    client = httpx.AsyncClient(transport=transport)

    out = asyncio.run(
        kimi_extract(
            "今晚致妤會晚回來，大概 7:30 才到", "好喔",
            channel="at-home", ts="2026-05-02 09:00:00",
            client=client, log_path=tmp_path / "memory.log",
        )
    )
    assert len(out) == 1
    f = out[0]
    assert f.type == "semantic"
    assert "7:30" in f.text
    assert f.valid_to_hint == "2026-05-03"
    assert f.importance == 3


def test_kimi_extract_long_lived_fact_no_valid_to(monkeypatch, tmp_path):
    monkeypatch.setenv("SYNTHETIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "plugins.memory.sqlite_vec.extract._resolve_hermes_home", lambda: tmp_path
    )
    facts = [
        {
            "type": "semantic",
            "text": "禮揚 最近在追 sleep medicine 的 RCT",
            "entity": "禮揚.研究興趣",
            "importance": 2,
        }
    ]
    transport = _FakeTransport(*_mock_synthetic_response(facts))
    client = httpx.AsyncClient(transport=transport)

    out = asyncio.run(
        kimi_extract(
            "最近在追 sleep medicine", "了解，要幫你 follow up 嗎",
            channel="cattia", ts="2026-05-02 09:00:00",
            client=client, log_path=tmp_path / "memory.log",
        )
    )
    assert len(out) == 1
    assert out[0].valid_to_hint is None
    assert out[0].entity == "禮揚.研究興趣"


def test_kimi_extract_drops_malformed_rows(monkeypatch, tmp_path):
    monkeypatch.setenv("SYNTHETIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "plugins.memory.sqlite_vec.extract._resolve_hermes_home", lambda: tmp_path
    )
    facts = [
        {"type": "semantic", "text": "good fact"},
        {"type": "garbage", "text": "bad type"},      # dropped
        {"type": "episodic"},                           # missing text → dropped
        {"type": "semantic", "text": "  "},             # blank text → dropped
    ]
    transport = _FakeTransport(*_mock_synthetic_response(facts))
    client = httpx.AsyncClient(transport=transport)

    out = asyncio.run(
        kimi_extract(
            "u", "a", channel="cattia", ts="2026-05-02 09:00:00",
            client=client, log_path=tmp_path / "memory.log",
        )
    )
    assert len(out) == 1
    assert out[0].text == "good fact"


def test_kimi_extract_5xx_raises_extracterror(monkeypatch, tmp_path):
    monkeypatch.setenv("SYNTHETIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "plugins.memory.sqlite_vec.extract._resolve_hermes_home", lambda: tmp_path
    )
    transport = _FakeTransport(503, {"error": "down"})
    client = httpx.AsyncClient(transport=transport)
    with pytest.raises(ExtractError):
        asyncio.run(
            kimi_extract(
                "u", "a", channel="cattia", ts="2026-05-02 09:00:00",
                client=client, log_path=tmp_path / "memory.log",
            )
        )


def test_kimi_extract_no_api_key_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("SYNTHETIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "plugins.memory.sqlite_vec.extract._resolve_hermes_home", lambda: tmp_path
    )  # auth.json absent
    with pytest.raises(ExtractError, match="API key"):
        asyncio.run(
            kimi_extract(
                "u", "a", channel="cattia", ts="2026-05-02 09:00:00",
                log_path=tmp_path / "memory.log",
            )
        )


def test_kimi_extract_reads_auth_json_when_no_env(monkeypatch, tmp_path):
    monkeypatch.delenv("SYNTHETIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "plugins.memory.sqlite_vec.extract._resolve_hermes_home", lambda: tmp_path
    )
    auth = {
        "credential_pool": {
            "custom:synthetic": [
                {"id": "test", "access_token": "syn_test_xxx"},
            ]
        }
    }
    (tmp_path / "auth.json").write_text(json.dumps(auth), encoding="utf-8")
    transport = _FakeTransport(*_mock_synthetic_response([]))
    client = httpx.AsyncClient(transport=transport)

    out = asyncio.run(
        kimi_extract(
            "x", "y", channel="cattia", ts="2026-05-02 09:00:00",
            client=client, log_path=tmp_path / "memory.log",
        )
    )
    assert out == []
    # The Authorization header carried the auth.json token.
    assert transport.calls[0].headers["Authorization"] == "Bearer syn_test_xxx"



# ===========================================================================
# Additional parser shapes discovered during live smoke test
# ===========================================================================


def test_parse_json_list_extracted_memories_key():
    """Kimi K2.5 with response_format=json_object often wraps the answer in
    a dict with key 'extracted_memories' (sometimes alongside an 'analysis'
    field showing its reasoning). Both must be parsed correctly."""
    payload = (
        '{"analysis": "the user mentions...", '
        '"extracted_memories": [{"type":"semantic","text":"a"}]}'
    )
    out = _parse_json_list(payload)
    assert out == [{"type": "semantic", "text": "a"}]


def test_parse_json_list_bare_single_fact_dict():
    """Kimi sometimes returns a single fact as a flat dict instead of a list.
    We detect that shape by the presence of canonical fact keys."""
    payload = (
        '{"type": "episodic", "text": "致妤今晚 7:30", '
        '"entity": "禮揚.家庭", "importance": 2}'
    )
    out = _parse_json_list(payload)
    assert len(out) == 1
    assert out[0]["text"] == "致妤今晚 7:30"


def test_parse_json_list_arbitrary_dict_falls_back_to_first_list():
    """If neither canonical keys nor fact-shape is present, the first
    list-valued field is returned. Defensive against future Kimi changes."""
    payload = '{"weird_unique_key": [{"type":"semantic","text":"x"}]}'
    out = _parse_json_list(payload)
    assert out == [{"type": "semantic", "text": "x"}]
