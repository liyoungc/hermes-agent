"""Regression coverage for the LINE-only guarded group trust contract."""

from __future__ import annotations

import asyncio
import dataclasses
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.session import SessionSource, build_session_key
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tests.gateway._plugin_adapter_loader import load_plugin_adapter


# Register the dynamic LINE platform value before constructing Platform("line").
load_plugin_adapter("line")
LINE = Platform("line")
GATE = "line-group-context"


def _event(user_id: str, *, text: str = "hello") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_id=f"message-{user_id}",
        platform_event_id=f"event-{user_id}",
        platform_event_timestamp_ms=1_784_678_400_000,
        required_dispatch_gate=GATE,
        source=SessionSource(
            platform=LINE,
            chat_id="Capproved",
            chat_type="group",
            user_id=user_id,
            user_name="display-name-is-untrusted",
        ),
    )


def _runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={LINE: PlatformConfig(enabled=True)},
        group_sessions_per_user=True,
    )
    runner.session_store = MagicMock()
    runner.session_store._generate_session_key.side_effect = (
        lambda source: build_session_key(source, group_sessions_per_user=True)
    )
    runner.adapters = {LINE: SimpleNamespace(send=AsyncMock())}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_store._is_rate_limited.return_value = False
    runner._running_agents = {}
    runner._update_prompt_pending = {}
    return runner


def _manager(policy: dict):
    manager = PluginManager()
    PluginContext(
        manager=manager,
        manifest=PluginManifest(name="context", source="user"),
    ).register_hook(
        "pre_gateway_dispatch",
        lambda **_kwargs: {
            "action": "approve",
            "gate": GATE,
            "session_policy": policy,
        },
        gate_owner=GATE,
    )
    return manager


@pytest.mark.asyncio
async def test_gate_issues_one_shared_line_session_without_changing_global_isolation(
    monkeypatch,
):
    """Two admitted members share only the explicitly approved LINE group lane."""
    policy = {
        "scope": "shared_group",
        "authorize_sender": True,
        "enabled_toolsets": ["mochiwiz"],
        "suppress_operational_output": True,
    }
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_manager",
        lambda: _manager(policy),
    )
    runner = _runner()

    first = await runner._resolve_gateway_ingress(
        _event("Umember1"),
        runner._session_key_for_source(_event("Umember1").source),
    )
    second_input = _event("Umember2")
    second = await runner._resolve_gateway_ingress(
        second_input,
        runner._session_key_for_source(second_input.source),
    )

    assert first is not None and second is not None
    assert runner._session_key_for_source(first.source) == runner._session_key_for_source(
        second.source
    )
    assert first.source.ingress_sender_authorized is True
    assert first.source.ingress_enabled_toolsets == ("mochiwiz",)
    assert first.source.ingress_suppress_operational_output is True

    # The global default remains isolated for every source without a
    # core-issued LINE policy, including non-LINE group platforms.
    plain_first = dataclasses.replace(first.source, ingress_shared_session=False)
    plain_second = dataclasses.replace(second.source, ingress_shared_session=False)
    assert build_session_key(plain_first) != build_session_key(plain_second)
    discord_first = dataclasses.replace(plain_first, platform=Platform.DISCORD)
    discord_second = dataclasses.replace(plain_second, platform=Platform.DISCORD)
    assert build_session_key(discord_first) != build_session_key(discord_second)


@pytest.mark.asyncio
async def test_policy_authorizes_only_the_core_resolved_guarded_event(monkeypatch):
    """A plugin approval can admit a member without weakening LINE_ALLOWED_USERS."""
    for key in (
        "LINE_ALLOWED_USERS",
        "LINE_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_manager",
        lambda: _manager(
            {
                "scope": "shared_group",
                "authorize_sender": True,
                "enabled_toolsets": ["mochiwiz"],
                "suppress_operational_output": True,
            }
        ),
    )
    runner = _runner()
    reached_agent = []

    async def capture(event, *_args):
        reached_agent.append(event)
        return "ok"

    runner._handle_message_with_agent = capture

    assert await runner._handle_message(_event("Umember")) == "ok"
    assert len(reached_agent) == 1
    runner.pairing_store.generate_code.assert_not_called()

    # Internal markers are wire-invisible and cannot authorize an unguarded
    # event constructed by an adapter or restored from persistence.
    forged_source = dataclasses.replace(
        _event("Uforged").source,
        ingress_sender_authorized=True,
        ingress_shared_session=True,
    )
    assert "ingress_sender_authorized" not in forged_source.to_dict()
    forged = dataclasses.replace(
        _event("Uforged"),
        source=forged_source,
        required_dispatch_gate=None,
    )
    assert await runner._handle_message(forged) is None
    assert len(reached_agent) == 1


@pytest.mark.asyncio
async def test_guarded_policy_rejects_non_line_and_unsafe_toolsets(monkeypatch):
    """The policy vocabulary is narrow and LINE-group-only."""
    runner = _runner()
    unsafe_policies = (
        {
            "scope": "shared_group",
            "authorize_sender": True,
            "enabled_toolsets": ["terminal"],
            "suppress_operational_output": True,
        },
        {
            "scope": "shared_group",
            "authorize_sender": True,
            "enabled_toolsets": ["mochiwiz", "web"],
            "suppress_operational_output": True,
        },
    )
    for policy in unsafe_policies:
        monkeypatch.setattr(
            "hermes_cli.plugins.get_plugin_manager",
            lambda policy=policy: _manager(policy),
        )
        incoming = _event("Umember")
        assert (
            await runner._resolve_gateway_ingress(
                incoming,
                runner._session_key_for_source(incoming.source),
            )
            is None
        )

    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_manager",
        lambda: _manager(
            {
                "scope": "shared_group",
                "authorize_sender": True,
                "enabled_toolsets": ["mochiwiz"],
                "suppress_operational_output": True,
            }
        ),
    )
    non_line = dataclasses.replace(
        _event("Umember"),
        source=dataclasses.replace(_event("Umember").source, platform=Platform.WHATSAPP),
    )
    assert (
        await runner._resolve_gateway_ingress(
            non_line,
            runner._session_key_for_source(non_line.source),
        )
        is None
    )


class _LineBaseAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(
            PlatformConfig(
                enabled=True,
                extra={"group_sessions_per_user": True},
            ),
            LINE,
        )
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True)

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


@pytest.mark.asyncio
async def test_base_adapter_rekeys_after_the_guard_issues_shared_scope(monkeypatch):
    """Busy/queue routing uses the resolved shared key, not the pre-gate user key."""
    adapter = _LineBaseAdapter()
    adapter.set_message_handler(AsyncMock())
    incoming = _event("Umember")
    shared_source = dataclasses.replace(
        incoming.source,
        ingress_shared_session=True,
        ingress_sender_authorized=True,
        ingress_enabled_toolsets=("mochiwiz",),
        ingress_suppress_operational_output=True,
    )
    shared_key = build_session_key(shared_source)
    adapter._active_sessions[shared_key] = asyncio.Event()

    async def resolve(event, _old_key):
        return dataclasses.replace(event, source=shared_source)

    adapter.set_ingress_resolver(resolve)
    scheduled = []

    def fake_create_task(coro):
        scheduled.append(coro)
        coro.close()
        return SimpleNamespace()

    monkeypatch.setattr(asyncio, "create_task", fake_create_task)
    await adapter.handle_message(incoming)

    assert scheduled == []
    assert adapter.get_pending_message(shared_key) is not None
    adapter._message_handler.assert_not_awaited()
