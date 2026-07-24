"""Regression coverage for the LINE-only guarded group trust contract."""

from __future__ import annotations

import asyncio
import dataclasses
from collections import OrderedDict
import sys
import threading
import types
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
    adapter.set_ingress_rebinder(
        lambda _parents, child, _session_key: child
    )
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


@pytest.mark.asyncio
async def test_guarded_group_agent_has_only_mochiwiz_and_no_operational_callbacks(
    monkeypatch,
):
    """Tool/status/approval/clarify/compaction rails are absent for the group."""
    import gateway.run as gateway_run

    created = []

    class FakeAgent:
        def __init__(self, **kwargs):
            self.session_id = kwargs["session_id"]
            self.enabled_toolsets = kwargs["enabled_toolsets"]
            self.model = kwargs["model"]
            self.tools = []
            self.context_compressor = SimpleNamespace(
                last_prompt_tokens=0,
                context_length=200_000,
            )
            self.session_prompt_tokens = 0
            self.session_completion_tokens = 0
            created.append(self)

        def run_conversation(
            self,
            user_message,
            conversation_history=None,
            task_id=None,
            **_kwargs,
        ):
            return {
                "failed": False,
                "final_response": "done",
                "messages": [{"role": "assistant", "content": "done"}],
                "api_calls": 1,
            }

        def interrupt(self, *_args, **_kwargs):
            return None

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "0")

    import hermes_cli.tools_config as tools_config
    from tools import approval

    monkeypatch.setattr(
        tools_config,
        "_get_platform_tools",
        lambda *_args, **_kwargs: {"terminal", "web", "memory"},
    )
    register_notify = MagicMock()
    monkeypatch.setattr(approval, "register_gateway_notify", register_notify)

    source = SessionSource(
        platform=LINE,
        chat_id="Capproved",
        chat_type="group",
        user_id="Umember",
        ingress_shared_session=True,
        ingress_sender_authorized=True,
        ingress_enabled_toolsets=("mochiwiz",),
        ingress_suppress_operational_output=True,
    )
    session_key = build_session_key(source)
    session_entry = SimpleNamespace(
        session_key=session_key,
        session_id="session-1",
    )
    adapter = MagicMock()
    adapter.SUPPORTS_MESSAGE_EDITING = True
    adapter.supports_status_text = True
    adapter.send = AsyncMock()
    adapter.get_pending_message.return_value = None
    adapter.send_typing = AsyncMock()
    adapter.stop_typing = AsyncMock()

    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {LINE: adapter}
    runner.config = SimpleNamespace(
        streaming=None,
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    )
    runner.hooks = SimpleNamespace(loaded_hooks=True, emit=AsyncMock())
    runner.session_store = SimpleNamespace(
        _entries={session_key: session_entry},
        _save=lambda: None,
        _record_gateway_session_peer=lambda *_args: None,
    )
    runner._session_db = MagicMock()
    runner._agent_cache = OrderedDict()
    runner._agent_cache_lock = threading.Lock()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._session_run_generation = {}
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._pending_skills_reload_notes = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._draining = False
    runner._get_proxy_url = lambda: None
    runner._resolve_session_agent_runtime = lambda **_kwargs: (
        "test-model",
        {"provider": "test", "api_key": "token"},
    )
    runner._resolve_session_reasoning_config = lambda **_kwargs: None
    runner._resolve_turn_agent_config = lambda message, model, runtime: {
        "model": model,
        "runtime": runtime,
    }
    runner._load_service_tier = lambda: None
    runner._extract_cache_busting_config = lambda _config: ()
    runner._thread_metadata_for_source = lambda *_args, **_kwargs: None
    runner._sync_telegram_topic_binding = MagicMock()
    runner._release_running_agent_state = MagicMock()

    result = await asyncio.wait_for(
        runner._run_agent(
            message="record this",
            context_prompt="",
            history=[],
            source=source,
            session_id="session-1",
            session_key=session_key,
        ),
        timeout=2,
    )

    assert result["final_response"] == "done"
    assert len(created) == 1
    agent = created[0]
    assert agent.enabled_toolsets == ["mochiwiz"]
    assert agent.tool_progress_callback is None
    assert agent.step_callback is None
    assert agent.event_callback is None
    assert agent.interim_assistant_callback is None
    assert agent.status_callback is None
    assert agent.notice_callback is None
    assert agent.background_review_callback is None
    assert agent.clarify_callback is None
    register_notify.assert_not_called()
    adapter.send.assert_not_awaited()
