"""Tests for the LINE platform adapter."""
import base64
import hashlib
import hmac as _hmac

import pytest

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _line_sig(body: bytes, secret: str) -> str:
    """Compute valid LINE HMAC-SHA256 signature (base64-encoded)."""
    h = _hmac.new(secret.encode("utf-8"), body, hashlib.sha256)
    return base64.b64encode(h.digest()).decode("utf-8")


def _make_adapter(platform=None, **extra):
    """Create a LineAdapter with test credentials."""
    from gateway.platforms.line import LineAdapter

    cfg = PlatformConfig(
        enabled=True,
        extra={
            "channel_access_token": "test-token",
            "channel_secret": "test-secret",
            **extra,
        },
    )
    p = platform or Platform.LINE
    return LineAdapter(cfg, platform=p)


# ---------------------------------------------------------------------------
# Chunk 1: Platform enum
# ---------------------------------------------------------------------------

class TestLinePlatformEnum:
    def test_line_enum_exists(self):
        assert Platform.LINE.value == "line"

    def test_line_lynx_enum_exists(self):
        assert Platform.LINE_LYNX.value == "line_lynx"


class TestLineConfigDetection:
    def test_line_connected_when_token_set(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "tok")
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "sec")
        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.LINE in config.get_connected_platforms()

    def test_line_not_connected_without_token(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.LINE not in config.get_connected_platforms()

    def test_line_lynx_connected_when_token_set(self, monkeypatch):
        monkeypatch.setenv("LINE_LYNX_CHANNEL_ACCESS_TOKEN", "tok")
        monkeypatch.setenv("LINE_LYNX_CHANNEL_SECRET", "sec")
        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.LINE_LYNX in config.get_connected_platforms()
