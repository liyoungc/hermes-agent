"""LINE Messaging API platform adapter.

Supports two LINE Bot accounts (Platform.LINE and Platform.LINE_LYNX).
Uses Push API for all outbound messages (reply_token TTL too short for LLM).
Per-group persona routing via MessageEvent.auto_skill.
"""

import asyncio
import base64
import hashlib
import hmac
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SDK availability guard
# ---------------------------------------------------------------------------

LINE_SDK_AVAILABLE = False
try:
    from linebot.v3.webhooks import (
        WebhookParser,
        MessageEvent as LineMessageEvent,
        InvalidSignatureError,
        TextMessageContent,
        ImageMessageContent,
        StickerMessageContent,
        AudioMessageContent,
    )
    LINE_SDK_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WEBHOOK_PORT = {
    "line": 18791,
    "line_lynx": 18792,
}
DEFAULT_WEBHOOK_PATH = {
    "line": "/webhook/line",
    "line_lynx": "/webhook/line/lynx",
}
LINE_PUSH_API_URL = "https://api.line.me/v2/bot/message/push"
LINE_CONTENT_API_URL = "https://api-data.line.me/v2/bot/message/{message_id}/content"
MAX_TEXT_LENGTH = 4900  # LINE limit is 5000; buffer of 100


def check_line_requirements() -> bool:
    """Check whether LINE SDK and credentials are available."""
    if not LINE_SDK_AVAILABLE:
        logger.warning("[line] line-bot-sdk not installed. Run: pip install line-bot-sdk")
        return False
    has_main = bool(os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
    has_lynx = bool(os.getenv("LINE_LYNX_CHANNEL_ACCESS_TOKEN"))
    if not has_main and not has_lynx:
        logger.warning("[line] No LINE credentials found in environment")
        return False
    return True


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _strip_markdown(text: str) -> str:
    """Remove markdown formatting markers; preserve text content."""
    # Code blocks (``` ... ```)
    text = re.sub(r"```[a-zA-Z]*\n?(.*?)```", r"\1", text, flags=re.DOTALL)
    # Inline code
    text = re.sub(r"`(.+?)`", r"\1", text)
    # Headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Bold + italic (**...**)
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    # Italic (_..._)
    text = re.sub(r"_(.*?)_", r"\1", text)
    # Links [text](url)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Blockquotes
    text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)
    return text.strip()


def _split_text(text: str, max_len: int = MAX_TEXT_LENGTH) -> List[str]:
    """Split text into chunks <= max_len chars, splitting on whitespace boundaries."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    while len(text) > max_len:
        split_pos = text.rfind(" ", 0, max_len)
        if split_pos <= 0:
            # No whitespace found — force split
            split_pos = max_len
        chunks.append(text[:split_pos].rstrip())
        text = text[split_pos:].lstrip()
    if text:
        chunks.append(text)
    return chunks

# ---------------------------------------------------------------------------
# LineAdapter
# ---------------------------------------------------------------------------

class LineAdapter(BasePlatformAdapter):
    """LINE Messaging API adapter.

    Instantiate once per LINE Bot account. Set platform=Platform.LINE for the
    main account and platform=Platform.LINE_LYNX for the hospital Lynx account.
    Each instance holds its own credentials (never shared).
    """

    def __init__(self, config: PlatformConfig, platform: Platform = Platform.LINE):
        super().__init__(config, platform)
        extra = config.extra or {}
        plat_key = platform.value  # "line" or "line_lynx"

        self.channel_access_token: str = (
            extra.get("channel_access_token")
            or os.getenv("LINE_CHANNEL_ACCESS_TOKEN" if plat_key == "line" else "LINE_LYNX_CHANNEL_ACCESS_TOKEN", "")
        )
        self.channel_secret: str = (
            extra.get("channel_secret")
            or os.getenv("LINE_CHANNEL_SECRET" if plat_key == "line" else "LINE_LYNX_CHANNEL_SECRET", "")
        )
        self.webhook_port: int = int(
            extra.get("webhook_port") or DEFAULT_WEBHOOK_PORT.get(plat_key, 18791)
        )
        self.webhook_path: str = (
            extra.get("webhook_path") or DEFAULT_WEBHOOK_PATH.get(plat_key, "/webhook/line")
        )
        if not self.webhook_path.startswith("/"):
            self.webhook_path = f"/{self.webhook_path}"

        self.group_personas: Dict[str, str] = extra.get("group_personas") or {}
        self.default_persona: str = extra.get("default_persona") or ""
        self.allow_from: List[str] = extra.get("allow_from") or []
        self.dm_policy: str = extra.get("dm_policy") or "open"
        self.group_policy: str = extra.get("group_policy") or "open"

        self._http_client: Optional[httpx.AsyncClient] = None
        self._runner = None

    async def connect(self) -> bool:
        if not LINE_SDK_AVAILABLE:
            logger.error("[line] line-bot-sdk not installed. Run: pip install line-bot-sdk")
            return False
        if not self.channel_access_token or not self.channel_secret:
            logger.error("[line] channel_access_token and channel_secret are required")
            return False

        from aiohttp import web

        self._http_client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self.channel_access_token}"},
            timeout=30.0,
        )

        app = web.Application()
        app.router.add_get("/health", lambda _: web.Response(text="ok"))
        app.router.add_post(self.webhook_path, self._handle_webhook)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        from aiohttp.web_runner import TCPSite
        site = TCPSite(self._runner, "127.0.0.1", self.webhook_port)
        await site.start()

        self._mark_connected()
        logger.info(
            "[line/%s] webhook listening on http://127.0.0.1:%s%s",
            self.platform.value, self.webhook_port, self.webhook_path,
        )
        return True

    async def disconnect(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        text = _strip_markdown(content)
        chunks = _split_text(text)
        for chunk in chunks:
            ok = await self._push_chunk(chat_id, chunk)
            if not ok:
                return SendResult(success=False, error="push failed")
        return SendResult(success=True)

    async def _push_chunk(self, chat_id: str, text: str) -> bool:
        if not self._http_client:
            logger.error("[line] not connected — cannot push message")
            return False
        payload = {"to": chat_id, "messages": [{"type": "text", "text": text}]}
        try:
            resp = await self._http_client.post(LINE_PUSH_API_URL, json=payload)
            if resp.status_code != 200:
                logger.error("[line] push failed %s: %s", resp.status_code, resp.text[:200])
                return False
            return True
        except Exception as exc:
            logger.error("[line] push error: %s", exc)
            return False

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "group" if chat_id.startswith("C") else "dm"}

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        pass  # LINE does not support typing indicators via API

    def format_message(self, content: str) -> str:
        return _strip_markdown(content)

    async def _handle_webhook(self, request):
        """Placeholder — full implementation in Task 5."""
        return None
