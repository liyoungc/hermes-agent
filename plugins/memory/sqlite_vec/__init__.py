"""Hermes V3 memory plugin — sqlite-vec store with two-tier (hot/cold) design.

W1 scope (this commit): schema bootstrap + provider stub registering with
the MemoryProvider ABC. Read path / write path / weekly promotion arrive in
W2 and W3 per spec docs/superpowers/specs/2026-05-02-hermes-memory-design.md.

Activate via $HERMES_HOME/config.yaml:

  memory:
    provider: sqlite_vec
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

from .store import init_db

logger = logging.getLogger(__name__)


def _default_db_path(hermes_home: str) -> Path:
    return Path(hermes_home).expanduser() / "memories" / "memory.db"


class SqliteVecMemoryProvider(MemoryProvider):
    """Hermes V3 long-term memory provider (W1 = schema only)."""

    @property
    def name(self) -> str:
        return "sqlite_vec"

    def is_available(self) -> bool:
        try:
            import sqlite_vec  # noqa: F401
        except ImportError:
            return False
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home")
        if not hermes_home:
            from hermes_constants import get_hermes_home
            hermes_home = str(get_hermes_home())
        self._db_path = _default_db_path(hermes_home)
        self._conn = init_db(self._db_path)
        logger.info("sqlite_vec memory ready at %s", self._db_path)

    def system_prompt_block(self) -> str:
        # W1: no system-prompt contribution. Persona stays in flat files
        # (SOUL.md, USER.md, life-dimensions.md) handled by built-in memory.
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        # W2 will implement actual retrieval. Empty return for now keeps
        # the plugin a no-op until we wire read_memory().
        return ""

    def sync_turn(self, user: str, assistant: str, **kwargs) -> None:
        # W3 will implement async write-back. No-op for W1.
        return None

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        # No model-facing tools; memory is implicit.
        return []

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Any:
        from tools.registry import tool_error
        return tool_error(f"sqlite_vec exposes no tools (got {tool_name!r})")

    def shutdown(self) -> None:
        if getattr(self, "_conn", None):
            self._conn.close()
            self._conn = None
