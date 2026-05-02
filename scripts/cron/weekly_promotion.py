#!/usr/bin/env python3
"""Cron entry point: Sun 03:00 UTC+8 weekly memory promotion.

Reads the last 7 days of pending episodes, runs one Kimi-thinking call to
produce a promotion diff, persists the diff as
~/.hermes/memories/pending_diffs/wk-YYYY-MM-DD.json, renders the digest
markdown, and posts it to #memory-review for user review.

Stdout ends with ``{"wakeAgent": false}`` so the cron framework skips
the agent run after we've handled delivery ourselves.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# The hermes container exposes the source tree at /opt/hermes but does not
# add it to sys.path; cron exec'd scripts inherit nothing. Insert it
# manually so plugin imports resolve.
sys.path.insert(0, "/opt/hermes")

# Load the user's .env so VOYAGE_API_KEY / DISCORD_BOT_TOKEN reach the
# plugin code; mirrors what run_agent.py does at module import.
try:
    from hermes_cli.env_loader import load_hermes_dotenv
    load_hermes_dotenv(hermes_home="/opt/data", project_env=None)
except Exception:
    pass

from plugins.memory.sqlite_vec.promotion import (  # noqa: E402
    db_path,
    memory_review_channel_id,
    weekly_promotion,
)
from plugins.memory.sqlite_vec.store import open_db  # noqa: E402


def main() -> int:
    conn = open_db(db_path(), check_same_thread=False)
    channel_id = memory_review_channel_id()
    summary = asyncio.run(
        weekly_promotion(conn, discord_channel_id=channel_id)
    )
    # Print human-readable summary to stdout for cron logs.
    print(json.dumps(summary, ensure_ascii=False, default=str))
    # Wake-gate: skip the agent run.
    print('{"wakeAgent": false}')
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
