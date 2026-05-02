#!/usr/bin/env python3
"""Cron entry point: Mon 03:00 UTC+8 weekly memory apply.

Loads the latest pending diff (purges any older than 14 days first),
checks for a rejection sentinel file (written by /memreview reject),
and either archives the diff as rejected or applies its
promote / dedup / expire actions atomically and stamps
``episodes.promoted_at`` on the candidate rows.
"""

from __future__ import annotations

import asyncio
import json
import sys

sys.path.insert(0, "/opt/hermes")

try:
    from hermes_cli.env_loader import load_hermes_dotenv
    load_hermes_dotenv(hermes_home="/opt/data", project_env=None)
except Exception:
    pass

from plugins.memory.sqlite_vec.promotion import (  # noqa: E402
    db_path,
    weekly_apply,
)
from plugins.memory.sqlite_vec.store import open_db  # noqa: E402


def main() -> int:
    conn = open_db(db_path(), check_same_thread=False)
    summary = asyncio.run(weekly_apply(conn))
    print(json.dumps(summary, ensure_ascii=False, default=str))
    print('{"wakeAgent": false}')
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
