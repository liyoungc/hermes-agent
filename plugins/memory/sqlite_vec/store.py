"""sqlite-vec backed memory store: schema bootstrap + connection helper.

W1 scope: schema only. Read/write paths come in W2/W3.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import sqlite_vec

logger = logging.getLogger(__name__)

VEC_DIM = 512  # voyage-3.5-lite output dimension we store

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

_VEC_VIRTUAL_TABLE_SQL = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS vec_facts USING vec0(
  fact_id INTEGER PRIMARY KEY,
  embedding int8[{VEC_DIM}] distance_metric=cosine
);
"""

# Triggers keep vec_facts in sync with semantic_facts. embedding is stored as
# raw int8 BLOB (512 bytes) on the relational side; vec0 needs vec_int8()
# wrapper to interpret it (without it, vec0 assumes float32).
_TRIGGERS_SQL = """
CREATE TRIGGER IF NOT EXISTS sf_after_insert
AFTER INSERT ON semantic_facts
BEGIN
  INSERT INTO vec_facts(fact_id, embedding) VALUES (NEW.id, vec_int8(NEW.embedding));
END;

CREATE TRIGGER IF NOT EXISTS sf_after_update_embedding
AFTER UPDATE OF embedding ON semantic_facts
BEGIN
  -- vec0 int8 columns reject UPDATE even via vec_int8(); use DELETE+INSERT.
  DELETE FROM vec_facts WHERE fact_id = NEW.id;
  INSERT INTO vec_facts(fact_id, embedding) VALUES (NEW.id, vec_int8(NEW.embedding));
END;

CREATE TRIGGER IF NOT EXISTS sf_after_delete
AFTER DELETE ON semantic_facts
BEGIN
  DELETE FROM vec_facts WHERE fact_id = OLD.id;
END;
"""


def open_db(db_path: Path, *, check_same_thread: bool = True) -> sqlite3.Connection:
    """Open a sqlite connection with sqlite-vec extension loaded.

    Pass ``check_same_thread=False`` when the connection will be reused
    across threads (e.g. the provider's prefetch worker pool). Caller is
    then responsible for serializing access via a lock.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=check_same_thread)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    return conn


def bootstrap_schema(conn: sqlite3.Connection) -> None:
    """Idempotently create tables, indexes, vec0 virtual table, and triggers."""
    conn.executescript(_SCHEMA_PATH.read_text())
    conn.executescript(_VEC_VIRTUAL_TABLE_SQL)
    conn.executescript(_TRIGGERS_SQL)
    conn.commit()


def init_db(db_path: Path, *, check_same_thread: bool = True) -> sqlite3.Connection:
    """Open + bootstrap. Returns a ready-to-use connection."""
    conn = open_db(db_path, check_same_thread=check_same_thread)
    bootstrap_schema(conn)
    return conn
