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
  embedding FLOAT[{VEC_DIM}]
);
"""

# Triggers keep vec_facts in sync with semantic_facts. embedding is stored as
# raw float32 BLOB on the relational side; vec0 reads the same bytes natively.
_TRIGGERS_SQL = """
CREATE TRIGGER IF NOT EXISTS sf_after_insert
AFTER INSERT ON semantic_facts
BEGIN
  INSERT INTO vec_facts(fact_id, embedding) VALUES (NEW.id, NEW.embedding);
END;

CREATE TRIGGER IF NOT EXISTS sf_after_update_embedding
AFTER UPDATE OF embedding ON semantic_facts
BEGIN
  UPDATE vec_facts SET embedding = NEW.embedding WHERE fact_id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS sf_after_delete
AFTER DELETE ON semantic_facts
BEGIN
  DELETE FROM vec_facts WHERE fact_id = OLD.id;
END;
"""


def open_db(db_path: Path) -> sqlite3.Connection:
    """Open a sqlite connection with sqlite-vec extension loaded."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
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


def init_db(db_path: Path) -> sqlite3.Connection:
    """Open + bootstrap. Returns a ready-to-use connection."""
    conn = open_db(db_path)
    bootstrap_schema(conn)
    return conn
