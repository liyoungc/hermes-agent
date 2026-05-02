-- Hermes V3 memory schema — episodes (hot raw) + semantic_facts (cold curated)
-- Spec: docs/superpowers/specs/2026-05-02-hermes-memory-design.md §3

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

-- Hot tier: raw turn-by-turn record. All Discord turns + cron synthetic land here.
CREATE TABLE IF NOT EXISTS episodes (
  id            INTEGER PRIMARY KEY,
  ts            TEXT NOT NULL,
  channel       TEXT NOT NULL,
  external_id   TEXT NOT NULL,
  role          TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  text          TEXT NOT NULL,
  synthetic     INTEGER NOT NULL DEFAULT 0,
  embedding     BLOB,
  metadata      TEXT,
  promoted_at   TEXT,
  UNIQUE(channel, external_id)
);
CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(ts);
CREATE INDEX IF NOT EXISTS idx_episodes_promoted_pending
  ON episodes(promoted_at, ts) WHERE promoted_at IS NULL;

-- Cold tier: curated facts. Cattia's actual working memory queries this.
CREATE TABLE IF NOT EXISTS semantic_facts (
  id                  INTEGER PRIMARY KEY,
  entity              TEXT,
  fact                TEXT NOT NULL,
  embedding           BLOB NOT NULL,
  source_episode_ids  TEXT,
  importance          INTEGER DEFAULT 2,
  hits                INTEGER DEFAULT 0,
  created_at          TEXT NOT NULL DEFAULT (datetime('now')),
  last_seen           TEXT,
  state               TEXT DEFAULT 'active' CHECK (state IN ('active', 'archived')),
  valid_from          TEXT NOT NULL DEFAULT (date('now')),
  valid_to            TEXT
);
CREATE INDEX IF NOT EXISTS idx_facts_entity ON semantic_facts(entity);
CREATE INDEX IF NOT EXISTS idx_facts_active ON semantic_facts(state, valid_to);
