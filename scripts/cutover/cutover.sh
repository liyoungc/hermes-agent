#!/usr/bin/env bash
# W4-1 cutover script — run on chococlaw when you've decided to retire
# MEMORY.md flat-file injection and commit fully to the sqlite_vec
# memory plugin.
#
# Spec target date: 2026-05-24, *after* observing at least one successful
# weekly review cycle on the new system.
#
# Idempotent — safe to re-run if interrupted partway.
#
# Usage:
#   ./scripts/cutover/cutover.sh             # dry run, prints planned actions
#   ./scripts/cutover/cutover.sh --commit    # actually do the work

set -euo pipefail

DRY_RUN=true
if [[ "${1:-}" == "--commit" ]]; then
  DRY_RUN=false
fi

today() { date -u +%Y-%m-%d; }
say() { echo "[cutover] $*"; }
do_or_say() {
  if $DRY_RUN; then
    say "(dry-run) $*"
  else
    say "$*"
    eval "$@"
  fi
}

HOME_DIR="${HERMES_HOME:-$HOME/.hermes}"
say "HERMES_HOME = ${HOME_DIR}"

# ---- 1. Pre-flight checks --------------------------------------------------

say "1. Pre-flight checks"
[[ -d "${HOME_DIR}/memories" ]] || { say "ERR no ${HOME_DIR}/memories"; exit 1; }
[[ -f "${HOME_DIR}/memories/memory.db" ]] || { say "ERR no memory.db — W1 hasn't shipped"; exit 1; }
say "  ✓ memory.db present"

if ! command -v docker >/dev/null; then
  say "WARN docker not on PATH — DB queries below will be skipped"
fi

# Confirm the new system has been writing recently (last 7 days).
if command -v docker >/dev/null; then
  ep_recent=$(docker exec hermes /opt/hermes/.venv/bin/python3 -c "
import sqlite3
conn = sqlite3.connect('/opt/data/memories/memory.db')
n = conn.execute(\"SELECT count(*) FROM episodes WHERE ts > datetime('now','-7 days')\").fetchone()[0]
print(n)
" 2>/dev/null || echo 0)
  if [[ "${ep_recent}" -lt 5 ]]; then
    say "WARN only ${ep_recent} episodes in the last 7 days. Either the gateway"
    say "     hasn't been used much OR the write path isn't actually firing."
    say "     Fix that BEFORE cutover, or the new system has nothing to retrieve."
  else
    say "  ✓ ${ep_recent} episodes recorded in the last 7 days"
  fi
fi

# ---- 2. Archive MEMORY.md --------------------------------------------------

ARCHIVE_NAME="MEMORY.md.archive-$(today)"
SRC="${HOME_DIR}/memories/MEMORY.md"
DST="${HOME_DIR}/memories/${ARCHIVE_NAME}"

say "2. Archive MEMORY.md → ${ARCHIVE_NAME}"
if [[ ! -f "${SRC}" ]]; then
  say "  - ${SRC} does not exist — already archived?"
else
  if [[ -f "${DST}" ]]; then
    say "  - ${DST} already exists — refusing to overwrite"
  else
    do_or_say "mv '${SRC}' '${DST}'"
    do_or_say "chmod 444 '${DST}'"
  fi
fi

# ---- 3. config.yaml: confirm provider=sqlite_vec ---------------------------

say "3. Confirm config.yaml memory.provider == sqlite_vec"
cfg="${HOME_DIR}/config.yaml"
if grep -qE '^[[:space:]]*provider:[[:space:]]*sqlite_vec' "${cfg}" 2>/dev/null; then
  say "  ✓ already set to sqlite_vec"
else
  say "  - provider not set — please edit ${cfg} manually:"
  say "      memory:"
  say "        provider: sqlite_vec"
fi

# ---- 4. Disable legacy memory crons ----------------------------------------

say "4. Disable legacy memory crons in jobs.json"
do_or_say "/usr/bin/env python3 - <<'PY'
import json, pathlib
p = pathlib.Path('${HOME_DIR}/cron/jobs.json')
if not p.exists():
    print('  - no jobs.json'); raise SystemExit(0)
data = json.loads(p.read_text())
legacy_names = {
    'Dimensions Memory Consolidation',
    'Forgetting Curve (Monthly Archive)',
    'Forgetting Curve',
}
changed = 0
for j in data.get('jobs', []):
    if j['name'] in legacy_names and j.get('enabled', False):
        j['enabled'] = False
        j['paused_at'] = '$(date -u +%Y-%m-%dT%H:%M:%SZ)'
        j['paused_reason'] = 'W4 cutover — replaced by sqlite_vec weekly_promotion'
        print(f'  ✓ disabled: {j[\"name\"]}')
        changed += 1
if changed:
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False))
else:
    print('  - no legacy jobs found (already disabled, or never installed)')
PY"

# ---- 5. Smoke test ---------------------------------------------------------

say "5. Smoke test: provider initializes + retrieves"
if command -v docker >/dev/null; then
  do_or_say "docker exec hermes /opt/hermes/.venv/bin/python3 -c '
from hermes_cli.env_loader import load_hermes_dotenv
load_hermes_dotenv(hermes_home=\"/opt/data\", project_env=None)
from agent.memory_manager import MemoryManager
from plugins.memory import load_memory_provider
mm = MemoryManager()
mm.add_provider(load_memory_provider(\"sqlite_vec\"))
mm.initialize_all(session_id=\"cutover-smoke\", platform=\"cli\", hermes_home=\"/opt/data\", agent_context=\"primary\")
out = mm.prefetch_all(\"我太太生日\")
print(\"prefetch returned:\", \"OK\" if out else \"EMPTY\")
mm.shutdown_all()
'"
fi

# ---- 6. Restart gateway ----------------------------------------------------

say "6. Restart gateway to pick up any config changes"
if command -v docker >/dev/null && [[ -d "${HOME}/Projects/hermes-agent" ]]; then
  do_or_say "(cd ${HOME}/Projects/hermes-agent && docker compose restart gateway)"
fi

# ---- Done ------------------------------------------------------------------

if $DRY_RUN; then
  say ""
  say "DRY RUN COMPLETE — no changes made. Re-run with --commit when ready."
  say ""
  say "After --commit, monitor for 24 hours via memory.log + #memory-review:"
  say "  - tail -f ~/.hermes/logs/memory.log"
  say "  - watch ~/.hermes/logs/memory_write_failures.jsonl size"
  say "  - confirm next Sunday's digest fires"
  say ""
  say "Rollback procedure: docs/runbooks/memory-rollback.md §3"
else
  say ""
  say "CUTOVER COMPLETE."
  say "  Archive at: ${DST}"
  say "  Legacy crons disabled in: ${HOME_DIR}/cron/jobs.json"
  say "  Gateway restarted."
  say ""
  say "Monitor for 24 hours then sanity-check via:"
  say "  docs/runbooks/memory-monitoring.md §6 (quick health check)"
fi
