# Hermes V3 cron scripts

These scripts are invoked by hermes-agent's cron scheduler. The scheduler
hardcodes `HERMES_HOME/scripts/` as the only path it will exec from
(security: prevents arbitrary script execution via path traversal), so
runtime copies must live at `~/.hermes/scripts/<name>.py` on each host.

The canonical source lives here in version control. Deploy via:

    cp scripts/cron/weekly_promotion.py ~/.hermes/scripts/
    cp scripts/cron/weekly_apply.py ~/.hermes/scripts/

Cron entries are added by adding rows to `~/.hermes/cron/jobs.json`
(see the `Hermes Weekly Memory Promotion` / `Hermes Weekly Memory Apply`
entries; expressions are in UTC — `0 19 * * 6` = Sun 03:00 UTC+8).

Both scripts emit `{"wakeAgent": false}` as the last stdout line so the
cron framework skips the agent run — delivery happens inside the script
via Discord HTTP POST.
