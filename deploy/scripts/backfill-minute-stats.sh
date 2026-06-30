#!/bin/bash
# ==============================================================================
# Backfill gold_t / xp_t for matches processed before migration 013.
#
# Migration 013 moved gold_t / xp_t out of player_minute_stats (minute=0
# sentinel) into the dedicated player_time_series_arrays table to avoid PK
# collision with real minute-0 stat rows.
#
# FIX: This script was originally written for migration 008 and still
# targeted player_minute_stats.  After migration 013 drops those columns and
# moves the data, the script must instead aggregate per-minute rows from
# player_minute_stats and insert them into player_time_series_arrays.
#
# Usage:
#   export PGPASSWORD="pass" && bash backfill-minute-stats.sh
#
# Safe to run multiple times — uses ON CONFLICT DO UPDATE.
# ==============================================================================

set -euo pipefail

export PGPASSWORD="${PGPASSWORD:-dota2}"
PGHOST="${PGHOST:-localhost}"
PGUSER="${PGUSER:-dota2}"
PGDATABASE="${PGDATABASE:-dota2}"

echo "Backfilling gold_t / xp_t from per-minute rows into player_time_series_arrays ..."

psql -h "$PGHOST" -U "$PGUSER" -d "$PGDATABASE" <<'SQL'
WITH aggregated AS (
    SELECT
        match_id,
        player_slot,
        jsonb_agg(gold ORDER BY minute) FILTER (WHERE gold IS NOT NULL) AS gold_t,
        jsonb_agg(xp   ORDER BY minute) FILTER (WHERE xp   IS NOT NULL) AS xp_t
    FROM player_minute_stats
    WHERE minute > 0
    GROUP BY match_id, player_slot
)
INSERT INTO player_time_series_arrays (match_id, player_slot, gold_t, xp_t)
SELECT match_id, player_slot, gold_t, xp_t
FROM aggregated
WHERE gold_t IS NOT NULL OR xp_t IS NOT NULL
ON CONFLICT (match_id, player_slot) DO UPDATE SET
    gold_t = EXCLUDED.gold_t,
    xp_t   = EXCLUDED.xp_t;
SQL

echo "Done. Affected rows: $(psql -h "$PGHOST" -U "$PGUSER" -d "$PGDATABASE" -t -A -c "SELECT COUNT(*) FROM player_time_series_arrays WHERE gold_t IS NOT NULL;")"
