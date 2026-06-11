#!/bin/bash
# ==============================================================================
# Backfill gold_t / xp_t for matches processed before migration 008.
#
# The parser (since build 008) stores golden/xp arrays as JSONB in
# player_minute_stats.gold_t / xp_t with minute=0 as a sentinel row.
# For older matches that were parsed before migration 008, those arrays
# are NULL. This script aggregates the existing per-minute rows (minute > 0)
# into arrays and inserts them as minute=0 sentinel rows.
#
# Usage:
#   PG_DSN="postgres://user:pass@host:5432/dota2" bash backfill-minute-stats.sh
#
# Safe to run multiple times — uses ON CONFLICT DO UPDATE.
# ==============================================================================

set -euo pipefail

PG_DSN="${PG_DSN:-postgres://dota2:dota2@localhost:5432/dota2?sslmode=disable}"

echo "Backfilling gold_t / xp_t from per-minute rows ..."

psql "${PG_DSN}" <<'SQL'
WITH aggregated AS (
    SELECT
        match_id,
        player_slot,
        jsonb_agg(gold ORDER BY minute) FILTER (WHERE gold IS NOT NULL) AS gold_t,
        jsonb_agg(xp ORDER BY minute)   FILTER (WHERE xp IS NOT NULL)   AS xp_t
    FROM player_minute_stats
    WHERE minute > 0
    GROUP BY match_id, player_slot
)
INSERT INTO player_minute_stats (match_id, player_slot, minute, gold_t, xp_t)
SELECT match_id, player_slot, 0, gold_t, xp_t
FROM aggregated
WHERE gold_t IS NOT NULL OR xp_t IS NOT NULL
ON CONFLICT (match_id, player_slot, minute) DO UPDATE SET
    gold_t = EXCLUDED.gold_t,
    xp_t   = EXCLUDED.xp_t;
SQL

echo "Done. Affected rows: $(psql "${PG_DSN}" -t -A -c "SELECT COUNT(*) FROM player_minute_stats WHERE minute = 0 AND gold_t IS NOT NULL;")"
