-- 013_separate_time_series_arrays.sql
-- Move gold_t / xp_t JSONB arrays out of player_minute_stats (minute=0
-- sentinel) into a dedicated table to eliminate PK conflict with real
-- minute-zero stat rows.
--
-- Background
-- ===========
-- Migration 008 added gold_t / xp_t columns to player_minute_stats and
-- stored the JSONB arrays at minute=0 as a "sentinel" because the PK
-- is (match_id, player_slot, minute). If the OpenDota API ever returns
-- actual minute=0 stat data (gold, xp, last_hits, denies), the sentinel
-- collides with the real row and the ON CONFLICT DO UPDATE silently
-- clobbers real data (issue #5).
--
-- Fix: create player_time_series_arrays with a 2-column PK
-- (match_id, player_slot) and move gold_t / xp_t there.

-- ============================================================================
-- 1. Create the dedicated time-series arrays table
-- ============================================================================
CREATE TABLE IF NOT EXISTS player_time_series_arrays (
    match_id    BIGINT NOT NULL,
    player_slot INT    NOT NULL,
    gold_t      JSONB,
    xp_t        JSONB,
    PRIMARY KEY (match_id, player_slot)
);

COMMENT ON TABLE  player_time_series_arrays IS 'Minute-by-minute gold/XP arrays (separated from player_minute_stats to avoid PK conflict at minute=0)';
COMMENT ON COLUMN player_time_series_arrays.gold_t IS 'Minute-by-minute gold array from OpenDota gold_t (JSONB)';
COMMENT ON COLUMN player_time_series_arrays.xp_t  IS 'Minute-by-minute XP array from OpenDota xp_t (JSONB)';

-- ============================================================================
-- 2. Migrate existing data from the sentinel rows
-- ============================================================================
INSERT INTO player_time_series_arrays (match_id, player_slot, gold_t, xp_t)
SELECT match_id, player_slot, gold_t, xp_t
FROM player_minute_stats
WHERE minute = 0
  AND (gold_t IS NOT NULL OR xp_t IS NOT NULL)
ON CONFLICT (match_id, player_slot) DO NOTHING;

-- ============================================================================
-- 3. Clean up — drop sentinel columns and orphaned rows
-- ============================================================================
ALTER TABLE player_minute_stats DROP COLUMN IF EXISTS gold_t;
ALTER TABLE player_minute_stats DROP COLUMN IF EXISTS xp_t;

-- Sentinel-only rows (minute=0 with only gold_t/xp_t and NULL for every
-- other column) are now useless. The parser never writes real per-minute
-- data at minute=0, so it is safe to remove them.
DELETE FROM player_minute_stats WHERE minute = 0
  AND gold IS NULL
  AND xp IS NULL
  AND last_hits IS NULL
  AND denies IS NULL;
