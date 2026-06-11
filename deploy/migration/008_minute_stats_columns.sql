-- 008_minute_stats_columns.sql
-- Add gold/xp time-series arrays to player_minute_stats for 10-minute
-- feature computation (avg_gold_10 / avg_xp_10).
--
-- The gold_t / xp_t columns store minute-by-minute arrays from OpenDota's
-- gold_t and xp_t player fields as JSONB. These are supplementary to the
-- existing per-minute rows (gold/xp per minute).
--
-- We use minute=0 as a sentinel — each player gets exactly one JSONB row
-- alongside their per-minute detail rows. This avoids changing the PK.

ALTER TABLE player_minute_stats
  ADD COLUMN IF NOT EXISTS gold_t JSONB,
  ADD COLUMN IF NOT EXISTS xp_t JSONB;

COMMENT ON COLUMN player_minute_stats.gold_t IS 'Minute-by-minute gold array from OpenDota gold_t (JSONB)';
COMMENT ON COLUMN player_minute_stats.xp_t IS 'Minute-by-minute XP array from OpenDota xp_t (JSONB)';
