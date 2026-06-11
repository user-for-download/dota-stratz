-- 009_gold_xp_10_features.sql
-- Add avg_gold_10 / avg_xp_10 columns to ML aggregate tables.
--
-- These features represent the average gold/XP during the first 10 minutes,
-- computed from the gold_t / xp_t JSONB arrays stored in player_minute_stats
-- (minute=0 sentinel row, written by the parser since migration 008).
--
-- The features follow the same pattern as avg_gpm/avg_xpm but capture
-- early-game performance rather than game-long GPM/XPM.

ALTER TABLE ml.team_hero_agg
  ADD COLUMN IF NOT EXISTS avg_gold_10 FLOAT,
  ADD COLUMN IF NOT EXISTS avg_xp_10   FLOAT;

ALTER TABLE ml.player_hero_agg
  ADD COLUMN IF NOT EXISTS avg_gold_10 FLOAT,
  ADD COLUMN IF NOT EXISTS avg_xp_10   FLOAT;

ALTER TABLE ml.hero_baseline_agg
  ADD COLUMN IF NOT EXISTS avg_gold_10 FLOAT,
  ADD COLUMN IF NOT EXISTS avg_xp_10   FLOAT;

COMMENT ON COLUMN ml.team_hero_agg.avg_gold_10    IS 'Average gold at minute 0-9 from gold_t array';
COMMENT ON COLUMN ml.team_hero_agg.avg_xp_10      IS 'Average XP at minute 0-9 from xp_t array';
COMMENT ON COLUMN ml.player_hero_agg.avg_gold_10  IS 'Average gold at minute 0-9 from gold_t array';
COMMENT ON COLUMN ml.player_hero_agg.avg_xp_10    IS 'Average XP at minute 0-9 from xp_t array';
COMMENT ON COLUMN ml.hero_baseline_agg.avg_gold_10 IS 'Average gold at minute 0-9 from gold_t array';
COMMENT ON COLUMN ml.hero_baseline_agg.avg_xp_10   IS 'Average XP at minute 0-9 from xp_t array';
