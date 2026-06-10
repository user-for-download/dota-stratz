-- 007_enhanced_features.sql
-- Add behavioral feature columns to ML aggregate tables.
-- These capture playstyle tendencies that the model can use to
-- differentiate drafts beyond raw stats:
--   firstblood_rate     — early aggression / laning dominance
--   avg_camps_stacked   — support utility / map efficiency
--   avg_vision_placed   — warding map control (obs_placed + sen_placed)
--
-- avg_gold_10 and avg_xp_10 are deferred: they require parser changes
-- to populate player_minute_stats (see issue #42).

ALTER TABLE ml.team_hero_agg
    ADD COLUMN IF NOT EXISTS firstblood_rate FLOAT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS avg_camps_stacked FLOAT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS avg_vision_placed FLOAT DEFAULT 0;

ALTER TABLE ml.player_hero_agg
    ADD COLUMN IF NOT EXISTS firstblood_rate FLOAT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS avg_camps_stacked FLOAT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS avg_vision_placed FLOAT DEFAULT 0;
