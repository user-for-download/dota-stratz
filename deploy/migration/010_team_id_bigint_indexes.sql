-- 010_team_id_bigint_indexes.sql
-- Fix ML schema: team_id type mismatch and add trainer-focused indexes.
--
-- Issue: ml.team_hero_agg.team_id and ml.team_h2h_agg.(team_id, enemy_team_id)
-- were INT but matches.radiant_team_id / dire_team_id are BIGINT. This can
-- silently truncate team IDs that exceed INT range (>2.1B).
--
-- Also adds composite indexes for the trainer's PIT aggregate queries which
-- now use LATERAL subqueries with start_time filtering instead of pre-computed
-- table joins.

-- ============================================================================
-- 1. Fix team_id type: INT → BIGINT
-- ============================================================================
ALTER TABLE ml.team_hero_agg
  ALTER COLUMN team_id TYPE BIGINT;

ALTER TABLE ml.team_h2h_agg
  ALTER COLUMN team_id TYPE BIGINT,
  ALTER COLUMN enemy_team_id TYPE BIGINT;

-- ============================================================================
-- 2. Trainer-focused composite indexes
--    These support the PIT LATERAL subqueries in features.py that filter by
--    m_hist.start_time < ds.start_time and join on team/hero/account.
-- ============================================================================

-- Training dataset: filter by patch, order by start_time for PIT
CREATE INDEX IF NOT EXISTS idx_matches_patch_result
  ON matches (patch, match_id)
  WHERE radiant_win IS NOT NULL;

-- PIT aggregate: historical matches sorted by time
CREATE INDEX IF NOT EXISTS idx_matches_patch_start
  ON matches (patch, start_time, match_id)
  WHERE radiant_win IS NOT NULL;

-- Team-hero PIT: join on hero + side, filter by time
CREATE INDEX IF NOT EXISTS idx_players_match_hero_side
  ON players (match_id, hero_id, is_radiant);

-- Player-hero PIT: join on account + hero
-- NOTE: The name idx_players_match_account_hero is intentionally distinct
-- from idx_players_account_hero (created in 001_core.sql on account_id,
-- hero_id) because IF NOT EXISTS checks by index name, not column list.
CREATE INDEX IF NOT EXISTS idx_players_match_account_hero
  ON players (match_id, account_id, hero_id);

-- Synergy/counter PIT: join picks_bans to find heroes in draft state
CREATE INDEX IF NOT EXISTS idx_picks_bans_match_order_pick_team
  ON picks_bans (match_id, "order", is_pick, team, hero_id);

-- Draft order queries for within-match draft state
CREATE INDEX IF NOT EXISTS idx_picks_bans_match_team_order_picks
  ON picks_bans (match_id, team, "order", hero_id)
  WHERE is_pick = TRUE;

-- Hero baseline PIT: aggregate by hero across many matches
CREATE INDEX IF NOT EXISTS idx_matches_patch_hero
  ON matches (patch, start_time)
  INCLUDE (match_id)
  WHERE radiant_win IS NOT NULL;
