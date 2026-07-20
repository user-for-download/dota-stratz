-- Migration 008: Performance indexes for training SQL and aggregate populators
-- These complement existing partial indexes with simpler composite patterns.

-- Matches: training query filters by patch + start_time for chronological split
CREATE INDEX IF NOT EXISTS idx_matches_patch_start_time
    ON matches (patch, start_time) WHERE radiant_win IS NOT NULL;

-- Picks_bans: training query joins on match_id + order, filters by is_pick
CREATE INDEX IF NOT EXISTS idx_picks_bans_match_order
    ON picks_bans (match_id, "order");

-- Players: aggregate populators join on match_id + hero_id
CREATE INDEX IF NOT EXISTS idx_players_match_hero_id
    ON players (match_id, hero_id);

-- Players: live prediction joins on account_id + hero_id for player_hero lookup
CREATE INDEX IF NOT EXISTS idx_players_account_hero_match
    ON players (account_id, hero_id, match_id);
