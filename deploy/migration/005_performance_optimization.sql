-- 005_performance_optimization.sql
-- Fix #4: Add generated columns + indexes for GROUP BY (time / 60) optimization.
-- These columns pre-compute the minute value so GROUP BY can use the index
-- instead of computing the expression for every row.

-- ============================================================================
-- 1. PLAYER_KILLS_LOG — add minute column + index
-- ============================================================================
ALTER TABLE player_kills_log
    ADD COLUMN IF NOT EXISTS minute INT GENERATED ALWAYS AS (time / 60) STORED;

CREATE INDEX IF NOT EXISTS idx_player_kills_log_match_minute
    ON player_kills_log (match_id, minute);

-- ============================================================================
-- 2. PLAYER_BUYBACK_LOG — add minute column + index
-- ============================================================================
ALTER TABLE player_buyback_log
    ADD COLUMN IF NOT EXISTS minute INT GENERATED ALWAYS AS (time / 60) STORED;

CREATE INDEX IF NOT EXISTS idx_player_buyback_log_match_minute
    ON player_buyback_log (match_id, minute);

-- ============================================================================
-- 3. PLAYER_OBS_LOG — add minute column + index
-- ============================================================================
ALTER TABLE player_obs_log
    ADD COLUMN IF NOT EXISTS minute INT GENERATED ALWAYS AS (time / 60) STORED;

CREATE INDEX IF NOT EXISTS idx_player_obs_log_match_minute
    ON player_obs_log (match_id, minute);

-- ============================================================================
-- 4. PLAYER_PURCHASE_LOG — add minute column + index
-- ============================================================================
ALTER TABLE player_purchase_log
    ADD COLUMN IF NOT EXISTS minute INT GENERATED ALWAYS AS (time / 60) STORED;

CREATE INDEX IF NOT EXISTS idx_player_purchase_log_match_minute
    ON player_purchase_log (match_id, minute);

-- ============================================================================
-- 5. OBJECTIVES — add minute column + index
-- ============================================================================
ALTER TABLE objectives
    ADD COLUMN IF NOT EXISTS minute INT GENERATED ALWAYS AS (time / 60) STORED;

CREATE INDEX IF NOT EXISTS idx_objectives_match_minute
    ON objectives (match_id, minute);

-- ============================================================================
-- 6. TEAMFIGHTS — add minute column + index
-- ============================================================================
ALTER TABLE teamfights
    ADD COLUMN IF NOT EXISTS minute INT GENERATED ALWAYS AS (start_time / 60) STORED;

CREATE INDEX IF NOT EXISTS idx_teamfights_match_minute
    ON teamfights (match_id, minute);

-- ============================================================================
-- 7. TEAMFIGHT_PLAYERS — add minute column + index
-- ============================================================================
ALTER TABLE teamfight_players
    ADD COLUMN IF NOT EXISTS minute INT GENERATED ALWAYS AS (start_time / 60) STORED;

CREATE INDEX IF NOT EXISTS idx_teamfight_players_match_minute
    ON teamfight_players (match_id, minute);

-- ============================================================================
-- ANALYZE to update statistics
-- ============================================================================
ANALYZE player_kills_log;
ANALYZE player_buyback_log;
ANALYZE player_obs_log;
ANALYZE player_purchase_log;
ANALYZE objectives;
ANALYZE teamfights;
ANALYZE teamfight_players;
