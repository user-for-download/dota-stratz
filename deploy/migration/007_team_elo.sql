-- Migration 007: Create ml.team_elo table for team strength ratings
-- Elo ratings are computed chronologically from all match results.
-- Used for post-hoc calibration in predict-match, NOT as neural network input.

CREATE TABLE IF NOT EXISTS ml.team_elo (
    team_id BIGINT PRIMARY KEY,
    elo FLOAT NOT NULL DEFAULT 1500.0
);

CREATE INDEX IF NOT EXISTS idx_team_elo_elo ON ml.team_elo (elo DESC);
