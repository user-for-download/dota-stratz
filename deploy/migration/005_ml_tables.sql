-- 005_ml_tables.sql
-- ML aggregate tables for LightGBM draft prediction training and real-time inference.
--
-- All six tables are patch-aware (partitioned by patch_id). The trainer populates
-- these during `make train PATCH=<id>`, and the inference API queries them at
-- /predict time to build feature vectors for the current draft state.
--
-- Tables use UNLOGGED for write speed during batch population (the trainer can
-- re-populate at any time). Each table has a GIST exclusion on (patch_id, hero_id)
-- to prevent duplicates during incremental upserts.
--
-- Bayesian shrinkage priors are applied in SQL (not in Python) so that both the
-- trainer (LightGBM dataset.py) and the inference API (features.py) see identical
-- prior-adjusted rates without duplicating the logic.

-- ============================================================================
-- 1. ml.team_hero_agg
--    Per-patch, per-team, per-hero pick/ban/win aggregates.
--    Team-level tendencies: how often does Team X pick/ban hero H in patch P?
-- ============================================================================
CREATE SCHEMA IF NOT EXISTS ml;

CREATE UNLOGGED TABLE IF NOT EXISTS ml.team_hero_agg (
    patch_id   INT  NOT NULL,
    team_id    INT  NOT NULL,
    hero_id    INT  NOT NULL,
    games      INT  NOT NULL DEFAULT 0,
    wins       INT  NOT NULL DEFAULT 0,
    bans       INT  NOT NULL DEFAULT 0,
    win_rate   FLOAT NOT NULL DEFAULT 0.5,  -- shrunk: (wins + prior) / (games + prior)
    avg_gpm    FLOAT,
    avg_xpm    FLOAT,
    avg_kills  FLOAT,
    avg_deaths FLOAT,
    avg_assists FLOAT,
    last_played BIGINT,
    PRIMARY KEY (patch_id, team_id, hero_id)
);

-- ============================================================================
-- 2. ml.player_hero_agg
--    Per-patch, per-account, per-hero stats.
--    Player mastery: how good is player X on hero H in patch P?
-- ============================================================================
CREATE UNLOGGED TABLE IF NOT EXISTS ml.player_hero_agg (
    patch_id   INT     NOT NULL,
    account_id BIGINT  NOT NULL,
    hero_id    INT     NOT NULL,
    games      INT     NOT NULL DEFAULT 0,
    wins       INT     NOT NULL DEFAULT 0,
    win_rate   FLOAT   NOT NULL DEFAULT 0.5,
    avg_gpm    FLOAT,
    avg_xpm    FLOAT,
    avg_kills  FLOAT,
    avg_deaths FLOAT,
    avg_assists FLOAT,
    avg_kda    FLOAT,
    lane_role  INT,
    last_played BIGINT,
    PRIMARY KEY (patch_id, account_id, hero_id)
);

-- ============================================================================
-- 3. ml.hero_synergy_agg
--    Per-patch, per-hero-pair synergy (same team).
--    Pair strength: how well do heroes A and B perform together in patch P?
-- ============================================================================
CREATE UNLOGGED TABLE IF NOT EXISTS ml.hero_synergy_agg (
    patch_id      INT   NOT NULL,
    hero_a        INT   NOT NULL,
    hero_b        INT   NOT NULL,
    games         INT   NOT NULL DEFAULT 0,
    wins          INT   NOT NULL DEFAULT 0,
    win_rate      FLOAT NOT NULL DEFAULT 0.5,
    PRIMARY KEY (patch_id, hero_a, hero_b)
);

-- ============================================================================
-- 4. ml.hero_counter_agg
--    Per-patch, per-hero-vs-enemy-hero matchup.
--    Counter strength: how does hero A perform against hero B in patch P?
-- ============================================================================
CREATE UNLOGGED TABLE IF NOT EXISTS ml.hero_counter_agg (
    patch_id      INT   NOT NULL,
    hero_id       INT   NOT NULL,
    enemy_hero_id INT   NOT NULL,
    games         INT   NOT NULL DEFAULT 0,
    wins          INT   NOT NULL DEFAULT 0,
    win_rate      FLOAT NOT NULL DEFAULT 0.5,
    avg_kd_diff   FLOAT,
    PRIMARY KEY (patch_id, hero_id, enemy_hero_id)
);

-- ============================================================================
-- 5. ml.team_h2h_agg
--    Per-patch, per-team-vs-team head-to-head.
--    Historic matchup: how does Team X fare against Team Y in patch P?
-- ============================================================================
CREATE UNLOGGED TABLE IF NOT EXISTS ml.team_h2h_agg (
    patch_id     INT  NOT NULL,
    team_id      INT  NOT NULL,
    enemy_team_id INT NOT NULL,
    games        INT  NOT NULL DEFAULT 0,
    wins         INT  NOT NULL DEFAULT 0,
    win_rate     FLOAT NOT NULL DEFAULT 0.5,
    PRIMARY KEY (patch_id, team_id, enemy_team_id)
);

-- ============================================================================
-- 6. ml.hero_baseline_agg
--    Per-patch hero global baselines (all teams, all players).
--    Used as fallback when team/player-specific data is sparse.
-- ============================================================================
CREATE UNLOGGED TABLE IF NOT EXISTS ml.hero_baseline_agg (
    patch_id       INT   NOT NULL,
    hero_id        INT   NOT NULL,
    total_picks    INT   NOT NULL DEFAULT 0,
    total_wins     INT   NOT NULL DEFAULT 0,
    total_bans     INT   NOT NULL DEFAULT 0,
    win_rate       FLOAT NOT NULL DEFAULT 0.5,
    pick_rate      FLOAT NOT NULL DEFAULT 0.0,
    ban_rate       FLOAT NOT NULL DEFAULT 0.0,
    avg_gpm        FLOAT,
    avg_xpm        FLOAT,
    avg_kills      FLOAT,
    avg_deaths     FLOAT,
    avg_assists    FLOAT,
    PRIMARY KEY (patch_id, hero_id)
);

-- ============================================================================
-- Helper: ensure _migrations table exists (idempotent)
-- ============================================================================
CREATE TABLE IF NOT EXISTS _migrations (name TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now());

INSERT INTO _migrations (name) VALUES ('005_ml_tables.sql') ON CONFLICT DO NOTHING;
