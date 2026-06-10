-- 008_ml_pit_ledgers.sql
-- Point-in-time correct materialized ledgers for ML feature extraction.
-- These feed the v1_pit_features.sql query used by the ML service.
--
-- Convention: team = 0 (Radiant), team = 1 (Dire), matching OpenDota API.
-- All ledgers filter to pro matches (leagueid > 0, lobby_type IN (1, 2)).
--
-- NOTE: The existing refresh_all_mv() function dynamically discovers all MVs
-- in the analytics schema via pg_class, so these will be auto-included.

-- ============================================================================
-- 1. HERO PAIR LEDGER
-- One row per (match, same-team hero pair) with win flag + start_time.
-- Drives point-in-time SYNERGY features.
-- ============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.hero_pair_ledger AS
SELECT
    m.match_id,
    m.start_time,
    LEAST(pb1.hero_id, pb2.hero_id)    AS hero_a,
    GREATEST(pb1.hero_id, pb2.hero_id) AS hero_b,
    CASE
        WHEN pb1.team = 0 THEN m.radiant_win::int
        ELSE (NOT m.radiant_win)::int
    END AS pair_won
FROM picks_bans pb1
JOIN picks_bans pb2
  ON pb1.match_id = pb2.match_id
 AND pb1.team     = pb2.team
 AND pb1.hero_id  < pb2.hero_id
JOIN matches m ON m.match_id = pb1.match_id
WHERE pb1.is_pick AND pb2.is_pick
  AND m.radiant_win IS NOT NULL
  AND m.leagueid > 0
  AND m.lobby_type IN (1, 2);

CREATE UNIQUE INDEX IF NOT EXISTS idx_hpl_pk
    ON analytics.hero_pair_ledger (match_id, hero_a, hero_b);
CREATE INDEX IF NOT EXISTS idx_hpl_pair_time
    ON analytics.hero_pair_ledger (hero_a, hero_b, start_time, match_id);

-- ============================================================================
-- 2. HERO COUNTER LEDGER
-- One row per (match, radiant_hero, dire_hero) ordered pair.
-- Drives point-in-time COUNTER features.
-- ============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.hero_counter_ledger AS
SELECT
    m.match_id,
    m.start_time,
    pr.hero_id              AS hero_id,
    pd.hero_id              AS enemy_hero_id,
    m.radiant_win::int      AS hero_won
FROM picks_bans pr
JOIN picks_bans pd
  ON pr.match_id = pd.match_id
 AND pr.team = 0
 AND pd.team = 1
JOIN matches m ON m.match_id = pr.match_id
WHERE pr.is_pick AND pd.is_pick
  AND m.radiant_win IS NOT NULL
  AND m.leagueid > 0
  AND m.lobby_type IN (1, 2);

CREATE UNIQUE INDEX IF NOT EXISTS idx_hcl_pk
    ON analytics.hero_counter_ledger (match_id, hero_id, enemy_hero_id);
CREATE INDEX IF NOT EXISTS idx_hcl_pair_time
    ON analytics.hero_counter_ledger (hero_id, enemy_hero_id, start_time, match_id);

-- ============================================================================
-- 3. TEAM HERO LEDGER
-- One row per (match, team_id, hero_id) with win flag + start_time.
-- Drives point-in-time TEAM-HERO familiarity features.
-- ============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.team_hero_ledger AS
SELECT
    m.match_id,
    m.start_time,
    CASE WHEN pb.team = 0 THEN m.radiant_team_id ELSE m.dire_team_id END AS team_id,
    pb.hero_id,
    CASE
        WHEN pb.team = 0 THEN m.radiant_win::int
        ELSE (NOT m.radiant_win)::int
    END AS team_won
FROM picks_bans pb
JOIN matches m ON m.match_id = pb.match_id
WHERE pb.is_pick
  AND m.radiant_win IS NOT NULL
  AND m.leagueid > 0
  AND m.lobby_type IN (1, 2)
  AND CASE WHEN pb.team = 0 THEN m.radiant_team_id ELSE m.dire_team_id END IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_thl_pk
    ON analytics.team_hero_ledger (match_id, team_id, hero_id);
CREATE INDEX IF NOT EXISTS idx_thl_team_hero_time
    ON analytics.team_hero_ledger (team_id, hero_id, start_time, match_id);

-- ============================================================================
-- 4. PREDICTIONS LOG
-- Served prediction log for drift monitoring and shadow promotion.
-- Modified by ml/logging_store.py and ml/backfill.py at runtime.
-- ============================================================================
CREATE TABLE IF NOT EXISTS analytics.predictions_log (
    id                   BIGSERIAL PRIMARY KEY,
    match_id             BIGINT      NOT NULL,
    model_id             TEXT        NOT NULL,
    git_sha              TEXT        NOT NULL,
    radiant_win_prob     DOUBLE PRECISION NOT NULL,
    radiant_win_prob_raw DOUBLE PRECISION,
    served_at            TIMESTAMPTZ NOT NULL DEFAULT now(),

    actual_label         INTEGER,
    resolved_at          TIMESTAMPTZ,

    UNIQUE (match_id, model_id, git_sha)
);

CREATE INDEX IF NOT EXISTS ix_predlog_unresolved
    ON analytics.predictions_log (match_id)
    WHERE actual_label IS NULL;

CREATE INDEX IF NOT EXISTS ix_predlog_model_time
    ON analytics.predictions_log (model_id, git_sha, served_at);

CREATE INDEX IF NOT EXISTS ix_predlog_patch
    ON analytics.predictions_log (match_id)
    INCLUDE (actual_label, radiant_win_prob, served_at);

-- ============================================================================
-- 5. MODEL DEPLOYMENTS
-- Tracks live / shadow deployments per model_id family.
-- Modified by ml/deploy.py at runtime.
-- ============================================================================
CREATE TABLE IF NOT EXISTS analytics.model_deployments (
    model_id            TEXT NOT NULL,
    git_sha             TEXT NOT NULL,
    role                TEXT NOT NULL CHECK (role IN ('live', 'shadow')),
    model_path          TEXT NOT NULL,
    promoted_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    baseline_brier      DOUBLE PRECISION,
    PRIMARY KEY (model_id, role)
);
