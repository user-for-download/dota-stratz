-- 002_ml.sql
-- Analytics schema, ML tables, and Point-in-Time (PIT) safe snapshots.
-- Merges and optimizes: 003_analytics, 005_ml_tables, 006_postgres_best_practices_fixes (roles),
-- 007_enhanced_features, 009_gold_xp_10_features, 010_team_id_bigint_indexes (ml part),
-- 011_hero_draft_slot_agg, 012_fix_ml_indexes (omits redundant), 014_pit_safe_snapshots, 015_snapshot_float_games.

CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS ml;

-- ============================================================================
-- 0. ROLES & PRIVILEGES
-- ============================================================================
DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'analytics_reader') THEN CREATE ROLE analytics_reader; END IF; END $$;
REVOKE ALL ON SCHEMA public FROM analytics_reader;
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM analytics_reader;
GRANT USAGE ON SCHEMA analytics, ml TO analytics_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO analytics_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA ml TO analytics_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA ml GRANT SELECT ON TABLES TO analytics_reader;

DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'analytics_writer') THEN CREATE ROLE analytics_writer; END IF; END $$;
GRANT USAGE ON SCHEMA public, analytics, ml TO analytics_writer;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO analytics_writer;
REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics FROM analytics_writer;
REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA analytics FROM analytics_writer;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA analytics TO analytics_writer;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA analytics TO analytics_writer;
ALTER DEFAULT PRIVILEGES IN SCHEMA analytics GRANT SELECT, INSERT, UPDATE ON TABLES TO analytics_writer;
ALTER DEFAULT PRIVILEGES IN SCHEMA analytics GRANT USAGE, SELECT ON SEQUENCES TO analytics_writer;

-- ML schema grants for analytics_writer (trainer/API write to ml.* tables)
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA ml TO analytics_writer;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA ml TO analytics_writer;
ALTER DEFAULT PRIVILEGES IN SCHEMA ml GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO analytics_writer;
ALTER DEFAULT PRIVILEGES IN SCHEMA ml GRANT USAGE, SELECT ON SEQUENCES TO analytics_writer;

-- ============================================================================
-- 1. ANALYTICS CONFIG & MATERIALIZED VIEWS
-- ============================================================================
CREATE TABLE IF NOT EXISTS analytics.shrinkage_config (
    metric_name VARCHAR PRIMARY KEY,
    prior_games FLOAT,
    prior_win_rate FLOAT DEFAULT 0.5,
    description TEXT
);

INSERT INTO analytics.shrinkage_config (metric_name, prior_games, prior_win_rate, description) VALUES
    ('team_hero_wr', 3.0, 0.5, 'Team-level hero picks'),
    ('player_hero_wr', 5.0, 0.5, 'Individual player mastery'),
    ('synergy_wr', 3.0, 0.5, 'Hero pair synergy'),
    ('counter_wr', 3.0, 0.5, 'Hero-vs-hero matchups')
ON CONFLICT (metric_name) DO NOTHING;

CREATE TABLE IF NOT EXISTS analytics.mv_refresh_log (
    refresh_id SERIAL PRIMARY KEY,
    view_name VARCHAR NOT NULL,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    rows_affected BIGINT,
    duration INTERVAL,
    error_message TEXT
);
CREATE INDEX IF NOT EXISTS idx_mv_refresh_log_view ON analytics.mv_refresh_log(view_name, started_at DESC);

CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.mv_team_hero_profile AS
WITH config AS (SELECT prior_games, prior_win_rate FROM analytics.shrinkage_config WHERE metric_name = 'team_hero_wr'),
team_hero_picks AS (
    SELECT
        CASE WHEN p.is_radiant THEN m.radiant_team_id ELSE m.dire_team_id END AS team_id,
        p.hero_id, COUNT(*) AS games_played, SUM(CASE WHEN p.win = 1 THEN 1 ELSE 0 END) AS wins,
        AVG(p.kills) AS avg_kills, AVG(p.deaths) AS avg_deaths, AVG(p.assists) AS avg_assists,
        AVG(p.gold_per_min) AS avg_gpm, AVG(p.xp_per_min) AS avg_xpm, AVG(p.hero_damage) AS avg_hero_damage,
        AVG(p.tower_damage) AS avg_tower_damage, MAX(m.start_time) AS last_played_time
    FROM players p INNER JOIN matches m ON p.match_id = m.match_id
    WHERE m.leagueid > 0 AND m.lobby_type IN (1, 2)
      AND CASE WHEN p.is_radiant THEN m.radiant_team_id ELSE m.dire_team_id END IS NOT NULL
    GROUP BY team_id, p.hero_id
),
team_hero_bans AS (
    SELECT
        CASE WHEN pb.team = 0 THEN m.radiant_team_id ELSE m.dire_team_id END AS team_id,
        pb.hero_id, COUNT(*) AS times_banned
    FROM picks_bans pb INNER JOIN matches m ON pb.match_id = m.match_id
    WHERE m.leagueid > 0 AND m.lobby_type IN (1, 2) AND pb.is_pick = FALSE
    GROUP BY CASE WHEN pb.team = 0 THEN m.radiant_team_id ELSE m.dire_team_id END, pb.hero_id
)
SELECT
    COALESCE(p.team_id, b.team_id) AS team_id, COALESCE(p.hero_id, b.hero_id) AS hero_id,
    COALESCE(p.games_played, 0) AS games_played, COALESCE(p.wins, 0) AS wins, COALESCE(b.times_banned, 0) AS times_banned,
    CASE WHEN COALESCE(p.games_played, 0) > 0
        THEN (COALESCE(p.wins, 0) + (c.prior_games * c.prior_win_rate)) / (p.games_played + c.prior_games)
        ELSE c.prior_win_rate END AS shrunk_win_rate,
    COALESCE(p.avg_kills, 0) AS avg_kills, COALESCE(p.avg_deaths, 0) AS avg_deaths, COALESCE(p.avg_assists, 0) AS avg_assists,
    COALESCE(p.avg_gpm, 0) AS avg_gpm, COALESCE(p.avg_xpm, 0) AS avg_xpm, COALESCE(p.avg_hero_damage, 0) AS avg_hero_damage,
    COALESCE(p.avg_tower_damage, 0) AS avg_tower_damage, p.last_played_time
FROM team_hero_picks p FULL OUTER JOIN team_hero_bans b ON p.team_id = b.team_id AND p.hero_id = b.hero_id CROSS JOIN config c;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_team_hero_profile_pk ON analytics.mv_team_hero_profile(team_id, hero_id);
CREATE INDEX IF NOT EXISTS idx_mv_team_hero_profile_team ON analytics.mv_team_hero_profile(team_id);
CREATE INDEX IF NOT EXISTS idx_mv_team_hero_profile_hero ON analytics.mv_team_hero_profile(hero_id);

CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.mv_hero_synergy AS
WITH config AS (SELECT prior_games, prior_win_rate FROM analytics.shrinkage_config WHERE metric_name = 'synergy_wr'),
hero_pairs AS (
    SELECT p1.hero_id AS hero_a, p2.hero_id AS hero_b, p1.match_id, CASE WHEN p1.is_radiant THEN m.radiant_win ELSE NOT m.radiant_win END AS won
    FROM players p1 INNER JOIN players p2 ON p1.match_id = p2.match_id AND p1.is_radiant = p2.is_radiant AND p1.hero_id < p2.hero_id
    INNER JOIN matches m ON p1.match_id = m.match_id
    WHERE m.leagueid > 0 AND m.lobby_type IN (1, 2)
)
SELECT hero_a, hero_b, COUNT(*) AS games_together, SUM(CASE WHEN won THEN 1 ELSE 0 END) AS wins_together,
       (SUM(CASE WHEN won THEN 1 ELSE 0 END) + (c.prior_games * c.prior_win_rate)) / (COUNT(*) + c.prior_games) AS shrunk_win_rate,
       SUM(CASE WHEN won THEN 1 ELSE 0 END)::FLOAT / NULLIF(COUNT(*), 0) AS raw_win_rate
FROM hero_pairs CROSS JOIN config c GROUP BY hero_a, hero_b, c.prior_games, c.prior_win_rate HAVING COUNT(*) >= 3;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_hero_synergy_pk ON analytics.mv_hero_synergy(hero_a, hero_b);
CREATE INDEX IF NOT EXISTS idx_mv_hero_synergy_hero_a ON analytics.mv_hero_synergy(hero_a);
CREATE INDEX IF NOT EXISTS idx_mv_hero_synergy_hero_b ON analytics.mv_hero_synergy(hero_b);

CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.mv_hero_counter AS
WITH config AS (SELECT prior_games, prior_win_rate FROM analytics.shrinkage_config WHERE metric_name = 'counter_wr'),
hero_matchups AS (
    SELECT p1.hero_id AS hero_id, p2.hero_id AS enemy_hero_id, p1.match_id,
           CASE WHEN p1.is_radiant THEN m.radiant_win ELSE NOT m.radiant_win END AS won, p1.kills - p1.deaths AS kd_diff
    FROM players p1 INNER JOIN players p2 ON p1.match_id = p2.match_id AND p1.is_radiant != p2.is_radiant
    INNER JOIN matches m ON p1.match_id = m.match_id WHERE m.leagueid > 0 AND m.lobby_type IN (1, 2)
)
SELECT hero_id, enemy_hero_id, COUNT(*) AS games_against, SUM(CASE WHEN won THEN 1 ELSE 0 END) AS wins_against,
       (SUM(CASE WHEN won THEN 1 ELSE 0 END) + (c.prior_games * c.prior_win_rate)) / (COUNT(*) + c.prior_games) AS shrunk_win_rate,
       AVG(kd_diff) AS avg_kd_diff, SUM(CASE WHEN won THEN 1 ELSE 0 END)::FLOAT / NULLIF(COUNT(*), 0) AS raw_win_rate
FROM hero_matchups CROSS JOIN config c GROUP BY hero_id, enemy_hero_id, c.prior_games, c.prior_win_rate HAVING COUNT(*) >= 3;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_hero_counter_pk ON analytics.mv_hero_counter(hero_id, enemy_hero_id);
CREATE INDEX IF NOT EXISTS idx_mv_hero_counter_hero ON analytics.mv_hero_counter(hero_id);
CREATE INDEX IF NOT EXISTS idx_mv_hero_counter_enemy ON analytics.mv_hero_counter(enemy_hero_id);

CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.mv_player_team_history AS
SELECT p.account_id, CASE WHEN p.is_radiant THEN m.radiant_team_id ELSE m.dire_team_id END AS team_id,
       COUNT(*) AS games_played, SUM(CASE WHEN p.win = 1 THEN 1 ELSE 0 END) AS wins,
       AVG(p.kills) AS avg_kills, AVG(p.deaths) AS avg_deaths, AVG(p.assists) AS avg_assists,
       AVG(p.gold_per_min) AS avg_gpm, AVG(p.xp_per_min) AS avg_xpm, AVG(p.hero_damage) AS avg_hero_damage,
       AVG(p.tower_damage) AS avg_tower_damage, AVG(p.hero_healing) AS avg_hero_healing, AVG(p.last_hits) AS avg_last_hits, AVG(p.denies) AS avg_denies,
       MIN(m.start_time) AS first_game_time, MAX(m.start_time) AS last_game_time,
       COUNT(CASE WHEN p.lane_role IN (1, 2) THEN 1 END) AS games_core, COUNT(CASE WHEN p.lane_role IN (3, 4, 5) THEN 1 END) AS games_support
FROM players p INNER JOIN matches m ON p.match_id = m.match_id
WHERE m.leagueid > 0 AND m.lobby_type IN (1, 2) AND p.account_id IS NOT NULL AND CASE WHEN p.is_radiant THEN m.radiant_team_id ELSE m.dire_team_id END IS NOT NULL
GROUP BY p.account_id, team_id;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_player_team_history_pk ON analytics.mv_player_team_history(account_id, team_id);
CREATE INDEX IF NOT EXISTS idx_mv_player_team_history_account ON analytics.mv_player_team_history(account_id);
CREATE INDEX IF NOT EXISTS idx_mv_player_team_history_team ON analytics.mv_player_team_history(team_id);

-- ============================================================================
-- 2. SNAPSHOT & TRACKING TABLES
-- ============================================================================
CREATE TABLE IF NOT EXISTS analytics.feature_snapshots_player_hero (
    snapshot_date DATE NOT NULL, account_id BIGINT NOT NULL, hero_id INT NOT NULL, games_played INT, wins INT, shrunk_win_rate FLOAT,
    avg_kills FLOAT, avg_deaths FLOAT, avg_assists FLOAT, avg_kda FLOAT, avg_gpm FLOAT, avg_xpm FLOAT, avg_hero_damage FLOAT,
    avg_tower_damage FLOAT, avg_hero_healing FLOAT, avg_last_hits FLOAT, primary_lane_role INT, role_flexibility INT,
    days_since_last_played INT, games_last_30d INT, games_last_90d INT, PRIMARY KEY (snapshot_date, account_id, hero_id)
);
CREATE INDEX IF NOT EXISTS idx_feature_snapshots_date ON analytics.feature_snapshots_player_hero(snapshot_date);

CREATE TABLE IF NOT EXISTS analytics.featurizer_runs (
    id INT PRIMARY KEY DEFAULT 1, last_snapshot_date DATE, last_run_timestamp TIMESTAMPTZ, matches_processed BIGINT,
    last_processed_match_id BIGINT, CONSTRAINT single_row_check CHECK (id = 1)
);
INSERT INTO analytics.featurizer_runs (id, last_snapshot_date, last_run_timestamp, matches_processed) VALUES (1, NULL, NULL, 0) ON CONFLICT (id) DO NOTHING;

-- ============================================================================
-- 3. ML AGGREGATE TABLES (with BIGINT fixes and columns merged)
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml.team_hero_agg (
    patch_id   INT  NOT NULL, team_id    BIGINT  NOT NULL, hero_id    INT  NOT NULL,
    games      INT  NOT NULL DEFAULT 0, wins       INT  NOT NULL DEFAULT 0, bans       INT  NOT NULL DEFAULT 0,
    win_rate   FLOAT NOT NULL DEFAULT 0.5, avg_gpm    FLOAT, avg_xpm    FLOAT, avg_kills  FLOAT, avg_deaths FLOAT, avg_assists FLOAT,
    firstblood_rate FLOAT DEFAULT 0, avg_camps_stacked FLOAT DEFAULT 0, avg_vision_placed FLOAT DEFAULT 0,
    avg_gold_10 FLOAT, avg_xp_10   FLOAT, last_played BIGINT,
    PRIMARY KEY (patch_id, team_id, hero_id)
);

CREATE TABLE IF NOT EXISTS ml.player_hero_agg (
    patch_id   INT     NOT NULL, account_id BIGINT  NOT NULL, hero_id    INT     NOT NULL,
    games      INT     NOT NULL DEFAULT 0, wins       INT     NOT NULL DEFAULT 0, win_rate   FLOAT   NOT NULL DEFAULT 0.5,
    avg_gpm    FLOAT, avg_xpm    FLOAT, avg_kills  FLOAT, avg_deaths FLOAT, avg_assists FLOAT, avg_kda    FLOAT, lane_role  INT,
    firstblood_rate FLOAT DEFAULT 0, avg_camps_stacked FLOAT DEFAULT 0, avg_vision_placed FLOAT DEFAULT 0,
    avg_gold_10 FLOAT, avg_xp_10   FLOAT, last_played BIGINT,
    PRIMARY KEY (patch_id, account_id, hero_id)
);

CREATE TABLE IF NOT EXISTS ml.hero_synergy_agg (
    patch_id      INT   NOT NULL, hero_a        INT   NOT NULL, hero_b        INT   NOT NULL,
    games         INT   NOT NULL DEFAULT 0, wins          INT   NOT NULL DEFAULT 0, win_rate      FLOAT NOT NULL DEFAULT 0.5,
    PRIMARY KEY (patch_id, hero_a, hero_b)
);

CREATE TABLE IF NOT EXISTS ml.hero_counter_agg (
    patch_id      INT   NOT NULL, hero_id       INT   NOT NULL, enemy_hero_id INT   NOT NULL,
    games         INT   NOT NULL DEFAULT 0, wins          INT   NOT NULL DEFAULT 0, win_rate      FLOAT NOT NULL DEFAULT 0.5, avg_kd_diff   FLOAT,
    PRIMARY KEY (patch_id, hero_id, enemy_hero_id)
);

CREATE TABLE IF NOT EXISTS ml.team_h2h_agg (
    patch_id     INT  NOT NULL, team_id      BIGINT  NOT NULL, enemy_team_id BIGINT NOT NULL,
    games        INT  NOT NULL DEFAULT 0, wins         INT  NOT NULL DEFAULT 0, win_rate     FLOAT NOT NULL DEFAULT 0.5,
    PRIMARY KEY (patch_id, team_id, enemy_team_id)
);

CREATE TABLE IF NOT EXISTS ml.hero_baseline_agg (
    patch_id       INT   NOT NULL, hero_id        INT   NOT NULL,
    total_picks    INT   NOT NULL DEFAULT 0, total_wins     INT   NOT NULL DEFAULT 0, total_bans     INT   NOT NULL DEFAULT 0,
    win_rate       FLOAT NOT NULL DEFAULT 0.5, pick_rate      FLOAT NOT NULL DEFAULT 0.0, ban_rate       FLOAT NOT NULL DEFAULT 0.0,
    avg_gpm        FLOAT, avg_xpm        FLOAT, avg_kills      FLOAT, avg_deaths     FLOAT, avg_assists    FLOAT,
    avg_gold_10    FLOAT, avg_xp_10      FLOAT,
    PRIMARY KEY (patch_id, hero_id)
);

CREATE TABLE IF NOT EXISTS ml.hero_draft_slot_agg (
    patch_id           INT      NOT NULL, hero_id            INT      NOT NULL, team_pick_ordinal  SMALLINT NOT NULL,
    games              INT      NOT NULL DEFAULT 0, wins               INT      NOT NULL DEFAULT 0, win_rate           FLOAT    NOT NULL DEFAULT 0.5,
    PRIMARY KEY (patch_id, hero_id, team_pick_ordinal),
    CONSTRAINT chk_hero_draft_slot_ordinal CHECK (team_pick_ordinal BETWEEN 1 AND 5)
);

-- ============================================================================
-- 4. PIT-SAFE SNAPSHOT TABLES (FLOAT games/wins applied where needed)
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml.team_hero_snapshot (
    as_of_date      DATE    NOT NULL, snapshot_tier   TEXT    NOT NULL DEFAULT 'daily', patch_id        INT     NOT NULL,
    team_id         BIGINT  NOT NULL, hero_id         INT     NOT NULL,
    games           FLOAT   NOT NULL DEFAULT 0, wins            FLOAT   NOT NULL DEFAULT 0, bans            INT     NOT NULL DEFAULT 0, win_rate        FLOAT   NOT NULL DEFAULT 0.5,
    avg_gpm         FLOAT, avg_xpm         FLOAT, avg_kills       FLOAT, avg_deaths      FLOAT, avg_assists     FLOAT,
    firstblood_rate     FLOAT, avg_camps_stacked   FLOAT, avg_vision_placed   FLOAT, avg_gold_10         FLOAT, avg_xp_10           FLOAT, last_played         BIGINT,
    PRIMARY KEY (patch_id, as_of_date, team_id, hero_id)
);
CREATE INDEX IF NOT EXISTS idx_team_hero_snapshot_lookup ON ml.team_hero_snapshot (patch_id, team_id, hero_id, as_of_date DESC);

CREATE TABLE IF NOT EXISTS ml.player_hero_snapshot (
    as_of_date      DATE    NOT NULL, snapshot_tier   TEXT    NOT NULL DEFAULT 'weekly', patch_id        INT     NOT NULL,
    account_id      BIGINT  NOT NULL, hero_id         INT     NOT NULL,
    games           FLOAT   NOT NULL DEFAULT 0, wins            FLOAT   NOT NULL DEFAULT 0, win_rate        FLOAT   NOT NULL DEFAULT 0.5,
    avg_gpm         FLOAT, avg_xpm         FLOAT, avg_kills       FLOAT, avg_deaths      FLOAT, avg_assists     FLOAT,
    avg_kda         FLOAT, lane_role       INT, firstblood_rate     FLOAT, avg_camps_stacked   FLOAT, avg_vision_placed   FLOAT,
    avg_gold_10         FLOAT, avg_xp_10           FLOAT, last_played         BIGINT,
    PRIMARY KEY (patch_id, as_of_date, account_id, hero_id)
);
CREATE INDEX IF NOT EXISTS idx_player_hero_snapshot_lookup ON ml.player_hero_snapshot (patch_id, account_id, hero_id, as_of_date DESC);

CREATE TABLE IF NOT EXISTS ml.hero_synergy_snapshot (
    as_of_date      DATE    NOT NULL, snapshot_tier   TEXT    NOT NULL DEFAULT 'weekly', patch_id        INT     NOT NULL,
    hero_a          INT     NOT NULL, hero_b          INT     NOT NULL,
    games           FLOAT   NOT NULL DEFAULT 0, wins            FLOAT   NOT NULL DEFAULT 0, win_rate        FLOAT   NOT NULL DEFAULT 0.5,
    PRIMARY KEY (patch_id, as_of_date, hero_a, hero_b)
);
CREATE INDEX IF NOT EXISTS idx_hero_synergy_snapshot_lookup ON ml.hero_synergy_snapshot (patch_id, hero_a, hero_b, as_of_date DESC);

CREATE TABLE IF NOT EXISTS ml.hero_counter_snapshot (
    as_of_date      DATE    NOT NULL, snapshot_tier   TEXT    NOT NULL DEFAULT 'weekly', patch_id        INT     NOT NULL,
    hero_id         INT     NOT NULL, enemy_hero_id   INT     NOT NULL,
    games           FLOAT   NOT NULL DEFAULT 0, wins            FLOAT   NOT NULL DEFAULT 0, win_rate        FLOAT   NOT NULL DEFAULT 0.5, avg_kd_diff     FLOAT,
    PRIMARY KEY (patch_id, as_of_date, hero_id, enemy_hero_id)
);
CREATE INDEX IF NOT EXISTS idx_hero_counter_snapshot_lookup ON ml.hero_counter_snapshot (patch_id, hero_id, enemy_hero_id, as_of_date DESC);

CREATE TABLE IF NOT EXISTS ml.team_h2h_snapshot (
    as_of_date      DATE    NOT NULL, snapshot_tier   TEXT    NOT NULL DEFAULT 'daily', patch_id        INT     NOT NULL,
    team_id         BIGINT  NOT NULL, enemy_team_id   BIGINT  NOT NULL,
    games           INT     NOT NULL DEFAULT 0, wins            INT     NOT NULL DEFAULT 0, win_rate        FLOAT   NOT NULL DEFAULT 0.5,
    PRIMARY KEY (patch_id, as_of_date, team_id, enemy_team_id)
);
CREATE INDEX IF NOT EXISTS idx_team_h2h_snapshot_lookup ON ml.team_h2h_snapshot (patch_id, team_id, enemy_team_id, as_of_date DESC);

CREATE TABLE IF NOT EXISTS ml.hero_baseline_snapshot (
    as_of_date      DATE    NOT NULL, snapshot_tier   TEXT    NOT NULL DEFAULT 'daily', patch_id        INT     NOT NULL,
    hero_id         INT     NOT NULL, total_picks     INT     NOT NULL DEFAULT 0, total_wins      INT     NOT NULL DEFAULT 0, total_bans      INT     NOT NULL DEFAULT 0,
    win_rate        FLOAT   NOT NULL DEFAULT 0.5, pick_rate       FLOAT   NOT NULL DEFAULT 0.0, ban_rate        FLOAT   NOT NULL DEFAULT 0.0,
    avg_gpm         FLOAT, avg_xpm         FLOAT, avg_kills       FLOAT, avg_deaths      FLOAT, avg_assists     FLOAT, avg_gold_10     FLOAT, avg_xp_10       FLOAT,
    PRIMARY KEY (patch_id, as_of_date, hero_id)
);
CREATE INDEX IF NOT EXISTS idx_hero_baseline_snapshot_lookup ON ml.hero_baseline_snapshot (patch_id, hero_id, as_of_date DESC);

CREATE TABLE IF NOT EXISTS ml.hero_draft_slot_snapshot (
    as_of_date           DATE     NOT NULL, snapshot_tier        TEXT     NOT NULL DEFAULT 'daily', patch_id             INT      NOT NULL,
    hero_id              INT      NOT NULL, team_pick_ordinal    SMALLINT NOT NULL,
    games                INT      NOT NULL DEFAULT 0, wins                 INT      NOT NULL DEFAULT 0, win_rate             FLOAT    NOT NULL DEFAULT 0.5,
    PRIMARY KEY (patch_id, as_of_date, hero_id, team_pick_ordinal),
    CONSTRAINT chk_snapshot_ordinal CHECK (team_pick_ordinal BETWEEN 1 AND 5)
);
CREATE INDEX IF NOT EXISTS idx_hero_draft_slot_snapshot_lookup ON ml.hero_draft_slot_snapshot (patch_id, hero_id, team_pick_ordinal, as_of_date DESC);
CREATE INDEX IF NOT EXISTS idx_hero_draft_slot_snapshot_hero_lookup ON ml.hero_draft_slot_snapshot (patch_id, hero_id, as_of_date DESC);

-- ============================================================================
-- 5. FUNCTIONS
-- ============================================================================
CREATE OR REPLACE FUNCTION analytics.refresh_all_mv()
RETURNS TABLE(view_name TEXT, duration INTERVAL, rows_affected BIGINT, status TEXT) AS $$
DECLARE v_start TIMESTAMPTZ; v_rows BIGINT; v_status TEXT; v_error TEXT; mv RECORD;
BEGIN
    FOR mv IN SELECT c.relname::TEXT AS name FROM pg_class c JOIN pg_namespace n ON c.relnamespace = n.oid WHERE n.nspname = 'analytics' AND c.relkind = 'm' ORDER BY c.relname
    LOOP
        v_start := clock_timestamp(); v_status := 'SUCCESS'; v_error := NULL;
        BEGIN EXECUTE format('REFRESH MATERIALIZED VIEW CONCURRENTLY analytics.%I', mv.name); EXECUTE format('SELECT COUNT(*) FROM analytics.%I', mv.name) INTO v_rows;
        EXCEPTION WHEN OTHERS THEN v_status := 'FAILED'; v_rows := 0; v_error := SQLERRM; END;
        INSERT INTO analytics.mv_refresh_log (view_name, started_at, completed_at, rows_affected, duration, error_message) VALUES (mv.name, v_start, clock_timestamp(), v_rows, clock_timestamp() - v_start, v_error);
        view_name := mv.name; duration := clock_timestamp() - v_start; rows_affected := v_rows; status := v_status; RETURN NEXT;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION analytics.update_feature_snapshots()
RETURNS TEXT AS $$
DECLARE run_record RECORD; target_date DATE; start_ts TIMESTAMPTZ := clock_timestamp(); rows_affected INT; total_rows INT := 0; snapshot_count INT := 0; max_mid BIGINT;
BEGIN
    SELECT * INTO run_record FROM analytics.featurizer_runs WHERE id = 1;
    FOR target_date IN
        SELECT DISTINCT DATE(TO_TIMESTAMP(start_time)) AS d
        FROM matches WHERE match_id > COALESCE(run_record.last_processed_match_id, 0) AND start_time > 0 ORDER BY d
    LOOP
        INSERT INTO analytics.feature_snapshots_player_hero (snapshot_date, account_id, hero_id, games_played, wins, shrunk_win_rate, avg_kills, avg_deaths, avg_assists, avg_kda, avg_gpm, avg_xpm, avg_hero_damage, avg_tower_damage, avg_hero_healing, avg_last_hits, primary_lane_role, role_flexibility, days_since_last_played, games_last_30d, games_last_90d)
        SELECT target_date, p.account_id, p.hero_id, COUNT(*), SUM(CASE WHEN p.win = 1 THEN 1 ELSE 0 END), (SUM(CASE WHEN p.win = 1 THEN 1 ELSE 0 END) + 2.5) / (COUNT(*) + 5.0), AVG(p.kills), AVG(p.deaths), AVG(p.assists), AVG(p.kda), AVG(p.gold_per_min), AVG(p.xp_per_min), AVG(p.hero_damage), AVG(p.tower_damage), AVG(p.hero_healing), AVG(p.last_hits), MODE() WITHIN GROUP (ORDER BY p.lane_role), COUNT(DISTINCT p.lane_role), (target_date - DATE(TO_TIMESTAMP(MAX(m.start_time))))::INT, COUNT(CASE WHEN m.start_time > EXTRACT(EPOCH FROM (target_date - INTERVAL '30 days')) THEN 1 END), COUNT(CASE WHEN m.start_time > EXTRACT(EPOCH FROM (target_date - INTERVAL '90 days')) THEN 1 END)
        FROM players p INNER JOIN matches m ON p.match_id = m.match_id WHERE m.leagueid > 0 AND m.lobby_type IN (1, 2) AND p.account_id IS NOT NULL AND DATE(TO_TIMESTAMP(m.start_time)) < target_date GROUP BY p.account_id, p.hero_id HAVING COUNT(*) >= 3
        ON CONFLICT (snapshot_date, account_id, hero_id) DO UPDATE SET games_played = EXCLUDED.games_played, wins = EXCLUDED.wins, shrunk_win_rate = EXCLUDED.shrunk_win_rate, avg_kills = EXCLUDED.avg_kills, avg_deaths = EXCLUDED.avg_deaths, avg_assists = EXCLUDED.avg_assists, avg_kda = EXCLUDED.avg_kda, avg_gpm = EXCLUDED.avg_gpm, avg_xpm = EXCLUDED.avg_xpm, avg_hero_damage = EXCLUDED.avg_hero_damage, avg_tower_damage = EXCLUDED.avg_tower_damage, avg_hero_healing = EXCLUDED.avg_hero_healing, avg_last_hits = EXCLUDED.avg_last_hits, primary_lane_role = EXCLUDED.primary_lane_role, role_flexibility = EXCLUDED.role_flexibility, days_since_last_played = EXCLUDED.days_since_last_played, games_last_30d = EXCLUDED.games_last_30d, games_last_90d = EXCLUDED.games_last_90d;
        GET DIAGNOSTICS rows_affected = ROW_COUNT;
        total_rows := total_rows + rows_affected;
        snapshot_count := snapshot_count + 1;
        SELECT MAX(match_id) INTO max_mid FROM matches WHERE DATE(TO_TIMESTAMP(start_time)) = target_date;
        UPDATE analytics.featurizer_runs SET last_snapshot_date = target_date, last_run_timestamp = start_ts, last_processed_match_id = GREATEST(COALESCE(last_processed_match_id, 0), COALESCE(max_mid, 0)) WHERE id = 1;
    END LOOP;
    IF snapshot_count = 0 THEN RETURN 'No new matches found. Snapshots up to date.'; END IF;
    UPDATE analytics.featurizer_runs SET matches_processed = matches_processed + total_rows WHERE id = 1;
    RETURN format('Generated/Updated %s snapshots across %s date(s) in %s', total_rows, snapshot_count, clock_timestamp() - start_ts);
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- 6. MIGRATIONS TRACKER
-- ============================================================================
CREATE TABLE IF NOT EXISTS _migrations (name TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now());
INSERT INTO _migrations (name) VALUES ('002_ml.sql') ON CONFLICT DO NOTHING;
