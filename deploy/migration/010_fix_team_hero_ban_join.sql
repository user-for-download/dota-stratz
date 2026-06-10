-- 010_fix_team_hero_ban_join.sql
--
-- Rebuilds mv_team_hero_profile to fix the FULL OUTER JOIN between
-- team_hero_picks and team_hero_bans that never matched.
--
-- Root cause
-- ----------
-- team_hero_picks derives its team_id from the actual team:
--   CASE WHEN p.is_radiant THEN m.radiant_team_id ELSE m.dire_team_id END
-- which is a BIGINT (actual team_id like 9530716).
--
-- team_hero_bans was using pb.team (0 or 1) as its team_id:
--   SELECT pb.team AS team_id, ...
-- so the FULL OUTER JOIN ON p.team_id = b.team_id never matched because
-- the types were different scales (BIGINT vs INT 0/1).
--
-- Fix: Derive the actual team_id from matches in team_hero_bans too:
--   CASE WHEN pb.team = 0 THEN m.radiant_team_id ELSE m.dire_team_id END
--
-- See audit finding #3.
-- ============================================================================

BEGIN;

DROP MATERIALIZED VIEW IF EXISTS analytics.mv_team_hero_profile CASCADE;

CREATE MATERIALIZED VIEW analytics.mv_team_hero_profile AS
WITH config AS (SELECT prior_games, prior_win_rate FROM analytics.shrinkage_config WHERE metric_name = 'team_hero_wr'),
team_hero_picks AS (
    SELECT
        CASE WHEN p.is_radiant THEN m.radiant_team_id ELSE m.dire_team_id END AS team_id,
        p.hero_id,
        COUNT(*) AS games_played,
        SUM(CASE WHEN p.win = 1 THEN 1 ELSE 0 END) AS wins,
        AVG(p.kills) AS avg_kills,
        AVG(p.deaths) AS avg_deaths,
        AVG(p.assists) AS avg_assists,
        AVG(p.gold_per_min) AS avg_gpm,
        AVG(p.xp_per_min) AS avg_xpm,
        AVG(p.hero_damage) AS avg_hero_damage,
        AVG(p.tower_damage) AS avg_tower_damage,
        MAX(m.start_time) AS last_played_time
    FROM players p
    INNER JOIN matches m ON p.match_id = m.match_id
    WHERE m.leagueid > 0 AND m.lobby_type IN (1, 2)
      AND CASE WHEN p.is_radiant THEN m.radiant_team_id ELSE m.dire_team_id END IS NOT NULL
    GROUP BY team_id, p.hero_id
),
team_hero_bans AS (
    SELECT
        CASE WHEN pb.team = 0 THEN m.radiant_team_id ELSE m.dire_team_id END AS team_id,
        pb.hero_id,
        COUNT(*) AS times_banned
    FROM picks_bans pb
    INNER JOIN matches m ON pb.match_id = m.match_id
    WHERE m.leagueid > 0 AND m.lobby_type IN (1, 2) AND pb.is_pick = FALSE
    GROUP BY CASE WHEN pb.team = 0 THEN m.radiant_team_id ELSE m.dire_team_id END, pb.hero_id
)
SELECT
    COALESCE(p.team_id, b.team_id) AS team_id,
    COALESCE(p.hero_id, b.hero_id) AS hero_id,
    COALESCE(p.games_played, 0) AS games_played,
    COALESCE(p.wins, 0) AS wins,
    COALESCE(b.times_banned, 0) AS times_banned,
    CASE
        WHEN COALESCE(p.games_played, 0) > 0
        THEN (COALESCE(p.wins, 0) + (c.prior_games * c.prior_win_rate)) / (p.games_played + c.prior_games)
        ELSE c.prior_win_rate
    END AS shrunk_win_rate,
    COALESCE(p.avg_kills, 0) AS avg_kills,
    COALESCE(p.avg_deaths, 0) AS avg_deaths,
    COALESCE(p.avg_assists, 0) AS avg_assists,
    COALESCE(p.avg_gpm, 0) AS avg_gpm,
    COALESCE(p.avg_xpm, 0) AS avg_xpm,
    COALESCE(p.avg_hero_damage, 0) AS avg_hero_damage,
    COALESCE(p.avg_tower_damage, 0) AS avg_tower_damage,
    p.last_played_time
FROM team_hero_picks p
FULL OUTER JOIN team_hero_bans b ON p.team_id = b.team_id AND p.hero_id = b.hero_id
CROSS JOIN config c;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_team_hero_profile_pk ON analytics.mv_team_hero_profile(team_id, hero_id);
CREATE INDEX IF NOT EXISTS idx_mv_team_hero_profile_team ON analytics.mv_team_hero_profile(team_id);
CREATE INDEX IF NOT EXISTS idx_mv_team_hero_profile_hero ON analytics.mv_team_hero_profile(hero_id);

DO $$
BEGIN
    RAISE NOTICE 'Migration 010 complete: rebuilt mv_team_hero_profile with fixed team_hero_bans JOIN';
END $$;

COMMIT;
