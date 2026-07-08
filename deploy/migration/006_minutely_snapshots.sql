-- ============================================================================
-- 006: Minutely Game State Snapshots for LiveDraftBERT Training
-- ============================================================================
-- Pre-computes per-minute game state features so the trainer doesn't need
-- to re-extract from raw tables every run. One row per minute per match.
-- ============================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS ml.minutely_snapshots AS
WITH match_minutes AS (
    SELECT m.match_id, m.duration, m.radiant_win,
           m.radiant_team_id, m.dire_team_id, m.patch,
           generate_series(0, m.duration / 60) AS minute
    FROM matches m
    WHERE m.radiant_win IS NOT NULL
      AND m.duration >= 600
),
gold_xp AS (
    SELECT g.match_id, g.minute,
           g.radiant_gold_adv,
           COALESCE(x.radiant_xp_adv, 0) AS radiant_xp_adv
    FROM match_gold_adv g
    LEFT JOIN match_xp_adv x ON x.match_id = g.match_id AND x.minute = g.minute
),
cumulative_towers AS (
    SELECT match_id, minute,
           SUM(CASE WHEN team = 0 AND type = 'tower_kill' THEN 1 ELSE 0 END)
               OVER (PARTITION BY match_id ORDER BY minute) AS radiant_towers,
           SUM(CASE WHEN team = 1 AND type = 'tower_kill' THEN 1 ELSE 0 END)
               OVER (PARTITION BY match_id ORDER BY minute) AS dire_towers
    FROM objectives
    WHERE type = 'tower_kill'
    GROUP BY match_id, minute, team, type
),
cumulative_barracks AS (
    SELECT match_id, minute,
           SUM(CASE WHEN team = 0 AND type = 'barracks_kill' THEN 1 ELSE 0 END)
               OVER (PARTITION BY match_id ORDER BY minute) AS radiant_barracks,
           SUM(CASE WHEN team = 1 AND type = 'barracks_kill' THEN 1 ELSE 0 END)
               OVER (PARTITION BY match_id ORDER BY minute) AS dire_barracks
    FROM objectives
    WHERE type = 'barracks_kill'
    GROUP BY match_id, minute, team, type
),
cumulative_rosh AS (
    SELECT match_id, minute,
           SUM(CASE WHEN team = 0 THEN 1 ELSE 0 END)
               OVER (PARTITION BY match_id ORDER BY minute) AS radiant_rosh,
           SUM(CASE WHEN team = 1 THEN 1 ELSE 0 END)
               OVER (PARTITION BY match_id ORDER BY minute) AS dire_rosh
    FROM objectives
    WHERE type = 'roshan_kill'
    GROUP BY match_id, minute, team
)
SELECT
    mm.match_id,
    mm.minute,
    mm.radiant_win,
    mm.radiant_team_id,
    mm.dire_team_id,
    mm.patch,
    COALESCE(gx.radiant_gold_adv, 0) AS radiant_gold_adv,
    COALESCE(gx.radiant_xp_adv, 0) AS radiant_xp_adv,
    COALESCE(ct.radiant_towers, 0) AS radiant_towers,
    COALESCE(ct.dire_towers, 0) AS dire_towers,
    COALESCE(cb.radiant_barracks, 0) AS radiant_barracks,
    COALESCE(cb.dire_barracks, 0) AS dire_barracks,
    COALESCE(cr.radiant_rosh, 0) AS radiant_rosh,
    COALESCE(cr.dire_rosh, 0) AS dire_rosh
FROM match_minutes mm
LEFT JOIN gold_xp gx ON gx.match_id = mm.match_id AND gx.minute = mm.minute
LEFT JOIN cumulative_towers ct ON ct.match_id = mm.match_id AND ct.minute = mm.minute
LEFT JOIN cumulative_barracks cb ON cb.match_id = mm.match_id AND cb.minute = mm.minute
LEFT JOIN cumulative_rosh cr ON cr.match_id = mm.match_id AND cr.minute = mm.minute
WITH DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_minutely_snapshots_pk
    ON ml.minutely_snapshots (match_id, minute);
CREATE INDEX IF NOT EXISTS idx_minutely_snapshots_patch
    ON ml.minutely_snapshots (patch);

-- Refresh function (call periodically or after new data ingestion)
CREATE OR REPLACE FUNCTION ml.refresh_minutely_snapshots()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY ml.minutely_snapshots;
END;
$$ LANGUAGE plpgsql;

INSERT INTO _migrations (name) VALUES ('006_minutely_snapshots.sql') ON CONFLICT DO NOTHING;
