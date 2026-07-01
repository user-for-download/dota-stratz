-- ============================================================================
-- schema-check.sql
-- Run BEFORE deploying the ML service to verify DB assumptions against the
-- actual Postgres schema. Run with:
--   docker exec -i dota2-postgres psql -U dota2 -d dota2 < scripts/schema-check.sql
-- ============================================================================

-- Tunable: set to the number of random match samples to inspect.
\set sample_n 5

-- ============================================================================
-- 1. picks_bans — team column convention
-- ============================================================================
-- Expected: team = 0 (Radiant), team = 1 (Dire).
-- This drives EVERY win-flag computation in the PIT ledgers.
SELECT '1. picks_bans.team convention' AS check_name;
SELECT team, COUNT(*) AS n,
       MIN(match_id) AS sample_match, MAX(match_id) AS sample_match
FROM picks_bans
WHERE is_pick
  AND team NOT IN (0, 1)
GROUP BY team;
-- If this returns ANY rows with team NOT IN (0,1), the convention is broken.
SELECT '   → PASS: All picks have team IN (0,1)'
WHERE NOT EXISTS (SELECT 1 FROM picks_bans WHERE is_pick AND team NOT IN (0, 1));

-- ============================================================================
-- 2. matches — pro-match coverage and data quality
-- ============================================================================
SELECT '2. Pro match coverage (leagueid > 0, lobby_type IN (1,2))' AS check_name;
SELECT COUNT(*) AS total_matches,
       COUNT(*) FILTER (WHERE leagueid > 0 AND lobby_type IN (1, 2)) AS pro_matches,
       COUNT(*) FILTER (WHERE radiant_win IS NULL) AS unresolved,
       COUNT(*) FILTER (WHERE duration <= 0) AS zero_duration,
       MIN(start_time) AS earliest, MAX(start_time) AS latest,
       COUNT(DISTINCT patch) AS n_patches,
       MIN(patch) AS min_patch, MAX(patch) AS max_patch
FROM matches;

-- Verify the indexes that the PIT SQL relies on exist.
SELECT '3. Required indexes' AS check_name;
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename IN ('matches', 'picks_bans', 'players')
  AND indexname IN ('idx_matches_pro_filter', 'idx_matches_start_time_brin',
                    'idx_matches_leagueid', 'idx_players_account_hero');

-- ============================================================================
-- 4. feature_snapshots_player_hero — snapshot coverage
-- ============================================================================
SELECT '4. feature_snapshots_player_hero coverage' AS check_name;
SELECT COUNT(*) AS total_rows,
       COUNT(DISTINCT snapshot_date) AS n_dates,
       MIN(snapshot_date) AS earliest, MAX(snapshot_date) AS latest,
       COUNT(DISTINCT account_id) AS n_players,
       COUNT(DISTINCT hero_id) AS n_heroes,
       COUNT(*) FILTER (WHERE games_played IS NULL) AS null_games,
       COUNT(*) FILTER (WHERE shrunk_win_rate IS NULL) AS null_wr
FROM analytics.feature_snapshots_player_hero;

-- ============================================================================
-- 5. refresh_all_mv() — dynamic MV discovery
-- ============================================================================
-- New PIT ledgers we'll add (008 migration):
--   analytics.hero_pair_ledger
--   analytics.hero_counter_ledger
--   analytics.team_hero_ledger
-- The existing refresh_all_mv() discovers MVs dynamically via pg_class,
-- so it will auto-include these. Verify:
SELECT '5. refresh_all_mv() will discover new MVs' AS check_name
WHERE EXISTS (
    SELECT 1 FROM pg_proc p
    JOIN pg_namespace n ON p.pronamespace = n.oid
    WHERE n.nspname = 'analytics'
      AND p.proname = 'refresh_all_mv'
      AND pg_get_functiondef(p.oid) LIKE '%pg_class%'
);

-- ============================================================================
-- 6. Sample picks_bans data (manual inspection)
-- ============================================================================
SELECT '6. Sample picks_bans rows (random matches)' AS check_name;
WITH sample_matches AS (
    SELECT match_id FROM matches
    WHERE leagueid > 0 AND lobby_type IN (1, 2) AND radiant_win IS NOT NULL
    ORDER BY random() LIMIT :sample_n
)
SELECT pb.match_id, pb."order", pb.is_pick, pb.hero_id, pb.team,
       CASE WHEN pb.team = 0 THEN 'Radiant' ELSE 'Dire' END AS side,
       m.radiant_team_id, m.dire_team_id,
       CASE
           WHEN pb.team = 0 THEN m.radiant_team_id::TEXT
           ELSE m.dire_team_id::TEXT
       END AS expected_team_id
FROM sample_matches sm
JOIN picks_bans pb ON pb.match_id = sm.match_id
JOIN matches m ON m.match_id = pb.match_id
WHERE pb.is_pick
ORDER BY pb.match_id, pb."order";

-- ============================================================================
-- 7. Check for post-game columns that would leak into features
-- ============================================================================
SELECT '7. Leakage guard — verify forbidden columns exist' AS check_name;
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'matches'
  AND column_name IN ('duration', 'radiant_win', 'tower_status_radiant',
                      'tower_status_dire', 'barracks_status_radiant',
                      'barracks_status_dire', 'radiant_score', 'dire_score')
ORDER BY column_name;

-- ============================================================================
-- SUMMARY
-- ============================================================================
SELECT '========================================' AS summary;
SELECT 'SCHEMA CHECK COMPLETE' AS result;
SELECT 'If all checks above show "PASS" or no unexpected rows, the DB is ready for the ML service.' AS note;
SELECT '========================================' AS summary;
