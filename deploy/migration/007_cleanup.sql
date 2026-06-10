-- 007_cleanup.sql
--
-- Drops tables, indexes, and functions that were part of removed features
-- (coordinator, trigger queue, raw_matches staging table, unused GIN indexes,
-- dead analytics materialized views, dead constants tables).
--
-- The corresponding CREATE statements have been removed from 001, 002, 003
-- so a fresh staging environment will not create them in the first place.
--
-- Safe to run on a running production database — all drops are IF EXISTS.

BEGIN;

-- ============================================================================
-- 1. RAW MATCHES STAGING TABLE (removed with coordinator / trigger queue)
-- ============================================================================
DROP TABLE IF EXISTS raw_matches CASCADE;
DROP FUNCTION IF EXISTS cleanup_raw_matches;

-- ============================================================================
-- 2. UNUSED GIN INDEXES (never queried, no planner benefit)
-- ============================================================================
DROP INDEX IF EXISTS idx_matches_metadata_gin;
DROP INDEX IF EXISTS idx_teamfight_players_ability_uses_gin;
DROP INDEX IF EXISTS idx_teamfight_players_item_uses_gin;
DROP INDEX IF EXISTS idx_teamfight_players_killed_gin;
DROP INDEX IF EXISTS idx_teamfight_players_deaths_pos_gin;

-- ============================================================================
-- 3. DEAD CONSTANTS TABLES (not referenced by Go code or ETL)
-- ============================================================================
DO $$ BEGIN
    -- Drop FK first so DROP TABLE CASCADE doesn't complain about ordering
    IF EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_hf_hero') THEN
        ALTER TABLE const_hero_facet DROP CONSTRAINT fk_hf_hero;
    END IF;
END $$;
DROP TABLE IF EXISTS const_hero_facet CASCADE;

-- ============================================================================
-- 4. DEAD ANALYTICS MATERIALIZED VIEWS
-- ============================================================================
DROP MATERIALIZED VIEW IF EXISTS analytics.mv_player_hero_profile CASCADE;

COMMIT;
