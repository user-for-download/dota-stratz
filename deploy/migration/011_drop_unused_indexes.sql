-- 007_drop_unused_indexes.sql
--
-- Drops low-selectivity indexes that provide negligible query benefit while
-- adding write overhead and disk usage.
--
-- idx_matches_radiant_win: boolean column with only 2 distinct values.
-- A B-tree index on a boolean cannot be used for selective filtering.
-- If specific queries filter on radiant_win = false, replace with a
-- partial index instead.
--
-- This migration is idempotent via IF EXISTS.

DROP INDEX IF EXISTS idx_matches_radiant_win;

-- Log completion (visible in docker logs during migration runs)
DO $$
BEGIN
    RAISE NOTICE 'Migration 007 complete: dropped idx_matches_radiant_win';
END $$;
