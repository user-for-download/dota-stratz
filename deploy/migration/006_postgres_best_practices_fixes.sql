-- 006_postgres_best_practices_fixes.sql
-- Applies Supabase Postgres best-practice fixes to the running database.
-- These changes are independent of the idempotent migration files above.
--
-- Changes:
--   1. Hardens analytics_writer: ALL PRIVILEGES -> SELECT, INSERT, UPDATE
--   2. Restricts analytics_reader: removes unnecessary public schema access
--      and adds granular ml schema access
--   3. Replaces CREATE TABLE IF NOT EXISTS ... PARTITION OF with bare
--      CREATE TABLE ... PARTITION OF (the calling function already guards
--      existence, and IF NOT EXISTS is PG13+ only)
-- ============================================================================

-- ============================================================================
-- 1. HARDEN analytics_writer PRIVILEGES
--    Least privilege: writer needs SELECT, INSERT, UPDATE — not DELETE/TRUNCATE
-- ============================================================================
REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics FROM analytics_writer;
REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA analytics FROM analytics_writer;

GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA analytics TO analytics_writer;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA analytics TO analytics_writer;

-- Default privileges for future tables created by the migration user
ALTER DEFAULT PRIVILEGES IN SCHEMA analytics
    GRANT SELECT, INSERT, UPDATE ON TABLES TO analytics_writer;
ALTER DEFAULT PRIVILEGES IN SCHEMA analytics
    GRANT USAGE, SELECT ON SEQUENCES TO analytics_writer;

-- ============================================================================
-- 2. RESTRICT analytics_reader
--    Remove unnecessary public schema exposure; add ml schema
-- ============================================================================
REVOKE ALL ON SCHEMA public FROM analytics_reader;
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM analytics_reader;

-- Ensure reader has access to ml schema (added in 005_ml_tables.sql)
GRANT USAGE ON SCHEMA ml TO analytics_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA ml TO analytics_reader;

-- Default privileges for future ml tables
ALTER DEFAULT PRIVILEGES IN SCHEMA ml
    GRANT SELECT ON TABLES TO analytics_reader;

-- ============================================================================
-- 3. FIX PARTITION FUNCTION (remove IF NOT EXISTS from PARTITION OF)
-- ============================================================================
CREATE OR REPLACE FUNCTION public.create_player_partition(
    partition_name TEXT, from_val BIGINT, to_val BIGINT
) RETURNS TEXT AS $$
BEGIN
    EXECUTE format('CREATE TABLE %I PARTITION OF players FOR VALUES FROM (%L) TO (%L)', partition_name, from_val, to_val);
    RETURN format('Partition %s created ( %s → %s )', partition_name, from_val, to_val);
EXCEPTION
    WHEN duplicate_table THEN RETURN format('Partition %s already exists', partition_name);
    WHEN SQLSTATE '42P17' THEN RETURN format('Partition %s range overlaps an existing partition: %s', partition_name, SQLERRM);
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Record migration
-- ============================================================================
INSERT INTO _migrations (name) VALUES ('006_postgres_best_practices_fixes.sql') ON CONFLICT DO NOTHING;
