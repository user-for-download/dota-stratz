-- 009_fix_check_constraints.sql
--
-- Drops CHECK constraints that conflict with deferrable-FK design and cause
-- infinite batch-requeue loops when violated.
--
-- Context
-- -------
-- The parser's batch_writer.go writes matches + players in a single pgx.Batch
-- transaction. CHECK constraints (duration > 0, kda >= 0) that fail raise
-- SQLSTATE 23514, which is NOT caught by the parser's isForeignKeyViolation()
-- guard (SQLSTATE 23503). The resulting batch failure triggers Nack+requeue,
-- creating an infinite loop for the entire batch even when only one row is
-- problematic (see audit findings #1 and #2).
--
-- The parser already validates match.Duration > 0 and routes invalid matches
-- to the DLQ via native dead-letter exchange (processor.go:149). The CHECK
-- constraint is therefore redundant for the ingestion path and harmful because
-- it can batch-kill healthy matches alongside a single zero-duration match.
--
-- For players.kda: OpenDota occasionally sends negative KDA values (edge case
-- when kills+assists < deaths in certain replay corruption scenarios). The
-- constraint adds no value — analytics queries filter kda >= 0 on read, and
-- the parser does not validate KDA before writing.
--
-- Also creates the pg_stat_statements extension (loaded via shared_preload
-- in compose.yaml but never made visible to queries — see compose.yaml line 18
-- comment).
--
-- Safe to run on a running production database — all DDL is IF EXISTS.
-- ============================================================================

BEGIN;

-- ============================================================================
-- 1. DROP CHECK constraints
-- ============================================================================

ALTER TABLE matches
    DROP CONSTRAINT IF EXISTS chk_duration_positive;

ALTER TABLE players
    DROP CONSTRAINT IF EXISTS chk_kda_non_negative;

-- ============================================================================
-- 2. CREATE pg_stat_statements extension
-- ============================================================================
-- The shared_preload_libraries entry in compose.yaml already loads the module
-- at postgres startup. This makes the functions and views visible.
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- ============================================================================
-- 3. Log completion
-- ============================================================================
DO $$
BEGIN
    RAISE NOTICE 'Migration 009 complete: dropped chk_duration_positive, chk_kda_non_negative, created pg_stat_statements ext';
END $$;

COMMIT;
