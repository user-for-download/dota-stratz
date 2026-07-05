-- Watermark-based query for OpenDota Explorer.
--
-- Used by id-fetcher when ingestion_checkpoints.last_parsed_match_id > 0
-- (i.e. the parser has committed at least one batch). Returns all matches
-- in the lookback window, ordered by match_id ASC.
--
-- The match_id filter is NOT applied in SQL — deduplication is handled
-- by the DB existence check in Run(). This prevents data loss when the
-- watermark is higher than some unparsed matches (e.g. after a backup
-- restore with a stale watermark).
--
-- The %%d placeholders are (in order):
--   1. lookback window in days (configurable via watermark_lookback_days)
--   2. comma-separated lobby_type list (same as matches.sql)
--   3. LIMIT / max results

SELECT match_id, start_time
FROM matches
WHERE start_time >= (EXTRACT(EPOCH FROM NOW() - INTERVAL '%d days'))::BIGINT
AND lobby_type IN (%s)
ORDER BY match_id ASC
LIMIT %d
