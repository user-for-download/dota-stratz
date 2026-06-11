-- Watermark-based query for OpenDota Explorer.
--
-- Used by id-fetcher when ingestion_checkpoints.last_parsed_match_id > 0
-- (i.e. the parser has committed at least one batch). Returns the
-- oldest matches strictly greater than the watermark, ordered by match_id
-- ASC, guaranteeing that no match is ever skipped regardless of backlog
-- depth.
--
-- ASC ordering + match_id > %%d is critical: every match above the
-- watermark appears in exactly one batch (the oldest ones first), so
-- even if there are 100K unparsed matches and the LIMIT only returns
-- 2.5K, the next cron tick will pick up the next 2.5K, and so on.
-- This avoids the permanent data-loss bug (issue #1) that occurred
-- with DESC + no match_id filter, where the watermark jumped past
-- older unprocessed matches that fell outside the LIMIT window.
--
-- The %%d placeholders are (in order):
--   1. lookback window in days (configurable via watermark_lookback_days)
--   2. match_id watermark (last_parsed_match_id from checkpoints)
--   3. comma-separated lobby_type list (same as matches.sql)
--   4. LIMIT / max results
--
-- Trade-off: OpenDota's public Explorer API does NOT support
-- `match_id > X` as a server-side filter, so we have to query a wider
-- window (or equal, if watermark_lookback_days defaults to
-- FETCH_LAST_COUNT_DAY) than the bootstrap path and filter in Go.
-- The Go-side filter is O(n) over a ~5x overscan, so the cost is
-- negligible vs. the saved OpenDota round-trip.

SELECT match_id, start_time
FROM matches
WHERE start_time >= (EXTRACT(EPOCH FROM NOW() - INTERVAL '%d days'))::BIGINT
AND match_id > %d
AND lobby_type IN (%s)
ORDER BY match_id ASC
LIMIT %d
