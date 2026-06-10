-- Watermark-based query for OpenDota Explorer.
--
-- Used by id-fetcher when ingestion_checkpoints.last_parsed_match_id > 0
-- (i.e. the parser has committed at least one batch). Returns the most
-- recent N days' worth of matches ordered by match_id DESC, so the Go
-- caller can filter to `match_id > <watermark>` and take the first
-- `batch_size` rows.
--
-- Trade-off: OpenDota's public Explorer API does NOT support
-- `match_id > X` as a server-side filter, so we have to query a wider
-- window (or equal, if watermark_lookback_days defaults to
-- FETCH_LAST_COUNT_DAY) than the bootstrap path and filter in Go.
-- The Go-side filter is O(n) over a ~5x overscan, so the cost is
-- negligible vs. the saved OpenDota round-trip.
--
-- The %%d placeholder is the lookback window in days (configurable via
-- watermark_lookback_days, default 30). The %%s placeholder is the
-- comma-separated lobby_type list, same as matches.sql.
--
-- ORDER BY match_id DESC ensures the Go filter sees the highest-id
-- matches first and bails out as soon as the cursor drops below the
-- watermark, avoiding scanning the whole result set.

SELECT match_id, start_time
FROM matches
WHERE start_time >= (EXTRACT(EPOCH FROM NOW() - INTERVAL '%d days'))::BIGINT
AND lobby_type IN (%s)
ORDER BY match_id DESC
