# Service Architecture

## ID Fetcher
**Purpose**: First pipeline stage. Cron-scheduled query of OpenDota Explorer API for recent match IDs.

**Key behaviors**:
- Self-scheduled via `robfig/cron` (`FETCH_SCHEDULE`, default `0 3 * * *`)
- Rolling time-window query (`start_time >= NOW() - N days`)
- Rate-limit avoidance via Redis-backed proxy pool
- Opt-in startup run (`ID_FETCHER_START_RUN`) waits for `PoolMinSize` proxies before firing
- Mutex flag prevents overlapping cron runs

## Detail Fetcher
**Purpose**: Consumes match IDs, fetches full match JSON from OpenDota, publishes to parser queue.

**Key behaviors**:
- Rate-limited proxy acquisition: `max_req_per_min=50` per proxy
- Retries with exponential backoff (max 3), then routes to DLQ
- `ErrMatchNotFound` (404) → ack + drop immediately
- Configurable concurrency (default 5 workers)

## Parser
**Purpose**: Terminal stage. Unmarshals JSON, batch-inserts into 20+ partitioned tables in a single transaction.

**Key behaviors**:
- Batch size: 100 matches or 2s timeout, whichever hits first
- FK violation (23503) → per-match fallback, offending match to DLQ
- `context.WithoutCancel` for `SendBatch`/`Commit` to prevent connection pool corruption during shutdown
- All child table helpers return errors (no swallowed failures)

## Proxy Manager
**Purpose**: Autonomous proxy pool. Fetches, validates, and maintains proxy health in Redis.

**Key behaviors**:
- Bootstrap: load from file + URL, deduplicate, validate, add survivors
- Refresh loop: periodic remote fetch (default 15min), top-up if below `PoolMinSize`
- Lease reaper: reclaims expired leases every 30s
- Supports HTTP/HTTPS CONNECT, SOCKS4, SOCKS5 via `MakeTransport`
- Source-fetch cooldown: 10-min guard against HTTP 429
