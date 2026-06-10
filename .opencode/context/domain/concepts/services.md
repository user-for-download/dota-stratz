# Service Architecture

## ID Fetcher
**Purpose**: First pipeline stage. Cron-scheduled query of OpenDota Explorer API for recent match IDs.

**Key behaviors**:
- Self-scheduled via `robfig/cron` (`FETCH_SCHEDULE`, default `*/5 * * * *` — every 5 min)
- Two query modes: bootstrap (rolling window, all matches in N days) and watermark (`match_id > last_parsed`, for incremental catch-up)
- Watermark query uses `ORDER BY match_id DESC` with a Go-side break-on-watermark filter — prevents the permanent pipeline stall that occurred with ASC+LIMIT (oldest matches always filled the result, blocking watermark advance)
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
- Bootstrap: load from file + URL (uses `limitedFetchWithRetry` so the cooldown kicks in before the follow-up top-up), deduplicate, validate, add survivors
- Refresh loop: periodic remote fetch (default 15min), top-up if below `PoolMinSize`
- Lease reaper: reclaims expired leases every 30s
- Transport supports HTTP/HTTPS CONNECT, SOCKS5, **SOCKS4** (native dialer — previously forced through `proxy.SOCKS5()` causing handshake failure on SOCKS4-only servers)
- Source-fetch cooldown: 10-min guard against HTTP 429 (both bootstrap and refresh respect it)

## Inference API
**Purpose**: Serves draft predictions via LightGBM binary classification models on HTTP :8080.

**Key behaviors**:
- Threaded connection pool (psycopg2 `ThreadedConnectionPool`) guarded by `threading.Lock`
- Broken connections are discarded via `putconn(conn, close=True)` on rollback failure — prevents pool poisoning (previously returned dead connections to the pool, causing cascading HTTP 500s)
- Six pre-fetched batch queries replace hundreds of individual lookups: baselines, team-hero, player-hero, synergy, counter, h2h
- Hot-reload endpoint (`/reload/:patch_id`) without restart
- Returns `{"status":"ok","patch_models_loaded":[...]}` on health check
