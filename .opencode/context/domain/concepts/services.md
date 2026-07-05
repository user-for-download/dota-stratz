# Service Architecture

## ID Fetcher
**Purpose**: First pipeline stage. Cron-scheduled query of OpenDota Explorer API for recent match IDs.

**Key behaviors**:
- Self-scheduled via `robfig/cron` (`FETCH_SCHEDULE`, default `*/5 * * * *` — every 5 min)
- Two query modes: **bootstrap** (rolling window `start_time >= NOW() - N days`) and **watermark** (`match_id > last_parsed_match_id`, for incremental catch-up)
- Watermark query uses `ORDER BY match_id ASC` with match_id filter pushed into SQL and `LIMIT` to bound result size — prevents data-loss bug where DESC+no-filter caused the watermark to skip past older unprocessed matches
- Watermark is read on startup from `ingestion_checkpoints.last_parsed_match_id` via the shared `checkpoint` package; if DB is unreachable, falls back to rolling-window path
- Rate-limit avoidance via Redis-backed proxy pool
- Opt-in startup run (`ID_FETCHER_START_RUN`) waits for `PoolMinSize` proxies before firing, shares cron trylock
- Mutex flag (buffered chan capacity-1) prevents overlapping cron runs
- Redis-sourced Prometheus metrics via `proxypool.NewRedisPoolCollector` for accurate pool size (replaces per-process gauge that diverged from Redis ground truth)
- `reconnectIfNeeded` verifies new connection is alive after reconnect
- Config validates required fields: RabbitMQ URL, queue names, Redis addr

## Detail Fetcher
**Purpose**: Consumes match IDs, fetches full match JSON from OpenDota, publishes to parser queue.

**Key behaviors**:
- Rate-limited proxy acquisition: `max_req_per_min=50` per proxy
- Retries with exponential backoff (max 3), then routes to DLQ
- Reconnect backoff uses `select` on `ctx.Done()` to prevent delayed shutdown
- `ErrMatchNotFound` (404) → ack + drop immediately
- Configurable concurrency (default 50 workers)
- Config validates required fields: RabbitMQ URL, queue names, Postgres DSN

## Parser
**Purpose**: Terminal stage. Unmarshals JSON, batch-inserts into 20+ partitioned tables in a single transaction.

**Key behaviors**:
- Batch size: 100 matches or 2s timeout, whichever hits first
- FK violation (23503) → per-match fallback, offending match to DLQ
- `context.WithoutCancel` for `SendBatch`/`Commit` to prevent connection pool corruption during shutdown
- Rollback uses 10-second timeout context to prevent indefinite blocking on unreachable DB
- All child table helpers return errors (no swallowed failures)
- Consumer uses `context.Context`-based `ConsumeWithReconnect(ctx, tag)` with context-aware backoff sleep

## Proxy Manager
**Purpose**: Autonomous proxy pool. Fetches, validates, and maintains proxy health in Redis.

**Key behaviors**:
- Bootstrap: load from file + URL (uses `limitedFetchWithRetry` so the cooldown kicks in before the follow-up top-up), deduplicate, validate, add survivors
- Refresh loop: periodic remote fetch (default 15min), top-up if below `PoolMinSize`
- Lease reaper: reclaims expired leases every 30s
- Transport supports HTTP/HTTPS CONNECT, SOCKS5, **SOCKS4** (native dialer — previously forced through `proxy.SOCKS5()` causing handshake failure on SOCKS4-only servers)
- Source-fetch cooldown: 10-min guard against HTTP 429 (both bootstrap and refresh respect it)
- Validator drains response body on non-200 status to prevent resource leaks
- Cooldown checks ZScore before re-adding proxy to prevent race with concurrent remove()

## Inference API
**Purpose**: Serves draft predictions via PyTorch DraftBERT (TorchScript JIT) on HTTP :8080 with Monte Carlo rollouts.

**Key behaviors**:
- Threaded connection pool (psycopg2 `ThreadedConnectionPool`) guarded by `threading.Lock`
- Broken connections are discarded via `putconn(conn, close=True)` on rollback failure — prevents pool poisoning
- `put_conn()` checks `_pool` inside lock to prevent race with `close_pool()`
- Model loading (file I/O) happens outside the lock to avoid blocking all threads during lazy load
- Six pre-fetched batch queries replace hundreds of individual lookups: baselines, team-hero, player-hero, synergy, counter, h2h
- NULL-safe: `.get()` defaults use `or 0` to handle NULL DB values (prevents `TypeError` on `int(None)`)
- **`POST /predict`** accepts `patch_id`, `first_pick_team`, `draft[]`, `radiant_team_id`, `dire_team_id`, `account_id` (optional, enables player-hero features), `num_recommendations`; returns top-5 hero scores + **reasoning** string explaining top picks
- **`POST /reload/{patch_id}`** — hot-reload a model without restart (requires `STRATZ_ADMIN_TOKEN` in Bearer auth header)
- `GET /health` — Returns `{"status":"ok","patch_models_loaded":[...]}`
