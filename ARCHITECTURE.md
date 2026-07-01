# Dota 2 Match Analysis System — Architecture

## Overview

An event-driven microservice pipeline that ingests Dota 2 match data from the [OpenDota API](https://www.opendota.com/), processes it through a multi-stage queue-based architecture, and stores the result in PostgreSQL for analytics and ML feature engineering.

**Language:** Go 1.26.3  
**Messaging:** RabbitMQ with dead-letter queues and automatic reconnection  
**Database:** PostgreSQL 16 (partitioned, with deferred FK constraints)  
**Caching/State:** Redis 7 (proxy pool backend + checkpoint state)  
**Observability:** Prometheus + Grafana (pre-configured dashboards and alerts)  
**Deployment:** Docker Compose with docker buildx bake

---

## Pipeline Data Flow

```
[OpenDota API] ──► [ID Fetcher] ──► [queue.match_ids] ──► [Detail Fetcher]
                                                                │
                                                                ▼
                                                        [queue.raw_matches]
                                                                │
                                                                ▼
                                                            [Parser]
                                                                │
                                                                ▼
                                                          [PostgreSQL]
                                                                │
                                                     (materialized views)
                                                                 │
                                                                 ▼
                                                     [Analytics Schema]
                                               (7 agg + 7 PIT snapshot tables)
                                                                │
                                                                ▼
                                                          [Trainer]
                                                    (LightGBM lambdarank)
                                                                │
                                                                ▼
                                                       [ML Models]
                                                    (per-patch .txt files)
                                                                │
                                                                ▼
                                                      [Inference API]
                                                    (FastAPI, port 8080)
```

**Trigger flow** (orchestration):

```
[ID Fetcher] ──(cron, e.g. "0 3 * * *")──► [OpenDota API] ──► [queue.match_ids] ──► [Detail Fetcher]
```

The ID Fetcher owns its own schedule (configurable via `FETCH_SCHEDULE`) and no longer depends on a coordinator service.

---

## Services

### 1. ID Fetcher

**Purpose:** First pipeline stage. Owns its own cron schedule. Queries the OpenDota Explorer API for matches within a rolling N-day window and publishes their IDs to RabbitMQ.

| Aspect | Detail |
|---|---|
| Package | `services/id-fetcher/` |
| Dependencies | Redis (proxy pool), RabbitMQ |
| Config | `config/config.yaml` (env-var substitution) |
| Schedule | `robfig/cron` driven by `FETCH_SCHEDULE` (e.g. `0 3 * * *`, `@every 24h`) |

**Packages:**
- `main.go` — Boots logger, RabbitMQ publisher, Redis-backed proxy pool, OpenDota client, fetcher, metrics server, and `robfig/cron` scheduler. Mutex flag prevents slow runs from overlapping the next tick. Opt-in startup fetch (`ID_FETCHER_START_RUN`) waits for the proxy pool to reach `ID_FETCHER_START_RUN_MIN_POOL_SIZE` before firing a one-shot run on boot, sharing the cron trylock.
- `internal/api/fetcher.go` — Single-shot fetch: calls `client.FetchMatches`, batches results, publishes to `queue.match_ids`
- `internal/api/opendota_client.go` — Executes embedded SQL via OpenDota Explorer API with `pool.WithProxy()`. Retries until the pool is exhausted or the context is cancelled. Uses `proxypool.MakeTransport` for HTTP proxy transport (supports HTTP/HTTPS and SOCKS5). Exits only on `context.Canceled` (not `DeadlineExceeded`) so per-request timeouts rotate to a fresh proxy instead of stopping retries.
- `internal/api/matches.sql` — Embedded query: `WHERE start_time >= (EXTRACT(EPOCH FROM NOW() - INTERVAL '%d days'))::BIGINT AND lobby_type IN (%s)`, where `%d` and `%s` are filled at runtime from `FETCH_LAST_COUNT_DAY` and `FETCH_LOBBY_TYPES`
- `internal/api/matches_watermark.sql` — Watermark-based query used after the parser has committed at least one batch. Uses `%d` placeholders filled at runtime with `fmt.Sprintf` for the lookback window, match_id filter, lobby_types, and LIMIT (configurable via `ID_FETCHER_WATERMARK_LOOKBACK_DAYS`, defaults to `FETCH_LAST_COUNT_DAY`).
- `internal/queue/` — RabbitMQ publisher with confirmed delivery. **Fresh channel per batch** (closed after publish) to prevent `NotifyPublish` listener memory leak. Uses `shutdown` channel for deadlock-safe reconnection (releases `closeMu` before `Close()` to prevent publisher deadlock on stalled TCP connections).
- `internal/config/` — YAML + env expansion. Parses `FETCH_LOBBY_TYPES` (comma-separated) into `[]int`. `watermark_lookback_days` defaults to `fetch_last_count_day` at runtime (not a static 30) so it never fails validation. New fields: `start_run`, `start_run_min_pool_size`, `start_run_max_wait`.
- `internal/metrics/` — `id_fetcher_pagination_runs_total`, `id_fetcher_match_ids_published_total`, `id_fetcher_api_calls_total`

**Key behaviors:**
- Self-scheduled: no external trigger required, no coordinator dependency
- Two query modes: **bootstrap** (rolling window `start_time >= NOW() - N days`) and **watermark** (`match_id > last_parsed_match_id ORDER BY match_id ASC LIMIT N`). Watermark pushed into SQL to guarantee correctness regardless of backlog depth
- Watermark is read on startup from `ingestion_checkpoints.last_parsed_match_id` via the shared `checkpoint` package. If DB is unreachable during bootstrap, falls back to rolling-window path (transient DB outage does not block fetches)
- **DB existence check** (Layer 3): Before publishing match IDs to the queue, checks the `matches` table for existing match IDs to prevent re-publishing already-parsed matches. Uses `matchExistsInDB` with array-based query (`SELECT match_id FROM matches WHERE match_id = ANY($1)`) for batch checking.
- Only fetches ranked (lobby_type 1,2) and normal (6) matches
- Rate-limit avoidance via proxy pool integration; infinite retry until success, pool exhaustion, or shutdown
- Redis-sourced Prometheus metrics via `NewRedisPoolCollector` for accurate pool size (replaces per-process gauge that diverged across services)

---

### 2. Detail Fetcher

**Purpose:** Second pipeline stage. Consumes match IDs, fetches full match JSON from OpenDota API, and publishes raw data to the parser queue.

| Aspect | Detail |
|---|---|
| Package | `services/detail-fetcher/` |
| Port | 9091 (metrics) |
| Dependencies | Redis (proxy pool), RabbitMQ |
| Config | `config/config.yaml` |

**Packages:**
- `internal/api/client.go` — HTTP client for `https://api.opendota.com/api/matches/{id}` with proxy pool integration: `AcquireWithRateLimit`, JSON validation, error classification. Uses `proxypool.MakeTransport` for the HTTP transport (HTTP/HTTPS proxy support only; SOCKS5 not needed for outbound API calls). Includes **direct connection fallback** as last resort before DLQ (attempts a direct HTTP request without proxy when all proxy-based retries fail). This prevents pipeline stalls when the free proxy pool has low validation rates (~3.4%).
- `internal/consumer/` — RabbitMQ consumer for `queue.match_ids` with DLQ binding and QoS. Auto-reconnection loop (1s → 30s exponential backoff).
- `internal/worker/` — Match processing: exponential backoff retries (max 5 — increased from 3 for better proxy rotation), handles `ErrMatchNotFound` (ack + drop), publishes to `queue.raw_matches` on success. Includes `matchExistenceChecker` interface + `postgresMatchChecker` implementation that checks the `matches` table via `SELECT EXISTS(...)` before fetching — skips (Acks) matches already committed to reduce redundant API calls. Logs at `Debug` level on skip, `Warn` on DB error (proceeds with fetch on error to avoid data loss). Configurable concurrency via `DETAIL_FETCHER_WORKER_CONCURRENCY` (default 50).
- `internal/publisher/` — RabbitMQ publisher for `queue.raw_matches` with publisher confirms, mutex-serialized to prevent concurrency trap. Uses `reconnectMu` to serialize reconnection attempts (prevents exchange re-declaration race during concurrent reconnect), and `shutdown` channel to prevent goroutine leaks on close (releases `closeMu` before `Close()` to avoid deadlock).
- `internal/metrics/` — `detail_fetcher_messages_received_total`, `detail_fetcher_fetches_total{result}`, `detail_fetcher_publishes_total{result}`, `detail_fetcher_dlq_routed_total`, `detail_fetcher_skipped_total`

**Key behaviors:**
- **`consumeWithReconnect`**: permanent output channel (never closed on reconnect), workers exit only via `<-ctx.Done()`. Prevents channel-close panics and in-flight message loss.
- Rate-limited proxy acquisition: `max_req_per_min` (default 50) per proxy
- Concurrency: configurable worker pool (default 50, increased from 5 for backfill throughput)
- All retries exhausted → direct connection fallback → routes to DLQ `queue.match_ids.dlq` (only if direct fallback also fails)
- Messages that produce `ErrMatchNotFound` (404) are acknowledged and dropped immediately
- DB existence check (`postgresMatchChecker`): checks `matches` table before fetching; skips and Acks matches already committed. Uses a 4-connection pgx pool separate from the parser's pool.

---

### 3. Parser

**Purpose:** Terminal pipeline stage. Consumes raw match JSON, unmarshals, validates, and batch-inserts into 20+ PostgreSQL tables in a single transaction.

| Aspect | Detail |
|---|---|
| Package | `services/parser/` |
| Port | 9093 (metrics + health) |
| Dependencies | PostgreSQL, RabbitMQ |
| Config | `config/config.yaml` |

**Packages:**
- `internal/consumer/` — RabbitMQ consumer for `queue.raw_matches` with `ConsumeWithReconnect` (auto-reconnection), `NewDLQChannel()` for independent DLQ publishing
- `internal/worker/processor.go` — Batch processor: accumulates messages up to `batch_size` (100) or `fetch_timeout` (2s), validates envelopes and match JSON, handles FK violations with individual fallback, sends poison pills to DLQ
- `internal/repository/batch_writer.go` — Batch insert into 19+ tables per match in a single `pgx.Batch` transaction with `ON CONFLICT DO NOTHING` (idempotency). All child table helpers return errors. Uses `context.WithoutCancel` for I/O to prevent connection corruption on graceful shutdown.
- `internal/models/opendota.go` — All Go structs (~318 lines): `OpenDotaMatch`, `Player`, plus 15 nested event types using `json.RawMessage` for JSONB fields
- `internal/metrics/` — `parser_matches_parsed_total`, `parser_matches_failed_total`, `parser_batch_processing_duration_seconds`, `parser_batch_size`, `parser_dlq_messages_total`

**Tables written per match:**
| Table | Content |
|---|---|
| `raw_matches` | Staging: raw JSON blob, fetched_at, parsed_at |
| `matches` | Match header: duration, game mode, lobby type, region, patch, etc. |
| `players` | Per-player stats: hero, KDA, gold/xp, damage/heal, etc. (RANGE-partitioned by match_id) |
| `picks_bans` | Draft order: hero picks and bans |
| `objectives` | Game events: tower kills, Roshan kills, etc. |
| `chat` | In-game chat messages |
| `match_gold_adv` / `match_xp_adv` | Per-minute gold/xp advantage time series |
| `teamfights` / `teamfight_players` | Teamfight events and participant stats |
| `player_kills_log` / `player_buyback_log` | Kill and buyback event logs |
| `player_runes_log` / `player_purchase_log` | Rune pickups and item purchases (with `seq` column) |
| `player_obs_log` / `player_sen_log` | Observer/sentry ward placement events |
| `player_obs_left_log` / `player_sen_left_log` | Ward destruction events |
| `player_ability_upgrades_log` | Skill level-up events |
| `player_benchmarks` | Hero-specific benchmark percentiles |
| `player_permanent_buffs` | Permanent modifier events (e.g., Aghanim's Shard) |
| `player_neutral_item_history` | Neutral item acquisitions |
| `player_minute_stats` | Per-minute gold/xp stats (gold, xp, last_hits, denies per minute) |
| `player_time_series_arrays` | Minute-by-minute gold/XP arrays (gold_t, xp_t JSONB; PK: match_id, player_slot) |

**Key behaviors:**
- **FK violation fallback**: On SQLSTATE 23503, falls back to per-match inserts. The offending match goes to DLQ; healthy matches commit. Prevents pipeline deadlock on unseeded reference data (e.g., new Valve heroes).
- **Division-by-zero guard**: If `duration.Seconds() <= 0`, clamps to 0.001 before computing `matches_per_sec`.
- **`context.WithoutCancel`**: All batch I/O (`SendBatch`, `Commit`) uses orphaned context with a 30s deadline to prevent connection pool corruption during graceful shutdown.
- **Idempotent inserts**: `ON CONFLICT DO NOTHING` on all tables enables safe retry.
- **Checkpoint watermark**: The checkpoint upsert is queued in the SAME `pgx.Batch` as the match inserts — it runs inside the same transaction, so the watermark only advances when the entire batch commits. Uses `GREATEST(...)` to guarantee monotonicity: a late-arriving batch with a smaller match_id can never rewind the watermark.
- **gold_t/xp_t JSONB arrays**: The parser writes minute-by-minute gold/XP arrays to `player_time_series_arrays` (PK: match_id, player_slot) for early-game feature computation (avg_gold_10, avg_xp_10).  Migration 013 moved these from the `player_minute_stats` minute=0 sentinel to avoid PK collision with real minute-0 rows.

---

### 4. Proxy Manager

**Purpose:** Autonomous proxy pool manager. Fetches proxies from file + remote URL, validates them concurrently, maintains pool health in Redis via refresh/lease-reaper loops.

| Aspect | Detail |
|---|---|
| Package | `services/proxy-manager/` |
| Port | 9090 (metrics + health) |
| Dependencies | Redis |
| Config | Environment variables (no YAML) |

**Packages:**
- `internal/config/` — ~20 env vars loaded via godotenv
- `internal/source/` — `FromFile(path)` and `FromURL(ctx, url)`, both parse lines with `http://` scheme normalization. SOCKS prefix filtering was removed — SOCKS5 support is now handled transparently by `proxypool.MakeTransport`.
- `internal/validator/` — Concurrent proxy validator (150 workers default). `ValidateStream()` uses `proxypool.MakeTransport` for HTTP transport (now supports HTTP, HTTPS, SOCKS4, and SOCKS5 proxy schemes). Performs GET, drains 256 bytes. Reports progress every 30s. Guaranteed goroutine cleanup via `defer close(progressDone)`.

**Key behaviors:**
- **Bootstrap:** Load from local file + remote URL, deduplicate, validate, add survivors to pool
- **Refresh loop:** Periodic fetch from remote source (default 15min), validate fresh proxies, top-up if below min
- **Revalidation:** Full pool revalidation every 60min — validates ALL proxies in Redis, removes dead ones
- **Lease reaper:** Reclaims expired leases every 30s
- **Source-fetch cooldown:** 10-minute guard prevents HTTP 429 on rapid successive fetches
- **Top-up:** If pool drops below `PoolMinSize` (20), fetches more immediately

**Configuration (environment variables):**
| Variable | Default | Description |
|---|---|---|
| `PROXY_FILE_PATH` | `deploy/proxy.txt` | Static proxy list file |
| `PROXY_REFRESH_SOURCE_URL` | *(ProxyScrape URL)* | Remote proxy source (empty = disabled) |
| `PROXY_REFRESH_INTERVAL_MIN` | 15 | Refresh loop interval |
| `PROXY_VALIDATION_CONCURRENCY` | 150 | Concurrent validation workers |
| `PROXY_VALIDATION_TIMEOUT_SEC` | 10 | Per-proxy validation timeout |
| `PROXY_POOL_MAX_SIZE` | 2000 | Maximum pool size |
| `PROXY_POOL_MIN_SIZE` | 20 | Minimum pool size (triggers top-up) |
| `PROXY_LEASE_DURATION_SEC` | 120 | Acquired proxy lease duration |
| `PROXY_ROTATION_STRATEGY` | `timestamp` | Pool strategy (timestamp, random) |
| `PROXY_REVALIDATION_INTERVAL_MIN` | 60 | Full pool revalidation interval (0 = disabled) |

---

### 5. Trainer

**Purpose:** Batch CLI service. Computes patch-aware aggregate tables from PostgreSQL and trains LightGBM lambdarank models for draft prediction.

| Aspect | Detail |
|---|---|---|
| Package | `services/trainer/` |
| Dependencies | PostgreSQL (psycopg2 + SQLAlchemy) |
| Memory | **8G** (patch 60 with 154k draft slots + 14 tables + cross-patch lookback) |
| Config | Environment variables (`TRAINER_*`) |

**Pipeline stages:**
1. **Aggregate population** — **7** populator functions compute `ml.*_agg` tables per patch via bulk INSERT (ON CONFLICT DO UPDATE). Queries aggregate match data grouped by team/hero/player/patch, filtering `WHERE radiant_win IS NOT NULL`. Includes `hero_draft_slot_agg` (pick-position win rates with `WHERE team_pick_ordinal <= 5` filter to handle All Draft game_mode=22 matches with extra picks), and `avg_gold_10`/`avg_xp_10` from gold_t/xp_t JSONB arrays.
2. **Snapshot population** — **7** additional populator functions compute `ml.*_snapshot` tables at point-in-time granularity. Daily-precision tables (`team_hero`, `hero_baseline`, `hero_draft_slot`, `team_h2h`) use per-snapshot-date buckets; weekly tables (`player_hero`, `synergy`, `counter`) use 7-day buckets for row-count management. The 4 sparse combo-keyed snapshot tables (`team_hero`, `player_hero`, `synergy`, `counter`) include **cross-patch lookback** (`lookback_patches=2`, `prior_patch_weight=0.5`) that reweights prior-patch games to combat hero-combo sparsity. `team_hero_snapshot` uses a two-phase query (pre-computed prior-patch aggregate + per-date current-patch PIT LATERAL) to avoid a 5M-row cross-product. The 3 dense tables (`hero_baseline`, `hero_draft_slot`, `team_h2h`) stay single-patch. `games`/`wins` are FLOAT on the 4 lookback tables (fractional from weighted counts); INT on the 3 dense tables.
3. **Feature extraction** — `TRAINING_FEATURES_SQL` computes **70 aggregate + 160 one-hot hero ID = 230-dim feature vectors** via a single query with `LATERAL` subqueries. Includes:
   - Team/Player hero aggregates from PIT-safe snapshots
   - Synergy/counter stats from already-picked allies/enemies
   - Head-to-head records and hero baseline stats
   - **Recent form**: Last 20 games win rate, games played, KDA (recency-weighted)
   - **Meta drift**: Win rate and pick rate delta over 7-day rolling window
   - **Sequence context**: Pick position, team picks so far, enemy picks so far
   - **Team strategy**: Push/gank/fight scores from team's hero history
   - Low-game missingness flags, delta features, role interactions
4. **Training** — LightGBM **binary classification** with early stopping (50 rounds), 85/15 chronological train/val split. **Platt scaling calibration** via LogisticRegression post-training for calibrated probabilities.
5. **Output** — Model (`model_patch_N.txt`), calibrator (`calibrator_patch_N.json`), metadata, and `feature_schema_patch_N.json` (230-column contract).
6. **Validation** — Patch 60 achieves: **binary_logloss 0.6894** with 230 features (was 0.6885 with 219). New features add context for recent form, meta shifts, and team strategies.

**Key behaviors:**
- Idempotent: aggregates use INSERT ON CONFLICT DO UPDATE
- Feature column order frozen in `feature_schema.json` (230 columns); API loads this to guarantee agreement
- Platt scaling calibration: raw LightGBM scores → calibrated 0-1 probabilities
- Draft phase reconstruction uses per-patch patterns from `DRAFT_PATTERNS` dict
- Bayesian shrinkage applied to win rates (`prior_games` / `prior_win_rate`)
- Target is relative to the picking team: `(df["radiant_win"] == (df["team"] == 0)).astype(int)`
- `POST /predict` accepts optional `account_id` and `radiant_team_id`/`dire_team_id` for personalized predictions
- Response includes calibrated `pick_probability` and `win_probability` fields

---

### 6. Inference API

**Purpose:** Online FastAPI service. Loads trained LightGBM models and serves draft predictions via HTTP.

| Aspect | Detail |
|---|---|
| Package | `services/api/` |
| Port | 8080 (health `/health`, predict `/predict`) |
| Dependencies | PostgreSQL (psycopg2 `ThreadedConnectionPool`) |
| Config | Environment variables (`API_*`) |

**Endpoints:**
- `GET /health` — Returns `{"status":"ok","patch_models_loaded":[...]}`
- `POST /predict` — Accepts `patch_id`, `first_pick_team`, `draft[]`, `radiant_team_id`, `dire_team_id`, optional `account_id`, optional `num_recommendations`. Returns top-5 hero scores with **reasoning** string
- `POST /reload/{patch_id}` — Hot-reload a model (requires `STRATZ_ADMIN_TOKEN` in Bearer auth header)

**Prediction flow:**
1. Client sends current draft state (picks/bans, teams, patch) with optional `account_id` for personalized player-hero features
2. API validates draft order against per-patch pattern, reconstructs `DraftContext`
3. Pre-fetches batch aggregate data (baselines, team-hero, H2H) per request using six batch queries
4. For each candidate hero not yet picked/banned, builds **230-dim feature vector** (70 aggregate + 160 one-hot hero ID) and scores via LightGBM
5. Applies Platt scaling calibration for calibrated win probabilities
6. Re-ranks using 1-ply look-ahead minimax search
7. Returns top-N recommendations with calibrated `pick_probability` and `win_probability`

**Feature categories (230 total):**
- `th_*` (15): Team-hero aggregate — team's historical stats with this hero
- `ph_*` (15): Player-hero aggregate — player's personal stats with this hero
- `sy_*` (2): Synergy — win rate when paired with already-picked allies
- `co_*` (3): Counter — win rate when facing already-picked enemies
- `h2h_*` (2): Head-to-head — team vs team historical record
- `bl_*` (12): Hero baseline — global pick/ban/win rates
- `hds_*` (2): Hero draft-slot — win rate at this pick position
- `rf_*` (3): Recent form — last 20 games win rate, games, KDA
- `md_*` (2): Meta drift — win rate and pick rate trend over 7 days
- `seq_*` (3): Sequence context — pick position, team/enemy picks so far
- `ts_*` (3): Team strategy — push/gank/fight scores from team history
- `is_pick`, `team` (2): Draft context flags
- `oh_hero_*` (160): One-hot hero ID encoding

**Model improvements (v2):**
- Platt scaling calibration: raw scores → calibrated 0-1 probabilities
- 1-ply look-ahead minimax: evaluates opponent's best response before recommending
- Recent form features: captures improving/declining players
- Meta drift features: adapts to win rate trends within a patch
- Team strategy features: push/gank/fight pattern recognition

**Key behaviors:**
- Thread-safe: uses `psycopg2.pool.ThreadedConnectionPool` guarded by `threading.Lock`
- Feature computation mirrors trainer logic exactly (same 230-column order via `feature_schema.json`)
- NULL-safe: all aggregate lookups guarded by `_float()`/`_int()` helpers
- Draft patterns dynamically selected by `patch_id` from per-patch dict
- Optional `account_id` enables player-hero aggregate features

---

## Shared Library

**Module:** `github.com/dota-stratz/shared/go-common` at `shared/go-common/`

### Packages

| Package | File(s) | Exports | Description |
|---|---|---|---|
| `cache` | `redis.go` | `Connect(addr, password, db)` | Redis connection with 3-retry ping |
| `db` | `postgres.go` | `Connect(ctx, dsn)` | pgxpool connection with ping |
| `mq` | `rabbitmq.go`, `queue.go`, `consumer.go`, `publisher.go` | `Connect(url)`, `QueueConfig`, `DeclareQueueWithDLQ(ch, cfg)`, `Consumer`, `Publisher` | AMQP connection + channel. Extended with shared queue declaration (`DeclareQueueWithDLQ`), auto-reconnecting `Consumer` with `ConsumeWithReconnect`, and `Publisher` with publisher confirms + automatic reconnect. Previously each service duplicated reconnection logic and queue declaration; these are now centralized in the shared package. |
| `logger` | `logger.go` | `InitLogger()`, `Sync()`, `Log` | Global zap.Logger from `LOG_LEVEL` |
| `checkpoint` | `checkpoint.go` | `ReadWatermark(ctx, pool)`, `CheckpointPipelineParser`, `CheckpointPipelineIDFetcher` | Shared constants + SQL for `ingestion_checkpoints` table — single source of truth for `last_parsed_match_id` column, used by parser (writer) and id-fetcher (reader) |
| `proxypool` | `pool.go`, `classify.go`, `metrics.go`, `transport.go`, `socks4.go`, `redis_collector.go` | `Pool`, `Acquire`, `Release`, `WithProxy`, `Report`, `MakeTransport`, `NewRedisPoolCollector` | Redis-backed proxy pool (~710 lines). `MakeTransport(proxyStr, timeout)` builds an `*http.Transport` for any scheme: HTTP/HTTPS CONNECT, SOCKS5, SOCKS4 (native dialer). `NewRedisPoolCollector` provides Redis-ground-truth Prometheus metrics. Shared by all services that make HTTP requests through proxies. |

### proxypool — Redis data structures

```
dota2:proxies                  ZSET   — available proxies (score = timestamp)
dota2:proxies:leases           HASH  — proxy → lease-expiry mapping
dota2:proxies:failures:{hash} STRING — failure count
dota2:proxies:cooldown:{hash} STRING — cooldown expiry
dota2:proxies:ratelimit:{hash} STRING — rate-limit tracking
```

**Key features:**
- Atomic Lua scripts for acquire, release, and rate-limit ops
- Failure classification: `HardFailure`/`BadStatus` → permanent removal, `RateLimited` → cooldown, `Timeout` → incremental counter → removal at threshold (3)
- `AcquireWithRateLimit(maxPerMin)` — enforces per-proxy rate limits
- Prometheus metrics: pool size gauges, removed/cooldown/reap counters, validation latency histograms
- `crc64.Checksum` + `base36` for proxy hash keys (~10× faster than SHA256+hex)
- `UnixMicro()` for ZSET scores (avoids float64 precision loss from UnixNano)

### mq — RabbitMQ shared patterns

```
shared/go-common/mq/
├── rabbitmq.go    — Connect(url) — basic AMQP connection + channel
├── queue.go       — QueueConfig, DeclareQueueWithDLQ(ch, cfg) — queue + DLQ declaration
├── consumer.go    — Consumer with ConsumeWithReconnect — auto-reconnecting consumer
└── publisher.go   — Publisher with confirms + reconnect — reconnecting publisher
```

**Key features:**
- **`QueueConfig`** — Single struct holding queue name, DLQ name, and message TTL. Provides a single source of truth for queue topology configuration.
- **`DeclareQueueWithDLQ(ch, cfg)`** — Idempotent queue + DLQ declaration with dead-letter exchange binding and 24h TTL on the DLQ. Replaces duplicated declarations across 4 service files.
- **`Consumer`** — Wraps a RabbitMQ connection with automatic reconnection. `ConsumeWithReconnect(done, tag)` returns a channel that survives broker restarts (exponential backoff 1s → 30s). The channel is never closed on reconnect — only on permanent shutdown via the `done` channel.
- **`Publisher`** — Wraps a RabbitMQ connection with publisher confirms and automatic reconnection. Thread-safe `Publish(ctx, queue, body)` with 5s confirm timeout. Handles connection loss transparently via background `NotifyClose` listener. Mutex-layered reconnection (`closeMu`, `publishMu`, `reconnectMu`) prevents deadlocks and exchange races during concurrent reconnect.

**Refactored services** — Previously each service duplicated queue declaration, reconnection logic, and publisher confirm setup. These are now thin wrappers around the shared mq primitives:

| Service | File | Wraps |
|---------|------|-------|
| Parser | `consumer/consumer.go` | `mq.Consumer` with `ConsumeWithReconnect` |
| Detail Fetcher | `consumer/consumer.go` | `mq.Consumer` (manual reconnect loop preserves shutdown ordering) |
| Detail Fetcher | `publisher/publisher.go` | `mq.Publisher` with `RawMatchMessage` marshaling |
| ID Fetcher | `queue/rabbitmq_publisher.go` | `mq.DeclareQueueWithDLQ` + `mq.Connect` (keeps own per-batch channel pattern for `NotifyPublish` cleanup) |

---

## Database Schema

### Migration Files

| File | Description |
|---|---|---|
| `001__init.sql` | **Core schema**: `matches`, `players` (RANGE-partitioned, 5B per partition, up to 30B + catchall), all child event tables, indexes, `ingestion_checkpoints`, partition management functions + initialization. FKs are `DEFERRABLE INITIALLY DEFERRED` for batch-insert throughput. |
| `002_ml.sql` | **Analytics + ML schema**: Bayesian shrinkage config, materialized views (`mv_team_hero_profile`, `mv_hero_synergy`, `mv_hero_counter`, `mv_player_team_history`), roles/grants, 7 aggregate tables (`*_agg`), 7 PIT-safe snapshot tables (`*_snapshot` with BIGINT team_ids, FLOAT games/wins on sparse tables), 8 DESC lookup indexes, partition verification, feature schema functions. |
| `003_static.sql` | **Static reference data**: `const_game_mode`, `const_lobby_type`, `const_region`, `const_patch` (up to 7.41), `const_hero`, `const_item`, `const_ability` — table definitions + seed data combined. Internal FK constraints within constants schema. |

### Table Structure

**Core tables:**
- `matches` — One row per match with match_id PK, duration, game_mode, lobby_type, region, patch, scores, objectives, draft info
- `players` — RANGE-partitioned by match_id (5B per partition). PK: `(match_id, player_slot)`. Hero, KDA, gold/xp, damage, healing, items, camps stacked, etc.

**Child tables (linked via match_id + player_slot):**
- Event logs: `player_kills_log`, `player_buyback_log`, `player_runes_log`, `player_purchase_log`, `player_obs_log`, `player_sen_log`, `player_obs_left_log`, `player_sen_left_log`, `player_ability_upgrades_log`, `player_permanent_buffs`, `player_neutral_item_history`
- Match-level: `picks_bans` (PK: match_id, order), `objectives` (PK: match_id, time, type, team), `chat` (PK: match_id, time, slot)
- Time series: `match_gold_adv`, `match_xp_adv`
- Teamfights: `teamfights`, `teamfight_players`
- Staging: `raw_matches` (match_id PK, raw_json JSONB, fetched_at, parsed_at)
- Benchmarks: `player_benchmarks`
- Summary: `player_minute_stats`

**Analytics schema:**
- `shrinkage_config` — Bayesian prior parameters per stat
- Materialized views: team hero profiles, hero synergies/counters, player team history, player hero profiles
- `feature_snapshots_player_hero` — Point-in-time feature snapshots for ML
- `featurizer_runs` — Tracking table for snapshot generation runs

**ML schema (`ml`):**
- **7** UNLOGGED patch-aware **aggregate tables** (`team_hero_agg`, `player_hero_agg`, `hero_synergy_agg`, `hero_counter_agg`, `team_h2h_agg`, `hero_baseline_agg`, `hero_draft_slot_agg`) — population-wide aggregates, all columns INT, no PIT awareness. Tables are re-populated on each `make train` run (not incrementally maintained).
- **7** LOGGED **PIT-safe snapshot tables** (`team_hero_snapshot`, `player_hero_snapshot`, `hero_synergy_snapshot`, `hero_counter_snapshot`, `team_h2h_snapshot`, `hero_baseline_snapshot`, `hero_draft_slot_snapshot`) — per-date-bucket materializations with `as_of_date` column. Training features use LATERAL "most recent snapshot AS OF match start" lookups for PIT safety. Populated on each `make train` run.
- **Snapshot table design:**
  - **Two-tier resolution** — Tier 1 (daily): `team_hero`, `hero_baseline`, `hero_draft_slot`, `team_h2h`. Tier 2 (weekly): `player_hero`, `synergy`, `counter` — weekly keeps player_hero row count manageable (avoids hero×account×date combinatorics).
  - **Cross-patch lookback** — 4 sparse tables (`team_hero`, `player_hero`, `synergy`, `counter`) aggregate prior-patch matches with `patch_weight = 0.5` to fill NULL combos. 3 dense tables stay single-patch.
  - **Column types** — `games`/`wins` are `FLOAT` on lookback tables (fractional from prior_weight), `INT` on dense tables.
- `team_hero_agg` — Team+hero historical stats (games, wins, bans, avg GPM/XPM/KDA, firstblood_rate, camps_stacked, vision_placed, avg_gold_10, avg_xp_10)
- `player_hero_agg` — Per-account hero stats (lane role, avg KDA, firstblood_rate, avg_gold_10, avg_xp_10)
- `hero_synergy_agg` — Pairwise synergy win rate on same team (keyed by `LEAST(hero_a, hero_b)`)
- `hero_counter_agg` — Pairwise counter win rate vs enemy hero (incl. avg_kd_diff)
- `team_h2h_agg` — Head-to-head win rate between team pairs
- `hero_baseline_agg` — Global hero pick/ban rates, avg stats per patch (incl. avg_gold_10, avg_xp_10)
- `hero_draft_slot_agg` — Hero win rate per team-pick ordinal (1st pick through 5th pick)

**Checkpoint:**
- `ingestion_checkpoints` — Singleton row (id=1) tracking `fetch_status`, `checkpoint_timestamp`, `last_completed_match_id`, `fetch_progress`, `parse_progress`

---

## Deployment

### Docker Compose Profiles

| Profile | Services | Make target |
|---|---|---|
| `all` | Everything | `make up` / `make up-d` |
| `db` | postgres, rabbitmq, redis | `make up-db` / `make up-db-d` |
| `mon` | prometheus, grafana | `make up-mon` |
| `proxy` | proxy-manager (+ db) | `make up-proxy` |
| `fetcher` | id-fetcher, detail-fetcher (+ db) | `make up-fetcher` |
| `parser` | parser (+ db) | `make up-parser` |
| `api` | ml-inference-api (+ db) | `make up-api` / `make up-api-d` |
| `train` | ml-trainer (+ db) | `make train PATCH=N` |

ML targets that need PostgreSQL use `--profile db --profile api` or `--profile db --profile train`.

### Resource Limits
| Service | Memory | CPUs | Notes |
|---------|--------|------|-------|
| postgres | **6G** | **4.0** | shared_buffers=2GB, work_mem=128MB, maintenance_work_mem=512MB |
| rabbitmq | 512M | 1.0 | Increased from 256M to handle burst load from 200 concurrent workers |
| parser | **1G** | 1.0 | Increased from 256M to accommodate 100-match JSON batch processing (prevents OOM kills) |
| trainer | **8G** | **4.0** | Patch 60 (154k draft slots + 14 tables + cross-patch lookback) |
| api | 512M | 0.5 | |
| Most others | 128-256M | 0.5 | |

### Physical DB Backup / Recovery

For fast backup and restore of the PostgreSQL data directory (much faster than `pg_dump` for large databases):

```bash
make db-backup-physical                      # Snapshot to ./backups/pgdata_*.tar
make db-backups                              # List existing snapshots
make db-restore-physical DUMP=pgdata_xxx.tar  # Restore from snapshot
```

Postgres is briefly stopped during the operation to ensure filesystem-level consistency. Requires the `alpine` Docker image.

### Network

All services connect via `dota2-net` (bridge network). Each service is addressable by its Docker Compose service name (e.g., `prometheus`, `grafana`, `parser`).

### Operations

```bash
make check              # Format + vet + test (pre-commit gate)
make logs-parser        # Tail a specific service's logs
make replay-dlq         # Replay up to 500 match IDs from dead-letter queue
make replay-dlq-n N=1000   # Replay N messages from DLQ
make downv              # Stop services and remove project volumes
```

### Monitoring

**Prometheus scrape targets** (via Docker bridge network):
| Target | Port | Service |
|---|---|---|
| `proxy-manager:9090` | 9090 | Proxy Manager |
| `detail-fetcher:9091` | 9091 | Detail Fetcher |
| `localhost:9092` | 9092 | Prometheus self |
| `parser:9093` | 9093 | Parser |
| `id-fetcher:9094` | 9094 | ID Fetcher |
| `rabbitmq:15692` | 15692 | RabbitMQ |
| `api:8080` | 8080 | ML API |

**Alerting rules** (3 pre-configured):
| Alert | Condition | Severity |
|---|---|---|
| `ProxyPoolDepleted` | `dota2_proxy_pool_available < 20` for 2m | warning |
| `IngestionStalled` | `time() - max(timestamp(id_fetcher_match_ids_published_total > 0)) > 93600` for 5m | warning |
| `DLQDepthGrowing` | `sum(rabbitmq_queue_messages_ready{queue=~".*\\.dlq"}) > 50` for 5m | warning |

**Grafana:** Pre-provisioned datasource (Prometheus at `prometheus:9092`) and "Proxy Manager Overview" dashboard with panels for pool health, validation latency (p50/p95/p99), removal reasons, and rate-limiting.

---

## Key Architectural Patterns

1. **Event-driven pipeline** — Services communicate exclusively through RabbitMQ queues. No direct service-to-service RPC.

2. **Proxy pool abstraction** — All OpenDota API calls go through the Redis-backed proxy pool for automatic IP rotation, rate-limit avoidance, and failure classification. Proxies managed independently by proxy-manager.

3. **Idempotent ingestion** — All DB inserts use `ON CONFLICT DO NOTHING` to make repeated fetches safe.

4. **Self-scheduled ingest** — The ID Fetcher owns its own `robfig/cron` schedule; the rest of the pipeline is purely reactive.

5. **Dead-letter queues** — Every RabbitMQ queue has a DLQ with 24h TTL. Poison messages routed there for manual inspection.

6. **Graceful shutdown** — All services handle SIGINT/SIGTERM, drain in-flight work, use bounded wait groups with timeouts. Publisher reconnection loops are interruptible via `shutdown` channels (closed before mutex acquisition) so `Close()` never deadlocks with an in-progress reconnect.

7. **Automatic reconnection** — RabbitMQ consumers and publishers survive broker restarts via reconnection loops with exponential backoff. Output channels are never closed on reconnect; workers exit via context cancellation. Publisher reconnections are serialized by `reconnectMu` to prevent the `exchange` race (two goroutines closing each other's fresh connection). The `currentCons` pointer in the consumer reconnect goroutine is protected by a mutex to prevent data races during the shutdown window.

8. **Deferrable FK constraints** — Foreign keys on bulk-insert tables are `DEFERRABLE INITIALLY DEFERRED` to avoid row-level lock contention during batch writes.

9. _Removed — ID Fetcher now uses a rolling time window (`start_time >= NOW() - N days`) instead of composite-cursor pagination. Each cron tick is fully self-contained._

10. **FK violation isolation** — Parser detects SQLSTATE 23503 and falls back to per-match inserts, routing only the offending match to DLQ rather than requeueing the entire batch.

11. **Analytics layering** — Separate `analytics` schema provides Bayesian-shrunk win rates, hero synergies/counters, and feature snapshots. ML features now use **PIT-safe snapshot tables** (7 `ml.*_snapshot` tables with `as_of_date` buckets) queried via LATERAL "most recent AS OF match start" lookups, replacing the old flat joins that leaked future information.

12. **Monitoring-first** — Every service exposes Prometheus metrics. Three pre-configured alerts and a Grafana dashboard ship with the deployment.

13. **ML offline training + online inference** — Training is a batch CLI (Trainer) that computes 7 aggregate tables + 7 PIT snapshot tables and trains models offline; inference is a stateless HTTP service (API) that loads pre-trained models. The contract between them is `feature_schema.json` — a frozen **219-column** column-order manifest (58 snapshot aggregate + 1 playing-side indicator + 160 one-hot hero ID) written at training time and loaded at API startup. Models are per-patch to capture meta shifts between Dota 2 balance patches.
14. **Checkpoint-based watermarking** — The parser writes `last_parsed_match_id` to `ingestion_checkpoints` in the same transaction as the match batch (via `GREATEST()` upsert for monotonicity). The ID Fetcher reads this watermark on startup to switch from the rolling-window bootstrap path to the incremental watermark path (`match_id > last_parsed_match_id ORDER BY match_id ASC LIMIT N`), preventing re-fetch of already-parsed matches.

15. **PIT-safe snapshot aggregates with cross-patch lookback** — ML training uses 7 snapshot tables with per-date-bucket materializations queried via LATERAL "most recent AS OF match start" to prevent label leakage. **Two-tier resolution**: daily tables (team_hero, hero_baseline, hero_draft_slot, team_h2h) for granularity where row counts are manageable; weekly tables (player_hero, synergy, counter) to avoid hero×account×date combinatorics. **Cross-patch lookback** on 4 sparse combo-keyed tables (`lookback_patches=2`, `prior_patch_weight=0.5`) reweights prior-patch games to fractional FLOAT values, filling NULL combos that would otherwise fall to COALESCE defaults. Dense tables (>98% hit rate) stay single-patch. The daily `team_hero_snapshot` uses a two-phase query (pre-computed prior-patch aggregate + per-date PIT LATERAL) to avoid a 5M-row cross-product of dates × matches.
