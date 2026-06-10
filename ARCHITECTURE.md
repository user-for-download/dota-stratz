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
                                                    (ML aggregate tables)
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
- `internal/api/matches_watermark.sql` — Watermark-based query used after the parser has committed at least one batch. Uses `%%d` placeholder for the lookback window (configurable via `ID_FETCHER_WATERMARK_LOOKBACK_DAYS`, defaults to `FETCH_LAST_COUNT_DAY`).
- `internal/queue/` — RabbitMQ publisher with confirmed delivery. **Fresh channel per batch** (closed after publish) to prevent `NotifyPublish` listener memory leak. Uses `shutdown` channel for deadlock-safe reconnection (see Bug #1/#4).
- `internal/config/` — YAML + env expansion. Parses `FETCH_LOBBY_TYPES` (comma-separated) into `[]int`. `watermark_lookback_days` defaults to `fetch_last_count_day` at runtime (not a static 30) so it never fails validation. New fields: `start_run`, `start_run_min_pool_size`, `start_run_max_wait`.
- `internal/metrics/` — `id_fetcher_pagination_runs_total`, `id_fetcher_match_ids_published_total`, `id_fetcher_api_calls_total`

**Key behaviors:**
- Self-scheduled: no external trigger required, no coordinator dependency
- Rolling time window (`start_time >= NOW() - N days`) — single query per cron tick, no pagination
- Only fetches ranked (lobby_type 1,2) and normal (6) matches
- Rate-limit avoidance via proxy pool integration; infinite retry until success, pool exhaustion, or shutdown

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
- `internal/api/client.go` — HTTP client for `https://api.opendota.com/api/matches/{id}` with proxy pool integration: `AcquireWithRateLimit`, JSON validation, error classification. Uses `proxypool.MakeTransport` for the HTTP transport (HTTP/HTTPS proxy support only; SOCKS5 not needed for outbound API calls).
- `internal/consumer/` — RabbitMQ consumer for `queue.match_ids` with DLQ binding and QoS. Auto-reconnection loop (1s → 30s exponential backoff).
- `internal/worker/` — Match processing: exponential backoff retries (max 3), handles `ErrMatchNotFound` (ack + drop), publishes to `queue.raw_matches` on success
- `internal/publisher/` — RabbitMQ publisher for `queue.raw_matches` with publisher confirms, mutex-serialized to prevent concurrency trap. Uses `reconnectMu` to serialize reconnection attempts (prevents exchange race — Bug #3), and `shutdown` channel to prevent goroutine leaks on close (Bug #2).
- `internal/metrics/` — `detail_fetcher_messages_received_total`, `detail_fetcher_fetches_total{result}`, `detail_fetcher_publishes_total{result}`, `detail_fetcher_dlq_routed_total`

**Key behaviors:**
- **`consumeWithReconnect`**: permanent output channel (never closed on reconnect), workers exit only via `<-ctx.Done()`. Prevents channel-close panics and in-flight message loss.
- Rate-limited proxy acquisition: `max_req_per_min` (default 50) per proxy
- Concurrency: configurable worker pool (default 5)
- All retries exhausted → routes to DLQ `queue.raw_matches.dlq`
- Messages that produce `ErrMatchNotFound` (404) are acknowledged and dropped immediately

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

**Key behaviors:**
- **FK violation fallback**: On SQLSTATE 23503, falls back to per-match inserts. The offending match goes to DLQ; healthy matches commit. Prevents pipeline deadlock on unseeded reference data (e.g., new Valve heroes).
- **Division-by-zero guard**: If `duration.Seconds() <= 0`, clamps to 0.001 before computing `matches_per_sec`.
- **`context.WithoutCancel`**: All batch I/O (`SendBatch`, `Commit`) uses orphaned context to prevent connection pool corruption during graceful shutdown.
- **Idempotent inserts**: `ON CONFLICT DO NOTHING` on all tables enables safe retry.

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
| `PROXY_LEASE_DURATION_SEC` | 60 | Acquired proxy lease duration |
| `PROXY_ROTATION_STRATEGY` | `timestamp` | Pool strategy (timestamp, random) |

---

### 5. Trainer

**Purpose:** Batch CLI service. Computes patch-aware aggregate tables from PostgreSQL and trains LightGBM lambdarank models for draft prediction.

| Aspect | Detail |
|---|---|
| Package | `services/trainer/` |
| Dependencies | PostgreSQL (psycopg2 + SQLAlchemy) |
| Memory | ~2G (patch 58 with 372k draft slots) |
| Config | Environment variables (`TRAINER_*`) |

**Pipeline stages:**
1. **Aggregate population** — 6 populator functions compute `ml.*_agg` tables per patch via bulk INSERT (TRUNCATE + insert). Queries aggregate match data grouped by team/hero/player/patch.
2. **Feature extraction** — `TRAINING_FEATURES_SQL` computes 196-dim feature vectors (36 aggregate columns + 160 one-hot hero ID) with `LEAST`/`GREATEST` index-friendly joins (~11s for 108k draft slots). NULL-safe with `COALESCE`.
3. **Training** — LightGBM lambdarank with NDCG evaluation, 80/15 train/val split at match level. Writes model, metadata JSON, and `feature_schema.json` (column order contract with API).
4. **Output** — Model per patch (`model_patch_N.txt`), metadata (`model_patch_N_meta.json`), and shared `feature_schema.json` to `/models` volume.

**Key behaviors:**
- Idempotent: aggregates are TRUNCATE + re-insert on each run
- Feature column order is frozen in `feature_schema.json` at training time; API loads this to guarantee agreement
- Draft phase reconstruction uses per-patch patterns from `DRAFT_PATTERNS` dict with a normalizer that ensures 0 = first-pick team
- Bayesian shrinkage applied to win rates (`prior_games` / `prior_win_rate` config)

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
- `GET /health` — Returns status and list of loaded model patches
- `POST /predict` — Accepts draft state (`patch_id`, `first_pick_team`, `draft[]`), returns top-5 hero recommendations with model scores
- `POST /reload/{patch_id}` — Hot-reload a model (requires admin token)

**Prediction flow:**
1. Client sends current draft state (picks/bans, teams, patch)
2. API validates draft order against per-patch pattern, reconstructs `DraftContext`
3. Pre-fetches batch aggregate data (baselines, team-hero, H2H) per request
4. For each candidate hero not yet picked/banned, builds 196-dim feature vector (matching trainer's schema) and scores via LightGBM
5. Returns top-5 recommendations sorted by score

**Key behaviors:**
- Thread-safe: uses `psycopg2.pool.ThreadedConnectionPool` for concurrent requests
- Feature computation mirrors trainer logic exactly (same column order via `feature_schema.json`)
- NULL-safe: all aggregate lookups guarded by `_float()`/`_int()` helpers with sensible defaults
- Draft patterns dynamically selected by `patch_id` from per-patch dict (supports patches 8–60 with fallback)

---

## Shared Library

**Module:** `github.com/dota-stratz/shared/go-common` at `shared/go-common/`

### Packages

| Package | File(s) | Exports | Description |
|---|---|---|---|
| `cache` | `redis.go` | `Connect(addr, password, db)` | Redis connection with 3-retry ping |
| `db` | `postgres.go` | `Connect(ctx, dsn)` | pgxpool connection with ping |
| `mq` | `rabbitmq.go` | `Connect(url)` | AMQP connection + channel |
| `logger` | `logger.go` | `InitLogger()`, `Sync()`, `Log` | Global zap.Logger from `LOG_LEVEL` |
| `proxypool` | `pool.go`, `classify.go`, `metrics.go`, `transport.go` | `Pool`, `Acquire`, `Release`, `WithProxy`, `Report`, `MakeTransport` | Redis-backed proxy pool (~710 lines). `MakeTransport(proxyStr, timeout)` builds an `*http.Transport` for any scheme: HTTP/HTTPS CONNECT, SOCKS5, SOCKS4. Shared by all services that make HTTP requests through proxies. |

### proxypool — Redis data structures

```
dota2:proxies                  ZSET   — available proxies (score = timestamp)
dota2:proxies:leases           HASH  — proxy → lease-expiry mapping
dota2:proxies:failures:{proxy} STRING — failure count
dota2:proxies:cooldown:{proxy} STRING — cooldown expiry
dota2:proxies:ratelimit:{proxy} STRING — rate-limit tracking
```

**Key features:**
- Atomic Lua scripts for acquire, release, and rate-limit ops
- Failure classification: `HardFailure`/`BadStatus` → permanent removal, `RateLimited` → cooldown, `Timeout` → incremental counter → removal at threshold (3)
- `AcquireWithRateLimit(maxPerMin)` — enforces per-proxy rate limits
- Prometheus metrics: pool size gauges, removed/cooldown/reap counters, validation latency histograms
- `crc64.Checksum` + `base36` for proxy hash keys (~10× faster than SHA256+hex)
- `UnixMicro()` for ZSET scores (avoids float64 precision loss from UnixNano)

---

## Database Schema

### Migration Files

| File | Description |
|---|---|---|
| `001_core.sql` | Core schema: `matches`, `players` (RANGE-partitioned), all child event tables, indexes, `ingestion_checkpoints`, partition management functions + initialization. FKs are `DEFERRABLE INITIALLY DEFERRED` for batch-insert throughput. |
| `002_constants.sql` | Static reference data: `const_game_mode`, `const_lobby_type`, `const_region`, `const_patch`, `const_hero`, `const_item`, `const_ability` — table definitions + seed data combined. Internal FK constraints within constants schema. |
| `003_analytics.sql` | Analytics schema: Bayesian shrinkage config, materialized views (`mv_team_hero_profile`, `mv_hero_synergy`, `mv_hero_counter`, `mv_player_team_history`), `feature_snapshots_player_hero`, `featurizer_runs`, `refresh_all_mv()`, `update_feature_snapshots()`, roles. |
| `004_partition_verify.sql` | Idempotent assertion that the 6 expected `players` partitions exist. Performs no schema changes on a healthy database — safety check for CI/testing. |
| `005_ml_tables.sql` | 6 patch-aware ML aggregate tables in `ml` schema: `team_hero_agg`, `player_hero_agg`, `hero_synergy_agg`, `hero_counter_agg`, `team_h2h_agg`, `hero_baseline_agg`. All UNLOGGED for write speed. |
| `006_postgres_best_practices_fixes.sql` | Runtime fixes: grants on `ml` schema, `grant_ml_access()` function for new users. |

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
- 6 UNLOGGED patch-aware aggregate tables populated per-patch by the Trainer
- `team_hero_agg` — Team+hero historical stats (games, wins, bans, avg GPM/XPM/KDA)
- `player_hero_agg` — Per-account hero stats (lane role, avg KDA)
- `hero_synergy_agg` — Pairwise synergy win rate on same team (keyed by `LEAST(hero_a, hero_b)`)
- `hero_counter_agg` — Pairwise counter win rate vs enemy hero
- `team_h2h_agg` — Head-to-head win rate between team pairs
- `hero_baseline_agg` — Global hero pick/ban rates, avg stats per patch
- Tables are re-populated on each `make train` run (not incrementally maintained)

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
| postgres | 512M | 1.0 | |
| trainer | **2G** | 2.0 | Patch 58 (372k draft slots) requires ~1.6G |
| api | 512M | 0.5 | |
| Most others | 128-256M | 0.5 | |

### Physical DB Backup / Recovery

For fast backup and restore of the PostgreSQL data directory (much faster than `pg_dump` for large databases):

```bash
make db-backup-physical                      # Snapshot to ./backups/pgdata_*.tar.gz
make db-backups                              # List existing snapshots
make db-restore-physical DUMP=pgdata_xxx.tar.gz  # Restore from snapshot
```

Postgres is briefly stopped during the operation to ensure filesystem-level consistency. Requires the `alpine` Docker image.

### Network

All services connect via `dota2-net` (bridge). Prometheus and Grafana use `network_mode: host`.

### Operations

```bash
make check              # Format + vet + test (pre-commit gate)
make logs-parser        # Tail a specific service's logs
make replay-dlq         # Replay up to 500 match IDs from dead-letter queue
make replay-dlq-n N=1000   # Replay N messages from DLQ
make downv              # Stop services and remove project volumes
```

### Monitoring

**Prometheus scrape targets** (via host networking):
| Target | Port | Service |
|---|---|---|
| `localhost:9090` | 9090 | Proxy Manager |
| `localhost:9091` | 9091 | Detail Fetcher |
| `localhost:9092` | 9092 | Prometheus self |
| `localhost:9093` | 9093 | Parser |
| `localhost:9094` | 9094 | ID Fetcher |

**Alerting rules** (3 pre-configured):
| Alert | Condition | Severity |
|---|---|---|
| `ProxyPoolDepleted` | `dota2_proxy_pool_available < 20` for 2m | warning |
| `IngestionStalled` | `time() - max(timestamp(id_fetcher_match_ids_published_total > 0)) > 93600` for 5m | warning |
| `DLQDepthGrowing` | `sum(rabbitmq_queue_messages_ready{queue=~".*\\.dlq"}) > 50` for 5m | warning |

**Grafana:** Pre-provisioned datasource (Prometheus at `localhost:9092`) and "Proxy Manager Overview" dashboard with panels for pool health, validation latency (p50/p95/p99), removal reasons, and rate-limiting.

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

11. **Analytics layering** — Separate `analytics` schema provides Bayesian-shrunk win rates, hero synergies/counters, and point-in-time feature snapshots to avoid look-ahead bias in ML training.

12. **Monitoring-first** — Every service exposes Prometheus metrics. Three pre-configured alerts and a Grafana dashboard ship with the deployment.

13. **ML offline training + online inference** — Training is a batch CLI (Trainer) that computes aggregates and trains models offline; inference is a stateless HTTP service (API) that loads pre-trained models. The contract between them is `feature_schema.json` — a frozen column-order manifest written at training time and loaded at API startup. Models are per-patch to capture meta shifts between Dota 2 balance patches.
