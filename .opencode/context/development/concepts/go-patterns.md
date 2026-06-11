# Go Patterns & Shared Library

## Shared Library (`shared/go-common`)
```
github.com/dota-stratz/shared/go-common
```

### checkpoint
- Shared constants (`CheckpointPipelineParser`, `CheckpointPipelineIDFetcher`) + `ReadWatermark()` SQL
- Single source of truth for the `ingestion_checkpoints` column name (`last_parsed_match_id`), used by both parser (writer) and id-fetcher (reader)
- Returns `(watermark, ok, err)` triplet: `ok=false` when the row is missing (fresh DB) — callers fall back to rolling-window path

### proxypool (most complex package, ~710 lines)
- Redis-backed pool: proxies stored in ZSET (`dota2:proxies`), leases in HASH
- `MakeTransport(proxyStr, timeout)` — builds `*http.Transport` for HTTP/HTTPS/SOCKS4/SOCKS5
- **Native SOCKS4 dialer** (`socks4.go`) — previously forced through `proxy.SOCKS5()` which fails on SOCKS4-only servers
- `AcquireWithRateLimit(maxPerMin)` — enforces per-proxy rate limits via atomic Lua scripts
- Failure classification: `HardFailure`/`BadStatus` → permanent removal, `RateLimited` → cooldown, `Timeout` → counter → removal at threshold (3)
- `NewRedisPoolCollector(rdb, ...)` — Prometheus `Collector` that reads pool size directly from Redis (replaces per-process promauto gauges that diverged across services)
- Performance: `crc64.Checksum` + base36 for proxy hash keys (~10× faster than SHA256)
- ZSET scores use `UnixMicro()` (avoids float64 precision loss from `UnixNano`)

## Patterns Used Across Services
- **Config**: YAML + `os.Getenv` substitution with `:-default` syntax in `internal/config/`
- **Metrics**: Prometheus counters + histograms in `internal/metrics/` per service; `NewRedisPoolCollector` for Redis-ground-truth metrics
- **Health**: `GET /healthz` returning `"ok"` on every service (parser also pings Postgres)
- **Reconnection**: RabbitMQ consumers/publishers have reconnect loops with exponential backoff (1s→30s). Consumers use permanent output channels (never closed on reconnect), workers exit via context cancellation
- **Shutdown**: `shutdown` channel pattern for deadlock-safe reconnection interrupt (closed before mutex acquisition so `Close()` never deadlocks with an in-progress reconnect)
- **Publisher safety**: Mutex-serialized reconnections (`reconnectMu`) to prevent exchange race. Fresh AMQP channel per batch (closed after publish) to prevent `NotifyPublish` listener leak
- **Watermark dispatch**: ID Fetcher uses `bootstrapCheckpoint()` to read parser's `last_parsed_match_id` on startup. Watermark pushed into SQL (`match_id > %d ORDER BY match_id ASC LIMIT %d`) for oldest-first backlog drainage. Go-side filter removed — SQL now guarantees correct results
- **context.WithoutCancel**: Parser batch I/O (`SendBatch`, `Commit`) uses orphaned context to prevent connection pool corruption during graceful shutdown
