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

## Shared mq Package (`shared/go-common/mq/`)
- **`Connect(url)`** — basic AMQP connection + channel (legacy helper)
- **`QueueConfig`** + **`DeclareQueueWithDLQ(ch, cfg)`** — single source of truth for queue + DLQ declaration (24h TTL, DLX binding). Previously duplicated across 4 service files.
- **`Consumer`** with `ConsumeWithReconnect(done, tag)` — auto-reconnecting consumer. Exponential backoff (1s→30s). Permanent output channel (never closed on reconnect). Used by parser and detail-fetcher.
- **`Publisher`** with `Publish(ctx, queue, body)` — publisher confirms with automatic reconnect. Mutex-layered safety (`closeMu`, `publishMu`, `reconnectMu`). Handles connection loss transparently via background `NotifyClose` listener. Used by detail-fetcher.
- **Refactored services** — Parser, detail-fetcher (consumer + publisher), and id-fetcher now wrap shared mq primitives instead of duplicating queue declaration and reconnection logic.

## Patterns Used Across Services
- **Config**: YAML + `os.Getenv` substitution with `:-default` syntax in `internal/config/`
- **Metrics**: Prometheus counters + histograms in `internal/metrics/` per service; `NewRedisPoolCollector` for Redis-ground-truth metrics
- **Health**: `GET /healthz` returning `"ok"` on every service (parser also pings Postgres)
- **Reconnection**: RabbitMQ consumers/publishers use the shared `mq.Consumer`/`mq.Publisher` primitives. Consumers use permanent output channels (never closed on reconnect), workers exit via context cancellation. Exponential backoff (1s→30s)
- **Shutdown**: `shutdown` channel pattern for deadlock-safe reconnection interrupt (closed before mutex acquisition so `Close()` never deadlocks with an in-progress reconnect). Detail-fetcher uses a separate close function to delay AMQP connection close until after in-flight workers drain
- **Publisher safety**: Mutex-serialized reconnections (`reconnectMu`) to prevent exchange race. ID-fetcher uses fresh AMQP channel per batch (closed after publish) to prevent `NotifyPublish` listener leak; detail-fetcher uses shared channel with `publishMu` serialization
- **Watermark dispatch**: ID Fetcher uses `bootstrapCheckpoint()` to read parser's `last_parsed_match_id` on startup. Watermark pushed into SQL (`match_id > %d ORDER BY match_id ASC LIMIT %d`) for oldest-first backlog drainage
- **context.WithoutCancel**: Parser batch I/O (`SendBatch`, `Commit`) uses orphaned context to prevent connection pool corruption during graceful shutdown
- **Parser error handling**: `processor.Run()` returns `ErrFatalPanic` on recovered panic; `main.go` calls `os.Exit(1)` at the top level. This preserves zombie-service prevention while keeping `Run()` testable
