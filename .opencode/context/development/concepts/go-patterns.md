# Go Patterns & Shared Library

## Shared Library (`shared/go-common`)
```
github.com/dota-stratz/shared/go-common
```

### proxypool (most complex package, ~710 lines)
- Redis-backed pool: proxies stored in ZSET (`dota2:proxies`), leases in HASH
- `MakeTransport(proxyStr, timeout)` — builds `*http.Transport` for HTTP/HTTPS/SOCKS4/SOCKS5
- `AcquireWithRateLimit(maxPerMin)` — enforces per-proxy rate limits via atomic Lua scripts
- Failure classification: `HardFailure`/`BadStatus` → permanent removal, `RateLimited` → cooldown, `Timeout` → counter → removal at threshold (3)
- Performance: `crc64.Checksum` + base36 for proxy hash keys (~10× faster than SHA256)
- ZSET scores use `UnixMicro()` (avoids float64 precision loss from `UnixNano`)

## Patterns Used Across Services
- **Config**: YAML + `os.Getenv` substitution in `internal/config/`
- **Metrics**: Prometheus counters + histograms in `internal/metrics/` per service
- **Health**: `GET /healthz` returning `"ok"` on every service
- **Reconnection**: RabbitMQ consumers/publishers have reconnect loops with exponential backoff (1s→30s)
- **Shutdown**: `shutdown` channel pattern for deadlock-safe reconnection interrupt
- **Publisher safety**: Mutex-serialized reconnections (`reconnectMu`) to prevent exchange race
