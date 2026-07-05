# Monitoring

**Core concept**: Pre-configured Prometheus + Grafana stack with service-level metrics and 3 alerting rules.

## Scrape Targets (Docker bridge network)
| Target | Port | Service |
|--------|------|---------|
| `proxy-manager:9090` | 9090 | Proxy Manager |
| `detail-fetcher:9091` | 9091 | Detail Fetcher |
| `localhost:9092` | 9092 | Prometheus self |
| `parser:9093` | 9093 | Parser |
| `id-fetcher:9094` | 9094 | ID Fetcher |
| `rabbitmq:15692` | 15692 | RabbitMQ |
| `api:8080` | 8080 | ML API |

## Alerts
| Alert | Condition | Severity |
|-------|-----------|----------|
| `ProxyPoolDepleted` | Available < 20 for 2m | warning |
| `IngestionStalled` | No match IDs published for >26h | warning |
| `DLQDepthGrowing` | Any DLQ queue >50 messages for 5m | warning |

## Grafana
- Pre-provisioned datasource: Prometheus at `prometheus:9092`
- Dashboard: "Proxy Manager Overview" (pool health, validation latency p50/p95/p99, removal reasons, cooldown rate)
- Auto-provisioned via `deploy/grafana/provisioning/`

## Key Bug Fixes (Recent)
- Consumer reconnect backoff now respects `ctx.Done()` to prevent delayed shutdown
- `put_conn()` checks `_pool` inside lock to prevent race with `close_pool()`
- Proxy cooldown checks ZScore before re-adding to prevent race with concurrent remove()
- Response body drained on non-200 status in validator and source fetcher
- Batch writer rollback uses 10-second timeout context
- Config validation for required RabbitMQ/Postgres fields at startup
