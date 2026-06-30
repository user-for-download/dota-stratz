# Monitoring

**Core concept**: Pre-configured Prometheus + Grafana stack with service-level metrics and 3 alerting rules.

## Scrape Targets (bridge network — was host network prior to HIGH-2/MEDIUM-12 fixes)
| Port | Service |
|------|---------|
| 9090 | Proxy Manager |
| 9091 | Detail Fetcher |
| 9092 | Prometheus self |
| 9093 | Parser |
| 9094 | ID Fetcher |

## Alerts
| Alert | Condition | Severity |
|-------|-----------|----------|
| `ProxyPoolDepleted` | Available < 20 for 2m | warning |
| `IngestionStalled` | No match IDs published for >26h | warning |
| `DLQDepthGrowing` | Any DLQ queue >50 messages for 5m | warning |

## Grafana
- Pre-provisioned datasource: Prometheus at `prometheus:9090` (was `localhost:9092` with host networking)
- Dashboard: "Proxy Manager Overview" (pool health, validation latency p50/p95/p99, removal reasons)
- Auto-provisioned via `deploy/grafana/provisioning/`
