# Docker Compose Architecture

**Core concept**: Profile-based Docker Compose with `docker buildx bake` for multi-service orchestration.

**Last updated**: July 2026

## Profiles
| Profile | Services | Make Target |
|---------|----------|-------------|
| `all` | Everything | `make up` / `make up-d` |
| `db` | postgres, rabbitmq, redis | `make up-db` / `make up-db-d` |
| `mon` | prometheus, grafana | `make up-mon` |
| `proxy` | proxy-manager (+ db) | `make up-proxy` |
| `fetcher` | id-fetcher, detail-fetcher (+ db) | `make up-fetcher` |
| `parser` | parser (+ db) | `make up-parser` |
| `api` | ml-inference-api (+ db) | `make up-api` / `make up-api-d` |
| `train` | ml-trainer (manual exec, also needs db) | `make train` |

ML targets that need postgres use `--profile db --profile api` or `--profile db --profile train`.

## Resource Limits
| Service | Memory | CPUs | Notes |
|---------|--------|------|-------|
| postgres | **2G** | 1.0 | Tuned shared_buffers/work_mem (was 512M — HIGH-1 fix) |
| trainer | **2G** | 2.0 | Patch 58 (372k draft slots) requires ~1.6G |
| api | 512M | 0.5 | |
| Most others | 128-256M | 0.5 | |

## Notable Fixes Applied
- **Postgres memory**: Increased from 512M → 2G with tuned shared_buffers/work_mem (HIGH-1)
- **Grafana networking**: Removed `network_mode: host`, uses Docker bridge with port mapping (HIGH-2)
- **RabbitMQ**: Set `RABBITMQ_VM_MEMORY_HIGH_WATERMARK: 0.5` to prevent OOM (HIGH-3)
- **id-fetcher**: Added `depends_on postgres: condition: service_healthy` (HIGH-4)
- **Prometheus**: Removed `network_mode: host`, uses bridge networking (MEDIUM-12)
- **Resource limits**: Added for Prometheus and Grafana (LOW-18)
- **RabbitMQ definitions**: Removed hardcoded users — `init.sh` now creates from `.env` (CRITICAL-1)
- **Trainer no longer sleeps**: Removed `command: ["sleep", "infinity"]` from trainer service — Dockerfile `CMD` (`python -m trainer.main`) now runs normally (BLOCKER B3)
- **proxy.txt validation**: Healthcheck now includes `test -s /app/proxy.txt` to verify proxy file exists and is non-empty (W12)
- **Asyncio Semaphore**: `asyncio.Semaphore(2)` limits concurrent WebSocket evaluations to prevent unbounded task creation (W3)
- **WebSocket cleanup**: `onclose` handler clears all orphaned connection resolvers to prevent hanging Promises (B6)
- **Proxy.txt read-only mount**: `deploy/proxy.txt` mounted `:ro` with comment explaining it's validated once at startup

## Health Checks
- API uses `python3 -c "import urllib.request..."` (slim image has no wget/curl)
- Other services use `wget` (busybox-based images)

## Volumes
`pg-data`, `rmq-data`, `redis-data`, `prometheus-data`, `grafana-data`, `ml-models`

## Build
```bash
make bake           # Build all Go service images via docker buildx bake
make bake-parser    # Build single service image
```
