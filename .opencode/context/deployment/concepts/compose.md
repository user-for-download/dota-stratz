# Docker Compose Architecture

**Core concept**: Profile-based Docker Compose with `docker buildx bake` for multi-service orchestration.

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
| postgres | 512M | 1.0 | |
| trainer | **2G** | 2.0 | Patch 58 (372k draft slots) requires ~1.6G |
| api | 512M | 0.5 | |
| Most others | 128-256M | 0.5 | |

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
