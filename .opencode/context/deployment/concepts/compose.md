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
| `api` | ml-inference-api (+ db) | `make up-api` |
| `train` | ml-trainer (manual exec) | — |

## Volumes
`pg-data`, `rmq-data`, `redis-data`, `prometheus-data`, `grafana-data`, `ml-models`

## Build
```bash
make bake           # Build all Go service images via docker buildx bake
make bake-parser    # Build single service image
```
