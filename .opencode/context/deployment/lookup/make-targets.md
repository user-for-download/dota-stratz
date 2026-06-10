# Makefile Targets Reference

## Core
| Target | Action |
|--------|--------|
| `make up` / `make up-d` | Start all services (foreground/background) |
| `make down` | Stop all services |
| `make check` | Format + vet + test (pre-commit gate) |
| `make build` | Build all Go binaries |

## Database
| Target | Action |
|--------|--------|
| `make psql` | Open psql shell |
| `make migrate` | Apply pending migrations |
| `make db-reset` | Drop + recreate + migrate (DESTRUCTIVE) |
| `make db-backup-physical` | Snapshot pg data directory |
| `make db-restore-physical DUMP=...` | Restore from snapshot |

## ML
| Target | Action |
|--------|--------|
| `make train PATCH=N` | Train LightGBM model for patch N (uses profiles `db`+`train`) |
| `make train-agg-only PATCH=N` | Populate aggregates only, skip training |
| `make up-api-d` | Start inference API on :8080 (uses profiles `db`+`api`) |
| `make down-api` | Stop inference API |
| `make test-api` | Smoke-test health + /predict endpoints |
| `make reload-api PATCH=N` | Hot-reload model (no restart) |
| `make migrate-ml` | Apply ML migration only |
| `make build-ml-images` | Build trainer + api Docker images |

## RabbitMQ
| Target | Action |
|--------|--------|
| `make replay-dlq` | Replay up to 500 DLQ messages |
| `make replay-dlq-n N=1000` | Replay N DLQ messages |

## Profiles
| Target | Containers |
|--------|-----------|
| `make up-db-d` | postgres, redis, rabbitmq |
| `make up-proxy` | + proxy-manager |
| `make up-fetcher` | + id-fetcher, detail-fetcher |
| `make up-parser` | + parser |
| `make up-api-d` | + ml-inference-api |
| `make train` | runs existing db+train containers |
