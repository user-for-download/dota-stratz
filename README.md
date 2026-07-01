# Dota 2 Match Analysis System

An event-driven pipeline that ingests, processes, and stores Dota 2 match data from the [OpenDota API](https://www.opendota.com/) into PostgreSQL for analytics and ML feature engineering.

## Architecture

```
OpenDota API  ──►  ID Fetcher  ──►  Detail Fetcher  ──►  Parser  ──►  PostgreSQL
                       │                                      │
                       │                         (analytics materialized views)
                       │                                            │
                  Proxy Manager  ──►  Redis (proxy pool)            │
                                                                    ▼
                                                              [Trainer]
                                                                    │
                                                                    ▼
                                                              [ML Models]
                                                                    │
                                                                    ▼
                                                              [Inference API]  :8080
```

Five microservices (4 Go + 2 Python) connected via RabbitMQ message queues, with a Redis-backed proxy pool for API rate-limit avoidance. The ID Fetcher owns its own schedule (cron-based). The ML pipeline (Trainer + API) sits downstream of PostgreSQL, consuming aggregated data for LightGBM lambdarank model training and serving draft predictions.

**See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full system design, service details, database schema, and deployment topology.**

## Quick Start

### Prerequisites

- Go 1.26.3+
- Docker + Docker Compose (with buildx)
- Make

### Setup

```bash
# 1. Clone and enter the project
git clone <repo-url> dota-stratz
cd dota-stratz

# 2. Create environment file
make env
#    → Copies deploy/.env.example to deploy/.env
#    → Edit deploy/.env to set POSTGRES_PASSWORD and review proxy settings

# 3. Sync environment (ensures deploy/.env and root .env stay in sync)
make env-sync

# 4. Start the infrastructure
make up-db
#    → Starts PostgreSQL, RabbitMQ, and Redis in Docker

# 5. Run migrations (first-time only - runs automatically on postgres startup)
make migrate

# 6. Build all services locally
make build
#    → Produces binaries in ./bin/

# 7. Run services (each in its own terminal, or use Docker)
make run-proxy-manager   # Manages the HTTP proxy pool
make run-id-fetcher      # Owns cron schedule, queries match IDs
make run-detail-fetcher  # Fetches match details from OpenDota
make run-parser          # Parses and inserts match data into Postgres
```

### Docker (full deployment)

```bash
# Start everything
make up

# Start in background
make up-d

# Or start specific profiles
make up-db       # Just the data layer (postgres, redis, rabbitmq)
make up-db-d     # Data layer in background
make up-proxy    # Data layer + proxy-manager
make up-fetcher  # Data layer + id-fetcher + detail-fetcher
make up-parser   # Data layer + parser
make up-api-d    # Data layer + ML inference API (background)
make up-mon      # Monitoring (Prometheus + Grafana)

# ML pipeline
make train PATCH=60      # Train LightGBM model for patch 60
make train-agg-only PATCH=60  # Populate aggregates only
make test-api             # Smoke test the inference API
make reload-api PATCH=60  # Hot-reload model (no restart)

# Build Docker images
make bake
make bake-parser    # Build a single service image

# Stop
make down

# Stop and remove volumes (destructive)
make downv      # Will prompt for confirmation
```

## Configuration

Configuration is managed through `deploy/.env`. Copy from `deploy/.env.example` and edit:

```bash
cp deploy/.env.example deploy/.env
```

### Essential Variables

| Variable | Default | Required | Description |
|---|---|---|---|
| `POSTGRES_PASSWORD` | `changeme` | Yes | PostgreSQL password |
| `PROXY_FILE_PATH` | `deploy/proxy.txt` | Yes | Static proxy list (bootstrap) |
| `PROXY_REFRESH_SOURCE_URL` | *(scrape URL)* | No | Remote proxy source (empty to disable) |
| `RABBITMQ_DEFAULT_PASS` | `guest` | No | RabbitMQ password |
| `FETCH_LAST_COUNT_DAY` | `360` | Yes | Rolling window (days) for ID Fetcher queries |
| `DETAIL_FETCHER_WORKER_CONCURRENCY` | `5` | No | Concurrent workers for detail fetcher |
| `FETCH_LOBBY_TYPES` | `1,2,6` | Yes | Comma-separated lobby types to fetch |
| `LOG_LEVEL` | `info` | No | Log level (debug, info, warn, error) |

See `deploy/.env.example` for all variables across 10 configuration sections.

## Development

### Workspace

The project uses a Go workspace spanning 5 modules, plus 2 Python services:

```
go.work (Go modules)
├── services/detail-fetcher
├── services/id-fetcher
├── services/parser
├── services/proxy-manager
└── shared/go-common

services/ (Python — no workspace)
├── trainer/      # LightGBM lambdarank training (batch CLI)
└── api/          # FastAPI inference server (:8080)
```

### Commands

```bash
make build       # Build all service binaries into ./bin
make test        # Run all tests
make test-race   # Run tests with race detector
make vet         # Run go vet on all modules
make fmt         # Format all Go code
make lint        # Run golangci-lint (requires external install)
make tidy        # Run go mod tidy across all modules
make check       # Format + vet + test (one-shot quality gate)
make bake        # Build all Docker images via docker buildx bake
```

### Running Locally

```bash
# Requires infrastructure running (make up-db or make up-db-d)
make run-proxy-manager   # port 9090
make run-id-fetcher      # port 9094
make run-detail-fetcher  # port 9091
make run-parser          # port 9093
```

### Database Access

```bash
make psql                 # Open psql shell
make migrate              # Re-apply all SQL migrations
make db-reset             # Drop and recreate database
make db-backup-physical   # Snapshot Postgres data directory (fast, stops briefly)
make db-restore-physical  # Restore from a physical snapshot
make db-backups           # List physical backups
```

### Redis Access

```bash
make redis-cli      # Open redis-cli
make redis-flush    # Flush all Redis data (DESTRUCTIVE)
make proxies-show   # Inspect proxy pool state
```

## Project Structure

```
├── services/
│   ├── detail-fetcher/    # Match detail fetcher (API consumer)
│   ├── id-fetcher/        # Match ID fetcher (API explorer queries, self-cron)
│   ├── parser/            # Match parser & DB writer
│   ├── proxy-manager/     # Autonomous proxy pool manager
│   ├── trainer/           # LightGBM lambdarank training (Python)
│   └── api/               # FastAPI inference server (Python, :8080)
├── shared/
│   └── go-common/         # Shared library
│       ├── cache/         # Redis connection helper
│       ├── db/            # PostgreSQL connection helper
│       ├── logger/        # Structured logging (zap)
│       ├── mq/            # RabbitMQ connection + queue declaration + reconnecting consumer/publisher
│       └── proxypool/     # Redis-backed proxy pool (Lua + Prometheus)
│           └── transport.go  # HTTP transport builder (HTTP + SOCKS5 proxy)
├── deploy/
│   ├── compose.yaml       # Docker Compose with profiles
│   ├── docker-bake.hcl    # Buildx bake config
│   ├── .env.example       # Environment variable template
│   ├── migration/         # SQL migration files (001__init, 002_ml, 003_static)
│   ├── rabbitmq/          # RabbitMQ definitions + init script
│   ├── prometheus/        # Prometheus config + alert rules
│   └── grafana/           # Pre-provisioned dashboards
├── .opencode/             # AI context system (29 files)
├── Makefile               # Build/deploy/test orchestration
├── go.work                # Go workspace
├── ARCHITECTURE.md        # Full system architecture docs
└── README.md              # This file
```

## Monitoring

### Metrics

Each service exposes Prometheus metrics (ML services expose via `/metrics` on their API port):

| Service | Port | Endpoint |
|---|---|---|
| Proxy Manager | 9090 | `/metrics` |
| Detail Fetcher | 9091 | `/metrics` |
| Parser | 9093 | `/metrics` |
| ID Fetcher | 9094 | `/metrics` |
| ML API | 8080 | `/metrics` |

### Alerts

Three pre-configured alerting rules ship with the deployment:

- **DLQDepthGrowing** — Dead-letter queue exceeds 50 messages for 5 minutes
- **ProxyPoolDepleted** — Available proxies fall below minimum threshold (20)
- **IngestionStalled** — No match IDs published for 15+ minutes

### Grafana

Start with:

```bash
make up-mon    # Prometheus + Grafana
```

Access Grafana at `http://localhost:3000` (default: admin/admin). A pre-built "Proxy Manager Overview" dashboard is auto-provisioned.

## RabbitMQ Management

Access the management UI at `http://localhost:15672` (default: guest/guest). Key queues:

| Queue | Purpose |
|---|---|
| `queue.match_ids` | Match IDs from ID Fetcher → Detail Fetcher |
| `queue.raw_matches` | Raw match JSON from Detail Fetcher → Parser |
| `queue.*.dlq` | Dead-letter queues (one per source queue) |

## Pipeline Lifecycle

1. **ID Fetcher** cron fires on its configured schedule (default: daily at midnight for incremental, or every 5 min for continuous), queries OpenDota Explorer for matches within a rolling N-day window (or uses watermark-based query to skip already-parsed matches), publishes IDs in batches to RabbitMQ. Includes DB existence check (Layer 3) to prevent re-publishing match IDs already in the database. Optional startup fetch runs on boot before the first cron tick (enabled via `ID_FETCHER_START_RUN=true`)
2. **Detail Fetcher** consumes match IDs, fetches full match JSON from OpenDota API (via proxy pool with 5 retries + direct connection fallback before DLQ), publishes raw data. Includes DB existence check (`postgresMatchChecker`) to skip matches already committed — acts as a safety net for DLQ replays and race conditions. Configurable concurrency (default 50 workers). Exposes `detail_fetcher_skipped_total` metric.
3. **Parser** consumes raw JSON, accumulates batches (size 100 or 2s timeout), validates, and bulk-inserts into 20+ PostgreSQL tables (including gold_t/xp_t JSONB arrays for early-game features). Memory: 1G.
4. Failures route to **dead-letter queues** for manual inspection and replay
5. **Analytics materialized views** refresh periodically for ML feature extraction
6. **Trainer** computes **7 patch-aware aggregate tables** + **7 PIT-safe snapshot tables** and trains LightGBM **binary classification** models with **Platt scaling calibration**. Feature vector: **230 dimensions** (70 aggregate + 160 one-hot hero ID). New features include recent form (last 20 games), meta drift (7-day trend), sequence context, and team strategy scores. Validated performance: **binary_logloss 0.6894** (patch 60). Outputs calibrated `pick_probability` and `win_probability`.
7. **Inference API** loads trained models and calibrators, serves draft predictions via HTTP (`POST /predict`), returns top-N hero recommendations with calibrated probabilities and reasoning

## License

See `LICENSE` (if applicable).
