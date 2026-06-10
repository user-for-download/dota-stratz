# Dota 2 Match Analysis System

An event-driven pipeline that ingests, processes, and stores Dota 2 match data from the [OpenDota API](https://www.opendota.com/) into PostgreSQL for analytics and ML feature engineering.

## Architecture

```
OpenDota API  ──►  ID Fetcher  ──►  Detail Fetcher  ──►  Parser  ──►  PostgreSQL
                       │                                      │
                       │                         (analytics materialized views)
                       │
                  Proxy Manager  ──►  Redis (proxy pool)
```

Four Go microservices connected via RabbitMQ message queues, with a Redis-backed proxy pool for API rate-limit avoidance. The ID Fetcher owns its own schedule (cron-based) and no longer requires a coordinator service.

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
make up-mon      # Monitoring (Prometheus + Grafana)

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
| `FETCH_LAST_COUNT_DAY` | `30` | Yes | Rolling window (days) for ID Fetcher queries |
| `FETCH_LOBBY_TYPES` | `1,2,6` | Yes | Comma-separated lobby types to fetch |
| `LOG_LEVEL` | `info` | No | Log level (debug, info, warn, error) |

See `deploy/.env.example` for all variables across 10 configuration sections.

## Development

### Workspace

The project uses a Go workspace spanning 5 modules:

```
go.work
├── services/detail-fetcher
├── services/id-fetcher
├── services/parser
├── services/proxy-manager
└── shared/go-common
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
make deps        # Download dependencies for all modules
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
│   └── proxy-manager/     # Autonomous proxy pool manager
├── shared/
│   └── go-common/         # Shared library
│       ├── cache/         # Redis connection helper
│       ├── db/            # PostgreSQL connection helper
│       ├── logger/        # Structured logging (zap)
│       ├── mq/            # RabbitMQ connection helper
│       └── proxypool/     # Redis-backed proxy pool (Lua + Prometheus)
│           └── transport.go  # HTTP transport builder (HTTP + SOCKS5 proxy)
├── deploy/
│   ├── compose.yaml       # Docker Compose with profiles
│   ├── docker-bake.hcl    # Buildx bake config
│   ├── .env.example       # Environment variable template
│   ├── migration/         # SQL migration files (001_core–004_verify)
│   ├── prometheus/        # Prometheus config + alert rules
│   └── grafana/           # Pre-provisioned dashboards
├── Makefile               # Build/deploy/test orchestration
├── go.work                # Go workspace
├── ARCHITECTURE.md        # Full system architecture docs
└── README.md              # This file
```

## Monitoring

### Metrics

Each service exposes Prometheus metrics:

| Service | Port | Endpoint |
|---|---|---|
| Proxy Manager | 9090 | `/metrics` |
| ID Fetcher | 9094 | `/metrics` |
| Detail Fetcher | 9091 | `/metrics` |
| Parser | 9093 | `/metrics` |

### Alerts

Three pre-configured alerting rules ship with the deployment:

- **DLQDepthCritical** — Dead-letter queue grows beyond 10 messages
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

1. **ID Fetcher** cron fires on its configured schedule (default: daily at 03:00 UTC), queries OpenDota Explorer for matches within a rolling N-day window, publishes IDs in batches to RabbitMQ. Optional startup fetch runs on boot before the first cron tick (enabled via `ID_FETCHER_START_RUN=true`)
2. **Detail Fetcher** consumes match IDs, fetches full match JSON from OpenDota API (via proxy pool), publishes raw data
3. **Parser** consumes raw JSON, accumulates batches (size 100 or 2s timeout), validates, and bulk-inserts into PostgreSQL
4. Failures route to **dead-letter queues** for manual inspection and replay
5. **Analytics materialized views** refresh periodically for ML feature extraction

## License

See `LICENSE` (if applicable).
