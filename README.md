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
                                                         (PyTorch DraftBERT)
                                                                    │
                                                                    ▼
                                                              [ML Models]
                                                         (TorchScript .pt)
                                                                    │
                                                                    ▼
                                                       [Inference API]  :8080
                                                                    │
                                                                    ▼
                                                        [Frontend]  :80
                                                  (Draft Predictor UI)
```

Six microservices (4 Go + 2 Python) + 1 Nginx frontend, connected via RabbitMQ message queues, with a Redis-backed proxy pool for API rate-limit avoidance. The ML pipeline uses PyTorch DraftBERT (Transformer + MLP multi-modal architecture) for draft prediction with Monte Carlo rollouts for strategic lookahead.

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
#    → Edit deploy/.env to set POSTGRES_PASSWORD and review settings

# 3. Sync environment
make env-sync

# 4. Start infrastructure
make up-db

# 5. Start all services
make up
```

### Docker Deployment

```bash
make up                   # Start everything
make up-d                 # Start in background
make up-db                # Data layer only (postgres, redis, rabbitmq)
make up-api-d             # ML API in background
make up-mon               # Monitoring (Prometheus + Grafana)

# ML Pipeline
make train PATCH=60       # Train PyTorch DraftBERT for patch 60
make train-agg-only       # Populate aggregates only
make reload-api PATCH=60  # Hot-reload model

# Stop
make down
make downv                # Stop and remove volumes (destructive)
```

## ML Pipeline

| Component | Technology | Description |
|-----------|-----------|-------------|
| **Trainer** | PyTorch DraftBERT | Transformer (128d, 4 heads, 3 layers) + MLP (59 tabular features) |
| **Inference** | TorchScript JIT | <2ms batched CPU inference via C++ graph |
| **Lookahead** | Monte Carlo Rollouts | 40 simulations per top-15 candidate, batch-evaluated |
| **Features** | 59 aggregate + sequence | Team/player hero stats, synergy, counter, H2H, draft-slot |
| **Calibration** | BCEWithLogitsLoss | Direct logit training, sigmoid output |
| **Early Stopping** | Patience-based | Stops training when validation loss plateaus (patience=5) |
| **Label Fix** | make_target() | Correctly handles Dire team labels (1 - radiant_win) |

## Configuration

Configuration is managed through `deploy/.env`. See `deploy/.env.example` for all variables.

### Key ML Variables

| Variable | Default | Description |
|---|---|---|
| `TRAINER_NUM_THREADS` | 12 | CPU threads for PyTorch training |
| `TRAINER_BATCH_SIZE` | 256 | Training batch size |
| `TRAINER_EPOCHS` | 15 | Training epochs |
| `TRAINER_LR` | 1e-4 | Learning rate |
| `TRAINER_WEIGHT_DECAY` | 1e-3 | AdamW weight decay |
| `TRAINER_MAX_SEQ_LEN` | 50 | Max draft sequence length |
| `TRAINER_D_MODEL` | 128 | Transformer embedding dimension |
| `TRAINER_NHEAD` | 4 | Attention heads |
| `TRAINER_NUM_LAYERS` | 3 | Transformer layers |

## Project Structure

```
├── services/
│   ├── id-fetcher/        # Match ID fetcher (Go, self-cron)
│   ├── detail-fetcher/    # Match detail fetcher (Go)
│   ├── parser/            # Match parser & DB writer (Go)
│   ├── proxy-manager/     # Proxy pool manager (Go)
│   ├── trainer/           # PyTorch DraftBERT training (Python)
│   ├── api/               # FastAPI inference API (Python, :8080)
│   └── frontend/          # Nginx draft predictor UI (:80)
├── shared/go-common/      # Shared Go library (mq, db, proxypool)
├── deploy/                # Docker Compose, migrations, monitoring
├── Makefile               # Build/deploy/test orchestration
├── ARCHITECTURE.md        # Full system architecture
└── README.md              # This file
```

## Monitoring

| Service | Port | Endpoint |
|---|---|---|
| Proxy Manager | 9090 | `/metrics` |
| Detail Fetcher | 9091 | `/metrics` |
| Parser | 9093 | `/metrics` |
| ID Fetcher | 9094 | `/metrics` |
| ML API | 8080 | `/metrics` |

Grafana at `http://localhost:3000` (admin/admin). Prometheus at `http://localhost:9092`.

## License

See `LICENSE` (if applicable).
