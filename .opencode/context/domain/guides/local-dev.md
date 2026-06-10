# Running the Stack Locally

## Prerequisites
- Docker Engine 24+ with Compose v2
- Go 1.26+
- Python 3.12+ (ML services only)
- ~10GB free disk for PostgreSQL data + Docker images

## Quick Start
```bash
# 1. Start the data layer
make up-db-d

# 2. Proxy manager (needs internet for OpenDota API)
make up-proxy

# 3. Wait for proxy pool to populate (~2 min)
docker logs dota2-proxy-manager --tail=20

# 4. Start fetchers + parser
make up-fetcher
make up-parser

# 5. Verify
docker ps --format "table {{.Names}}\t{{.Status}}"
```

## ML Pipeline (after data ingested)
```bash
make migrate-ml              # Create ML aggregate tables
make build-ml                # Build ML Docker images
make train                   # Auto-detect patch, train model
make up-api-d                # Start inference API
make test-api                # Smoke test
```

## Monitoring
```bash
make up-mon                  # Start Prometheus + Grafana
# Grafana: http://localhost:3000 (admin / admin)
```

## Stopping
```bash
make down                    # Stop all
make downv                   # Stop + remove volumes (DESTRUCTIVE)
```
