# Quick Dev Commands

## Start/Stop
```bash
make up-d            # Start everything in background
make up-db-d         # Start data layer only (postgres, redis, rabbitmq)
make down            # Stop everything
make logs-parser     # Tail parser logs
```

## Build/Test
```bash
make check           # fmt + vet + test (pre-commit gate)
make build           # Build all Go binaries into ./bin
make bake            # Build all Docker images
make test            # Run all Go tests
```

## Database
```bash
make psql            # Open psql shell on running container
make migrate         # Apply pending SQL migrations
make migrate-ml      # Apply ML migration only
make db-reset        # Drop + recreate + migrate (DESTRUCTIVE)
make db-backup-physical   # Snapshot pg data directory
```

## ML
```bash
make train PATCH=60                     # Train binary classification model for patch 60
make up-api-d                           # Start inference API on :8080 in background
make test-api                           # Smoke test health + /predict
make reload-api PATCH=60                # Hot-reload model (no restart, requires STRATZ_ADMIN_TOKEN)
```

**Note**: Models use **219 features** (59 aggregate + 160 one-hot hero ID). Platt scaling calibration (fit on validation set only) provides calibrated win probabilities. Training uses `binary` objective since all draft slots share the same `radiant_win` target.

## RabbitMQ
```bash
make replay-dlq           # Replay up to 500 DLQ messages
make replay-dlq-n N=1000  # Replay N DLQ messages
```

## Redis
```bash
make redis-cli       # Open redis-cli
make proxies-show    # Show proxy pool state
```
