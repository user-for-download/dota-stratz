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
make build-ml-images                    # Rebuild after code changes to trainer/api
```

**Note**: Models use 198 features (was 196) after adding player-hero `account_id` integration. Training uses `binary` objective (not `lambdarank`) since all draft slots in a match share the same `radiant_win` target.

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
