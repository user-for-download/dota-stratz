# ==============================================================================
# Dota 2 Match Analysis System
# ==============================================================================

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ------------------------------------------------------------------------------
# Paths / Tools
# ------------------------------------------------------------------------------

COMPOSE_FILE  := deploy/compose.yaml
ENV_FILE      := deploy/.env
ROOT_ENV_FILE := .env
MIGRATION_DIR := deploy/migration

COMPOSE := docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE)

# ------------------------------------------------------------------------------
# Services / Modules
# ------------------------------------------------------------------------------

SERVICES := detail-fetcher id-fetcher parser proxy-manager
MODULES  := shared/go-common $(addprefix services/,$(SERVICES))

# ------------------------------------------------------------------------------
# Containers / Defaults
# ------------------------------------------------------------------------------

POSTGRES_USER ?= dota2
POSTGRES_DB   ?= dota2

POSTGRES_CONTAINER := dota2-postgres
REDIS_CONTAINER    := dota2-redis
RABBITMQ_CONTAINER := dota2-rabbitmq

DB_PHYSICAL_BACKUP_DIR ?= ./backups

# ------------------------------------------------------------------------------
# Help styling
# ------------------------------------------------------------------------------

CYAN   := \033[36m
YELLOW := \033[33m
RESET  := \033[0m

# ==============================================================================
# Help
# ==============================================================================

.PHONY: help
help: ## Show this help
	@echo ""
	@echo "$(YELLOW)Dota 2 Match Analysis System$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z0-9_%.-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-24s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ==============================================================================
# Environment
# ==============================================================================

.PHONY: env
env: ## Create deploy/.env from deploy/.env.example if missing
	@if [ ! -f "$(ENV_FILE)" ]; then \
		if [ -f deploy/.env.example ]; then \
			echo "Creating $(ENV_FILE) from deploy/.env.example..."; \
			cp deploy/.env.example "$(ENV_FILE)"; \
			echo "$(YELLOW)Review $(ENV_FILE) before starting services.$(RESET)"; \
		else \
			echo "$(YELLOW)deploy/.env.example not found.$(RESET)"; \
			exit 1; \
		fi; \
	else \
		echo "$(ENV_FILE) already exists."; \
	fi

.PHONY: env-sync
env-sync: env ## Sync deploy/.env to root .env legacy copy
	@cp "$(ENV_FILE)" "$(ROOT_ENV_FILE)"
	@echo "Synced $(ENV_FILE) -> $(ROOT_ENV_FILE)"

# ==============================================================================
# Docker Compose
# ==============================================================================

.PHONY: up
up: env-sync ## Start all services
	$(COMPOSE) --profile all up

.PHONY: up-d
up-d: env-sync ## Start all services in background
	$(COMPOSE) --profile all up -d

.PHONY: up-db
up-db: env-sync ## Start postgres, redis, rabbitmq
	$(COMPOSE) --profile db up

.PHONY: up-db-d
up-db-d: env-sync ## Start postgres, redis, rabbitmq in background
	$(COMPOSE) --profile db up -d

.PHONY: up-mon
up-mon: env-sync ## Start prometheus and grafana
	$(COMPOSE) --profile mon up

.PHONY: up-proxy
up-proxy: env-sync ## Start proxy-manager plus data layer
	$(COMPOSE) --profile db --profile proxy up

.PHONY: up-fetcher
up-fetcher: env-sync ## Start id-fetcher and detail-fetcher plus data layer
	$(COMPOSE) --profile db --profile fetcher up

.PHONY: up-parser
up-parser: env-sync ## Start parser plus data layer
	$(COMPOSE) --profile db --profile parser up

.PHONY: down
down: ## Stop services
	$(COMPOSE) --profile all down

.PHONY: downv
downv: ## Stop services and remove project volumes
	@echo "$(YELLOW)WARNING: This removes project volumes and deletes local data.$(RESET)"
	@read -p "Continue? [y/N] " ans && [ "$${ans:-N}" = "y" ]
	$(COMPOSE) --profile all down -v

.PHONY: restart
restart: down up ## Restart all services

.PHONY: ps
ps: ## Show compose services
	$(COMPOSE) ps

.PHONY: logs
logs: ## Tail all logs
	$(COMPOSE) --profile all logs -f

.PHONY: logs-%
logs-%: ## Tail logs for a service, e.g. make logs-parser
	$(COMPOSE) logs -f --tail=100 $*

# ==============================================================================
# Database
# ==============================================================================

.PHONY: psql
psql: ## Open psql shell
	docker exec -it $(POSTGRES_CONTAINER) psql -U $(POSTGRES_USER) -d $(POSTGRES_DB)

.PHONY: migrate
migrate: ## Apply pending SQL migrations
	@echo "$(YELLOW)Applying migrations from $(MIGRATION_DIR)...$(RESET)"
	@docker exec -i $(POSTGRES_CONTAINER) psql \
		-U $(POSTGRES_USER) \
		-d $(POSTGRES_DB) \
		-v ON_ERROR_STOP=1 \
		-c "CREATE TABLE IF NOT EXISTS _migrations (name TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now());"
	@for f in $$(find "$(MIGRATION_DIR)" -maxdepth 1 -name '*.sql' | sort); do \
		name=$$(basename "$$f"); \
		applied=$$(docker exec -i $(POSTGRES_CONTAINER) psql \
			-U $(POSTGRES_USER) \
			-d $(POSTGRES_DB) \
			-tAc "SELECT 1 FROM _migrations WHERE name = '$$name';"); \
		if [ "$$applied" = "1" ]; then \
			echo "  SKIP  $$name"; \
		else \
			echo "  APPLY $$name"; \
			( echo "BEGIN;"; cat "$$f"; echo "INSERT INTO _migrations (name) VALUES ('$$name') ON CONFLICT DO NOTHING; COMMIT;" ) | \
			docker exec -i $(POSTGRES_CONTAINER) psql \
				-U $(POSTGRES_USER) \
				-d $(POSTGRES_DB) \
				-v ON_ERROR_STOP=1 || exit 1; \
		fi; \
	done
	@echo "$(CYAN)Migrations complete.$(RESET)"

.PHONY: db-reset
db-reset: ## Drop and recreate database, then run migrations
	@echo "$(YELLOW)WARNING: This drops and recreates database $(POSTGRES_DB).$(RESET)"
	@read -p "Continue? [y/N] " ans && [ "$${ans:-N}" = "y" ]
	docker exec -i $(POSTGRES_CONTAINER) psql \
		-U $(POSTGRES_USER) \
		-d postgres \
		-v ON_ERROR_STOP=1 \
		-c "DROP DATABASE IF EXISTS $(POSTGRES_DB) WITH (FORCE);" \
		-c "CREATE DATABASE $(POSTGRES_DB);"
	$(MAKE) migrate

.PHONY: db-backup-physical
db-backup-physical: ## Create physical Postgres volume backup, stops Postgres briefly
	@echo "$(YELLOW)Creating physical backup of $(POSTGRES_CONTAINER)...$(RESET)"
	@mkdir -p "$(DB_PHYSICAL_BACKUP_DIR)"
	@backup="pgdata_$$(date +%Y%m%d_%H%M%S).tar"; \
	echo "Stopping $(POSTGRES_CONTAINER)..."; \
	docker stop "$(POSTGRES_CONTAINER)"; \
	echo "Writing backup to $(DB_PHYSICAL_BACKUP_DIR)/$$backup..."; \
	docker run --rm \
		--volumes-from "$(POSTGRES_CONTAINER)" \
		-v "$$(pwd)/$(DB_PHYSICAL_BACKUP_DIR):/backup" \
		alpine \
		tar cf "/backup/$$backup" -C /var/lib/postgresql/data .; \
	status=$$?; \
	echo "Starting $(POSTGRES_CONTAINER)..."; \
	docker start "$(POSTGRES_CONTAINER)"; \
	if [ $$status -ne 0 ]; then \
		echo "$(YELLOW)Backup failed.$(RESET)"; \
		exit $$status; \
	fi; \
	echo "$(CYAN)Backup complete: $(DB_PHYSICAL_BACKUP_DIR)/$$backup$(RESET)"

.PHONY: db-restore-physical
db-restore-physical: ## Restore physical Postgres backup. Usage: make db-restore-physical DUMP=pgdata_xxx.tar
	@if [ -z "$(DUMP)" ]; then \
		echo "Usage: make db-restore-physical DUMP=pgdata_xxx.tar"; \
		echo "Backups directory: $(DB_PHYSICAL_BACKUP_DIR)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)WARNING: This will WIPE current Postgres data in $(POSTGRES_CONTAINER).$(RESET)"
	@read -p "Continue? [y/N] " ans && [ "$${ans:-N}" = "y" ]
	@dump_path="$(DUMP)"; \
	if [ ! -f "$$dump_path" ]; then \
		dump_path="$(DB_PHYSICAL_BACKUP_DIR)/$(DUMP)"; \
	fi; \
	if [ ! -f "$$dump_path" ]; then \
		echo "Backup file not found: $(DUMP)"; \
		exit 1; \
	fi; \
	echo "Stopping $(POSTGRES_CONTAINER)..."; \
	docker stop "$(POSTGRES_CONTAINER)"; \
	echo "Restoring $$dump_path..."; \
	docker run --rm \
		--volumes-from "$(POSTGRES_CONTAINER)" \
		-v "$$(pwd):/work" \
		alpine \
		sh -c "rm -rf /var/lib/postgresql/data/* && tar xf /work/$$dump_path -C /var/lib/postgresql/data"; \
	status=$$?; \
	echo "Starting $(POSTGRES_CONTAINER)..."; \
	docker start "$(POSTGRES_CONTAINER)"; \
	if [ $$status -ne 0 ]; then \
		echo "$(YELLOW)Restore failed.$(RESET)"; \
		exit $$status; \
	fi; \
	echo "$(CYAN)Restore complete from $$dump_path$(RESET)"

.PHONY: db-backups
db-backups: ## List physical DB backups
	@mkdir -p "$(DB_PHYSICAL_BACKUP_DIR)"
	@ls -lh "$(DB_PHYSICAL_BACKUP_DIR)"/*.tar 2>/dev/null || echo "No backups found in $(DB_PHYSICAL_BACKUP_DIR)"

# ==============================================================================
# Redis
# ==============================================================================

.PHONY: redis-cli
redis-cli: ## Open redis-cli
	docker exec -it $(REDIS_CONTAINER) redis-cli

.PHONY: redis-flush
redis-flush: ## Flush Redis data
	@echo "$(YELLOW)WARNING: This flushes all Redis data.$(RESET)"
	@read -p "Continue? [y/N] " ans && [ "$${ans:-N}" = "y" ]
	docker exec -it $(REDIS_CONTAINER) redis-cli FLUSHALL

.PHONY: proxies-show
proxies-show: ## Show proxy pool state in Redis
	@echo "$(CYAN)Available proxies:$(RESET)"
	@docker exec -it $(REDIS_CONTAINER) redis-cli ZRANGE dota2:proxies 0 -1 WITHSCORES
	@echo ""
	@echo "$(CYAN)Leased proxies:$(RESET)"
	@docker exec -it $(REDIS_CONTAINER) redis-cli HGETALL dota2:proxies:leases

# ==============================================================================
# RabbitMQ / DLQ
# ==============================================================================

.PHONY: replay-dlq
replay-dlq: ## Replay up to 500 match IDs from DLQ
	bash deploy/scripts/replay-dlq.sh 500

.PHONY: replay-dlq-dry
replay-dlq-dry: ## Dry-run DLQ replay
	bash deploy/scripts/replay-dlq.sh 500 --dry-run

.PHONY: replay-dlq-n
replay-dlq-n: ## Replay N messages from DLQ, e.g. make replay-dlq-n N=1000
	@if [ -z "$(N)" ]; then \
		echo "Usage: make replay-dlq-n N=1000"; \
		exit 1; \
	fi
	bash deploy/scripts/replay-dlq.sh $(N)

# ==============================================================================
# Go
# ==============================================================================

.PHONY: tidy
tidy: ## Run go mod tidy for all modules
	@for mod in $(MODULES); do \
		echo "==> go mod tidy: $$mod"; \
		(cd "$$mod" && go mod tidy) || exit 1; \
	done

.PHONY: fmt
fmt: ## Format Go code
	gofmt -s -w services shared

.PHONY: vet
vet: ## Run go vet for all modules
	@for mod in $(MODULES); do \
		echo "==> go vet: $$mod"; \
		(cd "$$mod" && go vet ./...) || exit 1; \
	done

.PHONY: test
test: ## Run tests for all modules
	@for mod in $(MODULES); do \
		echo "==> go test: $$mod"; \
		(cd "$$mod" && go test ./...) || exit 1; \
	done

.PHONY: test-race
test-race: ## Run tests with race detector
	@for mod in $(MODULES); do \
		echo "==> go test -race: $$mod"; \
		(cd "$$mod" && go test -race ./...) || exit 1; \
	done

.PHONY: lint
lint: ## Run golangci-lint for all modules
	@command -v golangci-lint >/dev/null 2>&1 || { \
		echo "$(YELLOW)golangci-lint is not installed.$(RESET)"; \
		exit 1; \
	}
	@for mod in $(MODULES); do \
		echo "==> golangci-lint: $$mod"; \
		(cd "$$mod" && golangci-lint run ./...) || exit 1; \
	done

.PHONY: check
check: fmt vet test-race ## Format, vet, and test (with race detector)

.PHONY: build
build: ## Build all Go services into ./bin
	@mkdir -p bin
	@for svc in $(SERVICES); do \
		echo "==> building $$svc"; \
		(cd "services/$$svc" && go build -o "../../bin/$$svc" .) || exit 1; \
	done
	@echo "$(CYAN)Binaries written to ./bin$(RESET)"

# ==============================================================================
# Local Run
# ==============================================================================

.PHONY: run-%
run-%: env-sync ## Run service locally, e.g. make run-parser
	@if [[ " $(SERVICES) " =~ " $* " ]]; then \
		go run ./services/$*; \
	else \
		echo "Unknown service: $*"; \
		echo "Valid services: $(SERVICES)"; \
		exit 1; \
	fi

# ==============================================================================
# ML Training & Inference API
# ==============================================================================

.PHONY: train
train: ## Train LightGBM model: make train PATCH=<id> (default: auto-detect)
	$(COMPOSE) --profile db --profile train run --rm trainer $(if $(PATCH),--patch $(PATCH),)

.PHONY: train-agg-only
train-agg-only: ## Populate aggregate tables only: make train-agg-only PATCH=<id>
	$(COMPOSE) --profile db --profile train run --rm trainer $(if $(PATCH),--patch $(PATCH),) --agg-only

.PHONY: up-api
up-api: ## Start ML inference API (foreground)
	$(COMPOSE) --profile db --profile api up

.PHONY: up-api-d
up-api-d: ## Start ML inference API (background)
	$(COMPOSE) --profile db --profile api up -d

.PHONY: down-api
down-api: ## Stop ML inference API
	$(COMPOSE) --profile db --profile api down

.PHONY: reload-api
reload-api: ## Hot-reload model for a patch: make reload-api PATCH=<id> TOKEN=<admin_token>
	@if [ -z "$(PATCH)" ]; then \
		echo "Usage: make reload-api PATCH=<patch_id> [TOKEN=<admin_token>]"; \
		exit 1; \
	fi
	@token=$${TOKEN:-$$(grep -oP '^STRATZ_ADMIN_TOKEN=\K.*' $(ENV_FILE) 2>/dev/null || echo "")}; \
	curl -X POST -H "Authorization: Bearer $$token" http://localhost:$(or $(API_PORT),8080)/reload/$(PATCH)

.PHONY: test-api
test-api: ## Quick smoke-test the inference API
	@echo "Testing API health..."; \
	curl -s http://localhost:$(or $(API_PORT),8080)/health | python3 -m json.tool; \
	echo ""; \
	echo "Testing /predict with 4-step draft (phase 1 bans)..."; \
	curl -s -X POST http://localhost:$(or $(API_PORT),8080)/predict \
		-H "Content-Type: application/json" \
		-d '{"patch_id":60,"first_pick_team":0,"draft":[{"hero_id":1,"is_pick":false,"team":0,"order":1},{"hero_id":2,"is_pick":false,"team":0,"order":2},{"hero_id":3,"is_pick":false,"team":1,"order":3},{"hero_id":4,"is_pick":false,"team":1,"order":4}]}' | python3 -m json.tool

.PHONY: migrate-ml
migrate-ml: ## Apply only the ML migration (002_ml.sql)
	@echo "$(YELLOW)Applying ML migration...$(RESET)"
	@name="002_ml.sql"; \
	applied=$$(docker exec -i $(POSTGRES_CONTAINER) psql \
		-U $(POSTGRES_USER) \
		-d $(POSTGRES_DB) \
		-tAc "SELECT 1 FROM _migrations WHERE name = '$$name';"); \
	if [ "$$applied" = "1" ]; then \
		echo "  SKIP  $$name (already applied)"; \
	else \
		echo "  APPLY $$name"; \
		docker exec -i $(POSTGRES_CONTAINER) psql \
			-U $(POSTGRES_USER) \
			-d $(POSTGRES_DB) \
			-v ON_ERROR_STOP=1 < $(MIGRATION_DIR)/$$name || exit 1; \
		docker exec -i $(POSTGRES_CONTAINER) psql \
			-U $(POSTGRES_USER) \
			-d $(POSTGRES_DB) \
			-v ON_ERROR_STOP=1 \
			-c "INSERT INTO _migrations (name) VALUES ('$$name') ON CONFLICT DO NOTHING;" || exit 1; \
		echo "$(CYAN)ML migration applied.$(RESET)"; \
	fi

# ==============================================================================
# Docker Buildx Bake
# ==============================================================================

.PHONY: bake
bake: ## Build all service images
	cd deploy && docker buildx bake --allow=fs.read=.. -f docker-bake.hcl

.PHONY: bake-%
bake-%: ## Build one image, e.g. make bake-parser
	cd deploy && docker buildx bake --allow=fs.read=.. -f docker-bake.hcl $*

# ==============================================================================
# Maintenance
# ==============================================================================

.PHONY: clean
clean: ## Remove local build artifacts
	rm -rf bin
	@echo "Cleaned ./bin"

.PHONY: nuke
nuke: ## DESTRUCTIVE: stop compose stack and remove project volumes
	@echo "$(YELLOW)WARNING: This removes this project's compose containers and volumes.$(RESET)"
	@read -p "Continue? [y/N] " ans && [ "$${ans:-N}" = "y" ]
	$(COMPOSE) --profile all down -v --remove-orphans
	@echo "$(CYAN)Project stack removed.$(RESET)"
