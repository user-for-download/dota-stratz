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
# Containers
# ------------------------------------------------------------------------------

POSTGRES_USER ?= dota2
POSTGRES_DB   ?= dota2
POSTGRES_CONTAINER := dota2-postgres

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
env: ## Create deploy/.env from .env.example if missing
	@test -f "$(ENV_FILE)" || cp deploy/.env.example "$(ENV_FILE)" 2>/dev/null && echo "$(ENV_FILE) ready."

.PHONY: env-sync
env-sync: env ## Sync deploy/.env to root .env
	@cp "$(ENV_FILE)" "$(ROOT_ENV_FILE)"

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

.PHONY: down
down: ## Stop services
	$(COMPOSE) --profile all down

.PHONY: downv
downv: ## Stop and remove volumes (destructive)
	@echo "$(YELLOW)WARNING: removes project volumes.$(RESET)"
	@read -p "Continue? [y/N] " ans && [ "$${ans:-N}" = "y" ]
	$(COMPOSE) --profile all down -v

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
		-U $(POSTGRES_USER) -d $(POSTGRES_DB) -v ON_ERROR_STOP=1 \
		-c "CREATE TABLE IF NOT EXISTS _migrations (name TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now());"
	@for f in $$(find "$(MIGRATION_DIR)" -maxdepth 1 -name '*.sql' | sort); do \
		name=$$(basename "$$f"); \
		applied=$$(docker exec -i $(POSTGRES_CONTAINER) psql \
			-U $(POSTGRES_USER) -d $(POSTGRES_DB) \
			-tAc "SELECT 1 FROM _migrations WHERE name = '$$name';"); \
		if [ "$$applied" = "1" ]; then \
			echo "  SKIP  $$name"; \
		else \
			echo "  APPLY $$name"; \
			( echo "BEGIN;"; cat "$$f"; echo "INSERT INTO _migrations (name) VALUES ('$$name') ON CONFLICT DO NOTHING; COMMIT;" ) | \
			docker exec -i $(POSTGRES_CONTAINER) psql \
				-U $(POSTGRES_USER) -d $(POSTGRES_DB) -v ON_ERROR_STOP=1 || exit 1; \
		fi; \
	done
	@echo "$(CYAN)Migrations complete.$(RESET)"

.PHONY: db-reset
db-reset: ## Drop and recreate database, then run migrations
	@echo "$(YELLOW)WARNING: drops $(POSTGRES_DB).$(RESET)"
	@read -p "Continue? [y/N] " ans && [ "$${ans:-N}" = "y" ]
	docker exec -i $(POSTGRES_CONTAINER) psql -U $(POSTGRES_USER) -d postgres -v ON_ERROR_STOP=1 \
		-c "DROP DATABASE IF EXISTS $(POSTGRES_DB) WITH (FORCE);" \
		-c "CREATE DATABASE $(POSTGRES_DB);"
	$(MAKE) migrate

# ==============================================================================
# Go Toolchain
# ==============================================================================

.PHONY: tidy
tidy: ## Run go mod tidy for all modules
	@for mod in $(MODULES); do echo "==> tidy: $$mod"; (cd "$$mod" && go mod tidy) || exit 1; done

.PHONY: fmt
fmt: ## Format Go code
	gofmt -s -w services shared

.PHONY: vet
vet: ## Run go vet for all modules
	@for mod in $(MODULES); do echo "==> vet: $$mod"; (cd "$$mod" && go vet ./...) || exit 1; done

.PHONY: test
test: ## Run Go tests
	@for mod in $(MODULES); do echo "==> test: $$mod"; (cd "$$mod" && go test ./...) || exit 1; done

.PHONY: lint
lint: ## Run golangci-lint
	@for mod in $(MODULES); do echo "==> lint: $$mod"; (cd "$$mod" && golangci-lint run ./...) || exit 1; done

.PHONY: check
check: fmt vet test ## Format, vet, and test

# ==============================================================================
# ML Training
# ==============================================================================

.PHONY: train
train: ## Train DraftBERT: make train PATCH=<id>
	$(COMPOSE) --profile db --profile train run --rm \
		--entrypoint python trainer -m trainer.main \
		$(if $(PATCH),--patch $(PATCH),)

.PHONY: train-live
train-live: ## Train LiveDraftBERT: make train-live PATCH=<id>
	$(COMPOSE) --profile db --profile train run --rm \
		--entrypoint python trainer -m trainer.main \
		$(if $(PATCH),--patch $(PATCH),) --live

.PHONY: train-agg-only
train-agg-only: ## Populate aggregates + embeddings: make train-agg-only PATCH=<id>
	$(COMPOSE) --profile db --profile train run --rm \
		--entrypoint python trainer -m trainer.main \
		$(if $(PATCH),--patch $(PATCH),) --agg-only

.PHONY: lr-find
lr-find: ## Run LR Range Test: make lr-find PATCH=<id> [--live]
	$(COMPOSE) --profile db --profile train run --rm \
		--entrypoint python trainer -m trainer.main \
		$(if $(PATCH),--patch $(PATCH),) --lr-find $(if $(live),--live,)

# ==============================================================================
# Inference API
# ==============================================================================

.PHONY: up-api
up-api: ## Start API (foreground)
	$(COMPOSE) --profile db --profile api up

.PHONY: up-api-d
up-api-d: ## Start API (background)
	$(COMPOSE) --profile db --profile api up -d

.PHONY: down-api
down-api: ## Stop API
	$(COMPOSE) --profile db --profile api down

.PHONY: reload-api
reload-api: ## Hot-reload model: make reload-api PATCH=<id>
	@token=$${TOKEN:-$$(grep -oP '^STRATZ_ADMIN_TOKEN=\K.*' $(ENV_FILE) 2>/dev/null || echo "")}; \
	curl -s -X POST -H "Authorization: Bearer $$token" \
		http://localhost:$(or $(API_PORT),8080)/reload/$(PATCH) | python3 -m json.tool

.PHONY: test-api
test-api: ## Smoke-test API health + predict
	@curl -s http://localhost:$(or $(API_PORT),8080)/health | python3 -m json.tool

# ==============================================================================
# Docker Build
# ==============================================================================

.PHONY: bake
bake: ## Build all Docker images
	cd deploy && docker buildx bake --allow=fs.read=.. -f docker-bake.hcl

.PHONY: bake-%
bake-%: ## Build one image, e.g. make bake-api
	cd deploy && docker buildx bake --allow=fs.read=.. -f docker-bake.hcl $*

# ==============================================================================
# Cleanup
# ==============================================================================

.PHONY: clean
clean: ## Remove build artifacts
	rm -rf bin

.PHONY: clean-all
clean-all: ## DESTRUCTIVE: nuke all Docker containers/volumes/images
	@echo "$(YELLOW)Nuking Docker...$(RESET)"
	-docker stop $$(docker ps -aq) 2>/dev/null || true
	-docker rm $$(docker ps -aq) 2>/dev/null || true
	-docker network prune -f
	-docker volume prune -f
	-docker rmi -f $$(docker images -qa) 2>/dev/null || true
