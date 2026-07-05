# dota-stratz Context System

## Core (AI agent standards)
- `core/standards/code-quality.md` — Go code conventions, patterns, review criteria
- `core/standards/documentation.md` — Documentation style and structure
- `core/standards/test-coverage.md` — Testing requirements and patterns
- `core/workflows/code-review.md` — Code review workflow for this project
- `core/workflows/task-delegation-basics.md` — Delegation rules and subagent routing

## Domain (project knowledge)
 - `domain/concepts/pipeline.md` — Event-driven pipeline data flow (ML: 7 aggregate tables, 218-dim features, stale-row protection, configurable pro-match filtering)
- `domain/concepts/services.md` — Service architecture (ID Fetcher, Detail Fetcher, Parser, Proxy Manager, Trainer, API)
- `domain/concepts/database.md` — Database schema, migrations (001–013), 7 ML aggregate tables + player_time_series_arrays
- `domain/lookup/services.md` — Service ports, deps, config quick reference
- `domain/lookup/ports.md` — Port allocation map
- `domain/lookup/env-vars.md` — Environment variable catalog
- `domain/guides/local-dev.md` — Running the stack locally

## Deployment
- `deployment/concepts/compose.md` — Docker Compose profiles and bake
- `deployment/concepts/monitoring.md` — Prometheus + Grafana setup
- `deployment/lookup/make-targets.md` — Makefile targets reference

## Development
- `development/concepts/go-patterns.md` — Shared library, proxypool, patterns (now includes native SOCKS4 dialer)
- `development/guides/branch-strategy.md` — Git workflow
- `development/lookup/quick-commands.md` — Common dev commands

## Context System (this repo)
- `context-system/operations/harvest.md` — Harvest summaries into context
- `context-system/operations/extract.md` — Extract from docs/code/URLs
- `context-system/operations/organize.md` — Restructure flat files
- `context-system/standards/mvi.md` — Minimal Viable Information principle
- `context-system/standards/structure.md` — Function-based directory rules
