# dota-stratz — Full Project Audit

> **Audit date:** 2026-07-06  
> **Scope:** ALL services, shared libraries, deploy configs, frontend, API  
> **Files analyzed:** ~100 source files (57 Go, 30+ Python, 10+ JS/HTML, 5 SQL migrations, 5 shell scripts)  
> **Skills used:** python-performance-optimization, supabase-postgres-best-practices, pandas-pro, machine-learning, golang-pro, golang-database, fastapi, docker-expert, rabbitmq-development

---

## 🔴 CRITICAL BUGS

| # | Service | File | Line(s) | Issue | Detail |
|---|---------|------|---------|-------|--------|
| **C1** | API | `api/live_features.py` | 332-333 | **Mega creeps logic inverted** | `mega_radiant` checks `dire_barracks >= 6` (Radiant's own barracks destroyed) instead of `radiant_barracks >= 6`. The model receives inverted mega-creeps signals for late-game predictions. |
| **C2** | Frontend | `app.js`, `live.js` | 7, 4 | **WebSocket hardcoded to `ws://`** | Both URLs use unencrypted `ws://`. Pages loaded over HTTPS will have WebSocket connections blocked by browsers (mixed content). Features silently break in production. |
| **C3** | API | `api/app.py` | 89-95 | **CORS only allows port 80** | `allow_origins` is hardcoded to `localhost`, `localhost:80`, `127.0.0.1:80`. Any other port (3000, 5173, production domain) is blocked. |
| **C4** | Go Shared | `shared/go-common/logger/logger.go` | 21-46 | **`sync.Once` prevents test re-initialization** | `logOnce.Do()` fires once ever. Tests that reset `Log = nil` and call `InitLogger()` again get nil. `TestInitLogger_SetsGlobal` will fail at runtime. |
| **C5** | Go ID-Fetcher | `id-fetcher/internal/queue/rabbitmq_publisher.go` | 231-253 | **`reconnectIfNeeded` returns nil on failed reconnect** | After `p.mu.Unlock()`, another goroutine can call `Close()`. `reconnect()` sees `p.closed` and returns without reconnecting. `reconnectIfNeeded` then returns `nil` error on a dead connection → nil-pointer panic in `publishChannel`. |
| **C6** | Deploy | `deploy/compose.yaml` | 374 | **Trainer writes models as root, API reads as non-root** | `trainer` runs as `user: "0"` (root) to write to `ml-models` volume. `api` doesn't specify user → default non-root may not read root-owned files. |
| **C7** | Deploy | `deploy/migration/002_ml.sql` | 21-29 | **`analytics_writer` has NO access to `ml` schema** | Grants exist for `public` and `analytics` schemas but NOT `ml`. Trainer/API writes to `ml.*` tables will fail with "permission denied". |
| **C8** | Deploy | `deploy/rabbitmq/Dockerfile` | 17 | **`definitions.json` is dead code** | Copied into the image but `rabbitmq.conf` has no `management.load_definitions` directive. The file is never loaded. Contains no queues/exchanges anyway. |

---

## 🟠 HIGH-SEVERITY BUGS

| # | Service | File | Line(s) | Issue | Detail |
|---|---------|------|---------|-------|--------|
| **H1** | API | `api/live_predict.py` | 123-158 | **`predict_with_cache` is dead code** | The method exists for caching transformer/static embeddings, but the WebSocket handler calls `predict()` directly every tick. Transformer + static MLP recompute on every WebSocket tick (~2ms each), defeating the caching architecture. |
| **H2** | API | `api/live_predict.py` | 82-99 | **LivePredictor has no thread safety** | Unlike `Predictor` (which uses `threading.RLock`), `LivePredictor` has no locking. Concurrent requests trigger redundant model loads for the same `patch_id`. |
| **H3** | API | `api/model_live.py` | 19 | **Default `num_dynamic_features=15` vs actual 24** | Constructor default is 15, but `live_features.py` defines 24. If `LiveDraftBERT()` is instantiated without the metadata JSON, it crashes with a dimension mismatch. Currently mitigated by `live_predict.py` passing the correct value from schema. |
| **H4** | API | `api/app.py` | 578-661 | **Live WS outer loop blocks match switching** | Inner polling loop runs until match ends. Client cannot send a new message to switch matches. Must close and reopen WebSocket. |
| **H5** | API | `api/live_predict.py` | 350 | **`buyback_diff` direction inverted vs other diffs** | Uses `dire_X - radiant_X` while all other diffs use `radiant_X - dire_X`. Model receives an inconsistent signal. |
| **H6** | Go Detail-Fetcher | `detail-fetcher/internal/config/config.go` | 102-104 | **Config requires DSN but main.go treats it as optional** | Config always rejects empty DSN, making the "optional" code path in main.go unreachable dead code. Inconsistency between validation and runtime. |
| **H7** | Go ID-Fetcher | `id-fetcher/internal/api/opendota_client.go` | 168-191 | **Contradictory comments on SQL watermark filtering** | Line 170 says "match_id filter is NOT in SQL." Line 190 says "SQL already guarantees match_id > watermark." The SQL has no such filter. |
| **H8** | Go Detail-Fetcher | `detail-fetcher/internal/consumer/consumer.go` | 29-37 | **`ConsumeTag` is dead code** | Duplicate of `Consume`, never called anywhere. |
| **H9** | Deploy | `deploy/compose.yaml` | 220-250 | **`detail-fetcher` has NO memory/CPU resource limits** | Unlike every other service, it has no `deploy.resources.limits`. Can consume unlimited memory/CPU. |
| **H10** | Deploy | `deploy/compose.yaml` | 434 | **`frontend` healthcheck uses `curl` which may not exist** | nginx:alpine image doesn't include `curl`. Healthcheck will fail with "curl: not found". |
| **H11** | Deploy | `deploy/docker-bake.hcl` | 6, 46 | **`frontend` not in default bake group; inconsistent build context** | Default group only includes 4 Go services. Running `docker bake` doesn't build frontend/api/trainer. Also, bake context differs from compose.yaml. |
| **H12** | Deploy | `deploy/models/` | varies | **Feature schema mismatch between baseline (216) and patch_60 (219)** | `feature_schema_baseline.json` has 216 features; `feature_schema_patch_60.json` has 219. If API loads baseline schema but model expects 219, dimension mismatch crash. |
| **H13** | Trainer | `aggregates.py` | 767 | **`[prior_weight] * 22` fragility** | If SQL `%s` count changes, parameter list silently mismatches. No assertion or count check. |

---

## 🟡 MEDIUM-SEVERITY BUGS

| # | Service | File | Line(s) | Issue | Detail |
|---|---------|------|---------|-------|--------|
| **M1** | API | `api/lookahead.py` | 36-38 | **Private attribute access across modules** | `lookahead` accesses `predictor._lock`, `predictor._models`, `predictor._schemas`. Fragile coupling. |
| **M2** | API | `api/app.py` | 126-127 | **Sync `/predict` endpoint** | Uses `def` (not `async def`). Each prediction blocks a thread pool thread. Under load, thread pool exhaustion. |
| **M3** | API | `api/multitask.py` | 55-68 | **`classify_slot` hardcodes patch 60 ban count** | Hardcodes 12 bans. Ignores `next_slot` parameter. Incorrect for other patches. |
| **M4** | Go Shared | `shared/go-common/mq/publisher.go` | 188-235 | **Undocumented lock ordering constraint** | `reconnectMu` → `closeMu` order is fragile. Any future code acquiring in reverse order would deadlock. |
| **M5** | Go ID-Fetcher | `id-fetcher/internal/api/opendota_client.go` | 83 | **`fmt.Sprintf` for SQL construction** | Anti-pattern. Safe today (integer values), but fragile if inputs change. |
| **M6** | Go Parser | `parser/internal/models/opendota.go` | 424-441 | **`FlexString.UnmarshalJSON` stores raw JSON bytes** | Non-string types stored as raw bytes (`"true"`, `"42"`). Correct for current use but surprising behavior. |
| **M7** | Deploy | `deploy/compose.yaml` | 189-191 | **`.env` and `.env.example` are out of sync** | Missing `TRAINER_LOOKBACK_PATCHES`, `TRAINER_PRIOR_PATCH_WEIGHT`, `TRAINER_LEAGUE_ONLY`, `TRAINER_LOBBY_TYPES` from `.env`. Section ordering differs. |
| **M8** | Deploy | `deploy/compose.yaml` | 46 | **`ID_FETCHER_START_RUN=true` contradicts compose default** | `.env` sets `true`, compose defaults to `false`. Misleading. |
| **M9** | Deploy | `deploy/migration/002_ml.sql` | 308-321 | **`update_feature_snapshots()` scans ALL matches** | O(n) per date on the entire matches table. Slow for large databases. |
| **M10** | Deploy | `deploy/scripts/seed-constants.sh` | 26-31 | **Writes to `/tmp/` which may be limited** | Concurrent runs or tmpfs limits could cause failures. |
| **M11** | Deploy | `deploy/scripts/check-proxy.sh` | 18 | **Overwrites `proxy.txt` even with zero proxies** | If no valid proxies found, file is emptied. Proxy-manager loses all proxies. |
| **M12** | Deploy | `deploy/scripts/replay-dlq.sh` | 191-215 | **`trap` overwrites previous traps** | Multiple `trap 'rm -f ...' EXIT` calls. Only the last one executes on exit. |
| **M13** | Deploy | `deploy/scripts/backfill-minute-stats.sh` | 23 | **Connects to localhost by default** | Assumes Postgres on localhost:5432. Inconsistent with other scripts using `docker exec`. |
| **M14** | Deploy | `deploy/grafana/proxy-manager.json` | 120-141 | **Two dashboard panels overlap** | "Cooldown Rate" and "Rate-Limited (Proactive)" both at `y: 12`. One overlays the other. |

---

## 🟠 DESIGN LIMITATIONS (not bugs)

| # | Service | File | Lines | Issue |
|---|---------|------|-------|-------|
| **L1** | Trainer | `aggregates.py` | 784-807 | **Prior bans not weighted** — `_team_hero_bans_prior()` uses `COUNT(*)` without `prior_weight`. |
| **L2** | Trainer | `train_live.py` | 77-86 | **Full forward pass during training** — Caching is inference-only; training needs backprop through all params. ✅ Correct behavior. |
| **L3** | Trainer | `train_live.py` | 66 | **No DataLoader** — Manual tensor slicing. Acceptable for CPU-only. |
| **L4** | Trainer | `dataset_live.py` | 135-156 | **Python loop for data loading** — One-time ~2s step. ✅ Not a bottleneck. |
| **L5** | Trainer | `dataset_pt.py` | 33-37 | **Row-by-row tensor population** — One-time ~200ms. ✅ Acceptable. |
| **L6** | Trainer | `live_features.py` | 214-233 | **Simplified death timers** — No buyback-aware clearing. Acceptable approximation. |
| **L7** | Trainer | `live_features.py` | 199-201 | **Aegis detection heuristic** — Roshan steals/denies can false-positive. |
| **L8** | Go | `proxypool/socks4.go` | 64 | **SOCKS4 uses `SetDeadline` instead of context** — Graceful shutdown may not interrupt SOCKS4 connections quickly. |
| **L9** | Go | `proxypool/pool.go` | 658-758 | **`WithProxy` has 100+ lines of nested error handling** — Hard to reason about. |
| **L10** | API | `live_predict.py` | 161-180 | **Synchronous `requests.get` blocks executor threads** — Called from async WebSocket handlers via `run_in_executor`. |

---

## 🔵 TEST COVERAGE GAPS

| # | Module | Tests Exist? | Missing Coverage |
|---|--------|-------------|-----------------|
| **T1** | Trainer `train_pt.py` | ❌ None | TorchScript export, dummy shapes, metadata writing |
| **T2** | Trainer `train_live.py` | ❌ None | Custom training loop, gradient clipping, early stopping |
| **T3** | Trainer `model_pt.py` / `model_live.py` | ❌ None | Forward pass, shape assertions, padding mask, edge cases |
| **T4** | Trainer `live_features.py` | ❌ None | All 24 dynamic features (death timers, aegis, momentum) |
| **T5** | Trainer `db.py` | ❌ None | `fetch_patch_id`, `load_heroes` |
| **T6** | Trainer `config.py` | ❌ None | Env var parsing, DSN construction |
| **T7** | Trainer `features.py` | ✅ Partial | `make_target`, `feature_column_names` only — no SQL contract tests |
| **T8** | Trainer `dataset_pt.py` | ✅ Partial | Split ratios and shapes — no prefix augmentation, NaN handling |
| **T9** | Trainer `aggregates.py` | ✅ Partial | Helpers only — no populator functions tested |
| **T10** | API `predictor.py` | ❌ None | No tests for prediction logic |
| **T11** | API `reasoning.py` | ❌ None | No tests for reasoning generation |
| **T12** | API `lookahead.py` | ❌ None | No tests for Monte Carlo rollouts |
| **T13** | API `draft_state.py` | ❌ None | No tests for draft validation |
| **T14** | Go `proxypool` | ❌ None | No tests for proxy pool operations |
| **T15** | Go `mq` | ❌ None | No tests for RabbitMQ publisher/consumer |

---

## 🟢 THINGS DONE RIGHT

| Area | Good Practice |
|------|---------------|
| **SQL Injection** | `_VALID_TABLES` frozenset guard prevents injection via f-string table name |
| **Transaction Safety** | `_clean_patch_rows` does NOT commit, making DELETE+INSERT atomic |
| **PIT Safety** | Training features use LATERAL `ORDER BY as_of_date DESC LIMIT 1` against snapshot tables |
| **Cross-patch Lookback** | Sparse combo-keyed tables include prior-patch weighted data |
| **Bayesian Shrinkage** | `_shrunk_wr()` applies prior-based shrinkage to win rates |
| **LiveDraftBERT Architecture** | `encode_draft()`/`forward_dynamic()` separation designed for caching |
| **Feature Contract** | Schemas written to JSON at training time, consumed by API |
| **Go Concurrency** | `sync.Once`, `sync.RWMutex`, channel-based error propagation in Go services |
| **Docker** | Multi-stage builds, non-root users (except trainer for volume writes) |
| **Monitoring** | Prometheus metrics on all Go services + Grafana dashboards |
| **Dead-letter queues** | Every RabbitMQ queue has DLQ with 24h TTL |

---

## SUMMARY STATS

| Metric | Value |
|--------|-------|
| Files analyzed | ~100 (57 Go, 30+ Python, 10+ JS/HTML, 5 SQL, 5 shell scripts) |
| **Critical bugs** | **8** (C1-C8) |
| **High-severity bugs** | **13** (H1-H13) |
| **Medium-severity bugs** | **14** (M1-M14) |
| Design limitations | 10 (L1-L10) |
| Missing test coverage | 15 modules (T1-T15) |
| **Total distinct issues** | **45** |
| Good practices | 11 |

---

*Generated by OpenAgent using golang-pro, golang-database, fastapi, docker-expert, rabbitmq-development, python-performance-optimization, supabase-postgres-best-practices, pandas-pro, and machine-learning skills.*
