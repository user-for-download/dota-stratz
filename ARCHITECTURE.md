# Dota 2 Match Analysis System — Architecture

## Overview

An event-driven microservice pipeline that ingests Dota 2 match data from the [OpenDota API](https://www.opendota.com/), processes it through a multi-stage queue-based architecture, and stores the result in PostgreSQL for analytics and ML feature engineering.

**Language:** Go 1.26.3 / Python 3.12  
**ML Framework:** PyTorch 2.2+ (DraftBERT Transformer + MLP)  
**Messaging:** RabbitMQ with dead-letter queues and automatic reconnection  
**Database:** PostgreSQL 16 (partitioned, with deferred FK constraints)  
**Caching/State:** Redis 7 (proxy pool backend + checkpoint state)  
**Observability:** Prometheus + Grafana (pre-configured dashboards and alerts)  
**Deployment:** Docker Compose with docker buildx bake

---

## Pipeline Data Flow

```
[OpenDota API] ──► [ID Fetcher] ──► [queue.match_ids] ──► [Detail Fetcher]
                                                                │
                                                                ▼
                                                        [queue.raw_matches]
                                                                │
                                                                ▼
                                                            [Parser]
                                                                │
                                                                ▼
                                                          [PostgreSQL]
                                                                │
                                                     (materialized views)
                                                                 │
                                                                 ▼
                                                     [Analytics Schema]
                                               (7 agg + 7 PIT snapshot tables)
                                                                 │
                                                                 ▼
                                                          [Trainer]
                                                    (PyTorch DraftBERT)
                                                                 │
                                                                 ▼
                                                       [ML Models]
                                                    (TorchScript .pt files)
                                                                 │
                                                                 ▼
                                                      [Inference API]
                                                    (FastAPI, port 8080)
                                                                 │
                                                                 ▼
                                                     [Frontend (Nginx)]
                                                       (Draft Predictor, port 80)
```

---

## Services

### 1-4. Go Microservices (ID Fetcher, Detail Fetcher, Parser, Proxy Manager)

Event-driven pipeline services connected via RabbitMQ. See original documentation for details on each service.

| Service | Package | Port | Purpose |
|---------|---------|------|---------|
| ID Fetcher | `services/id-fetcher/` | 9094 | Cron-scheduled OpenDota match ID discovery |
| Detail Fetcher | `services/detail-fetcher/` | 9091 | Match JSON download via proxy pool |
| Parser | `services/parser/` | 9093 | JSON parsing + bulk PostgreSQL insert |
| Proxy Manager | `services/proxy-manager/` | 9090 | Redis-backed proxy pool management |

---

### 5. Trainer (PyTorch DraftBERT)

**Purpose:** Batch CLI service. Populates aggregate/snapshot tables and trains PyTorch DraftBERT model for draft prediction.

| Aspect | Detail |
|---|---|
| Package | `services/trainer/` |
| Framework | PyTorch 2.2+ (CPU-only in Docker) |
| Dependencies | PostgreSQL (psycopg2 + SQLAlchemy) |
| Config | Environment variables (`TRAINER_*`) |

**Pipeline stages:**
1. **Aggregate population** — 7 populator functions compute `ml.*_agg` tables per patch
2. **Snapshot population** — 7 PIT-safe snapshot tables with cross-patch lookback
3. **Training** — PyTorch DraftBERT model with configurable hyperparameters
4. **Export** — TorchScript JIT compilation for CPU inference

**Model architecture — MultiModalDraftBERT:**
- **Sequence branch:** Transformer encoder (d_model=128, nhead=4, 3 layers)
  - Hero embedding: `nn.Embedding(165, 128, padding_idx=0)`
  - Action embedding: `nn.Embedding(5, 128, padding_idx=0)` — tokens 1-4 for RadBan/DireBan/RadPick/DirePick
  - Positional embedding: `nn.Embedding(max_seq_len, 128)`
  - Embedding dropout: 0.3
  - Mean pooling over sequence dimension
- **Tabular branch:** MLP for 59 continuous features
  - Input LayerNorm → Linear(59, 64) → ReLU → Dropout(0.3)
- **Fusion head:** Linear(192, 64) → ReLU → Dropout(0.3) → Linear(64, 1)
- **Output:** Raw logit for P(Radiant wins) via BCEWithLogitsLoss
- **Parameter count:** ~639K

**Training configuration (all from `deploy/.env`):**

| Parameter | Env Variable | Default | Description |
|-----------|-------------|---------|-------------|
| CPU threads | `TRAINER_NUM_THREADS` | 12 | `torch.set_num_threads()` |
| Batch size | `TRAINER_BATCH_SIZE` | 256 | DataLoader batch size |
| Epochs | `TRAINER_EPOCHS` | 15 | Training epochs |
| Learning rate | `TRAINER_LR` | 1e-4 | AdamW learning rate |
| Weight decay | `TRAINER_WEIGHT_DECAY` | 1e-3 | AdamW weight decay |
| Max sequence length | `TRAINER_MAX_SEQ_LEN` | 50 | Draft sequence padding length |
| Transformer dim | `TRAINER_D_MODEL` | 128 | Embedding dimension |
| Attention heads | `TRAINER_NHEAD` | 4 | Multi-head attention |
| Transformer layers | `TRAINER_NUM_LAYERS` | 3 | Encoder depth |

**Key behaviors:**
- TorchScript export uses `copy.deepcopy()` before `.cpu()` to avoid severing optimizer references
- Dummy tensors use non-zero hero IDs (5, 10, 15) to prevent `to_padded_tensor` crash
- `ReduceLROnPlateau(patience=2, factor=0.5)` for adaptive learning rate
- Chronological train/val split (oldest → train, newest → val)
- All hyperparameters configurable via `.env` — zero code changes needed

**Output:**
- `draftbert_compiled_{patch_id}.pt` — TorchScript model for API
- `draftbert_weights_{patch_id}.pt` — PyTorch state dict for resume
- `model_patch_{patch_id}_meta.json` — Training metadata
- `feature_schema_patch_{patch_id}.json` — Feature contract (59 aggregate columns + `max_seq_len`)

---

### 6. Inference API (FastAPI + TorchScript)

**Purpose:** Online FastAPI service. Loads TorchScript models and serves draft predictions via HTTP + WebSocket.

| Aspect | Detail |
|---|---|
| Package | `services/api/` |
| Port | 8080 |
| Framework | FastAPI + PyTorch JIT |
| Memory | 2G (PyTorch + MCTS batch inference) |

**Endpoints:**
- `GET /health` — Health check with DB ping and loaded patches
- `POST /predict` — Draft prediction with MCTS rollouts
- `POST /predict-match` — 5v5 composition evaluation
- `POST /reload/{patch_id}` — Hot-reload model (admin token)
- `WS /ws/draft` — Real-time MCTS progress streaming

**Prediction flow:**
1. Client sends draft state (picks/bans, teams, patch)
2. API validates draft order against per-patch pattern (`DRAFT_PATTERNS`)
3. Pre-fetches batch aggregate data (7 queries per request)
4. Builds sequence tensors (heroes + actions) and tabular features
5. **Batched TorchScript inference** — single matrix multiply for all candidates (~2ms)
6. Team-hero proficiency boosts (65%+ WR → +0.25 score boost)
7. **Monte Carlo rollouts** — 40 simulations × 15 top candidates, batch-evaluated
8. Returns top-N recommendations with reasoning

**WebSocket streaming (`/ws/draft`):**
- Queue-based async bridge between sync executor (MCTS) and async WebSocket
- Progress packets streamed per-candidate during MCTS evaluation
- Frontend shows real-time MCTS overlay with progress bar and top picks

**Feature categories (59 aggregate + sequence):**
- `th_*` (14): Team-hero aggregate
- `ph_*` (15): Player-hero aggregate
- `sy_*` (2): Synergy with allies
- `co_*` (3): Counter vs enemies
- `h2h_*` (2): Head-to-head record
- `bl_*` (12): Hero baseline stats
- `hds_*` (2): Hero draft-slot win rate
- Derived (4): Missingness flags, delta features, role interactions

**Key behaviors:**
- Thread-safe: `threading.RLock` for model swap, `BoundedSemaphore` for DB pool
- Model loading (file I/O) happens outside the lock
- NULL-safe: `_float()`/`_int()` helpers for all aggregate lookups
- Draft patterns validated per-patch from `DRAFT_PATTERNS` dict
- `for_team` parameter enables per-team recommendations (API inverts for Dire)

---

### 7. Frontend (Draft Predictor)

**Purpose:** Interactive web application for draft prediction with real-time MCTS streaming.

| Aspect | Detail |
|---|---|
| Package | `services/frontend/` |
| Port | 80 (Nginx) |
| Stack | Vanilla HTML/CSS/JS, WebSocket |
| Proxy | `/api/*` → `dota2-ml-api:8080`, `/ws/*` → WebSocket upgrade |

**Features:**
- Team selection (1000 teams) with mutual exclusion
- First-pick side toggle (changes draft order pattern)
- Real-time MCTS progress overlay (WebSocket streaming)
- 24-slot Captain's Mode draft with ban/pick phases
- Dual-column recommendation panels (bans + picks)
- Hero image grid with highlighting and tooltip predictions

---

## Shared Library

**Module:** `shared/go-common/`

| Package | Description |
|---|---|
| `mq/` | RabbitMQ connection, queue declaration, auto-reconnecting consumer/publisher with confirms |
| `db/` | pgxpool connection helper |
| `cache/` | Redis connection helper |
| `logger/` | Structured logging (zap) |
| `checkpoint/` | Pipeline watermark tracking |
| `proxypool/` | Redis-backed proxy pool with Lua scripts, SOCKS4/5 support, Prometheus metrics |

---

## Database Schema

### Migration Files

| File | Description |
|---|---|
| `001__init.sql` | Core schema: matches, players (RANGE-partitioned), event tables, indexes |
| `002_ml.sql` | Analytics + ML: 7 aggregate tables + 7 PIT-safe snapshot tables |
| `003_static.sql` | Static reference data: heroes, items, abilities, game modes |

### ML Schema (`ml`)

**7 aggregate tables** (per-patch, re-populated on each training run):
- `team_hero_agg`, `player_hero_agg`, `hero_synergy_agg`, `hero_counter_agg`
- `team_h2h_agg`, `hero_baseline_agg`, `hero_draft_slot_agg`

**7 PIT-safe snapshot tables** (per-date-bucket with `as_of_date`):
- `team_hero_snapshot`, `player_hero_snapshot`, `hero_synergy_snapshot`, `hero_counter_snapshot`
- `team_h2h_snapshot`, `hero_baseline_snapshot`, `hero_draft_slot_snapshot`

4 sparse tables use cross-patch lookback (`lookback_patches=2`, `prior_patch_weight=0.5`). 3 dense tables stay single-patch.

---

## Deployment

### Docker Compose

| Profile | Services |
|---|---|
| `all` | Everything |
| `db` | postgres, rabbitmq, redis |
| `train` | trainer + db |
| `api` | inference API + db |
| `mon` | prometheus, grafana |

### Resource Limits

| Service | Memory | CPUs |
|---------|--------|------|
| postgres | 6G | 4.0 |
| trainer | 8G | 12 |
| api | 2G | 2 |
| rabbitmq | 512M | 1.0 |
| parser | 1G | 1.0 |

---

## Key Architectural Patterns

1. **Event-driven pipeline** — Services communicate exclusively through RabbitMQ
2. **Proxy pool abstraction** — All OpenDota calls go through Redis-backed proxy pool
3. **Idempotent ingestion** — `ON CONFLICT DO NOTHING` on all DB inserts
4. **Dead-letter queues** — Every queue has a DLQ with 24h TTL
5. **PIT-safe snapshots** — Training uses LATERAL "most recent AS OF match start" lookups
6. **Cross-patch lookback** — Sparse aggregate tables weight prior-patch data to fill gaps
7. **TorchScript JIT** — Model compiled to C++ graph for <2ms CPU inference
8. **Monte Carlo rollouts** — 40 random draft completions per candidate, batch-evaluated
9. **WebSocket MCTS streaming** — Queue-based async bridge for real-time progress updates
10. **Configurable training** — All hyperparameters in `.env`, zero code changes needed
