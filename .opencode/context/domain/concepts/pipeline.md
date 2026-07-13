# Pipeline Data Flow

**Core concept**: Event-driven microservice pipeline that ingests Dota 2 matches from OpenDota API through a multi-stage queue architecture into PostgreSQL for analytics and ML.

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
```

## Key Points
- **No coordinator**: ID Fetcher owns its cron schedule (`FETCH_SCHEDULE`), rest of pipeline is reactive
- **Queue isolation**: `queue.match_ids` (IDs), `queue.raw_matches` (full JSON). Each has a DLQ
- **Idempotent inserts**: All DB writes use `ON CONFLICT DO NOTHING` — safe retry
- **FK violation fallback**: Parser detects 23503, routes offending match to DLQ, commits healthy matches
- **Graceful shutdown**: SIGINT/SIGTERM drains in-flight work via bounded wait groups with timeouts

## Services
| Stage | Service | Reads From | Writes To |
|-------|---------|-----------|-----------|
| 1 | ID Fetcher | OpenDota Explorer API | `queue.match_ids` |
| 2 | Detail Fetcher | `queue.match_ids`, OpenDota API | `queue.raw_matches` |
| 3 | Parser | `queue.raw_matches` | PostgreSQL (20+ tables) |
| 4 | **Trainer** | PostgreSQL (ml.aggregates) | ML model files, feature schema, Drafting Bot components |
| 5 | **API** | PostgreSQL (ml.aggregates) + model files | Predictions via HTTP :8080 |
| — | Proxy Manager | — | Redis (proxy pool) |

**ML downstream**: After data lands in PostgreSQL, the Trainer computes **7** patch-aware aggregate tables (`team_hero_agg`, `player_hero_agg`, `hero_synergy_agg`, `hero_counter_agg`, `team_h2h_agg`, `hero_baseline_agg`, `hero_draft_slot_agg`), filtering out matches where `radiant_win IS NULL` to avoid abandoned-match pollution. All 7 tables are LOGGED. It then trains a **PyTorch DraftBERT** model (Transformer + MLP multi-modal architecture, ~639K parameters) with BCEWithLogitsLoss. The inference API loads TorchScript JIT models for <2ms CPU inference and uses Monte Carlo rollouts for strategic lookahead, and supports autonomous Drafting Bots for strategy simulation.

**Feature vectors**: 143 aggregate columns (63 hand-crafted + 80 SVD embeddings) + sequence data (hero picks/bans with action tokens) + patch_id embedding. The Transformer branch processes draft sequences; the MLP branch processes tabular features. The LiveDraftBERT adds 30 per-minute dynamic features (gold/xp advantage, towers, Roshan, teamfights, power spikes, vision, neutral items). All hyperparameters configurable via `deploy/.env`. Training uses prefix augmentation (~3× sequence expansion), chronological train/val split, and StandardScaler normalization.

**Drafting Bots**: Four self-contained components in `services/trainer/trainer/` enable autonomous draft simulation — `inference_cache.py` (in-memory aggregate cache), `draft_state.py` (59-dim RAM feature builder), `bot_greedy.py` (single-step lookahead), and `bot_mcts.py` (MCTS with UCB1 + DraftBERT value network). All bypass the API by loading the TorchScript model directly.
