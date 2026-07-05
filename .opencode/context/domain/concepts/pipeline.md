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
| 4 | **Trainer** | PostgreSQL (ml.aggregates) | ML model files, feature schema |
| 5 | **API** | PostgreSQL (ml.aggregates) + model files | Predictions via HTTP :8080 |
| — | Proxy Manager | — | Redis (proxy pool) |

**ML downstream**: After data lands in PostgreSQL, the Trainer computes **7** patch-aware aggregate tables (`team_hero_agg`, `player_hero_agg`, `hero_synergy_agg`, `hero_counter_agg`, `team_h2h_agg`, `hero_baseline_agg`, `hero_draft_slot_agg`), filtering out matches where `radiant_win IS NULL` to avoid abandoned-match pollution. All 7 tables are LOGGED (changed from UNLOGGED per CRITICAL-5 fix — crash-safe with VACUUM ANALYZE in the populator). It then trains LightGBM **binary classification** models. The inference API loads these models and serves draft predictions. Training uses `binary` objective (not `lambdarank`) because every draft slot in a match shares the same `radiant_win` target — lambdarank requires varied relevance within each group and would produce zero-gradient trees.

**Feature vectors**: 58 aggregate columns + 160 one-hot hero IDs = **218 total features** (up from previous 196). New columns include avg_gold_10/avg_xp_10 (early-game gold/XP from gold_t/xp_t JSONB arrays), hero draft-slot win rates (hds_*), low-game missingness flags (ph_is_new_player, th_is_new_team_hero), delta features (rel_th_win_rate, rel_ph_win_rate), and role interaction features (ph_vision_support_score, ph_gpm_carry_score).
