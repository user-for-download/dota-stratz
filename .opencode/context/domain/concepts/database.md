# Database Schema

**Core concept**: PostgreSQL 16 with RANGE-partitioned tables, deferred FK constraints, and an analytics schema for ML feature engineering.

## Migration Files
| File | Content |
|------|---------|
| `001_core.sql` | Core tables: `matches`, `players` (partitioned), all child event tables, indexes, FKs (DEFERRABLE INITIALLY DEFERRED) |
| `002_constants.sql` | Static reference: `const_game_mode`, `const_lobby_type`, `const_region`, `const_patch`, `const_hero`, `const_item`, `const_ability` |
| `003_analytics.sql` | Bayesian shrinkage config, MV `mv_team_hero_profile`, `mv_hero_synergy`, `mv_hero_counter`, `mv_player_team_history`, feature snapshots, roles, refresh functions |
| `004_partition_verify.sql` | Idempotent assertion that 6 expected `players` partitions exist (safety check) |
| `005_ml_tables.sql` | 6 initial ML aggregate tables in `ml` schema (originally UNLOGGED, changed to LOGGED per CRITICAL-5 fix) |
| `006_postgres_best_practices_fixes.sql` | Runtime fixes: grants on `ml` schema, `grant_ml_access()` function |
| `007_enhanced_features.sql` | Adds 3 behavioral feature columns (`firstblood_rate`, `avg_camps_stacked`, `avg_vision_placed`) to `ml.team_hero_agg` and `ml.player_hero_agg` |
| `008_minute_stats_columns.sql` | Adds `gold_t` / `xp_t` JSONB columns to `player_minute_stats` for early-game (min 0-9) gold/XP computation |
| `009_gold_xp_10_features.sql` | Adds `avg_gold_10` / `avg_xp_10` columns to `ml.team_hero_agg`, `ml.player_hero_agg`, `ml.hero_baseline_agg` |
| `010_team_id_bigint_indexes.sql` | Fixes `team_id` in ML tables from INT→BIGINT; adds PIT-focused composite indexes for trainer LATERAL queries |
| `011_hero_draft_slot_agg.sql` | Adds `ml.hero_draft_slot_agg` table (7th ML table) for hero pick-position win rates (ordinal 1-5) |
| `012_fix_ml_indexes.sql` | Drops redundant indexes; CHECK constraint on `team_pick_ordinal` (guarded — 011 already creates it) |
| `013_separate_time_series_arrays.sql` | Moves `gold_t`/`xp_t` JSONB from `player_minute_stats` (minute=0 sentinel) into dedicated `player_time_series_arrays` table to eliminate PK conflict with real minute-zero rows |
| `014_performance_optimization.sql` | Adds generated columns (`minute = time / 60`) + composite indexes `(match_id, minute)` to 7 event tables for accelerated GROUP BY queries. Created in response to audit finding W4. |

## Key Tables
- **`matches`** — Match header: duration, game_mode, lobby_type, region, patch_id, scores
- **`players`** — RANGE-partitioned by match_id (5B per partition). PK: `(match_id, player_slot)`
- **`picks_bans`** — Draft order with hero picks/bans per match
- **`team_games`** — Team-level match results (used for H2H aggregates)
- **`raw_matches`** — Staging table with raw JSON blob

## Analytics Schema
- Materialized views provide Bayesian-shrunk win rates for team-hero, hero synergy, hero counter
- `feature_snapshots_player_hero` — Point-in-time snapshots to avoid look-ahead bias
- `refresh_all_mv()` — Function to refresh all MVs CONCURRENTLY with logging

## ML Schema (`ml`)
- **7** patch-aware aggregate tables for PyTorch DraftBERT training and inference
- Tables are LOGGED for crash safety (previously UNLOGGED — changed per CRITICAL-5; VACUUM ANALYZE added to aggregate populator to compensate for write performance)
- All aggregate queries filter `WHERE radiant_win IS NOT NULL` to exclude abandoned matches from win-rate calculations (prevents ~3-5% deflation)
- avg_gold_10 / avg_xp_10 computed from `gold_t`/`xp_t` JSONB arrays now stored in `player_time_series_arrays` (separated from `player_minute_stats` minute=0 sentinel by migration `013` to avoid PK conflict)
- **Stale row protection**: `_clean_patch_rows` deletes rows for the current patch_id before re-inserting (so disappeared rows don't persist), and `_analyze_ml_tables()` runs `VACUUM ANALYZE` on all 7 tables after every full populate cycle to reclaim bloat and update query planner stats
- **Configurable match filtering**: All seven populators apply the same `TRAINER_LEAGUE_ONLY` / `TRAINER_LOBBY_TYPES` filter (replaces old hardcoded `leagueid > 0` that was only in `populate_h2h`)
- Trainer's `TRAINING_FEATURES_SQL` computes **59 aggregate + 160 one-hot hero ID = 219-dim feature vectors** via a single query with `LEAST`/`GREATEST` index-friendly joins on synergy aggregates, and `LATERAL` subqueries for PIT synergy/counter lookups. The feature schema is exported to JSON (`feature_schema_patch_{patch_id}.json`) with explicit `continuous_features`, `categorical_features`, and `embedding_features` lists for API contract enforcement. Player-hero features use real data when `account_id` is available at inference time, otherwise fall back to hardcoded defaults to avoid train-serving skew.

### ML Aggregate Tables
| Table | Rows (patch 58) | Purpose |
|-------|-----------------|---------|
| `team_hero_agg` | 35,240 | Team+hero historical stats (games, wins, bans, avg stats, avg_gold_10, avg_xp_10, firstblood_rate, camps_stacked, vision_placed) |
| `player_hero_agg` | 39,845 | Player+hero historical stats per account (includes lane_role, kda, avg_gold_10, avg_xp_10) |
| `hero_synergy_agg` | 7,332 | Pairwise hero synergy win rates on same team |
| `hero_counter_agg` | 15,216 | Pairwise hero counter win rates vs enemy (incl. avg_kd_diff) |
| `team_h2h_agg` | 4,846 | Team head-to-head win rates |
| `hero_baseline_agg` | 126 | Global hero pick/ban rates and avg stats per patch (incl. avg_gold_10, avg_xp_10) |
| `hero_draft_slot_agg` | ~1,200 | Hero win rate per team-pick ordinal (1st–5th pick position) |

> **Perf fix**: `player_purchase_log` and 6 other event tables now have `minute = time / 60` generated columns with composite `(match_id, minute)` indexes, reducing GROUP BY query time during training data extraction (migration `014`).

### Snapshot Function
`analytics.update_feature_snapshots()` iterates over ALL distinct dates since the last run (not just `MAX(DATE(...))`) — previously, if multiple days' matches were ingested between cron runs, only the most recent day was captured and intermediate days were permanently lost.
