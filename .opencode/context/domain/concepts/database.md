# Database Schema

**Core concept**: PostgreSQL 16 with RANGE-partitioned tables, deferred FK constraints, and an analytics schema for ML feature engineering.

## Migration Files
| File | Content |
|------|---------|
| `001_core.sql` | Core tables: `matches`, `players` (partitioned), all child event tables, indexes, FKs (DEFERRABLE INITIALLY DEFERRED) |
| `002_constants.sql` | Static reference: `const_game_mode`, `const_lobby_type`, `const_region`, `const_patch`, `const_hero`, `const_item`, `const_ability` |
| `003_analytics.sql` | Bayesian shrinkage config, MV `mv_team_hero_profile`, `mv_hero_synergy`, `mv_hero_counter`, `mv_player_team_history`, feature snapshots, roles, refresh functions |
| `004_partition_verify.sql` | Idempotent assertion that 6 expected `players` partitions exist (safety check) |
| `005_ml_tables.sql` | 6 patch-aware ML aggregate tables in `ml` schema |

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
- 6 patch-aware aggregate tables for LightGBM training and inference
- Tables are UNLOGGED for write speed (re-populated on each train run)
- Trainer's `TRAINING_FEATURES_SQL` computes 196-dim feature vectors (36 aggregate + 160 one-hot hero ID) via a single query with `LEAST`/`GREATEST` index-friendly joins on synergy aggregates (~11s for 108k rows)

### ML Aggregate Tables
| Table | Rows (patch 58) | Purpose |
|-------|-----------------|---------|
| `team_hero_agg` | 35,240 | Team+hero historical stats (games, wins, bans, avg stats) |
| `player_hero_agg` | 39,845 | Player+hero historical stats per account |
| `hero_synergy_agg` | 7,332 | Pairwise hero synergy win rates on same team |
| `hero_counter_agg` | 15,216 | Pairwise hero counter win rates vs enemy |
| `team_h2h_agg` | 4,846 | Team head-to-head win rates |
| `hero_baseline_agg` | 126 | Global hero pick/ban rates and avg stats per patch |
