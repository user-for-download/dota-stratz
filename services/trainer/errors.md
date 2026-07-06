# Trainer Service — Bug & Error Audit

> **Audit date:** 2026-07-06  
> **Scope:** `services/trainer/` — all 12 source modules, 3 test files, Dockerfile, requirements.txt  
> **Skills used:** python-performance-optimization, supabase-postgres-best-practices, pandas-pro, machine-learning  
> **Files analyzed:** 15

---

## 🔴 CRITICAL BUGS

| # | File | Line | Severity | Issue | Detail |
|---|------|------|----------|-------|--------|
| **C1** | `train_live.py` | 77-86 | **HIGH** | **Transformer + Static MLP not cached per match** | The model has `encode_draft()` and `forward_dynamic()` methods designed to compute transformer/static embeddings once per match and cache them. The training loop calls `model(heroes, actions, static, dynamic)` (the full `forward()`) every batch, causing the *entire* transformer + MLP to recompute for every minute of every match, every epoch. For a dataset with 100K samples across 15 epochs, this wastes ~90% of compute. |
| **C2** | `dataset_live.py` | 135-156 | **HIGH** | **Pure Python loop over millions of rows** | `for i in range(len(dynamic_df)):` iterates in Python over every single row of the dynamic features DataFrame. For a patch with 10K matches × ~45 minutes each = 450K rows, this is 450K Python iterations doing dict lookups and list appends. Should be vectorized with NumPy boolean masks. |
| **C3** | `dataset_pt.py` | 33-37 | **MED** | **Row-by-row tensor population** | `DraftSequenceDataset.__init__` loops `for i in range(n)` populating `self.heroes[i,:len(h)]` one sample at a time. For a dataset with 200K prefix-augmented sequences, this is 200K Python iterations. Could be vectorized using `torch.nn.utils.rnn.pad_sequence` or batched array assignment. |
| **C4** | `aggregates.py` | 767 | **MED** | **`[prior_weight] * 22` fragility** | The `_team_hero_prior_agg` function uses `[prior_weight] * 22 + [min_patch, patch_id]` as query parameters. If anyone modifies the SQL and adds or removes a single `%s`, the parameter count silently mismatches (psycopg2 raises "not enough arguments" at runtime). No assertion or count check exists. |

---

## 🟡 DOCUMENTATION / COMMENT BUGS

| # | File | Line | Issue | Detail |
|---|------|------|-------|--------|
| **D1** | `live_features.py` | 22 | **Wrong count in comment** | `# Dynamic feature columns (26 features)` → there are **24** columns in `DYNAMIC_FEATURE_COLUMNS`, not 26. (The docstring header at line 10 correctly says "24 dynamic features".) |
| **D2** | `tests/conftest.py` | 21-22 | **Wrong count in comment** | `"""The 58 aggregate feature columns..."""` → `feature_column_names(include_onehot=False)` returns **59** columns. |

### 59 aggregate column breakdown

| Group | Count | Columns |
|-------|-------|---------|
| `is_pick`, `team` | 2 | Draft context |
| `th_*` | 14 | Team-hero stats (games, wins, win_rate, bans, avg_gpm, avg_xpm, avg_kills, avg_deaths, avg_assists, firstblood_rate, avg_camps_stacked, avg_vision_placed, avg_gold_10, avg_xp_10) |
| `ph_*` | 15 | Player-hero stats (+ avg_kda, lane_role) |
| `sy_*` | 2 | Synergy with allies |
| `co_*` | 3 | Counter vs enemies (+ avg_kd_diff) |
| `h2h_*` | 2 | Head-to-head |
| `bl_*` | 13 | Hero baseline (+ pick_rate, ban_rate) |
| `hds_*` | 2 | Draft-slot win rate |
| `ph_is_new_player`, `th_is_new_team_hero` | 2 | Low-game flags |
| `rel_th_win_rate`, `rel_ph_win_rate` | 2 | Delta features |
| `ph_vision_support_score`, `ph_gpm_carry_score` | 2 | Role interactions |
| **Total** | **59** | |

---

## 🟠 PERFORMANCE ISSUES

| # | File | Lines | Severity | Issue |
|---|------|-------|----------|-------|
| **P1** | `live_features.py` | 198-261 | **HIGH** | **All feature engineering in Pandas after SQL** — The SQL loads ALL raw tick data (kills, objectives, items, buybacks, wards, teamfights) into memory, then does ALL feature computation in Pandas with `groupby().rolling().cumsum()`. For large patches (50K+ matches), intermediate DataFrames from `generate_series(minute)` could exceed available RAM. |
| **P2** | `features.py` | 48-302 | **MED** | **14 LATERAL joins per draft step** — The main `training_features_sql()` query uses 7 CROSS JOIN LATERAL subqueries, each doing a filtered lookup against snapshot tables. For a patch with 10K matches × ~50 draft steps = 500K rows, this executes 3.5M lateral subqueries. Each lateral query does an ORDER BY + LIMIT 1 on a filtered snapshot table. |
| **P3** | `model_pt.py` | 67 | **LOW** | **`pos_emb` recomputed every forward pass** — `torch.arange(S, device=heroes.device)` creates a position tensor from scratch every call. Could pre-compute and register as `self.register_buffer('position_ids', ...)`. |
| **P4** | `aggregates.py` | 836-891 | **MED** | **Daily snapshot cross-product** — `populate_team_hero_snapshot` does `CROSS JOIN LATERAL` over every combination of date × (team_id, hero_id). For a 90-day patch with 5K team-hero combos, this is 450K lateral subquery executions. |

---

## 🔵 DESIGN / CONSISTENCY ISSUES

| # | File | Lines | Issue |
|---|------|-------|-------|
| **I1** | `aggregates.py` | 784-807 | **Prior bans not weighted** — `_team_hero_bans_prior()` uses `COUNT(*)` without applying `prior_weight`, while `_team_hero_prior_agg()` applies `prior_weight` to games/wins. Bans from prior patches are treated equally to current-patch bans, but games are discounted. Minor inconsistency. |
| **I2** | `train_live.py` | 174-201 | **Separate DB connection for metadata upsert** — `_upsert_model_meta()` opens a NEW psycopg2 connection instead of reusing the existing `conn` parameter passed to `train_live_model()`. Wastes a connection. |
| **I3** | `train_live.py` | 66 | **No DataLoader** — The LiveDraftBERT training loop uses manual tensor slicing with `torch.randperm` instead of `torch.utils.data.DataLoader`. Bypasses `num_workers`, `prefetch_factor`, `pin_memory` — all irrelevant for CPU-only but unusual. |
| **I4** | `live_features.py` | 214-233 | **Simplified death timer model** — Death timers use rolling kill windows (1/2/3 min) scaled by game phase. No buyback-aware death clearing. A buyback immediately negates a death, but this model treats the hero as dead for the full window. |
| **I5** | `live_features.py` | 199-201 | **Aegis detection is a 5-min sum of Roshan kills** — Aegis is detected by checking if a Roshan kill happened within 5 minutes. But Roshan kills without Aegis pickup (enemy steals, denies) would falsely indicate Aegis. Also, a team killing Roshan twice within 5 minutes doesn't stack Aegis. |

---

## 🟣 TEST COVERAGE GAPS

| # | Module | Tests Exist? | Coverage What's Missing |
|---|--------|-------------|------------------------|
| **T1** | `train_pt.py` | ❌ None | No test for TorchScript export, dummy tensor shapes, metadata writing, scheduler behavior |
| **T2** | `train_live.py` | ❌ None | No test for the custom tensor-slicing training loop, gradient clipping, early stopping |
| **T3** | `model_pt.py` / `model_live.py` | ❌ None | No forward-pass tests, no shape assertions, no padding mask verification, no edge cases (all padding, single hero) |
| **T4** | `live_features.py` | ❌ None | No test for any of the 24 dynamic features (death timers, aegis windows, momentum, power spikes, etc.) |
| **T5** | `db.py` | ❌ None | No test for `fetch_patch_id` auto-detection, `load_heroes` |
| **T6** | `config.py` | ❌ None | No test for env var parsing, DSN construction, property defaults |
| **T7** | `features.py` | ✅ Partial | Tests exist for `make_target` and `feature_column_names` but NOT for `training_features_sql()`, `write_schema()`, or the SQL feature contract |
| **T8** | `dataset_pt.py` | ✅ Partial | Tests for split ratios and tensor shapes but NOT for prefix augmentation logic, NaN handling, large datasets |
| **T9** | `aggregates.py` | ✅ Partial | Tests for `_clean_patch_rows` and `_match_extra_where` but NOT for ANY of the 14 populator functions, `_analyze_ml_tables`, or `populate_all` |

---

## ✅ THINGS DONE RIGHT

Despite the issues above, the codebase has several strong points:

| Area | Good Practice |
|------|---------------|
| **SQL Injection** | `_VALID_TABLES` frozenset guard prevents injection via f-string table name in `_clean_patch_rows` |
| **Transaction Safety** | `_clean_patch_rows` does NOT commit, making DELETE+INSERT atomic with the caller's transaction |
| **PIT Safety** | Training features use LATERAL `ORDER BY as_of_date DESC LIMIT 1` against snapshot tables, avoiding lookahead bias |
| **Cross-patch Lookback** | Sparse combo-keyed tables (team_hero, player_hero, synergy, counter) include prior-patch weighted data |
| **Error Handling** | `main.py` has proper `try/except/finally` with `conn.close()` and `eng.dispose()` |
| **Bayesian Shrinkage** | `_shrunk_wr()` applies prior-based shrinkage to win rates, preventing overfitting on sparse data |
| **Architecture** | LiveDraftBERT has `encode_draft()`/`forward_dynamic()` separation designed for caching (even if training doesn't use it) |
| **Contract** | Feature schemas are written to JSON at training time and consumed by the API to guarantee column order agreement |
| **Consistent Filtering** | All 7 aggregate populators use the same config-driven `_match_extra_where()` filters (previously only h2h filtered) |
| **Stale Row Protection** | `_clean_patch_rows` deletes before inserting, preventing stale data from persisting after corrections |

---

## SUMMARY STATS

| Metric | Value |
|--------|-------|
| Files analyzed | 15 (12 source + 3 test + Dockerfile + requirements.txt) |
| **Critical bugs** | **4** (C1-C4) |
| Documentation bugs | 2 (D1-D2) |
| Performance issues | 4 (P1-P4) |
| Design issues | 5 (I1-I5) |
| Missing test coverage | 7 modules (T1-T7 partially) |
| **Total distinct issues** | **22** |
| **Good practices** | **8/10** |

---

*Generated by OpenAgent using python-performance-optimization, supabase-postgres-best-practices, pandas-pro, and machine-learning skills.*
