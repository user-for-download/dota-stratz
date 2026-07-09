"""Feature computation for the inference API.

This module mirrors the trainer's ``features.py`` to produce identical
feature vectors at prediction time. The contract between training and
inference is ``feature_schema.json``, written by the trainer and loaded
at API startup.

NULL safety: all lookups use ``_float()`` / ``_int()`` guards to prevent
``float(None)`` crashes when aggregate tables have NULL values (e.g. a hero
was picked but has no synergy data yet for this patch).

Performance: Team-hero, baseline, and h2h queries are pre-fetched in bulk
per request (see ``BatchContext``) so only synergy and counter queries are
made per-hero.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import db as db_
from .draft_state import DraftContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NULL-safe helpers
# ---------------------------------------------------------------------------

_FLOAT_DEFAULTS: dict[str, float] = {
    "win_rate": 0.5,
    "pick_rate": 0.0,
    "ban_rate": 0.0,
    "avg_gpm": 0.0,
    "avg_xpm": 0.0,
    "avg_kills": 0.0,
    "avg_deaths": 0.0,
    "avg_assists": 0.0,
    "avg_kda": 0.0,
    "avg_kd_diff": 0.0,
    "firstblood_rate": 0.0,
    "avg_camps_stacked": 0.0,
    "avg_vision_placed": 0.0,
    "avg_gold_10": 0.0,
    "avg_xp_10": 0.0,
    "hds_win_rate": 0.5,
}

_INT_DEFAULTS: dict[str, int] = {
    "games": 0,
    "wins": 0,
    "bans": 0,
    "total_picks": 0,
    "total_wins": 0,
    "total_bans": 0,
    "lane_role": 0,
    "hds_games": 0,
}


def _float(val: Any, key: str = "") -> float:
    """Safely convert a value to float, returning a sensible default if None."""
    if val is None:
        return _FLOAT_DEFAULTS.get(key, 0.0)
    try:
        return float(val)
    except (TypeError, ValueError):
        return _FLOAT_DEFAULTS.get(key, 0.0)


def _int(val: Any, key: str = "") -> int:
    """Safely convert a value to int, returning a sensible default if None."""
    if val is None:
        return _INT_DEFAULTS.get(key, 0)
    try:
        return int(val)
    except (TypeError, ValueError):
        return _INT_DEFAULTS.get(key, 0)


# ---------------------------------------------------------------------------
# Schema management
# ---------------------------------------------------------------------------


def load_schema(model_dir: str | Path, patch_id: int) -> dict[str, Any]:
    """Load the patch-specific feature schema written by the trainer.

    Uses ``feature_schema_patch_{patch_id}.json`` so that each patch has
    its own schema, preventing dimension mismatches when multiple patches
    are trained independently (Bug #5).
    """
    path = Path(model_dir) / f"feature_schema_patch_{patch_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature schema not found at {path}. "
            "Has the trainer been run for this deployment?"
        )
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Batch pre-fetch context
# ---------------------------------------------------------------------------


@dataclass
class BatchContext:
    """Pre-fetched aggregate data shared across all hero evaluations.

    Storing these in a single dataclass avoids N+1 query patterns: instead
    of making per-hero round-trips for baseline, team-hero, synergy,
    counter, h2h, and draft-slot data, we fetch them all in 7 queries and
    distribute the results here.
    """

    baselines: dict[int, dict]
    """hero_id → baseline row dict (from fetch_baselines_batch)."""

    team_hero_agg: dict[int, dict]
    """hero_id → team-hero agg dict (from fetch_team_hero_agg_batch), may be empty."""

    player_hero_agg: dict[int, dict]
    """hero_id → player-hero agg dict (from fetch_player_hero_agg_batch).
    Falls back to empty dict when account_id is not available — the feature
    builder uses hardcoded defaults in that case."""

    synergy: dict[int, tuple[float, int]]
    """hero_id → (avg_synergy_win_rate, count) — batch pre-fetched."""

    counter: dict[int, tuple[float, int, float]]
    """hero_id → (avg_counter_win_rate, count, avg_kd_diff) — batch pre-fetched."""

    h2h_row: dict | None
    """Single head-to-head row for the team pair (same for all heroes)."""

    hero_draft_slot: dict[int, tuple[float, int]]
    """hero_id → (hds_win_rate, hds_games) — pick-position win rate for the
    current team_pick_ordinal. Falls back to ``(0.5, 0)`` per-hero."""

    hero_embs: dict[int, list[float]]
    team_emb: list[float]
    player_emb: list[float]
    hero_spatial_embs: dict[int, list[float]]
    """16-D SVD embedding vector for the player."""


def pre_fetch_batch(
    patch_id: int,
    hero_ids: list[int],
    team_id: int | None,
    enemy_team_id: int | None,
    ctx: DraftContext | None = None,
    account_id: int | None = None,
) -> BatchContext:
    """Pre-fetch all per-hero aggregate data in bulk (7 queries total).

    This replaces hundreds of individual ``fetch_baseline``,
    ``fetch_team_hero_agg``, ``fetch_player_hero_agg``,
    ``fetch_synergy_avg``, ``fetch_counter_avg``,
    ``fetch_h2h``, and ``fetch_hero_draft_slot_batch`` calls
    with 7 batched queries.
    """
    baselines = db_.fetch_baselines_batch(patch_id, hero_ids) if hero_ids else {}
    team_hero_agg = (
        db_.fetch_team_hero_agg_batch(patch_id, team_id, hero_ids)
        if team_id is not None and hero_ids
        else {}
    )
    player_hero_agg = (
        db_.fetch_player_hero_agg_batch(patch_id, account_id, hero_ids)
        if account_id is not None and hero_ids
        else {}
    )
    synergy = db_.fetch_synergy_batch(patch_id, hero_ids, ctx.ally_picks) if ctx and hero_ids else {}
    counter = db_.fetch_counter_batch(patch_id, hero_ids, ctx.enemy_picks) if ctx and hero_ids else {}
    h2h_row = (
        db_.fetch_h2h(patch_id, team_id, enemy_team_id)
        if team_id is not None and enemy_team_id is not None
        else None
    )
    # Hero draft-slot (pick-position) aggregate — uses the upcoming pick
    # ordinal for the recommending team (e.g. if team has 2 picks already,
    # the next pick is ordinal 3).
    team_pick_ordinal = len(ctx.ally_picks) + 1 if ctx else 1
    hero_draft_slot = (
        db_.fetch_hero_draft_slot_batch(patch_id, hero_ids, team_pick_ordinal)
        if hero_ids
        else {}
    )
    he, te, pe, hse = db_.fetch_embeddings(patch_id, hero_ids, team_id, account_id)
    return BatchContext(
        baselines=baselines,
        team_hero_agg=team_hero_agg,
        player_hero_agg=player_hero_agg,
        synergy=synergy,
        counter=counter,
        h2h_row=h2h_row,
        hero_draft_slot=hero_draft_slot,
        hero_embs=he,
        team_emb=te,
        player_emb=pe,
        hero_spatial_embs=hse,
    )


# ---------------------------------------------------------------------------
# Feature vector construction
# ---------------------------------------------------------------------------


def build_feature_vector(
    hero_id: int,
    ctx: DraftContext,
    patch_id: int,
    batch: BatchContext,
    schema: dict[str, Any],
    embeddings: dict[str, list[float]] | None = None,
    max_hero_id: int = 160,
) -> np.ndarray:
    """Build the full feature vector (numeric columns + hero embeddings) for a
    candidate hero.

    This function MUST produce the same feature vector, in the same column
    order, as the trainer's ``training_features_sql`` query for the model to produce
    valid predictions.

    Most aggregate lookups use the pre-fetched ``BatchContext``. Only
    synergy and counter queries are made per-hero because they depend on
    the current draft state (which allies/enemies are locked in).

    Parameters
    ----------
    hero_id : int
        The candidate hero (1-160).
    ctx : DraftContext
        Current draft state (taken heroes, turn, etc.).
    patch_id : int
        Current Dota 2 patch.
    batch : BatchContext
        Pre-fetched aggregate data (see ``pre_fetch_batch``).
    schema : dict
        Feature schema dict from ``feature_schema.json``.
    max_hero_id : int
        Maximum hero ID for one-hot encoding (default 160).
    """
    # We need the same order as feature_column_names(include_onehot=False)
    # from the trainer. Build a dict keyed by column name.
    vec: dict[str, float] = {}

    # -- Draft context (must match trainer's feature_column_names order) --
    vec["is_pick"] = 1.0  # inference always recommends picks
    vec["team"] = float(ctx.recommending_team)  # 0 = radiant, 1 = dire
    vec["draft_phase_id"] = float(ctx.draft_phase_id)  # CM phase (0-5)

    # -- Team-hero aggregates (from pre-fetched batch dict) --
    th = batch.team_hero_agg.get(hero_id)
    vec["th_games"] = _float(th.get("games") if th else None, "games")
    vec["th_wins"] = _float(th.get("wins") if th else None, "wins")
    vec["th_win_rate"] = _float(th.get("win_rate") if th else None, "win_rate")
    vec["th_bans"] = _float(th.get("bans") if th else None, "bans")
    vec["th_avg_gpm"] = _float(th.get("avg_gpm") if th else None, "avg_gpm")
    vec["th_avg_xpm"] = _float(th.get("avg_xpm") if th else None, "avg_xpm")
    vec["th_avg_kills"] = _float(th.get("avg_kills") if th else None, "avg_kills")
    vec["th_avg_deaths"] = _float(th.get("avg_deaths") if th else None, "avg_deaths")
    vec["th_avg_assists"] = _float(th.get("avg_assists") if th else None, "avg_assists")
    vec["th_firstblood_rate"] = _float(th.get("firstblood_rate") if th else None, "firstblood_rate")
    vec["th_avg_camps_stacked"] = _float(th.get("avg_camps_stacked") if th else None, "avg_camps_stacked")
    vec["th_avg_vision_placed"] = _float(th.get("avg_vision_placed") if th else None, "avg_vision_placed")
    vec["th_avg_gold_10"] = _float(th.get("avg_gold_10") if th else None, "avg_gold_10")
    vec["th_avg_xp_10"] = _float(th.get("avg_xp_10") if th else None, "avg_xp_10")

    # -- Player-hero aggregates (from pre-fetched batch dict) --
    # Falls back to hardcoded defaults when account_id is unavailable
    # (spectator mode). Real data is used when the caller provides an
    # account_id, reducing train-serving skew (issue #11).
    ph = batch.player_hero_agg.get(hero_id)
    vec["ph_games"] = _float(ph.get("games") if ph else None, "games")
    vec["ph_wins"] = _float(ph.get("wins") if ph else None, "wins")
    vec["ph_win_rate"] = _float(ph.get("win_rate") if ph else None, "win_rate")
    vec["ph_avg_gpm"] = _float(ph.get("avg_gpm") if ph else None, "avg_gpm")
    vec["ph_avg_xpm"] = _float(ph.get("avg_xpm") if ph else None, "avg_xpm")
    vec["ph_avg_kills"] = _float(ph.get("avg_kills") if ph else None, "avg_kills")
    vec["ph_avg_deaths"] = _float(ph.get("avg_deaths") if ph else None, "avg_deaths")
    vec["ph_avg_assists"] = _float(ph.get("avg_assists") if ph else None, "avg_assists")
    vec["ph_avg_kda"] = _float(ph.get("avg_kda") if ph else None, "avg_kda")
    vec["ph_lane_role"] = _float(ph.get("lane_role") if ph else None, "lane_role")
    vec["ph_firstblood_rate"] = _float(ph.get("firstblood_rate") if ph else None, "firstblood_rate")
    vec["ph_avg_camps_stacked"] = _float(ph.get("avg_camps_stacked") if ph else None, "avg_camps_stacked")
    vec["ph_avg_vision_placed"] = _float(ph.get("avg_vision_placed") if ph else None, "avg_vision_placed")
    vec["ph_avg_gold_10"] = _float(ph.get("avg_gold_10") if ph else None, "avg_gold_10")
    vec["ph_avg_xp_10"] = _float(ph.get("avg_xp_10") if ph else None, "avg_xp_10")

    # -- Synergy with allies (from pre-fetched batch dict) --
    sy = batch.synergy.get(hero_id)
    vec["sy_avg_win_rate"] = _float(sy[0] if sy else None, "win_rate")
    # sy_n_teammates = len(ally_picks): counts the number of already-picked allies at
    # this draft state. The training SQL uses COUNT(*) over pb2 rows in the LATERAL join,
    # which is semantically equivalent (LEFT JOIN preserves all prior pick rows).
    vec["sy_n_teammates"] = float(sy[1] if sy else 0)

    # -- Counter vs enemies (from pre-fetched batch dict) --
    co = batch.counter.get(hero_id)
    vec["co_avg_win_rate"] = _float(co[0] if co else None, "win_rate")
    # co_n_enemies = len(enemy_picks): counts the number of already-picked enemies at
    # this draft state. Semantically equivalent to the training SQL's COUNT(*) over
    # opposing-team pb2 rows in the LATERAL join.
    vec["co_n_enemies"] = float(co[1] if co else 0)
    vec["co_avg_kd_diff"] = _float(co[2] if co else None, "avg_kd_diff")

    # -- Head-to-head (from pre-fetched batch — same for all heroes) --
    h2h = batch.h2h_row
    vec["h2h_win_rate"] = _float(h2h.get("win_rate") if h2h else None, "win_rate")
    vec["h2h_games"] = _float(h2h.get("games") if h2h else None, "games")

    # -- Hero draft-slot (pick-position) win rate --
    hds = batch.hero_draft_slot.get(hero_id)
    vec["hds_win_rate"] = _float(hds[0] if hds else None, "hds_win_rate")
    vec["hds_games"] = float(_int(hds[1] if hds else None, "hds_games"))

    # -- Hero baseline (from pre-fetched batch dict) --
    bl = batch.baselines.get(hero_id)
    vec["bl_total_picks"] = _float(bl.get("total_picks") if bl else None, "total_picks")
    vec["bl_total_wins"] = _float(bl.get("total_wins") if bl else None, "total_wins")
    vec["bl_total_bans"] = _float(bl.get("total_bans") if bl else None, "total_bans")
    vec["bl_win_rate"] = _float(bl.get("win_rate") if bl else None, "win_rate")
    vec["bl_pick_rate"] = _float(bl.get("pick_rate") if bl else None, "pick_rate")
    vec["bl_ban_rate"] = _float(bl.get("ban_rate") if bl else None, "ban_rate")
    vec["bl_avg_gpm"] = _float(bl.get("avg_gpm") if bl else None, "avg_gpm")
    vec["bl_avg_xpm"] = _float(bl.get("avg_xpm") if bl else None, "avg_xpm")
    vec["bl_avg_kills"] = _float(bl.get("avg_kills") if bl else None, "avg_kills")
    vec["bl_avg_deaths"] = _float(bl.get("avg_deaths") if bl else None, "avg_deaths")
    vec["bl_avg_assists"] = _float(bl.get("avg_assists") if bl else None, "avg_assists")
    vec["bl_avg_gold_10"] = _float(bl.get("avg_gold_10") if bl else None, "avg_gold_10")
    vec["bl_avg_xp_10"] = _float(bl.get("avg_xp_10") if bl else None, "avg_xp_10")

    # -- Task 4: Low-game missingness flags --
    ph_games_val = vec.get("ph_games", 0.0)
    th_games_val = vec.get("th_games", 0.0)
    vec["ph_is_new_player"] = 1.0 if ph_games_val < 5 else 0.0
    vec["th_is_new_team_hero"] = 1.0 if th_games_val < 5 else 0.0

    # -- Task 5: Draft-state delta features --
    bl_wr = vec.get("bl_win_rate", 0.5)
    th_wr = vec.get("th_win_rate", 0.5)
    ph_wr = vec.get("ph_win_rate", 0.5)
    vec["rel_th_win_rate"] = th_wr - bl_wr
    vec["rel_ph_win_rate"] = ph_wr - bl_wr

    # -- Task 6: Role interaction features --
    ph_lane_role_val = vec.get("ph_lane_role", 0.0)
    ph_vision_val = vec.get("ph_avg_vision_placed", 0.0)
    ph_gpm_val = vec.get("ph_avg_gpm", 0.0)
    vec["ph_vision_support_score"] = ph_vision_val if ph_lane_role_val == 5 else 0.0
    vec["ph_gpm_carry_score"] = ph_gpm_val if ph_lane_role_val == 1 else 0.0

    # -- Task 7: Macro Composition Constraints --
    team_gpm = vec.get("bl_avg_gpm", 0.0)
    team_xpm = vec.get("bl_avg_xpm", 0.0)
    for ally_id in ctx.ally_picks:
        ally_bl = batch.baselines.get(ally_id, {})
        team_gpm += _float(ally_bl.get("avg_gpm"), "avg_gpm")
        team_xpm += _float(ally_bl.get("avg_xpm"), "avg_xpm")
    vec["team_gpm_budget"] = team_gpm
    vec["team_xpm_budget"] = team_xpm

    # -- Task 8: Pick Propensity (team comfort pick signal) --
    th_games_val = vec.get("th_games", 0.0)
    bl_total_picks = vec.get("bl_total_picks", 1.0)
    vec["team_pick_propensity"] = th_games_val / max(bl_total_picks, 1.0)

    # -- Task 9: Semantic SVD Embeddings --
    he = batch.hero_embs.get(hero_id, [0.0] * 32)
    for i in range(32):
        vec[f"hero_emb_{i}"] = he[i]
    for i in range(16):
        vec[f"team_emb_{i}"] = batch.team_emb[i]
    for i in range(16):
        vec[f"player_emb_{i}"] = batch.player_emb[i]

    hse = batch.hero_spatial_embs.get(hero_id, [0.0] * 16)
    for i in range(16):
        vec[f"hero_spatial_emb_{i}"] = hse[i]

    # Build numeric array in the exact column order from the schema
    numeric_values = []
    aggregate_cols = schema["aggregate_columns"]
    for col in aggregate_cols:
        numeric_values.append(vec.get(col, 0.0))
    numeric = np.array(numeric_values, dtype=np.float32)

    # --- Standardize Features ---
    drift_stats = schema.get("drift_stats")
    if drift_stats:
        means = np.array(drift_stats["mean"], dtype=np.float32)
        stds = np.array(drift_stats["std"], dtype=np.float32)
        numeric = (numeric - means) / stds

    # Add hero_id as a numeric value (DO NOT scale categorical IDs!)
    numeric = np.append(numeric, float(hero_id))

    # --- Feature Drift Detection (on standardized values) ---
    if drift_stats:
        z_scores = np.abs(numeric[:len(means)])
        max_z = float(np.max(z_scores))
        if max_z > 4.0:
            drift_col = aggregate_cols[int(np.argmax(z_scores))] if int(np.argmax(z_scores)) < len(aggregate_cols) else "unknown"
            logger.warning("Feature drift detected on %s! Z-Score: %.2f. Data may be anomalous.", drift_col, max_z)

    # Resolve embedding vector for this hero
    n_embeddings = schema.get("n_embeddings", 32)
    emb_list = embeddings.get(str(hero_id), [0.0] * n_embeddings)
    emb_array = np.array(emb_list, dtype=np.float32)

    return np.concatenate([numeric, emb_array])
