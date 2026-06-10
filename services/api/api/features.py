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
}

_INT_DEFAULTS: dict[str, int] = {
    "games": 0,
    "wins": 0,
    "bans": 0,
    "total_picks": 0,
    "total_wins": 0,
    "total_bans": 0,
    "lane_role": 0,
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


def load_schema(model_dir: str | Path) -> dict[str, Any]:
    """Load the feature schema written by the trainer.

    Returns the parsed JSON dict, or raises FileNotFoundError.
    """
    path = Path(model_dir) / "feature_schema.json"
    if not path.exists():
        raise FileNotFoundError(
            f"feature_schema.json not found at {path}. "
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
    of making per-hero round-trips for baseline, team-hero, and h2h data,
    we fetch them all in 2-3 queries and distribute the results here.
    """

    baselines: dict[int, dict]
    """hero_id → baseline row dict (from fetch_baselines_batch)."""

    team_hero_agg: dict[int, dict]
    """hero_id → team-hero agg dict (from fetch_team_hero_agg_batch), may be empty."""

    h2h_row: dict | None
    """Single head-to-head row for the team pair (same for all heroes)."""


def pre_fetch_batch(
    patch_id: int,
    hero_ids: list[int],
    team_id: int | None,
    enemy_team_id: int | None,
) -> BatchContext:
    """Pre-fetch all per-hero aggregate data in bulk (2-3 queries total).

    This replaces hundreds of individual ``fetch_baseline``,
    ``fetch_team_hero_agg``, and ``fetch_h2h`` calls with 2-3 batched
    queries.
    """
    baselines = db_.fetch_baselines_batch(patch_id, hero_ids) if hero_ids else {}
    team_hero_agg = (
        db_.fetch_team_hero_agg_batch(patch_id, team_id, hero_ids)
        if team_id and hero_ids
        else {}
    )
    h2h_row = (
        db_.fetch_h2h(patch_id, team_id, enemy_team_id)
        if team_id and enemy_team_id
        else None
    )
    return BatchContext(
        baselines=baselines,
        team_hero_agg=team_hero_agg,
        h2h_row=h2h_row,
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
    max_hero_id: int = 160,
) -> np.ndarray:
    """Build the full feature vector (numeric columns + one-hot) for a
    candidate hero.

    This function MUST produce the same feature vector, in the same column
    order, as the trainer's ``extract_features`` for the model to produce
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
    cols = schema["columns"]  # authoritative column order
    n_features_total = schema["n_features"]

    # We need the same order as feature_column_names(include_onehot=False)
    # from the trainer. Build a dict keyed by column name.
    vec: dict[str, float] = {}

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

    # -- Player-hero aggregates --
    # At inference time we don't always know the account_id. We skip
    # player-hero features if team_id is unknown (spectator mode).
    # The model is trained to tolerate all-zeros for these features.
    vec["ph_games"] = 0.0
    vec["ph_wins"] = 0.0
    vec["ph_win_rate"] = 0.5
    vec["ph_avg_gpm"] = 0.0
    vec["ph_avg_xpm"] = 0.0
    vec["ph_avg_kills"] = 0.0
    vec["ph_avg_deaths"] = 0.0
    vec["ph_avg_assists"] = 0.0
    vec["ph_avg_kda"] = 0.0
    vec["ph_lane_role"] = 0.0

    # -- Synergy with allies (per-hero: depends on current allies) --
    sy_wr, sy_cnt = db_.fetch_synergy_avg(patch_id, hero_id, ctx.ally_picks)
    vec["sy_avg_win_rate"] = _float(sy_wr, "win_rate")
    vec["sy_n_teammates"] = float(sy_cnt)

    # -- Counter vs enemies (per-hero: depends on current enemies) --
    co_wr, co_cnt = db_.fetch_counter_avg(patch_id, hero_id, ctx.enemy_picks)
    vec["co_avg_win_rate"] = _float(co_wr, "win_rate")
    vec["co_n_enemies"] = float(co_cnt)

    # -- Head-to-head (from pre-fetched batch — same for all heroes) --
    h2h = batch.h2h_row
    vec["h2h_win_rate"] = _float(h2h.get("win_rate") if h2h else None, "win_rate")
    vec["h2h_games"] = _float(h2h.get("games") if h2h else None, "games")

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

    # Build numeric array in the exact column order from the schema
    numeric_values = []
    aggregate_cols = schema["aggregate_columns"]
    for col in aggregate_cols:
        numeric_values.append(vec.get(col, 0.0))
    numeric = np.array(numeric_values, dtype=np.float32)

    # One-hot encode hero_id
    onehot = np.zeros(max_hero_id, dtype=np.float32)
    if 1 <= hero_id <= max_hero_id:
        onehot[hero_id - 1] = 1.0

    return np.concatenate([numeric, onehot])
