"""Feature computation for the LightGBM lambdarank training pipeline.

Each row in the training dataset corresponds to one element of a draft
(pick or ban decision in a specific match). Features describe the state
of the draft at that moment: which heroes are already picked/banned,
team and player historical aggregates, synergy/counter stats, etc.

Column names and their order are written to ``feature_schema.json`` alongside
the trained model file. The inference API loads this JSON at startup to
guarantee column-order agreement between training and inference.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column naming convention
# ---------------------------------------------------------------------------
# Prefixes help the model distinguish feature types:
#   th_  — team-hero aggregate
#   ph_  — player-hero aggregate
#   sy_  — synergy (hero pair on same team)
#   co_  — counter (hero vs enemy hero)
#   h2h_ — head-to-head (team vs team)
#   bl_  — hero baseline (global, all teams)
#   oh_  — one-hot encoded hero ID (sparse indicator)

# ---------------------------------------------------------------------------
# SQL query: build training features
# ---------------------------------------------------------------------------

TRAINING_FEATURES_SQL = """
    WITH draft_slots AS (
        SELECT
            pb.match_id,
            pb.hero_id,
            pb.is_pick,
            pb.team,
            pb."order",
            m.radiant_team_id,
            m.dire_team_id,
            m.radiant_win,
            m.patch AS patch_id,
            p.account_id
        FROM picks_bans pb
        INNER JOIN matches m ON pb.match_id = m.match_id
        LEFT JOIN players p
               ON p.match_id = pb.match_id
              AND p.hero_id = pb.hero_id
              AND p.is_radiant = (pb.team = 0)
        WHERE m.patch = %(patch_id)s
          AND pb."order" IS NOT NULL
    )
    SELECT
        ds.match_id,
        ds.hero_id,
        ds.is_pick::INT AS is_pick,
        ds.team,
        ds."order" AS "order",
        ds.radiant_team_id,
        ds.dire_team_id,
        ds.radiant_win,

        -- Team-hero aggregate (from the perspective of the picking team)
        COALESCE(th.games, 0)       AS th_games,
        COALESCE(th.wins, 0)        AS th_wins,
        COALESCE(th.win_rate, 0.5)  AS th_win_rate,
        COALESCE(th.bans, 0)        AS th_bans,
        COALESCE(th.avg_gpm, 0)     AS th_avg_gpm,
        COALESCE(th.avg_xpm, 0)     AS th_avg_xpm,
        COALESCE(th.avg_kills, 0)   AS th_avg_kills,
        COALESCE(th.avg_deaths, 0)  AS th_avg_deaths,
        COALESCE(th.avg_assists, 0) AS th_avg_assists,
        COALESCE(th.firstblood_rate, 0) AS th_firstblood_rate,
        COALESCE(th.avg_camps_stacked, 0) AS th_avg_camps_stacked,
        COALESCE(th.avg_vision_placed, 0) AS th_avg_vision_placed,

        -- Player-hero aggregate (only for picks — NULL for bans)
        COALESCE(ph.games, 0)       AS ph_games,
        COALESCE(ph.wins, 0)        AS ph_wins,
        COALESCE(ph.win_rate, 0.5)  AS ph_win_rate,
        COALESCE(ph.avg_gpm, 0)     AS ph_avg_gpm,
        COALESCE(ph.avg_xpm, 0)     AS ph_avg_xpm,
        COALESCE(ph.avg_kills, 0)   AS ph_avg_kills,
        COALESCE(ph.avg_deaths, 0)  AS ph_avg_deaths,
        COALESCE(ph.avg_assists, 0) AS ph_avg_assists,
        COALESCE(ph.avg_kda, 0)     AS ph_avg_kda,
        COALESCE(ph.lane_role, 0)   AS ph_lane_role,
        COALESCE(ph.firstblood_rate, 0) AS ph_firstblood_rate,
        COALESCE(ph.avg_camps_stacked, 0) AS ph_avg_camps_stacked,
        COALESCE(ph.avg_vision_placed, 0) AS ph_avg_vision_placed,

        -- Synergy with already-picked allies (uses LEAST/GREATEST for index)
        COALESCE(sy_avg.wr, 0.5)    AS sy_avg_win_rate,
        COALESCE(sy_avg.cnt, 0)     AS sy_n_teammates,

        -- Counter vs already-picked enemies
        COALESCE(co_avg.wr, 0.5)    AS co_avg_win_rate,
        COALESCE(co_avg.cnt, 0)     AS co_n_enemies,

        -- Team head-to-head
        COALESCE(h2h.win_rate, 0.5) AS h2h_win_rate,
        COALESCE(h2h.games, 0)      AS h2h_games,

        -- Hero baseline
        COALESCE(bl.total_picks, 0)   AS bl_total_picks,
        COALESCE(bl.total_wins, 0)    AS bl_total_wins,
        COALESCE(bl.total_bans, 0)    AS bl_total_bans,
        COALESCE(bl.win_rate, 0.5)    AS bl_win_rate,
        COALESCE(bl.pick_rate, 0)     AS bl_pick_rate,
        COALESCE(bl.ban_rate, 0)      AS bl_ban_rate,
        COALESCE(bl.avg_gpm, 0)       AS bl_avg_gpm,
        COALESCE(bl.avg_xpm, 0)       AS bl_avg_xpm,
        COALESCE(bl.avg_kills, 0)     AS bl_avg_kills,
        COALESCE(bl.avg_deaths, 0)    AS bl_avg_deaths,
        COALESCE(bl.avg_assists, 0)   AS bl_avg_assists

    FROM draft_slots ds

    LEFT JOIN ml.team_hero_agg th
        ON th.patch_id = ds.patch_id
       AND th.team_id  = CASE ds.team WHEN 0 THEN ds.radiant_team_id ELSE ds.dire_team_id END
       AND th.hero_id  = ds.hero_id

    LEFT JOIN ml.player_hero_agg ph
        ON ph.patch_id    = ds.patch_id
       AND ph.hero_id     = ds.hero_id
       AND ph.account_id  = ds.account_id

    -- Synergy with already-picked teammates (LEAST/GREATEST uses PK index)
    LEFT JOIN LATERAL (
        SELECT
            AVG(s.win_rate)::FLOAT AS wr,
            COUNT(*)::INT          AS cnt
        FROM picks_bans pb2
        INNER JOIN ml.hero_synergy_agg s
            ON s.patch_id = ds.patch_id
           AND s.hero_a = LEAST(ds.hero_id, pb2.hero_id)
           AND s.hero_b = GREATEST(ds.hero_id, pb2.hero_id)
        WHERE pb2.match_id  = ds.match_id
          AND pb2."order"   < ds."order"
          AND pb2.is_pick   = TRUE
          AND pb2.team      = ds.team
    ) sy_avg ON TRUE

    -- Counter vs already-picked enemies
    LEFT JOIN LATERAL (
        SELECT
            AVG(c.win_rate)::FLOAT AS wr,
            COUNT(*)::INT          AS cnt
        FROM picks_bans pb2
        INNER JOIN ml.hero_counter_agg c
            ON c.patch_id = ds.patch_id
           AND c.hero_id  = ds.hero_id
           AND c.enemy_hero_id = pb2.hero_id
        WHERE pb2.match_id  = ds.match_id
          AND pb2."order"   < ds."order"
          AND pb2.is_pick   = TRUE
          AND pb2.team     != ds.team
    ) co_avg ON TRUE

    -- Head-to-head
    LEFT JOIN ml.team_h2h_agg h2h
        ON h2h.patch_id       = ds.patch_id
       AND h2h.team_id        = CASE ds.team WHEN 0 THEN ds.radiant_team_id ELSE ds.dire_team_id END
       AND h2h.enemy_team_id  = CASE ds.team WHEN 0 THEN ds.dire_team_id ELSE ds.radiant_team_id END

    -- Hero baseline
    LEFT JOIN ml.hero_baseline_agg bl
        ON bl.patch_id = ds.patch_id
       AND bl.hero_id  = ds.hero_id

    ORDER BY ds.match_id, ds."order"
"""


def feature_column_names(include_onehot: bool = True, max_hero_id: int = 160) -> list[str]:
    """Return the ordered list of feature column names.

    This is the source of truth for the training/API column contract.
    """
    cols = [
        # Draft context (side + pick-vs-ban) — critical context for the model
        "is_pick", "team",
        # Team-hero aggregates
        "th_games", "th_wins", "th_win_rate", "th_bans",
        "th_avg_gpm", "th_avg_xpm", "th_avg_kills", "th_avg_deaths", "th_avg_assists",
        "th_firstblood_rate", "th_avg_camps_stacked", "th_avg_vision_placed",
        # Player-hero aggregates
        "ph_games", "ph_wins", "ph_win_rate",
        "ph_avg_gpm", "ph_avg_xpm", "ph_avg_kills", "ph_avg_deaths", "ph_avg_assists",
        "ph_avg_kda", "ph_lane_role",
        "ph_firstblood_rate", "ph_avg_camps_stacked", "ph_avg_vision_placed",
        # Synergy
        "sy_avg_win_rate", "sy_n_teammates",
        # Counter
        "co_avg_win_rate", "co_n_enemies",
        # Team head-to-head
        "h2h_win_rate", "h2h_games",
        # Hero baseline
        "bl_total_picks", "bl_total_wins", "bl_total_bans",
        "bl_win_rate", "bl_pick_rate", "bl_ban_rate",
        "bl_avg_gpm", "bl_avg_xpm", "bl_avg_kills", "bl_avg_deaths", "bl_avg_assists",
    ]
    if include_onehot:
        cols.extend(f"oh_hero_{i}" for i in range(1, max_hero_id + 1))
    return cols


# ---------------------------------------------------------------------------
# Feature extraction from raw SQL result
# ---------------------------------------------------------------------------

def make_target(df: pd.DataFrame) -> np.ndarray:
    """Return the target vector: 1 if the picker's team won, 0 otherwise.

    The target is relative to the picking team, NOT absolute radiant_win:
      team=0 (Radiant): target = radiant_win
      team=1 (Dire):    target = 1 - radiant_win  (i.e., Dire won)

    Previously this function returned bare ``radiant_win``, which meant
    the model was trained to associate strong Dire picks with failure
    (since a Dire win sets radiant_win=0 → target=0). This inversion
    caused the model to learn the inverse for all Dire training rows
    (Bug #2).

    For bans we still assign a target (the team that banned won/lost) so
    that the model learns to associate bans with outcomes too.
    """
    return (df["radiant_win"] == (df["team"] == 0)).astype(int).values


def make_group(df: pd.DataFrame) -> np.ndarray:
    """Return the query group array for LightGBM lambdarank.

    Each match is one query group. The group size is the number of
    draft slots (picks + bans) for that match.
    """
    return df.groupby("match_id", sort=False).size().values


def extract_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    max_hero_id: int = 160,
) -> np.ndarray:
    """Build the full feature matrix including one-hot hero encoding.

    *df* is the result of ``TRAINING_FEATURES_SQL``.
    *feature_cols* is the list of non-onehot column names.
    """
    numeric = df[feature_cols].fillna(0).values.astype(np.float32)

    # One-hot encode hero_id (1-indexed, 1..max_hero_id)
    hero_ids = df["hero_id"].values
    n = len(hero_ids)
    onehot = np.zeros((n, max_hero_id), dtype=np.float32)
    for i in range(n):
        hid = int(hero_ids[i])
        if 1 <= hid <= max_hero_id:
            onehot[i, hid - 1] = 1.0

    return np.concatenate([numeric, onehot], axis=1)


def write_schema(model_dir: str | Path, max_hero_id: int = 160) -> None:
    """Write ``feature_schema.json`` — the authoritative column-order contract.

    The inference API loads this file to build its feature vectors in
    *exactly* the same order as training.
    """
    cols = feature_column_names(include_onehot=True, max_hero_id=max_hero_id)
    schema = {
        "columns": cols,
        "n_features": len(cols),
        "max_hero_id": max_hero_id,
        "aggregate_columns": feature_column_names(include_onehot=False),
        "onehot_prefix": "oh_hero_",
    }
    path = Path(model_dir) / "feature_schema.json"
    path.write_text(json.dumps(schema, indent=2))
    logger.info("Wrote feature schema to %s (%d columns)", path, len(cols))
