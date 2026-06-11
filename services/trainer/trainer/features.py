"""Feature computation for the LightGBM binary classification training pipeline.

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
            m.start_time,
            p.account_id,
            p.player_slot
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

        -- Team-hero PIT aggregate (history before this match only)
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
        COALESCE(th.avg_gold_10, 0) AS th_avg_gold_10,
        COALESCE(th.avg_xp_10, 0) AS th_avg_xp_10,

        -- Player-hero PIT aggregate (only for picks — NULL for bans)
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
        COALESCE(ph.avg_gold_10, 0) AS ph_avg_gold_10,
        COALESCE(ph.avg_xp_10, 0) AS ph_avg_xp_10,

        -- Synergy with already-picked allies (PIT-correct using historical matches)
        COALESCE(sy_avg.wr, 0.5)    AS sy_avg_win_rate,
        COALESCE(sy_avg.cnt, 0)     AS sy_n_teammates,

        -- Counter vs already-picked enemies (PIT-correct)
        COALESCE(co_avg.wr, 0.5)    AS co_avg_win_rate,
        COALESCE(co_avg.cnt, 0)     AS co_n_enemies,

        -- Team head-to-head PIT
        COALESCE(h2h.win_rate, 0.5) AS h2h_win_rate,
        COALESCE(h2h.games, 0)      AS h2h_games,

        -- Hero baseline PIT
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
        COALESCE(bl.avg_assists, 0)   AS bl_avg_assists,
        COALESCE(bl.avg_gold_10, 0)   AS bl_avg_gold_10,
        COALESCE(bl.avg_xp_10, 0)     AS bl_avg_xp_10,

        -- Low-game missingness flags (PIT-correct)
        CASE WHEN COALESCE(ph.games, 0) < 5 THEN 1 ELSE 0 END AS ph_is_new_player,
        CASE WHEN COALESCE(th.games, 0) < 5 THEN 1 ELSE 0 END AS th_is_new_team_hero,

        -- Draft-state delta features (PIT-correct)
        (COALESCE(th.win_rate, 0.5) - COALESCE(bl.win_rate, 0.5)) AS rel_th_win_rate,
        (COALESCE(ph.win_rate, 0.5) - COALESCE(bl.win_rate, 0.5)) AS rel_ph_win_rate,

        -- Role interaction features
        CASE WHEN COALESCE(ph.lane_role, 0) = 5 THEN COALESCE(ph.avg_vision_placed, 0) ELSE 0 END AS ph_vision_support_score,
        CASE WHEN COALESCE(ph.lane_role, 0) = 1 THEN COALESCE(ph.avg_gpm, 0) ELSE 0 END AS ph_gpm_carry_score

    FROM draft_slots ds

    -- ── Team-hero PIT aggregate ──────────────────────────────────────────
    -- Computes aggregates from historical matches only (start_time < ds.start_time).
    -- Bans are now correctly populated from picks_bans (was hardcoded to 0).
    LEFT JOIN LATERAL (
        SELECT
            COUNT(*) FILTER (WHERE p_hist.match_id IS NOT NULL) AS games,
            SUM(CASE WHEN p_hist.win = 1 THEN 1 ELSE 0 END)
                FILTER (WHERE p_hist.match_id IS NOT NULL) AS wins,
            COALESCE(AVG(p_hist.gold_per_min)::FLOAT, 0) AS avg_gpm,
            COALESCE(AVG(p_hist.xp_per_min)::FLOAT, 0) AS avg_xpm,
            COALESCE(AVG(p_hist.kills)::FLOAT, 0) AS avg_kills,
            COALESCE(AVG(p_hist.deaths)::FLOAT, 0) AS avg_deaths,
            COALESCE(AVG(p_hist.assists)::FLOAT, 0) AS avg_assists,
            COUNT(*) FILTER (WHERE pb_ban.match_id IS NOT NULL) AS bans,
            COALESCE(AVG(p_hist.firstblood_claimed)::FLOAT, 0) AS firstblood_rate,
            COALESCE(AVG(p_hist.camps_stacked)::FLOAT, 0) AS avg_camps_stacked,
            COALESCE(AVG(p_hist.obs_placed + p_hist.sen_placed)::FLOAT, 0) AS avg_vision_placed,
            COALESCE(AVG(gold10.avg_gold_10)::FLOAT, 0) AS avg_gold_10,
            COALESCE(AVG(xp10.avg_xp_10)::FLOAT, 0) AS avg_xp_10
        FROM matches m_hist
        LEFT JOIN players p_hist
            ON p_hist.match_id = m_hist.match_id
           AND p_hist.hero_id = ds.hero_id
           AND CASE WHEN p_hist.is_radiant THEN m_hist.radiant_team_id ELSE m_hist.dire_team_id END
               = CASE ds.team WHEN 0 THEN ds.radiant_team_id ELSE ds.dire_team_id END
        -- Bans: count times this team banned this hero in historical matches
        LEFT JOIN picks_bans pb_ban
            ON pb_ban.match_id = m_hist.match_id
           AND pb_ban.hero_id = ds.hero_id
           AND pb_ban.is_pick = FALSE
           AND CASE WHEN pb_ban.team = 0 THEN m_hist.radiant_team_id ELSE m_hist.dire_team_id END
               = CASE ds.team WHEN 0 THEN ds.radiant_team_id ELSE ds.dire_team_id END
        LEFT JOIN LATERAL (
            SELECT AVG(arr.elem::numeric) AS avg_gold_10
            FROM player_minute_stats pms,
            LATERAL jsonb_array_elements_text(pms.gold_t) WITH ORDINALITY AS arr(elem, pos)
            WHERE pms.match_id = m_hist.match_id
              AND pms.player_slot = p_hist.player_slot
              AND pms.minute = 0
              AND pos <= 10
        ) gold10 ON TRUE
        LEFT JOIN LATERAL (
            SELECT AVG(arr.elem::numeric) AS avg_xp_10
            FROM player_minute_stats pms,
            LATERAL jsonb_array_elements_text(pms.xp_t) WITH ORDINALITY AS arr(elem, pos)
            WHERE pms.match_id = m_hist.match_id
              AND pms.player_slot = p_hist.player_slot
              AND pms.minute = 0
              AND pos <= 10
        ) xp10 ON TRUE
        WHERE m_hist.patch = ds.patch_id
          AND m_hist.start_time < ds.start_time   -- ⬅ PIT: exclude current + future matches
          AND m_hist.radiant_win IS NOT NULL
    ) th ON TRUE

    -- ── Player-hero PIT aggregate ────────────────────────────────────────
    LEFT JOIN LATERAL (
        SELECT
            COUNT(*) AS games,
            SUM(CASE WHEN p_hist.win = 1 THEN 1 ELSE 0 END) AS wins,
            COALESCE(AVG(p_hist.gold_per_min)::FLOAT, 0) AS avg_gpm,
            COALESCE(AVG(p_hist.xp_per_min)::FLOAT, 0) AS avg_xpm,
            COALESCE(AVG(p_hist.kills)::FLOAT, 0) AS avg_kills,
            COALESCE(AVG(p_hist.deaths)::FLOAT, 0) AS avg_deaths,
            COALESCE(AVG(p_hist.assists)::FLOAT, 0) AS avg_assists,
            COALESCE(AVG(p_hist.kda)::FLOAT, 0) AS avg_kda,
            MODE() WITHIN GROUP (ORDER BY p_hist.lane_role) AS lane_role,
            COALESCE(AVG(p_hist.firstblood_claimed)::FLOAT, 0) AS firstblood_rate,
            COALESCE(AVG(p_hist.camps_stacked)::FLOAT, 0) AS avg_camps_stacked,
            COALESCE(AVG(p_hist.obs_placed + p_hist.sen_placed)::FLOAT, 0) AS avg_vision_placed,
            COALESCE(AVG(gold10.avg_gold_10)::FLOAT, 0) AS avg_gold_10,
            COALESCE(AVG(xp10.avg_xp_10)::FLOAT, 0) AS avg_xp_10
        FROM matches m_hist
        INNER JOIN players p_hist
            ON p_hist.match_id = m_hist.match_id
           AND p_hist.hero_id = ds.hero_id
           AND p_hist.account_id = ds.account_id
        LEFT JOIN LATERAL (
            SELECT AVG(arr.elem::numeric) AS avg_gold_10
            FROM player_minute_stats pms,
            LATERAL jsonb_array_elements_text(pms.gold_t) WITH ORDINALITY AS arr(elem, pos)
            WHERE pms.match_id = m_hist.match_id
              AND pms.player_slot = p_hist.player_slot
              AND pms.minute = 0
              AND pos <= 10
        ) gold10 ON TRUE
        LEFT JOIN LATERAL (
            SELECT AVG(arr.elem::numeric) AS avg_xp_10
            FROM player_minute_stats pms,
            LATERAL jsonb_array_elements_text(pms.xp_t) WITH ORDINALITY AS arr(elem, pos)
            WHERE pms.match_id = m_hist.match_id
              AND pms.player_slot = p_hist.player_slot
              AND pms.minute = 0
              AND pos <= 10
        ) xp10 ON TRUE
        WHERE m_hist.patch = ds.patch_id
          AND m_hist.start_time < ds.start_time   -- ⬅ PIT: exclude current + future matches
          AND m_hist.radiant_win IS NOT NULL
          AND ds.account_id IS NOT NULL
    ) ph ON TRUE

    -- ── Synergy PIT (already-picked allies in historical matches) ────────
    LEFT JOIN LATERAL (
        SELECT
            COALESCE(AVG(team_won)::FLOAT, 0.5) AS wr,
            COUNT(*)::INT AS cnt
        FROM picks_bans pb2
        CROSS JOIN LATERAL (
            SELECT
                CASE WHEN p_ally.is_radiant THEN m_hist.radiant_win
                     ELSE NOT m_hist.radiant_win END AS team_won
            FROM matches m_hist
            INNER JOIN players p_ally
                ON p_ally.match_id = m_hist.match_id
               AND p_ally.hero_id = pb2.hero_id
               AND p_ally.is_radiant = (pb2.team = 0)
            INNER JOIN players p_hero
                ON p_hero.match_id = m_hist.match_id
               AND p_hero.hero_id = ds.hero_id
               AND p_hero.is_radiant = p_ally.is_radiant
            WHERE m_hist.patch = ds.patch_id
              AND m_hist.start_time < ds.start_time
              AND m_hist.radiant_win IS NOT NULL
        ) pair_hist
        WHERE pb2.match_id = ds.match_id
          AND pb2."order"  < ds."order"
          AND pb2.is_pick  = TRUE
          AND pb2.team     = ds.team
        GROUP BY pb2.hero_id
    ) sy_avg ON TRUE

    -- ── Counter PIT (enemy picks in historical matches) ──────────────────
    LEFT JOIN LATERAL (
        SELECT
            COALESCE(AVG(team_won)::FLOAT, 0.5) AS wr,
            COUNT(*)::INT AS cnt
        FROM picks_bans pb2
        CROSS JOIN LATERAL (
            SELECT
                CASE WHEN p_ally.is_radiant THEN m_hist.radiant_win
                     ELSE NOT m_hist.radiant_win END AS team_won
            FROM matches m_hist
            INNER JOIN players p_ally
                ON p_ally.match_id = m_hist.match_id
               AND p_ally.hero_id = pb2.hero_id
               AND p_ally.is_radiant = (pb2.team = 0)
            INNER JOIN players p_hero
                ON p_hero.match_id = m_hist.match_id
               AND p_hero.hero_id = ds.hero_id
               AND p_hero.is_radiant != p_ally.is_radiant
            WHERE m_hist.patch = ds.patch_id
              AND m_hist.start_time < ds.start_time
              AND m_hist.radiant_win IS NOT NULL
        ) pair_hist
        WHERE pb2.match_id = ds.match_id
          AND pb2."order"  < ds."order"
          AND pb2.is_pick  = TRUE
          AND pb2.team    != ds.team
        GROUP BY pb2.hero_id
    ) co_avg ON TRUE

    -- ── Head-to-head PIT ─────────────────────────────────────────────────
    LEFT JOIN LATERAL (
        SELECT
            COUNT(*) AS games,
            COALESCE(AVG(CASE
                WHEN m_hist.radiant_team_id = my_team AND m_hist.dire_team_id = enemy_team
                    THEN CASE WHEN m_hist.radiant_win THEN 1.0 ELSE 0.0 END
                WHEN m_hist.dire_team_id = my_team AND m_hist.radiant_team_id = enemy_team
                    THEN CASE WHEN NOT m_hist.radiant_win THEN 1.0 ELSE 0.0 END
            END), 0.5)::FLOAT AS win_rate
        FROM matches m_hist
        CROSS JOIN (SELECT
            CASE ds.team WHEN 0 THEN ds.radiant_team_id ELSE ds.dire_team_id END AS my_team,
            CASE ds.team WHEN 0 THEN ds.dire_team_id ELSE ds.radiant_team_id END AS enemy_team
        ) teams
        WHERE m_hist.patch = ds.patch_id
          AND m_hist.start_time < ds.start_time
          AND m_hist.radiant_win IS NOT NULL
          AND ((m_hist.radiant_team_id = my_team AND m_hist.dire_team_id = enemy_team)
               OR (m_hist.dire_team_id = my_team AND m_hist.radiant_team_id = enemy_team))
    ) h2h ON TRUE

    -- ── Hero baseline PIT ────────────────────────────────────────────────
    LEFT JOIN LATERAL (
        WITH hero_stats AS (
            SELECT
                COUNT(*) AS total_picks,
                SUM(CASE WHEN p_hist.win = 1 THEN 1 ELSE 0 END) AS total_wins,
                COALESCE(AVG(p_hist.gold_per_min)::FLOAT, 0) AS avg_gpm,
                COALESCE(AVG(p_hist.xp_per_min)::FLOAT, 0) AS avg_xpm,
                COALESCE(AVG(p_hist.kills)::FLOAT, 0) AS avg_kills,
                COALESCE(AVG(p_hist.deaths)::FLOAT, 0) AS avg_deaths,
                COALESCE(AVG(p_hist.assists)::FLOAT, 0) AS avg_assists,
                COALESCE(AVG(gold10.avg_gold_10)::FLOAT, 0) AS avg_gold_10,
                COALESCE(AVG(xp10.avg_xp_10)::FLOAT, 0) AS avg_xp_10
            FROM matches m_hist
            INNER JOIN players p_hist
                ON p_hist.match_id = m_hist.match_id
               AND p_hist.hero_id = ds.hero_id
            LEFT JOIN LATERAL (
                SELECT AVG(arr.elem::numeric) AS avg_gold_10
                FROM player_minute_stats pms,
                LATERAL jsonb_array_elements_text(pms.gold_t) WITH ORDINALITY AS arr(elem, pos)
                WHERE pms.match_id = m_hist.match_id
                  AND pms.player_slot = p_hist.player_slot
                  AND pms.minute = 0
                  AND pos <= 10
            ) gold10 ON TRUE
            LEFT JOIN LATERAL (
                SELECT AVG(arr.elem::numeric) AS avg_xp_10
                FROM player_minute_stats pms,
                LATERAL jsonb_array_elements_text(pms.xp_t) WITH ORDINALITY AS arr(elem, pos)
                WHERE pms.match_id = m_hist.match_id
                  AND pms.player_slot = p_hist.player_slot
                  AND pms.minute = 0
                  AND pos <= 10
            ) xp10 ON TRUE
            WHERE m_hist.patch = ds.patch_id
              AND m_hist.start_time < ds.start_time
              AND m_hist.radiant_win IS NOT NULL
        ),
        hero_bans AS (
            SELECT COUNT(*) AS total_bans
            FROM matches m_hist
            INNER JOIN picks_bans pb_hist
                ON pb_hist.match_id = m_hist.match_id
               AND pb_hist.hero_id = ds.hero_id
               AND pb_hist.is_pick = FALSE
            WHERE m_hist.patch = ds.patch_id
              AND m_hist.start_time < ds.start_time
              AND m_hist.radiant_win IS NOT NULL
        ),
        total_matches AS (
            SELECT COUNT(DISTINCT match_id) AS total
            FROM matches m_hist
            WHERE m_hist.patch = ds.patch_id
              AND m_hist.start_time < ds.start_time
              AND m_hist.radiant_win IS NOT NULL
        )
        SELECT
            hs.total_picks,
            hs.total_wins,
            COALESCE(hb.total_bans, 0) AS total_bans,
            (hs.total_wins + 3.0 * 0.5) / (hs.total_picks + 3.0)::FLOAT AS win_rate,
            CASE WHEN tm.total > 0 THEN hs.total_picks::FLOAT / tm.total ELSE 0 END AS pick_rate,
            CASE WHEN tm.total > 0 THEN COALESCE(hb.total_bans, 0)::FLOAT / tm.total ELSE 0 END AS ban_rate,
            hs.avg_gpm, hs.avg_xpm, hs.avg_kills, hs.avg_deaths, hs.avg_assists,
            hs.avg_gold_10, hs.avg_xp_10
        FROM hero_stats hs
        CROSS JOIN total_matches tm
        LEFT JOIN hero_bans hb ON TRUE
    ) bl ON TRUE

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
        "th_avg_gold_10", "th_avg_xp_10",
        # Player-hero aggregates
        "ph_games", "ph_wins", "ph_win_rate",
        "ph_avg_gpm", "ph_avg_xpm", "ph_avg_kills", "ph_avg_deaths", "ph_avg_assists",
        "ph_avg_kda", "ph_lane_role",
        "ph_firstblood_rate", "ph_avg_camps_stacked", "ph_avg_vision_placed",
        "ph_avg_gold_10", "ph_avg_xp_10",
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
        "bl_avg_gold_10", "bl_avg_xp_10",
        # Task 4: Low-game flags
        "ph_is_new_player",
        "th_is_new_team_hero",
        # Task 5: Delta features
        "rel_th_win_rate",
        "rel_ph_win_rate",
        # Task 6: Role interactions
        "ph_vision_support_score",
        "ph_gpm_carry_score",
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
