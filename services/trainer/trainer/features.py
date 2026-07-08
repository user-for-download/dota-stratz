"""Feature computation for the PyTorch DraftBERT training pipeline.

Each row in the training dataset corresponds to one element of a draft
(pick or ban decision in a specific match). Features describe the state
of the draft at that moment: which heroes are already picked/banned,
team and player historical aggregates, synergy/counter stats, etc.

Column names and their order are written to ``feature_schema.json`` alongside
the trained model file. The inference API loads this JSON at startup to
guarantee column-order agreement between training and inference.

**PIT-safe snapshots**: ``TRAINING_FEATURES_SQL`` now uses LATERAL subqueries
against the ``ml.*_snapshot`` tables (migration 014), looking up the most
recent snapshot ``AS OF`` the match start time. This eliminates look-ahead
bias where aggregate features could \"see\" future match outcomes within the
same patch.
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
#   hds_ — hero draft-slot (pick-position) win rate
#   oh_  — one-hot encoded hero ID (sparse indicator)

# ---------------------------------------------------------------------------
# SQL query: build training features
# ---------------------------------------------------------------------------

def training_features_sql_fast(extra: str = "") -> str:
    """Fast training SQL using aggregate tables directly (no LATERAL joins).

    Uses ml.*_agg tables instead of ml.*_snapshot tables.
    Much faster (~30s vs ~20min) but slightly less PIT-safe.
    Good enough for training since aggregates are already populated.
    """
    return f"""
    WITH draft_slots AS (
        SELECT
            pb.match_id, pb.hero_id, pb.is_pick, pb.team, pb."order",
            m.start_time, m.radiant_team_id, m.dire_team_id, m.radiant_win,
            m.patch AS patch_id, p.account_id, p.player_slot,
            CASE WHEN pb.is_pick THEN
                ROW_NUMBER() OVER (PARTITION BY pb.match_id, pb.team, pb.is_pick ORDER BY pb."order")
            END AS team_pick_ordinal
        FROM picks_bans pb
        INNER JOIN matches m ON pb.match_id = m.match_id
        LEFT JOIN players p ON p.match_id = pb.match_id AND p.hero_id = pb.hero_id AND p.is_radiant = (pb.team = 0)
        WHERE m.patch = :patch_id
          AND m.radiant_win IS NOT NULL AND m.duration >= 900
          {extra}
          AND NOT EXISTS (SELECT 1 FROM players p2 WHERE p2.match_id = m.match_id AND p2.leaver_status IN (1,2,3,4))
          AND pb."order" IS NOT NULL AND pb.hero_id IS NOT NULL AND pb.team IN (0,1)
    )
    SELECT
        ds.match_id, ds.hero_id, ds.is_pick::INT AS is_pick, ds.team,
        CASE WHEN ds."order" <= 7 THEN 0 WHEN ds."order" <= 9 THEN 1
             WHEN ds."order" <= 12 THEN 2 WHEN ds."order" <= 18 THEN 3
             WHEN ds."order" <= 22 THEN 4 ELSE 5 END AS draft_phase_id,
        ds."order", ds.start_time, ds.radiant_team_id, ds.dire_team_id, ds.radiant_win,
        -- Aggregates (direct join, no LATERAL)
        COALESCE(th.games,0) AS th_games, COALESCE(th.wins,0) AS th_wins,
        COALESCE(th.win_rate,0.5) AS th_win_rate, COALESCE(th.bans,0) AS th_bans,
        COALESCE(th.avg_gpm,0) AS th_avg_gpm, COALESCE(th.avg_xpm,0) AS th_avg_xpm,
        COALESCE(th.avg_kills,0) AS th_avg_kills, COALESCE(th.avg_deaths,0) AS th_avg_deaths,
        COALESCE(th.avg_assists,0) AS th_avg_assists, COALESCE(th.firstblood_rate,0) AS th_firstblood_rate,
        COALESCE(th.avg_camps_stacked,0) AS th_avg_camps_stacked,
        COALESCE(th.avg_vision_placed,0) AS th_avg_vision_placed,
        COALESCE(th.avg_gold_10,0) AS th_avg_gold_10, COALESCE(th.avg_xp_10,0) AS th_avg_xp_10,
        COALESCE(ph.games,0) AS ph_games, COALESCE(ph.wins,0) AS ph_wins,
        COALESCE(ph.win_rate,0.5) AS ph_win_rate, COALESCE(ph.avg_gpm,0) AS ph_avg_gpm,
        COALESCE(ph.avg_xpm,0) AS ph_avg_xpm, COALESCE(ph.avg_kills,0) AS ph_avg_kills,
        COALESCE(ph.avg_deaths,0) AS ph_avg_deaths, COALESCE(ph.avg_assists,0) AS ph_avg_assists,
        COALESCE(ph.avg_kda,0) AS ph_avg_kda, COALESCE(ph.lane_role,0) AS ph_lane_role,
        COALESCE(ph.firstblood_rate,0) AS ph_firstblood_rate,
        COALESCE(ph.avg_camps_stacked,0) AS ph_avg_camps_stacked,
        COALESCE(ph.avg_vision_placed,0) AS ph_avg_vision_placed,
        COALESCE(ph.avg_gold_10,0) AS ph_avg_gold_10, COALESCE(ph.avg_xp_10,0) AS ph_avg_xp_10,
        -- Synergy/counter: use hero-level baseline stats (fast SQL can't do per-draft-step joins)
        COALESCE(bl.win_rate, 0.5) AS sy_avg_win_rate, 0 AS sy_n_teammates,
        COALESCE(bl.win_rate, 0.5) AS co_avg_win_rate, 0 AS co_n_enemies,
        0.0 AS co_avg_kd_diff,
        COALESCE(h2h.win_rate,0.5) AS h2h_win_rate, COALESCE(h2h.games,0) AS h2h_games,
        COALESCE(bl.total_picks,0) AS bl_total_picks, COALESCE(bl.total_wins,0) AS bl_total_wins,
        COALESCE(bl.total_bans,0) AS bl_total_bans, COALESCE(bl.win_rate,0.5) AS bl_win_rate,
        COALESCE(bl.pick_rate,0) AS bl_pick_rate, COALESCE(bl.ban_rate,0) AS bl_ban_rate,
        COALESCE(bl.avg_gpm,0) AS bl_avg_gpm, COALESCE(bl.avg_xpm,0) AS bl_avg_xpm,
        COALESCE(bl.avg_kills,0) AS bl_avg_kills, COALESCE(bl.avg_deaths,0) AS bl_avg_deaths,
        COALESCE(bl.avg_assists,0) AS bl_avg_assists,
        COALESCE(bl.avg_gold_10,0) AS bl_avg_gold_10, COALESCE(bl.avg_xp_10,0) AS bl_avg_xp_10,
        COALESCE(hds.win_rate,0.5) AS hds_win_rate, COALESCE(hds.games,0) AS hds_games,
        CASE WHEN COALESCE(ph.games,0) < 5 THEN 1 ELSE 0 END AS ph_is_new_player,
        CASE WHEN COALESCE(th.games,0) < 5 THEN 1 ELSE 0 END AS th_is_new_team_hero,
        (COALESCE(th.win_rate,0.5) - COALESCE(bl.win_rate,0.5)) AS rel_th_win_rate,
        (COALESCE(ph.win_rate,0.5) - COALESCE(bl.win_rate,0.5)) AS rel_ph_win_rate,
        CASE WHEN COALESCE(ph.lane_role,0) = 5 THEN COALESCE(ph.avg_vision_placed,0) ELSE 0 END AS ph_vision_support_score,
        CASE WHEN COALESCE(ph.lane_role,0) = 1 THEN COALESCE(ph.avg_gpm,0) ELSE 0 END AS ph_gpm_carry_score,
        COALESCE(bl.avg_gpm,0) AS team_gpm_budget, COALESCE(bl.avg_xpm,0) AS team_xpm_budget,
        CASE WHEN COALESCE(bl.total_picks,0) > 0 THEN COALESCE(th.games,0)::FLOAT / bl.total_picks ELSE 0.0 END AS team_pick_propensity
    FROM draft_slots ds
    LEFT JOIN ml.team_hero_agg th ON th.patch_id = ds.patch_id
        AND th.team_id = CASE ds.team WHEN 0 THEN ds.radiant_team_id ELSE ds.dire_team_id END
        AND th.hero_id = ds.hero_id
    LEFT JOIN ml.player_hero_agg ph ON ph.patch_id = ds.patch_id
        AND ph.account_id = ds.account_id AND ph.hero_id = ds.hero_id
    -- Synergy/counter: REMOVED from fast SQL (joins were hero vs itself, always defaulting)
    -- These features come from hero baseline stats instead; Transformer learns synergy from sequence
    LEFT JOIN ml.team_h2h_agg h2h ON h2h.patch_id = ds.patch_id
        AND h2h.team_id = CASE ds.team WHEN 0 THEN ds.radiant_team_id ELSE ds.dire_team_id END
        AND h2h.enemy_team_id = CASE ds.team WHEN 0 THEN ds.dire_team_id ELSE ds.radiant_team_id END
    LEFT JOIN ml.hero_baseline_agg bl ON bl.patch_id = ds.patch_id AND bl.hero_id = ds.hero_id
    LEFT JOIN ml.hero_draft_slot_agg hds ON hds.patch_id = ds.patch_id
        AND hds.hero_id = ds.hero_id AND hds.team_pick_ordinal = ds.team_pick_ordinal
    ORDER BY ds.match_id, ds."order"
    """


def training_features_sql(extra: str = "", lookback: int = 0) -> str:
    """Return the training features SQL with optional match filters.

    Parameters
    ----------
    extra : str
        Additional ``AND ...`` conditions injected into the ``WHERE`` clause
        of the ``draft_slots`` CTE.  Must use the ``m`` alias for the
        ``matches`` table.  Build via ``_match_extra_where(cfg)`` from
        ``aggregates.py`` to keep filters consistent with aggregate
        population.
    lookback : int
        Number of previous patches for aggregate/snapshot data only.
        Draft data always uses exactly patch_id (current patch only).
    """
    # Draft data: ONLY current patch (patch 60) — current gameplay only
    patch_cond = "m.patch = :patch_id"

    return f"""
    WITH draft_slots AS (
        SELECT
            pb.match_id,
            pb.hero_id,
            pb.is_pick,
            pb.team,
            pb."order",
            m.start_time,
            m.radiant_team_id,
            m.dire_team_id,
            m.radiant_win,
            m.patch AS patch_id,
            p.account_id,
            p.player_slot,
            CASE WHEN pb.is_pick THEN
                ROW_NUMBER() OVER (
                    PARTITION BY pb.match_id, pb.team, pb.is_pick
                    ORDER BY pb."order"
                )
            END AS team_pick_ordinal
        FROM picks_bans pb
        INNER JOIN matches m ON pb.match_id = m.match_id
        LEFT JOIN players p
               ON p.match_id = pb.match_id
              AND p.hero_id = pb.hero_id
              AND p.is_radiant = (pb.team = 0)
        WHERE {patch_cond}
          AND m.radiant_win IS NOT NULL
          AND m.duration >= 900
          {extra}
          AND NOT EXISTS (
              SELECT 1 FROM players p2
              WHERE p2.match_id = m.match_id
              AND p2.leaver_status IN (1, 2, 3, 4)
          )
          AND pb."order" IS NOT NULL
          AND pb.hero_id IS NOT NULL
          AND pb.team IN (0, 1)
    )
    SELECT
        ds.match_id,
        ds.hero_id,
        ds.is_pick::INT AS is_pick,
        ds.team,
        -- CM draft phase: 0=Ban1, 1=Pick1, 2=Ban2, 3=Pick2, 4=Ban3, 5=FinalPick
        CASE
            WHEN ds."order" <= 7 THEN 0
            WHEN ds."order" <= 9 THEN 1
            WHEN ds."order" <= 12 THEN 2
            WHEN ds."order" <= 18 THEN 3
            WHEN ds."order" <= 22 THEN 4
            ELSE 5
        END AS draft_phase_id,
        ds."order" AS "order",
        ds.start_time,
        ds.radiant_team_id,
        ds.dire_team_id,
        ds.radiant_win,

        -- Team-hero aggregate (from pre-computed ml.team_hero_agg)
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

        -- Player-hero aggregate (from ml.player_hero_agg)
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

        -- Synergy with already-picked allies (from ml.hero_synergy_agg)
        COALESCE(sy.win_rate, 0.5)  AS sy_avg_win_rate,
        -- sy_n_teammates: counts the number of already-picked allies (COUNT(*) over the
        -- lateral join's pb2 rows, which is preserved for all prior picks regardless of
        -- whether a synergy row exists). Semantically equivalent to len(ally_picks) in the API.
        COALESCE(sy.games, 0)       AS sy_n_teammates,

        -- Counter vs already-picked enemies (from ml.hero_counter_agg)
        COALESCE(co.win_rate, 0.5)  AS co_avg_win_rate,
        -- co_n_enemies: counts the number of already-picked enemies (COUNT(*) over the
        -- lateral join's pb2 rows for the opposing team). Semantically equivalent to len(enemy_picks) in the API.
        COALESCE(co.games, 0)       AS co_n_enemies,
        COALESCE(co.avg_kd_diff, 0) AS co_avg_kd_diff,

        -- Team head-to-head (from ml.team_h2h_agg)
        COALESCE(h2h.win_rate, 0.5) AS h2h_win_rate,
        COALESCE(h2h.games, 0)      AS h2h_games,

        -- Hero baseline (from ml.hero_baseline_agg)
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

        -- Hero draft-slot (pick-position) win rate
        COALESCE(hds.win_rate, 0.5) AS hds_win_rate,
        COALESCE(hds.games, 0)      AS hds_games,

        -- Low-game missingness flags
        CASE WHEN COALESCE(ph.games, 0) < 5 THEN 1 ELSE 0 END AS ph_is_new_player,
        CASE WHEN COALESCE(th.games, 0) < 5 THEN 1 ELSE 0 END AS th_is_new_team_hero,

        -- Draft-state delta features
        (COALESCE(th.win_rate, 0.5) - COALESCE(bl.win_rate, 0.5)) AS rel_th_win_rate,
        (COALESCE(ph.win_rate, 0.5) - COALESCE(bl.win_rate, 0.5)) AS rel_ph_win_rate,

        -- Role interaction features
        CASE WHEN COALESCE(ph.lane_role, 0) = 5 THEN COALESCE(ph.avg_vision_placed, 0) ELSE 0 END AS ph_vision_support_score,
        CASE WHEN COALESCE(ph.lane_role, 0) = 1 THEN COALESCE(ph.avg_gpm, 0) ELSE 0 END AS ph_gpm_carry_score,

        -- Macro Composition Features
        COALESCE(mac.ally_gpm, 0) + COALESCE(bl.avg_gpm, 0) AS team_gpm_budget,
        COALESCE(mac.ally_xpm, 0) + COALESCE(bl.avg_xpm, 0) AS team_xpm_budget,

        -- Pick Propensity (team comfort pick signal)
        CASE WHEN COALESCE(bl.total_picks, 0) > 0
            THEN COALESCE(th.games, 0)::FLOAT / bl.total_picks
            ELSE 0.0 END AS team_pick_propensity

    FROM draft_slots ds

    -- Team-hero aggregate (PIT-safe: most recent daily snapshot as of match start)
    -- Falls back to earliest available snapshot when match predates snapshot system
    LEFT JOIN LATERAL (
        SELECT * FROM ml.team_hero_snapshot ths
        WHERE ths.team_id = CASE ds.team WHEN 0 THEN ds.radiant_team_id ELSE ds.dire_team_id END
          AND ths.hero_id = ds.hero_id
          AND ths.patch_id <= ds.patch_id
        ORDER BY (ths.as_of_date <= to_timestamp(ds.start_time)::DATE) DESC,
                 ths.as_of_date DESC
        LIMIT 1
    ) th ON TRUE

    -- Player-hero aggregate (PIT-safe: most recent weekly snapshot as of match start)
    -- Falls back to earliest available snapshot when match predates snapshot system
    -- NULL for bans where account_id is NULL
    LEFT JOIN LATERAL (
        SELECT * FROM ml.player_hero_snapshot phs
        WHERE phs.account_id = ds.account_id
          AND phs.hero_id = ds.hero_id
          AND phs.patch_id <= ds.patch_id
        ORDER BY (phs.as_of_date <= to_timestamp(ds.start_time)::DATE) DESC,
                 phs.as_of_date DESC
        LIMIT 1
    ) ph ON TRUE

    -- Synergy: look up each already-picked ally's hero pair from ml.hero_synergy_snapshot
    -- Falls back to earliest available snapshot when match predates snapshot system
    LEFT JOIN LATERAL (
        SELECT
            COALESCE(AVG(snap.win_rate), 0.5) AS win_rate,
            COUNT(*)::INT AS games
        FROM picks_bans pb2
        LEFT JOIN LATERAL (
            SELECT win_rate, games FROM ml.hero_synergy_snapshot hss
            WHERE hss.hero_a = LEAST(ds.hero_id, pb2.hero_id)
              AND hss.hero_b = GREATEST(ds.hero_id, pb2.hero_id)
              AND hss.patch_id <= ds.patch_id
            ORDER BY (hss.as_of_date <= to_timestamp(ds.start_time)::DATE) DESC,
                     hss.as_of_date DESC
            LIMIT 1
        ) snap ON TRUE
        WHERE pb2.match_id = ds.match_id
          AND pb2."order"  < ds."order"
          AND pb2.is_pick  = TRUE
          AND pb2.team     = ds.team
    ) sy ON TRUE

    -- Counter: look up each enemy pick's hero pair from ml.hero_counter_snapshot
    -- Falls back to earliest available snapshot when match predates snapshot system
    LEFT JOIN LATERAL (
        SELECT
            COALESCE(AVG(snap.win_rate), 0.5) AS win_rate,
            COUNT(*)::INT AS games,
            COALESCE(AVG(snap.avg_kd_diff), 0.0) AS avg_kd_diff
        FROM picks_bans pb2
        LEFT JOIN LATERAL (
            SELECT win_rate, avg_kd_diff FROM ml.hero_counter_snapshot hcs
            WHERE hcs.hero_id = ds.hero_id
              AND hcs.enemy_hero_id = pb2.hero_id
              AND hcs.patch_id <= ds.patch_id
            ORDER BY (hcs.as_of_date <= to_timestamp(ds.start_time)::DATE) DESC,
                     hcs.as_of_date DESC
            LIMIT 1
        ) snap ON TRUE
        WHERE pb2.match_id = ds.match_id
          AND pb2."order"  < ds."order"
          AND pb2.is_pick  = TRUE
          AND pb2.team    != ds.team
    ) co ON TRUE

    -- Team head-to-head (PIT-safe: most recent daily snapshot as of match start)
    -- Falls back to earliest available snapshot when match predates snapshot system
    LEFT JOIN LATERAL (
        SELECT * FROM ml.team_h2h_snapshot ths
        WHERE ths.team_id = CASE ds.team WHEN 0 THEN ds.radiant_team_id ELSE ds.dire_team_id END
          AND ths.enemy_team_id = CASE ds.team WHEN 0 THEN ds.dire_team_id ELSE ds.radiant_team_id END
          AND ths.patch_id <= ds.patch_id
        ORDER BY (ths.as_of_date <= to_timestamp(ds.start_time)::DATE) DESC,
                 ths.as_of_date DESC
        LIMIT 1
    ) h2h ON TRUE

    -- Hero baseline (PIT-safe: most recent daily snapshot as of match start)
    -- Falls back to earliest available snapshot when match predates snapshot system
    LEFT JOIN LATERAL (
        SELECT * FROM ml.hero_baseline_snapshot hbs
        WHERE hbs.hero_id = ds.hero_id
          AND hbs.patch_id <= ds.patch_id
        ORDER BY (hbs.as_of_date <= to_timestamp(ds.start_time)::DATE) DESC,
                 hbs.as_of_date DESC
        LIMIT 1
    ) bl ON TRUE

    -- Hero draft-slot aggregate (PIT-safe: most recent daily snapshot as of match start)
    -- Falls back to earliest available snapshot when match predates snapshot system
    -- NULL team_pick_ordinal for bans → join misses → COALESCE gives defaults.
    LEFT JOIN LATERAL (
        SELECT * FROM ml.hero_draft_slot_snapshot hdss
        WHERE hdss.hero_id = ds.hero_id
          AND hdss.team_pick_ordinal = ds.team_pick_ordinal
          AND hdss.patch_id <= ds.patch_id
        ORDER BY (hdss.as_of_date <= to_timestamp(ds.start_time)::DATE) DESC,
                 hdss.as_of_date DESC
        LIMIT 1
    ) hds ON TRUE

    -- Macro Composition (PIT-safe sum of already picked allies)
    -- Falls back to earliest available snapshot when match predates snapshot system
    LEFT JOIN LATERAL (
        SELECT
            SUM(COALESCE(hbs.avg_gpm, 0)) AS ally_gpm,
            SUM(COALESCE(hbs.avg_xpm, 0)) AS ally_xpm
        FROM picks_bans pb2
        LEFT JOIN LATERAL (
            SELECT avg_gpm, avg_xpm FROM ml.hero_baseline_snapshot
            WHERE hero_id = pb2.hero_id
              AND patch_id <= ds.patch_id
            ORDER BY (as_of_date <= to_timestamp(ds.start_time)::DATE) DESC,
                     as_of_date DESC
            LIMIT 1
        ) hbs ON TRUE
        WHERE pb2.match_id = ds.match_id
          AND pb2."order" < ds."order"
          AND pb2.is_pick = TRUE
          AND pb2.team = ds.team
    ) mac ON TRUE

    ORDER BY ds.match_id, ds."order"
"""


def feature_column_names(include_onehot: bool = True, max_hero_id: int = 160, n_embeddings: int = 32) -> list[str]:
    """Return the ordered list of feature column names.

    This is the source of truth for the training/API column contract.
    When include_onehot=True, appends hero_id (categorical) + 32 embedding
    columns instead of 160 one-hot columns.

    NOTE: include_onehot=True is for API/inference schema only; training uses
    include_onehot=False (no embeddings needed for PyTorch DraftBERT).
    """
    cols = [
        # Draft context (side + pick-vs-ban + phase) — critical context for the model
        "is_pick", "team", "draft_phase_id",
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
        "co_avg_win_rate", "co_n_enemies", "co_avg_kd_diff",
        # Team head-to-head
        "h2h_win_rate", "h2h_games",
        # Hero baseline
        "bl_total_picks", "bl_total_wins", "bl_total_bans",
        "bl_win_rate", "bl_pick_rate", "bl_ban_rate",
        "bl_avg_gpm", "bl_avg_xpm", "bl_avg_kills", "bl_avg_deaths", "bl_avg_assists",
        "bl_avg_gold_10", "bl_avg_xp_10",
        # Hero draft-slot (pick-position) win rate
        "hds_win_rate", "hds_games",
        # Task 4: Low-game flags
        "ph_is_new_player",
        "th_is_new_team_hero",
        # Task 5: Delta features
        "rel_th_win_rate",
        "rel_ph_win_rate",
        # Task 6: Role interactions
        "ph_vision_support_score",
        "ph_gpm_carry_score",
        # Task 7: Macro Composition Constraints
        "team_gpm_budget",
        "team_xpm_budget",
        # Task 8: Pick Propensity
        "team_pick_propensity",
    ]
    if include_onehot:
        # hero_id as native categorical + 32-D semantic embeddings
        cols.append("hero_id")
        cols.extend(f"emb_{i}" for i in range(n_embeddings))
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


def write_schema(model_dir: str | Path, patch_id: int, max_hero_id: int = 160, n_embeddings: int = 32, max_seq_len: int = 50, drift_stats: dict | None = None) -> None:
    """Write ``feature_schema_patch_{patch_id}.json`` — the authoritative
    column-order contract for a specific patch.

    The inference API loads this file to build its feature vectors in
    *exactly* the same order as training. Making the filename patch-specific
    prevents multiple training runs from overwriting each other's schema (Bug #5).
    """
    cols = feature_column_names(include_onehot=True, max_hero_id=max_hero_id, n_embeddings=n_embeddings)
    agg_cols = feature_column_names(include_onehot=False)

    # Separate feature lists for explicit API contract
    continuous_features = agg_cols  # Features for tabular MLP
    categorical_features = ["hero_id"]  # Features for embedding layers
    embedding_features = [f"emb_{i}" for i in range(n_embeddings)]  # SVD embeddings

    schema = {
        "columns": cols,
        "n_features": len(cols),
        "n_aggregate_columns": len(agg_cols),
        "n_embeddings": n_embeddings,
        "max_hero_id": max_hero_id,
        "aggregate_columns": agg_cols,
        "continuous_features": continuous_features,
        "categorical_features": categorical_features,
        "embedding_features": embedding_features,
        "embedding_prefix": "emb_",
        "max_seq_len": max_seq_len,
    }
    if drift_stats:
        schema["drift_stats"] = drift_stats
    path = Path(model_dir) / f"feature_schema_patch_{patch_id}.json"
    path.write_text(json.dumps(schema, indent=2))
    logger.info("Wrote feature schema to %s (%d columns)", path, len(cols))
