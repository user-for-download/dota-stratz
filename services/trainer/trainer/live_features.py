"""Feature extraction for LiveDraftBERT training.

Extracts per-minute dynamic features capturing the 5 true pillars of live Dota 2:
1. Active Vulnerability (Death Timers & Buybacks)
2. Power Spikes (BKB, Blink, Aghs, Rapier)
3. Objectives (Towers, Barracks, Roshan, Courier)
4. Win Conditions (Mega Creeps)
5. Vision & Map Control (Wards)

26 dynamic features that capture actual game state, not just gold.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dynamic feature columns (26 features)
# ---------------------------------------------------------------------------

DYNAMIC_FEATURE_COLUMNS = [
    # Core advantages
    "radiant_gold_adv",
    "radiant_xp_adv",
    # Objectives
    "tower_diff",
    "roshan_diff",
    "ward_diff",
    "tf_diff",
    # Momentum
    "gold_adv_diff_1m",
    "xp_adv_diff_1m",
    "gold_adv_diff_3m",
    "xp_adv_diff_3m",
    # Time context
    "minute",
    "minute_sq",
    # Active Vulnerability (Death Timers & Buybacks)
    "radiant_dead_now",
    "dire_dead_now",
    "buyback_diff",
    # Power Spikes & Cheese
    "bkb_diff",
    "blink_diff",
    "aghs_diff",
    "rapier_diff",
    # Win Conditions
    "mega_creeps_radiant",
    "mega_creeps_dire",
    "courier_lost_diff",
    # Aegis (5-min window)
    "aegis_diff",
    # Barracks differential
    "barracks_diff",
]

TARGET_COLUMN = "radiant_win"

# ---------------------------------------------------------------------------
# SQL: Extract per-minute dynamic features
# ---------------------------------------------------------------------------

EXTRACT_DYNAMIC_SQL = """
WITH match_minutes AS (
    SELECT m.match_id, m.duration, m.radiant_win,
           generate_series(0, m.duration / 60) AS minute
    FROM matches m
    WHERE m.patch >= %(patch_id)s - 2 AND m.patch <= %(patch_id)s
      AND m.radiant_win IS NOT NULL
      AND m.duration >= 600
),
gold_xp AS (
    SELECT g.match_id, g.minute,
           g.radiant_gold_adv,
           COALESCE(x.radiant_xp_adv, 0) AS radiant_xp_adv
    FROM match_gold_adv g
    LEFT JOIN match_xp_adv x ON x.match_id = g.match_id AND x.minute = g.minute
    WHERE g.match_id IN (SELECT match_id FROM match_minutes)
),
kills AS (
    SELECT match_id, (time / 60) AS minute,
           SUM(CASE WHEN player_slot < 128 THEN 1 ELSE 0 END) AS r_kills,
           SUM(CASE WHEN player_slot >= 128 THEN 1 ELSE 0 END) AS d_kills
    FROM player_kills_log
    WHERE match_id IN (SELECT DISTINCT match_id FROM match_minutes)
    GROUP BY match_id, (time / 60)
),
objs AS (
    SELECT match_id, (time / 60) AS minute,
           SUM(CASE WHEN team = 0 AND type = 'tower_kill' THEN 1 ELSE 0 END) AS r_towers,
           SUM(CASE WHEN team = 1 AND type = 'tower_kill' THEN 1 ELSE 0 END) AS d_towers,
           SUM(CASE WHEN team = 0 AND type = 'barracks_kill' THEN 1 ELSE 0 END) AS r_barracks,
           SUM(CASE WHEN team = 1 AND type = 'barracks_kill' THEN 1 ELSE 0 END) AS d_barracks,
           SUM(CASE WHEN type = 'roshan_kill' AND team = 0 THEN 1 ELSE 0 END) AS r_rosh,
           SUM(CASE WHEN type = 'roshan_kill' AND team = 1 THEN 1 ELSE 0 END) AS d_rosh,
           SUM(CASE WHEN type = 'CHAT_MESSAGE_COURIER_LOST' AND team = 2 THEN 1 ELSE 0 END) AS r_couriers_lost,
           SUM(CASE WHEN type = 'CHAT_MESSAGE_COURIER_LOST' AND team = 3 THEN 1 ELSE 0 END) AS d_couriers_lost
    FROM objectives
    WHERE match_id IN (SELECT DISTINCT match_id FROM match_minutes)
    GROUP BY match_id, (time / 60)
),
wards AS (
    SELECT match_id, (time / 60) AS minute,
           SUM(CASE WHEN player_slot < 128 THEN 1 ELSE 0 END) AS r_obs,
           SUM(CASE WHEN player_slot >= 128 THEN 1 ELSE 0 END) AS d_obs
    FROM player_obs_log
    WHERE match_id IN (SELECT DISTINCT match_id FROM match_minutes)
    GROUP BY match_id, (time / 60)
),
tf AS (
    SELECT tf2.match_id, (tf2.start_time / 60) AS minute,
           SUM(CASE WHEN tp.gold_delta > 0 THEN 1 ELSE 0 END) AS r_wins,
           SUM(CASE WHEN tp.gold_delta < 0 THEN 1 ELSE 0 END) AS d_wins
    FROM teamfights tf2
    JOIN teamfight_players tp ON tf2.match_id = tp.match_id AND tf2.start_time = tp.start_time
    WHERE tf2.match_id IN (SELECT DISTINCT match_id FROM match_minutes)
      AND tp.player_slot < 128
    GROUP BY tf2.match_id, (tf2.start_time / 60)
),
items AS (
    SELECT match_id, (time / 60) AS minute,
           SUM(CASE WHEN player_slot < 128 AND key = 'black_king_bar' THEN 1 ELSE 0 END) AS r_bkb,
           SUM(CASE WHEN player_slot >= 128 AND key = 'black_king_bar' THEN 1 ELSE 0 END) AS d_bkb,
           SUM(CASE WHEN player_slot < 128 AND key = 'blink' THEN 1 ELSE 0 END) AS r_blink,
           SUM(CASE WHEN player_slot >= 128 AND key = 'blink' THEN 1 ELSE 0 END) AS d_blink,
           SUM(CASE WHEN player_slot < 128 AND key IN ('ultimate_scepter', 'aghanims_shard') THEN 1 ELSE 0 END) AS r_aghs,
           SUM(CASE WHEN player_slot >= 128 AND key IN ('ultimate_scepter', 'aghanims_shard') THEN 1 ELSE 0 END) AS d_aghs,
           SUM(CASE WHEN player_slot < 128 AND key = 'rapier' THEN 1 ELSE 0 END) AS r_rapier,
           SUM(CASE WHEN player_slot >= 128 AND key = 'rapier' THEN 1 ELSE 0 END) AS d_rapier
    FROM player_purchase_log
    WHERE match_id IN (SELECT DISTINCT match_id FROM match_minutes)
      AND key IN ('black_king_bar', 'blink', 'ultimate_scepter', 'aghanims_shard', 'rapier')
    GROUP BY match_id, (time / 60)
),
bbs AS (
    SELECT match_id, (time / 60) AS minute,
           SUM(CASE WHEN player_slot < 128 THEN 1 ELSE 0 END) AS r_buybacks,
           SUM(CASE WHEN player_slot >= 128 THEN 1 ELSE 0 END) AS d_buybacks
    FROM player_buyback_log
    WHERE match_id IN (SELECT DISTINCT match_id FROM match_minutes)
    GROUP BY match_id, (time / 60)
)
SELECT
    mm.match_id, mm.minute, mm.radiant_win,
    COALESCE(gx.radiant_gold_adv, 0) AS radiant_gold_adv,
    COALESCE(gx.radiant_xp_adv, 0) AS radiant_xp_adv,
    COALESCE(k.r_kills, 0) AS radiant_kills_tick,
    COALESCE(k.d_kills, 0) AS dire_kills_tick,
    COALESCE(o.r_towers, 0) AS radiant_towers_tick,
    COALESCE(o.d_towers, 0) AS dire_towers_tick,
    COALESCE(o.r_barracks, 0) AS radiant_barracks_tick,
    COALESCE(o.d_barracks, 0) AS dire_barracks_tick,
    COALESCE(o.r_rosh, 0) AS radiant_rosh_tick,
    COALESCE(o.d_rosh, 0) AS dire_rosh_tick,
    COALESCE(o.r_couriers_lost, 0) AS radiant_couriers_lost_tick,
    COALESCE(o.d_couriers_lost, 0) AS dire_couriers_lost_tick,
    COALESCE(w.r_obs, 0) AS radiant_obs_tick,
    COALESCE(w.d_obs, 0) AS dire_obs_tick,
    COALESCE(tf.r_wins, 0) AS radiant_tf_wins_tick,
    COALESCE(tf.d_wins, 0) AS dire_tf_wins_tick,
    COALESCE(i.r_bkb, 0) AS radiant_bkb_tick,
    COALESCE(i.d_bkb, 0) AS dire_bkb_tick,
    COALESCE(i.r_blink, 0) AS radiant_blink_tick,
    COALESCE(i.d_blink, 0) AS dire_blink_tick,
    COALESCE(i.r_aghs, 0) AS radiant_aghs_tick,
    COALESCE(i.d_aghs, 0) AS dire_aghs_tick,
    COALESCE(i.r_rapier, 0) AS radiant_rapier_tick,
    COALESCE(i.d_rapier, 0) AS dire_rapier_tick,
    COALESCE(b.r_buybacks, 0) AS radiant_buybacks_tick,
    COALESCE(b.d_buybacks, 0) AS dire_buybacks_tick
FROM match_minutes mm
LEFT JOIN gold_xp gx ON gx.match_id = mm.match_id AND gx.minute = mm.minute
LEFT JOIN kills k ON k.match_id = mm.match_id AND k.minute = mm.minute
LEFT JOIN objs o ON o.match_id = mm.match_id AND o.minute = mm.minute
LEFT JOIN wards w ON w.match_id = mm.match_id AND w.minute = mm.minute
LEFT JOIN tf ON tf.match_id = mm.match_id AND tf.minute = mm.minute
LEFT JOIN items i ON i.match_id = mm.match_id AND i.minute = mm.minute
LEFT JOIN bbs b ON b.match_id = mm.match_id AND b.minute = mm.minute
ORDER BY mm.match_id, mm.minute
"""


def extract_dynamic_features(engine, patch_id: int) -> pd.DataFrame:
    """Extract per-minute dynamic features capturing true game state."""
    logger.info("Extracting per-minute dynamic features for patch %s (lookback 2) ...", patch_id)

    df = pd.read_sql(EXTRACT_DYNAMIC_SQL, engine, params={"patch_id": patch_id})

    if df.empty:
        raise ValueError(f"No data found for patch {patch_id}")

    df = df.sort_values(["match_id", "minute"]).reset_index(drop=True)

    # --- Aegis Active State (5-minute rolling window) ---
    df["radiant_aegis"] = df.groupby("match_id")["radiant_rosh_tick"].rolling(5, min_periods=1).sum().reset_index(0, drop=True)
    df["dire_aegis"] = df.groupby("match_id")["dire_rosh_tick"].rolling(5, min_periods=1).sum().reset_index(0, drop=True)
    df["aegis_diff"] = (df["radiant_aegis"] > 0).astype(float) - (df["dire_aegis"] > 0).astype(float)

    # --- Cumulative states from tick events ---
    tick_cols = [c for c in df.columns if c.endswith("_tick")]
    for col in tick_cols:
        cum_col = col.replace("_tick", "")
        df[cum_col] = df.groupby("match_id")[col].cumsum()

    # --- Win Conditions: Mega Creeps ---
    df["mega_creeps_radiant"] = (df["dire_barracks"] >= 6).astype(float)
    df["mega_creeps_dire"] = (df["radiant_barracks"] >= 6).astype(float)

    # --- Active Vulnerability: Dead Heroes Now ---
    # Death timers scale with game time:
    #   Early (0-20): dead ~1 min
    #   Mid (20-40): dead ~2 min
    #   Late (40+): dead ~3 min
    # NOTE: r_kills = kills BY Radiant = Dire heroes dead
    #       d_kills = kills BY Dire   = Radiant heroes dead
    r_kills = df.groupby("match_id")["radiant_kills_tick"]
    d_kills = df.groupby("match_id")["dire_kills_tick"]

    # Dead RADIANT heroes (Dire got the kill)
    r_dead_early = d_kills.rolling(1, min_periods=1).sum().reset_index(0, drop=True) * (df["minute"] < 20).astype(float)
    r_dead_mid = d_kills.rolling(2, min_periods=1).sum().reset_index(0, drop=True) * ((df["minute"] >= 20) & (df["minute"] < 40)).astype(float)
    r_dead_late = d_kills.rolling(3, min_periods=1).sum().reset_index(0, drop=True) * (df["minute"] >= 40).astype(float)
    df["radiant_dead_now"] = r_dead_early + r_dead_mid + r_dead_late

    # Dead DIRE heroes (Radiant got the kill)
    d_dead_early = r_kills.rolling(1, min_periods=1).sum().reset_index(0, drop=True) * (df["minute"] < 20).astype(float)
    d_dead_mid = r_kills.rolling(2, min_periods=1).sum().reset_index(0, drop=True) * ((df["minute"] >= 20) & (df["minute"] < 40)).astype(float)
    d_dead_late = r_kills.rolling(3, min_periods=1).sum().reset_index(0, drop=True) * (df["minute"] >= 40).astype(float)
    df["dire_dead_now"] = d_dead_early + d_dead_mid + d_dead_late

    # --- Power Spike Differentials ---
    df["bkb_diff"] = df["radiant_bkb"] - df["dire_bkb"]
    df["blink_diff"] = df["radiant_blink"] - df["dire_blink"]
    df["aghs_diff"] = df["radiant_aghs"] - df["dire_aghs"]
    df["rapier_diff"] = df["radiant_rapier"] - df["dire_rapier"]

    # --- Buyback & Courier ---
    df["buyback_diff"] = df["dire_buybacks"] - df["radiant_buybacks"]
    df["courier_lost_diff"] = df["dire_couriers_lost"] - df["radiant_couriers_lost"]

    # --- Objectives Differentials ---
    df["tower_diff"] = df["radiant_towers"] - df["dire_towers"]
    df["barracks_diff"] = df["radiant_barracks"] - df["dire_barracks"]
    df["roshan_diff"] = df["radiant_rosh"] - df["dire_rosh"]
    df["ward_diff"] = df["radiant_obs"] - df["dire_obs"]
    df["tf_diff"] = df["radiant_tf_wins"] - df["dire_tf_wins"]

    # --- Momentum (Gold/XP swings) ---
    for col in ["radiant_gold_adv", "radiant_xp_adv"]:
        df[f"{col}_prev1"] = df.groupby("match_id")[col].shift(1).fillna(0)
        df[f"{col}_prev3"] = df.groupby("match_id")[col].shift(3).fillna(0)

    df["gold_adv_diff_1m"] = df["radiant_gold_adv"] - df["radiant_gold_adv_prev1"]
    df["xp_adv_diff_1m"] = df["radiant_xp_adv"] - df["radiant_xp_adv_prev1"]
    df["gold_adv_diff_3m"] = df["radiant_gold_adv"] - df["radiant_gold_adv_prev3"]
    df["xp_adv_diff_3m"] = df["radiant_xp_adv"] - df["radiant_xp_adv_prev3"]

    df["minute_sq"] = df["minute"] ** 2
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(float)

    logger.info("Final dataset: %d rows, %d matches, %.1f%% radiant win rate",
                len(df), df["match_id"].nunique(), df[TARGET_COLUMN].mean() * 100)

    return df
