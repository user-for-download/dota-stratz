"""Feature extraction for LiveDraftBERT training.

Extracts per-minute dynamic features capturing the full game state:
1. Core Advantages (Gold, XP)
2. Objectives (Towers T1-T4, Melee/Rax, Roshan, Ward, Teamfight)
3. Momentum (Gold/XP swings 1m, 3m)
4. Time Context (minute, minute_sin, minute_cos, day_night_sin)
5. Active Vulnerability (Death Timers & Buybacks)
6. Power Spikes (BKB, Blink, Aghs, Rapier)
7. Win Conditions (Mega Creeps)
8. Vision & Map Control (Deep Wards, Courier)
9. Aegis Control
10. Neutral Item Control

32 dynamic features (DYNAMIC_FEATURE_COLUMNS is source of truth).
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DYNAMIC_FEATURE_COLUMNS = [
    "radiant_gold_adv", "radiant_xp_adv",
    "t1_tower_diff", "t2_tower_diff", "t3_tower_diff", "t4_tower_diff",
    "melee_rax_diff", "range_rax_diff", "roshan_diff", "ward_diff", "tf_diff",
    "gold_adv_diff_1m", "xp_adv_diff_1m", "gold_adv_diff_3m", "xp_adv_diff_3m",
    "minute", "minute_sin", "minute_cos", "day_night_sin",
    "radiant_dead_now", "dire_dead_now", "buyback_diff",
    "bkb_diff", "blink_diff", "aghs_diff", "rapier_diff",
    "mega_creeps_radiant", "mega_creeps_dire", "courier_lost_diff", "aegis_diff",
    "deep_ward_diff",
    "neutral_tier_diff",
]

TARGET_COLUMN = "radiant_win"

EXTRACT_DYNAMIC_SQL = """
WITH match_minutes AS (
    SELECT m.match_id, m.duration, m.radiant_win,
           generate_series(0, m.duration / 60) AS minute
    FROM matches m
    WHERE m.patch >= %(patch_id)s - %(lookback)s AND m.patch <= %(patch_id)s
      AND m.radiant_win IS NOT NULL AND m.duration >= 600
),
gold_xp AS (
    SELECT g.match_id, g.minute, g.radiant_gold_adv,
           COALESCE(x.radiant_xp_adv, 0) AS radiant_xp_adv
    FROM match_gold_adv g
    LEFT JOIN match_xp_adv x ON x.match_id = g.match_id AND x.minute = g.minute
    WHERE g.match_id IN (SELECT match_id FROM match_minutes)
),
kills AS (
    SELECT match_id, minute,
           SUM(CASE WHEN player_slot < 128 THEN 1 ELSE 0 END) AS r_kills,
           SUM(CASE WHEN player_slot >= 128 THEN 1 ELSE 0 END) AS d_kills
    FROM player_kills_log
    WHERE match_id IN (SELECT DISTINCT match_id FROM match_minutes)
    GROUP BY match_id, minute
),
objs AS (
    SELECT match_id, minute,
           SUM(CASE WHEN team=0 AND type='tower_kill' AND COALESCE(key,'') LIKE '%%tower1%%' THEN 1 ELSE 0 END) AS r_t1,
           SUM(CASE WHEN team=1 AND type='tower_kill' AND COALESCE(key,'') LIKE '%%tower1%%' THEN 1 ELSE 0 END) AS d_t1,
           SUM(CASE WHEN team=0 AND type='tower_kill' AND COALESCE(key,'') LIKE '%%tower2%%' THEN 1 ELSE 0 END) AS r_t2,
           SUM(CASE WHEN team=1 AND type='tower_kill' AND COALESCE(key,'') LIKE '%%tower2%%' THEN 1 ELSE 0 END) AS d_t2,
           SUM(CASE WHEN team=0 AND type='tower_kill' AND COALESCE(key,'') LIKE '%%tower3%%' THEN 1 ELSE 0 END) AS r_t3,
           SUM(CASE WHEN team=1 AND type='tower_kill' AND COALESCE(key,'') LIKE '%%tower3%%' THEN 1 ELSE 0 END) AS d_t3,
           SUM(CASE WHEN team=0 AND type='tower_kill' AND COALESCE(key,'') LIKE '%%tower4%%' THEN 1 ELSE 0 END) AS r_t4,
           SUM(CASE WHEN team=1 AND type='tower_kill' AND COALESCE(key,'') LIKE '%%tower4%%' THEN 1 ELSE 0 END) AS d_t4,
           SUM(CASE WHEN team=0 AND type='barracks_kill' AND COALESCE(key,'') LIKE '%%melee%%' THEN 1 ELSE 0 END) AS r_melee,
           SUM(CASE WHEN team=1 AND type='barracks_kill' AND COALESCE(key,'') LIKE '%%melee%%' THEN 1 ELSE 0 END) AS d_melee,
           SUM(CASE WHEN team=0 AND type='barracks_kill' AND COALESCE(key,'') LIKE '%%range%%' THEN 1 ELSE 0 END) AS r_range,
           SUM(CASE WHEN team=1 AND type='barracks_kill' AND COALESCE(key,'') LIKE '%%range%%' THEN 1 ELSE 0 END) AS d_range,
           SUM(CASE WHEN type='roshan_kill' AND team=0 THEN 1 ELSE 0 END) AS r_rosh,
           SUM(CASE WHEN type='roshan_kill' AND team=1 THEN 1 ELSE 0 END) AS d_rosh,
           SUM(CASE WHEN type='CHAT_MESSAGE_COURIER_LOST' AND team=2 THEN 1 ELSE 0 END) AS r_couriers,
           SUM(CASE WHEN type='CHAT_MESSAGE_COURIER_LOST' AND team=3 THEN 1 ELSE 0 END) AS d_couriers
    FROM objectives
    WHERE match_id IN (SELECT DISTINCT match_id FROM match_minutes)
    GROUP BY match_id, minute
),
wards AS (
    SELECT match_id, minute,
           SUM(CASE WHEN player_slot < 128 THEN 1 ELSE 0 END) AS r_obs,
           SUM(CASE WHEN player_slot >= 128 THEN 1 ELSE 0 END) AS d_obs
    FROM player_obs_log
    WHERE match_id IN (SELECT DISTINCT match_id FROM match_minutes)
    GROUP BY match_id, minute
),
deep_wards AS (
    SELECT match_id, minute,
           SUM(CASE WHEN player_slot < 128 AND y > 128 THEN 1 ELSE 0 END) AS r_deep,
           SUM(CASE WHEN player_slot >= 128 AND y < 128 THEN 1 ELSE 0 END) AS d_deep
    FROM player_obs_log
    WHERE match_id IN (SELECT DISTINCT match_id FROM match_minutes)
    GROUP BY match_id, minute
),
tf AS (
    SELECT tf2.match_id, tf2.minute,
           SUM(CASE WHEN tp.gold_delta > 0 THEN 1 ELSE 0 END) AS r_wins,
           SUM(CASE WHEN tp.gold_delta < 0 THEN 1 ELSE 0 END) AS d_wins
    FROM teamfights tf2
    JOIN teamfight_players tp ON tf2.match_id = tp.match_id AND tf2.start_time = tp.start_time
    WHERE tf2.match_id IN (SELECT DISTINCT match_id FROM match_minutes) AND tp.player_slot < 128
    GROUP BY tf2.match_id, tf2.minute
),
items AS (
    SELECT match_id, minute,
           SUM(CASE WHEN player_slot < 128 AND key='black_king_bar' THEN 1 ELSE 0 END) AS r_bkb,
           SUM(CASE WHEN player_slot >= 128 AND key='black_king_bar' THEN 1 ELSE 0 END) AS d_bkb,
           SUM(CASE WHEN player_slot < 128 AND key='blink' THEN 1 ELSE 0 END) AS r_blink,
           SUM(CASE WHEN player_slot >= 128 AND key='blink' THEN 1 ELSE 0 END) AS d_blink,
           SUM(CASE WHEN player_slot < 128 AND key IN ('ultimate_scepter','aghanims_shard') THEN 1 ELSE 0 END) AS r_aghs,
           SUM(CASE WHEN player_slot >= 128 AND key IN ('ultimate_scepter','aghanims_shard') THEN 1 ELSE 0 END) AS d_aghs,
           SUM(CASE WHEN player_slot < 128 AND key='rapier' THEN 1 ELSE 0 END) AS r_rapier,
           SUM(CASE WHEN player_slot >= 128 AND key='rapier' THEN 1 ELSE 0 END) AS d_rapier
    FROM player_purchase_log
    WHERE match_id IN (SELECT DISTINCT match_id FROM match_minutes)
      AND key IN ('black_king_bar','blink','ultimate_scepter','aghanims_shard','rapier')
    GROUP BY match_id, minute
),
bbs AS (
    SELECT match_id, minute,
           SUM(CASE WHEN player_slot < 128 THEN 1 ELSE 0 END) AS r_buybacks,
           SUM(CASE WHEN player_slot >= 128 THEN 1 ELSE 0 END) AS d_buybacks
    FROM player_buyback_log
    WHERE match_id IN (SELECT DISTINCT match_id FROM match_minutes)
    GROUP BY match_id, minute
),
neutral_items AS (
    SELECT ni.match_id, ni.minute,
           COUNT(CASE WHEN ni.player_slot < 128 THEN 1 END) AS r_neutr,
           COUNT(CASE WHEN ni.player_slot >= 128 THEN 1 END) AS d_neutr
    FROM (
        SELECT match_id, player_slot, (time / 60) AS minute
        FROM player_neutral_item_history
        WHERE match_id IN (SELECT DISTINCT match_id FROM match_minutes)
    ) ni
    GROUP BY ni.match_id, ni.minute
)
SELECT
    mm.match_id, mm.minute, mm.radiant_win,
    COALESCE(gx.radiant_gold_adv, 0) AS radiant_gold_adv,
    COALESCE(gx.radiant_xp_adv, 0) AS radiant_xp_adv,
    COALESCE(k.r_kills, 0) AS radiant_kills_tick,
    COALESCE(k.d_kills, 0) AS dire_kills_tick,
    COALESCE(o.r_t1, 0) AS radiant_t1_tick, COALESCE(o.d_t1, 0) AS dire_t1_tick,
    COALESCE(o.r_t2, 0) AS radiant_t2_tick, COALESCE(o.d_t2, 0) AS dire_t2_tick,
    COALESCE(o.r_t3, 0) AS radiant_t3_tick, COALESCE(o.d_t3, 0) AS dire_t3_tick,
    COALESCE(o.r_t4, 0) AS radiant_t4_tick, COALESCE(o.d_t4, 0) AS dire_t4_tick,
    COALESCE(o.r_melee, 0) AS radiant_melee_tick, COALESCE(o.d_melee, 0) AS dire_melee_tick,
    COALESCE(o.r_range, 0) AS radiant_range_tick, COALESCE(o.d_range, 0) AS dire_range_tick,
    COALESCE(o.r_rosh, 0) AS radiant_rosh_tick, COALESCE(o.d_rosh, 0) AS dire_rosh_tick,
    COALESCE(o.r_couriers, 0) AS radiant_couriers_tick, COALESCE(o.d_couriers, 0) AS dire_couriers_tick,
    COALESCE(w.r_obs, 0) AS radiant_obs_tick, COALESCE(w.d_obs, 0) AS dire_obs_tick,
    COALESCE(dw.r_deep, 0) AS radiant_deep_tick, COALESCE(dw.d_deep, 0) AS dire_deep_tick,
    COALESCE(tf.r_wins, 0) AS radiant_tf_tick, COALESCE(tf.d_wins, 0) AS dire_tf_tick,
    COALESCE(i.r_bkb, 0) AS radiant_bkb_tick, COALESCE(i.d_bkb, 0) AS dire_bkb_tick,
    COALESCE(i.r_blink, 0) AS radiant_blink_tick, COALESCE(i.d_blink, 0) AS dire_blink_tick,
    COALESCE(i.r_aghs, 0) AS radiant_aghs_tick, COALESCE(i.d_aghs, 0) AS dire_aghs_tick,
    COALESCE(i.r_rapier, 0) AS radiant_rapier_tick, COALESCE(i.d_rapier, 0) AS dire_rapier_tick,
    COALESCE(b.r_buybacks, 0) AS radiant_buybacks_tick, COALESCE(b.d_buybacks, 0) AS dire_buybacks_tick,
    COALESCE(ni.r_neutr, 0) AS radiant_neutr_tick,
    COALESCE(ni.d_neutr, 0) AS dire_neutr_tick
FROM match_minutes mm
LEFT JOIN gold_xp gx ON gx.match_id = mm.match_id AND gx.minute = mm.minute
LEFT JOIN kills k ON k.match_id = mm.match_id AND k.minute = mm.minute
LEFT JOIN objs o ON o.match_id = mm.match_id AND o.minute = mm.minute
LEFT JOIN wards w ON w.match_id = mm.match_id AND w.minute = mm.minute
LEFT JOIN deep_wards dw ON dw.match_id = mm.match_id AND dw.minute = mm.minute
LEFT JOIN tf ON tf.match_id = mm.match_id AND tf.minute = mm.minute
LEFT JOIN items i ON i.match_id = mm.match_id AND i.minute = mm.minute
LEFT JOIN bbs b ON b.match_id = mm.match_id AND b.minute = mm.minute
LEFT JOIN neutral_items ni ON ni.match_id = mm.match_id AND ni.minute = mm.minute
ORDER BY mm.match_id, mm.minute
"""


def extract_dynamic_features(engine, patch_id: int, lookback: int = 2) -> pd.DataFrame:
    logger.info("Extracting per-minute dynamic features for patch %s (lookback %d) ...", patch_id, lookback)
    df = pd.read_sql(EXTRACT_DYNAMIC_SQL, engine, params={"patch_id": patch_id, "lookback": lookback})
    if df.empty:
        raise ValueError(f"No data found for patch {patch_id}")
    df = df.sort_values(["match_id", "minute"]).reset_index(drop=True)

    # Aegis (5-min rolling window) — using transform for memory efficiency
    df["radiant_aegis"] = df.groupby("match_id")["radiant_rosh_tick"].transform(
        lambda x: x.rolling(5, min_periods=1).sum()
    )
    df["dire_aegis"] = df.groupby("match_id")["dire_rosh_tick"].transform(
        lambda x: x.rolling(5, min_periods=1).sum()
    )
    df["aegis_diff"] = (df["radiant_aegis"] > 0).astype(float) - (df["dire_aegis"] > 0).astype(float)

    # Cumulative ticks
    tick_cols = [c for c in df.columns if c.endswith("_tick")]
    cum_cols = [c.replace("_tick", "") for c in tick_cols]
    df[cum_cols] = df.groupby("match_id")[tick_cols].cumsum()

    # Mega Creeps
    df["mega_creeps_radiant"] = ((df["dire_melee"] + df["dire_range"]) >= 6).astype(float)
    df["mega_creeps_dire"] = ((df["radiant_melee"] + df["radiant_range"]) >= 6).astype(float)

    # Dead heroes (rolling window, time-dependent death timer)
    # Using transform for memory efficiency — avoids intermediate MultiIndex
    minute_lt_20 = (df["minute"] < 20).astype(float)
    minute_20_40 = ((df["minute"] >= 20) & (df["minute"] < 40)).astype(float)
    minute_gte_40 = (df["minute"] >= 40).astype(float)

    def _rolling_dead(kills_col, window):
        return df.groupby("match_id")[kills_col].transform(
            lambda x: x.rolling(window, min_periods=1).sum()
        )

    df["radiant_dead_now"] = (
        _rolling_dead("dire_kills_tick", 1) * minute_lt_20 +
        _rolling_dead("dire_kills_tick", 2) * minute_20_40 +
        _rolling_dead("dire_kills_tick", 3) * minute_gte_40
    )
    df["dire_dead_now"] = (
        _rolling_dead("radiant_kills_tick", 1) * minute_lt_20 +
        _rolling_dead("radiant_kills_tick", 2) * minute_20_40 +
        _rolling_dead("radiant_kills_tick", 3) * minute_gte_40
    )

    # Differentials
    df["bkb_diff"] = df["radiant_bkb"] - df["dire_bkb"]
    df["blink_diff"] = df["radiant_blink"] - df["dire_blink"]
    df["aghs_diff"] = df["radiant_aghs"] - df["dire_aghs"]
    df["rapier_diff"] = df["radiant_rapier"] - df["dire_rapier"]
    df["buyback_diff"] = df["radiant_buybacks"] - df["dire_buybacks"]
    df["courier_lost_diff"] = df["radiant_couriers"] - df["dire_couriers"]
    df["t1_tower_diff"] = df["radiant_t1"] - df["dire_t1"]
    df["t2_tower_diff"] = df["radiant_t2"] - df["dire_t2"]
    df["t3_tower_diff"] = df["radiant_t3"] - df["dire_t3"]
    df["t4_tower_diff"] = df["radiant_t4"] - df["dire_t4"]
    df["melee_rax_diff"] = df["radiant_melee"] - df["dire_melee"]
    df["range_rax_diff"] = df["radiant_range"] - df["dire_range"]
    df["roshan_diff"] = df["radiant_rosh"] - df["dire_rosh"]
    df["ward_diff"] = df["radiant_obs"] - df["dire_obs"]
    df["deep_ward_diff"] = df["radiant_deep"] - df["dire_deep"]
    df["tf_diff"] = df["radiant_tf"] - df["dire_tf"]

    # Momentum
    for col in ["radiant_gold_adv", "radiant_xp_adv"]:
        df[f"{col}_prev1"] = df.groupby("match_id")[col].shift(1).fillna(0)
        df[f"{col}_prev3"] = df.groupby("match_id")[col].shift(3).fillna(0)
    df["gold_adv_diff_1m"] = df["radiant_gold_adv"] - df["radiant_gold_adv_prev1"]
    df["xp_adv_diff_1m"] = df["radiant_xp_adv"] - df["radiant_xp_adv_prev1"]
    df["gold_adv_diff_3m"] = df["radiant_gold_adv"] - df["radiant_gold_adv_prev3"]
    df["xp_adv_diff_3m"] = df["radiant_xp_adv"] - df["radiant_xp_adv_prev3"]
    # Cyclical time encoding (replaces minute_sq)
    # 5-min bounty rune cycle, 10-min day/night cycle
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 5.0)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 5.0)
    df["day_night_sin"] = np.sin(2 * np.pi * df["minute"] / 10.0)

    # Neutral items (real — time-filtered via player_neutral_item_history)
    df["neutral_tier_diff"] = df["radiant_neutr"] - df["dire_neutr"]

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(float)
    logger.info("Final dataset: %d rows, %d matches, %.1f%% radiant win rate",
                len(df), df["match_id"].nunique(), df[TARGET_COLUMN].mean() * 100)
    return df
