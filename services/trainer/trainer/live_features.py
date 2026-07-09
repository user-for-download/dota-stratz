"""Feature extraction for LiveDraftBERT training.

Extracts per-minute dynamic features capturing the full game state:
1. Active Vulnerability (Death Timers & Buybacks)
2. Power Spikes (BKB, Blink, Aghs, Rapier)
3. Objectives (Towers, Barracks, Roshan, Courier)
4. Win Conditions (Mega Creeps)
5. Vision & Map Control (Wards, Deep Vision)
6. Economy Distribution & Laning (Carry/Support NW, Scaling Threats)
7. Teamfight Execution (CC Effectiveness, Cohesion)
8. Neutral Item Control
9. Map Pressure (Tower Damage)

45 dynamic features that capture actual game state, not just gold.
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
    "minute", "minute_sq",
    "radiant_dead_now", "dire_dead_now", "buyback_diff",
    "bkb_diff", "blink_diff", "aghs_diff", "rapier_diff",
    "mega_creeps_radiant", "mega_creeps_dire", "courier_lost_diff", "aegis_diff",
    "rad_carry_nw_pct", "dire_carry_nw_pct", "carry_farm_diff", "support_nw_diff",
    "radiant_cs_adv",
    "save_item_diff", "aura_item_diff",
    "dewards_diff", "deep_ward_diff",
    "rune_control_diff",
    "tf_gold_swing_1m", "tf_xp_swing_1m",
    "map_confinement_diff", "scaling_threat_diff",
    "cc_effectiveness_diff", "neutral_tier_diff", "tower_damage_diff",
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
player_extras AS (
    SELECT p.match_id, p.player_slot,
           p.stuns, p.teamfight_participation, p.tower_damage, p.total_gold,
           COALESCE(pb.buff_stacks, 0) AS buff_stacks,
           COALESCE(ni.neutrals, 0) AS neutrals
    FROM players p
    LEFT JOIN (
        SELECT match_id, player_slot, SUM(stack_count) AS buff_stacks
        FROM player_permanent_buffs
        WHERE match_id IN (SELECT DISTINCT match_id FROM match_minutes)
        GROUP BY match_id, player_slot
    ) pb ON pb.match_id = p.match_id AND pb.player_slot = p.player_slot
    LEFT JOIN (
        SELECT match_id, player_slot, COUNT(*) AS neutrals
        FROM player_neutral_item_history
        WHERE match_id IN (SELECT DISTINCT match_id FROM match_minutes)
        GROUP BY match_id, player_slot
    ) ni ON ni.match_id = p.match_id AND ni.player_slot = p.player_slot
    WHERE p.match_id IN (SELECT DISTINCT match_id FROM match_minutes)
),
team_extras AS (
    SELECT match_id, 0 AS minute,
           SUM(CASE WHEN player_slot < 128 THEN stuns ELSE 0 END) AS r_stuns,
           SUM(CASE WHEN player_slot >= 128 THEN stuns ELSE 0 END) AS d_stuns,
           AVG(CASE WHEN player_slot < 128 THEN teamfight_participation END) AS r_tf_part,
           AVG(CASE WHEN player_slot >= 128 THEN teamfight_participation END) AS d_tf_part,
           SUM(CASE WHEN player_slot < 128 THEN tower_damage ELSE 0 END) AS r_td,
           SUM(CASE WHEN player_slot >= 128 THEN tower_damage ELSE 0 END) AS d_td,
           SUM(CASE WHEN player_slot < 128 THEN total_gold ELSE 0 END) AS r_gold,
           SUM(CASE WHEN player_slot >= 128 THEN total_gold ELSE 0 END) AS d_gold,
           SUM(CASE WHEN player_slot < 128 THEN buff_stacks ELSE 0 END) AS r_buffs,
           SUM(CASE WHEN player_slot >= 128 THEN buff_stacks ELSE 0 END) AS d_buffs,
           SUM(CASE WHEN player_slot < 128 THEN neutrals ELSE 0 END) AS r_neutr,
           SUM(CASE WHEN player_slot >= 128 THEN neutrals ELSE 0 END) AS d_neutr
    FROM player_extras GROUP BY match_id
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
    COALESCE(te.r_stuns, 0) AS radiant_stuns,
    COALESCE(te.d_stuns, 0) AS dire_stuns,
    COALESCE(te.r_tf_part, 0) AS radiant_tf_part,
    COALESCE(te.d_tf_part, 0) AS dire_tf_part,
    COALESCE(te.r_td, 0) AS radiant_td,
    COALESCE(te.d_td, 0) AS dire_td,
    COALESCE(te.r_gold, 0) AS radiant_gold_total,
    COALESCE(te.d_gold, 0) AS dire_gold_total,
    COALESCE(te.r_buffs, 0) AS radiant_buffs,
    COALESCE(te.d_buffs, 0) AS dire_buffs,
    COALESCE(te.r_neutr, 0) AS radiant_neutr,
    COALESCE(te.d_neutr, 0) AS dire_neutr
FROM match_minutes mm
LEFT JOIN gold_xp gx ON gx.match_id = mm.match_id AND gx.minute = mm.minute
LEFT JOIN kills k ON k.match_id = mm.match_id AND k.minute = mm.minute
LEFT JOIN objs o ON o.match_id = mm.match_id AND o.minute = mm.minute
LEFT JOIN wards w ON w.match_id = mm.match_id AND w.minute = mm.minute
LEFT JOIN deep_wards dw ON dw.match_id = mm.match_id AND dw.minute = mm.minute
LEFT JOIN tf ON tf.match_id = mm.match_id AND tf.minute = mm.minute
LEFT JOIN items i ON i.match_id = mm.match_id AND i.minute = mm.minute
LEFT JOIN bbs b ON b.match_id = mm.match_id AND b.minute = mm.minute
LEFT JOIN team_extras te ON te.match_id = mm.match_id
ORDER BY mm.match_id, mm.minute
"""


def extract_dynamic_features(engine, patch_id: int, lookback: int = 2) -> pd.DataFrame:
    logger.info("Extracting per-minute dynamic features for patch %s (lookback %d) ...", patch_id, lookback)
    df = pd.read_sql(EXTRACT_DYNAMIC_SQL, engine, params={"patch_id": patch_id, "lookback": lookback})
    if df.empty:
        raise ValueError(f"No data found for patch {patch_id}")
    df = df.sort_values(["match_id", "minute"]).reset_index(drop=True)

    # Aegis (5-min rolling window)
    df["radiant_aegis"] = df.groupby("match_id")["radiant_rosh_tick"].rolling(5, min_periods=1).sum().reset_index(0, drop=True)
    df["dire_aegis"] = df.groupby("match_id")["dire_rosh_tick"].rolling(5, min_periods=1).sum().reset_index(0, drop=True)
    df["aegis_diff"] = (df["radiant_aegis"] > 0).astype(float) - (df["dire_aegis"] > 0).astype(float)

    # Cumulative ticks
    tick_cols = [c for c in df.columns if c.endswith("_tick")]
    cum_cols = [c.replace("_tick", "") for c in tick_cols]
    df[cum_cols] = df.groupby("match_id")[tick_cols].cumsum()

    # Mega Creeps
    df["mega_creeps_radiant"] = ((df["dire_melee"] + df["dire_range"]) >= 6).astype(float)
    df["mega_creeps_dire"] = ((df["radiant_melee"] + df["radiant_range"]) >= 6).astype(float)

    # Dead heroes (rolling window)
    rk = df.groupby("match_id")["radiant_kills_tick"]
    dk = df.groupby("match_id")["dire_kills_tick"]
    df["radiant_dead_now"] = (
        dk.rolling(1, min_periods=1).sum().reset_index(0, drop=True) * (df["minute"] < 20).astype(float) +
        dk.rolling(2, min_periods=1).sum().reset_index(0, drop=True) * ((df["minute"] >= 20) & (df["minute"] < 40)).astype(float) +
        dk.rolling(3, min_periods=1).sum().reset_index(0, drop=True) * (df["minute"] >= 40).astype(float)
    )
    df["dire_dead_now"] = (
        rk.rolling(1, min_periods=1).sum().reset_index(0, drop=True) * (df["minute"] < 20).astype(float) +
        rk.rolling(2, min_periods=1).sum().reset_index(0, drop=True) * ((df["minute"] >= 20) & (df["minute"] < 40)).astype(float) +
        rk.rolling(3, min_periods=1).sum().reset_index(0, drop=True) * (df["minute"] >= 40).astype(float)
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
    df["tf_diff"] = df["radiant_tf_wins"] - df["dire_tf_wins"]

    # Momentum
    for col in ["radiant_gold_adv", "radiant_xp_adv"]:
        df[f"{col}_prev1"] = df.groupby("match_id")[col].shift(1).fillna(0)
        df[f"{col}_prev3"] = df.groupby("match_id")[col].shift(3).fillna(0)
    df["gold_adv_diff_1m"] = df["radiant_gold_adv"] - df["radiant_gold_adv_prev1"]
    df["xp_adv_diff_1m"] = df["radiant_xp_adv"] - df["radiant_xp_adv_prev1"]
    df["gold_adv_diff_3m"] = df["radiant_gold_adv"] - df["radiant_gold_adv_prev3"]
    df["xp_adv_diff_3m"] = df["radiant_xp_adv"] - df["radiant_xp_adv_prev3"]
    df["minute_sq"] = df["minute"] ** 2

    # Economy (placeholders for gold_t-based features)
    df["rad_carry_nw_pct"] = 0.2
    df["dire_carry_nw_pct"] = 0.2
    df["carry_farm_diff"] = 0.0
    df["support_nw_diff"] = df["radiant_gold_total"] - df["dire_gold_total"]
    df["radiant_cs_adv"] = 0.0
    df["save_item_diff"] = 0.0
    df["aura_item_diff"] = 0.0
    df["dewards_diff"] = 0.0
    df["rune_control_diff"] = 0.0
    df["tf_gold_swing_1m"] = 0.0
    df["tf_xp_swing_1m"] = 0.0

    # New features from player_extras
    df["map_confinement_diff"] = 0.0  # Placeholder (complex SQL needed for deaths_pos)
    df["scaling_threat_diff"] = df["radiant_buffs"] - df["dire_buffs"]
    r_cc = df["radiant_stuns"] * df["radiant_tf_part"].clip(lower=0.01)
    d_cc = df["dire_stuns"] * df["dire_tf_part"].clip(lower=0.01)
    df["cc_effectiveness_diff"] = r_cc - d_cc
    df["neutral_tier_diff"] = df["radiant_neutr"] - df["dire_neutr"]
    df["tower_damage_diff"] = df["radiant_td"] - df["dire_td"]

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(float)
    logger.info("Final dataset: %d rows, %d matches, %.1f%% radiant win rate",
                len(df), df["match_id"].nunique(), df[TARGET_COLUMN].mean() * 100)
    return df
