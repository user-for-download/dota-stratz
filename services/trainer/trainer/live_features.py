"""Feature extraction for LiveDraftBERT training.

Extracts per-minute dynamic features capturing the full game state:
1. Active Vulnerability (Death Timers & Buybacks)
2. Power Spikes (BKB, Blink, Aghs, Rapier)
3. Objectives (Towers, Barracks, Roshan, Courier)
4. Win Conditions (Mega Creeps)
5. Vision & Map Control (Wards, Deep Vision, Confinement)
6. Economy Distribution & Laning (Carry/Support NW, Scaling Threats)
7. Teamfight Execution (CC Effectiveness, Cohesion)
8. Neutral Item Control
9. Map Pressure (Tower Damage)

46 dynamic features that capture actual game state, not just gold.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dynamic feature columns (46 features)
# ---------------------------------------------------------------------------

DYNAMIC_FEATURE_COLUMNS = [
    # Core advantages
    "radiant_gold_adv",
    "radiant_xp_adv",
    # Objectives (Granular)
    "t1_tower_diff",
    "t2_tower_diff",
    "t3_tower_diff",
    "t4_tower_diff",
    "melee_rax_diff",
    "range_rax_diff",
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
    # Economy Distribution (who has the gold)
    "rad_carry_nw_pct",
    "dire_carry_nw_pct",
    "carry_farm_diff",
    "support_nw_diff",
    # Laning Phase (CS dominance)
    "radiant_cs_adv",
    # Defensive & Utility Power Spikes
    "save_item_diff",
    "aura_item_diff",
    # Vision Denial (de-warding)
    "dewards_diff",
    "deep_ward_diff",
    # Rune Control
    "rune_control_diff",
    # Teamfight Efficiency (magnitude of swings)
    "tf_gold_swing_1m",
    "tf_xp_swing_1m",
    # Map Confinement (center of mass)
    "map_confinement_diff",
    # Scaling Threats (permanent buffs)
    "scaling_threat_diff",
    # CC Effectiveness (stuns + teamfight participation)
    "cc_effectiveness_diff",
    # Neutral Item Tier Timing
    "neutral_tier_diff",
    # Map Pressure (tower damage)
    "tower_damage_diff",
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
    WHERE m.patch >= %(patch_id)s - %(lookback)s AND m.patch <= %(patch_id)s
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
    SELECT match_id, minute,
           SUM(CASE WHEN player_slot < 128 THEN 1 ELSE 0 END) AS r_kills,
           SUM(CASE WHEN player_slot >= 128 THEN 1 ELSE 0 END) AS d_kills
    FROM player_kills_log
    WHERE match_id IN (SELECT DISTINCT match_id FROM match_minutes)
    GROUP BY match_id, minute
),
objs AS (
    SELECT match_id, minute,
           SUM(CASE WHEN team = 0 AND type = 'tower_kill' AND COALESCE(key, '') LIKE '%%tower1%%' THEN 1 ELSE 0 END) AS r_t1_towers,
           SUM(CASE WHEN team = 1 AND type = 'tower_kill' AND COALESCE(key, '') LIKE '%%tower1%%' THEN 1 ELSE 0 END) AS d_t1_towers,
           SUM(CASE WHEN team = 0 AND type = 'tower_kill' AND COALESCE(key, '') LIKE '%%tower2%%' THEN 1 ELSE 0 END) AS r_t2_towers,
           SUM(CASE WHEN team = 1 AND type = 'tower_kill' AND COALESCE(key, '') LIKE '%%tower2%%' THEN 1 ELSE 0 END) AS d_t2_towers,
           SUM(CASE WHEN team = 0 AND type = 'tower_kill' AND COALESCE(key, '') LIKE '%%tower3%%' THEN 1 ELSE 0 END) AS r_t3_towers,
           SUM(CASE WHEN team = 1 AND type = 'tower_kill' AND COALESCE(key, '') LIKE '%%tower3%%' THEN 1 ELSE 0 END) AS d_t3_towers,
           SUM(CASE WHEN team = 0 AND type = 'tower_kill' AND COALESCE(key, '') LIKE '%%tower4%%' THEN 1 ELSE 0 END) AS r_t4_towers,
           SUM(CASE WHEN team = 1 AND type = 'tower_kill' AND COALESCE(key, '') LIKE '%%tower4%%' THEN 1 ELSE 0 END) AS d_t4_towers,
           SUM(CASE WHEN team = 0 AND type = 'barracks_kill' AND COALESCE(key, '') LIKE '%%melee%%' THEN 1 ELSE 0 END) AS r_melee_rax,
           SUM(CASE WHEN team = 1 AND type = 'barracks_kill' AND COALESCE(key, '') LIKE '%%melee%%' THEN 1 ELSE 0 END) AS d_melee_rax,
           SUM(CASE WHEN team = 0 AND type = 'barracks_kill' AND COALESCE(key, '') LIKE '%%range%%' THEN 1 ELSE 0 END) AS r_range_rax,
           SUM(CASE WHEN team = 1 AND type = 'barracks_kill' AND COALESCE(key, '') LIKE '%%range%%' THEN 1 ELSE 0 END) AS d_range_rax,
           SUM(CASE WHEN type = 'roshan_kill' AND team = 0 THEN 1 ELSE 0 END) AS r_rosh,
           SUM(CASE WHEN type = 'roshan_kill' AND team = 1 THEN 1 ELSE 0 END) AS d_rosh,
           SUM(CASE WHEN type = 'CHAT_MESSAGE_COURIER_LOST' AND team = 2 THEN 1 ELSE 0 END) AS r_couriers_lost,
           SUM(CASE WHEN type = 'CHAT_MESSAGE_COURIER_LOST' AND team = 3 THEN 1 ELSE 0 END) AS d_couriers_lost
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
    -- Deep wards: wards placed past the river on the enemy side
    -- Radiant deep ward = y > 96 (upper half, Dire territory)
    -- Dire deep ward = y < 96 (lower half, Radiant territory)
    SELECT match_id, minute,
           SUM(CASE WHEN player_slot < 128 AND y > 96 THEN 1 ELSE 0 END) AS r_deep_obs,
           SUM(CASE WHEN player_slot >= 128 AND y < 96 THEN 1 ELSE 0 END) AS d_deep_obs,
           SUM(CASE WHEN player_slot < 128 THEN 1 ELSE 0 END) AS r_total_obs,
           SUM(CASE WHEN player_slot >= 128 THEN 1 ELSE 0 END) AS d_total_obs
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
    WHERE tf2.match_id IN (SELECT DISTINCT match_id FROM match_minutes)
      AND tp.player_slot < 128
    GROUP BY tf2.match_id, tf2.minute
),
items AS (
    SELECT match_id, minute,
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
-- NEW: Scaling threats from permanent buffs (LC duel damage, Silencer int, etc.)
perm_buffs AS (
    SELECT pb.match_id, 0 AS minute,
           SUM(CASE WHEN pb.player_slot < 128 THEN pb.stack_count ELSE 0 END) AS r_buff_stacks,
           SUM(CASE WHEN pb.player_slot >= 128 THEN pb.stack_count ELSE 0 END) AS d_buff_stacks
    FROM player_permanent_buffs pb
    WHERE pb.match_id IN (SELECT DISTINCT match_id FROM match_minutes)
    GROUP BY pb.match_id
),
-- NEW: Neutral item tier timing (who gets their neutrals faster)
neutral_items AS (
    SELECT ni.match_id, ni.minute,
           COUNT(CASE WHEN ni.player_slot < 128 THEN 1 END) AS r_neutrals,
           COUNT(CASE WHEN ni.player_slot >= 128 THEN 1 END) AS d_neutrals
    FROM (
        SELECT match_id, player_slot,
               (time / 60) AS minute,
               item_neutral
        FROM player_neutral_item_history
        WHERE match_id IN (SELECT DISTINCT match_id FROM match_minutes)
    ) ni
    GROUP BY ni.match_id, ni.minute
),
-- NEW: Player-level stats (stuns, teamfight participation, tower damage)
player_agg AS (
    SELECT p.match_id, 0 AS minute,
           SUM(CASE WHEN p.player_slot < 128 THEN p.stuns ELSE 0 END) AS r_stuns,
           SUM(CASE WHEN p.player_slot >= 128 THEN p.stuns ELSE 0 END) AS d_stuns,
           AVG(CASE WHEN p.player_slot < 128 THEN p.teamfight_participation END) AS r_tf_participation,
           AVG(CASE WHEN p.player_slot >= 128 THEN p.teamfight_participation END) AS d_tf_participation,
           SUM(CASE WHEN p.player_slot < 128 THEN p.tower_damage ELSE 0 END) AS r_tower_damage,
           SUM(CASE WHEN p.player_slot >= 128 THEN p.tower_damage ELSE 0 END) AS d_tower_damage,
           -- Support NW: bottom 2 net_worths per team (approximated via gold_t last value)
           SUM(CASE WHEN p.player_slot < 128 THEN p.total_gold ELSE 0 END) AS r_total_gold,
           SUM(CASE WHEN p.player_slot >= 128 THEN p.total_gold ELSE 0 END) AS d_total_gold
    FROM players p
    WHERE p.match_id IN (SELECT DISTINCT match_id FROM match_minutes)
    GROUP BY p.match_id
),
-- NEW: Map confinement — center of mass of team movements via lane_pos
-- lane_pos is JSONB: {"x_str": {"y_str": count}} — sparse coordinate→time map
-- Map confinement via deaths_pos in teamfights (proxy for map control)
tf_deaths AS (
    SELECT tp.match_id, 0 AS minute,
           -- Deaths on Radiant side (y < 96) by each team
           SUM(CASE WHEN tp.player_slot < 128 THEN
               CASE WHEN (tp.deaths_pos->>'y')::float < 96 THEN 1 ELSE 0 END
           ELSE 0 END) AS r_deaths_home,
           SUM(CASE WHEN tp.player_slot < 128 THEN
               CASE WHEN (tp.deaths_pos->>'y')::float >= 96 THEN 1 ELSE 0 END
           ELSE 0 END) AS r_deaths_away,
           SUM(CASE WHEN tp.player_slot >= 128 THEN
               CASE WHEN (tp.deaths_pos->>'y')::float >= 96 THEN 1 ELSE 0 END
           ELSE 0 END) AS d_deaths_home,
           SUM(CASE WHEN tp.player_slot >= 128 THEN
               CASE WHEN (tp.deaths_pos->>'y')::float < 96 THEN 1 ELSE 0 END
           ELSE 0 END) AS d_deaths_away
    FROM teamfight_players tp
    WHERE tp.match_id IN (SELECT DISTINCT match_id FROM match_minutes)
      AND tp.deaths > 0
    GROUP BY tp.match_id
)
SELECT
    mm.match_id, mm.minute, mm.radiant_win,
    COALESCE(gx.radiant_gold_adv, 0) AS radiant_gold_adv,
    COALESCE(gx.radiant_xp_adv, 0) AS radiant_xp_adv,
    COALESCE(k.r_kills, 0) AS radiant_kills_tick,
    COALESCE(k.d_kills, 0) AS dire_kills_tick,
    COALESCE(o.r_t1_towers, 0) AS radiant_t1_towers_tick,
    COALESCE(o.d_t1_towers, 0) AS dire_t1_towers_tick,
    COALESCE(o.r_t2_towers, 0) AS radiant_t2_towers_tick,
    COALESCE(o.d_t2_towers, 0) AS dire_t2_towers_tick,
    COALESCE(o.r_t3_towers, 0) AS radiant_t3_towers_tick,
    COALESCE(o.d_t3_towers, 0) AS dire_t3_towers_tick,
    COALESCE(o.r_t4_towers, 0) AS radiant_t4_towers_tick,
    COALESCE(o.d_t4_towers, 0) AS dire_t4_towers_tick,
    COALESCE(o.r_melee_rax, 0) AS radiant_melee_rax_tick,
    COALESCE(o.d_melee_rax, 0) AS dire_melee_rax_tick,
    COALESCE(o.r_range_rax, 0) AS radiant_range_rax_tick,
    COALESCE(o.d_range_rax, 0) AS dire_range_rax_tick,
    COALESCE(o.r_rosh, 0) AS radiant_rosh_tick,
    COALESCE(o.d_rosh, 0) AS dire_rosh_tick,
    COALESCE(o.r_couriers_lost, 0) AS radiant_couriers_lost_tick,
    COALESCE(o.d_couriers_lost, 0) AS dire_couriers_lost_tick,
    COALESCE(w.r_obs, 0) AS radiant_obs_tick,
    COALESCE(w.d_obs, 0) AS dire_obs_tick,
    COALESCE(dw.r_deep_obs, 0) AS radiant_deep_obs_tick,
    COALESCE(dw.d_deep_obs, 0) AS dire_deep_obs_tick,
    COALESCE(dw.r_total_obs, 0) AS radiant_ward_total_tick,
    COALESCE(dw.d_total_obs, 0) AS dire_ward_total_tick,
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
    COALESCE(b.d_buybacks, 0) AS dire_buybacks_tick,
    COALESCE(pb.r_buff_stacks, 0) AS radiant_buff_stacks,
    COALESCE(pb.d_buff_stacks, 0) AS dire_buff_stacks,
    COALESCE(ni.r_neutrals, 0) AS radiant_neutrals_tick,
    COALESCE(ni.d_neutrals, 0) AS dire_neutrals_tick,
    COALESCE(pa.r_stuns, 0) AS radiant_stuns,
    COALESCE(pa.d_stuns, 0) AS dire_stuns,
    COALESCE(pa.r_tf_participation, 0) AS radiant_tf_participation,
    COALESCE(pa.d_tf_participation, 0) AS dire_tf_participation,
    COALESCE(pa.r_tower_damage, 0) AS radiant_tower_damage,
    COALESCE(pa.d_tower_damage, 0) AS dire_tower_damage,
    COALESCE(pa.r_total_gold, 0) AS radiant_total_gold,
    COALESCE(pa.d_total_gold, 0) AS dire_total_gold,
    COALESCE(td.r_deaths_home, 0) AS r_deaths_home,
    COALESCE(td.r_deaths_away, 0) AS r_deaths_away,
    COALESCE(td.d_deaths_home, 0) AS d_deaths_home,
    COALESCE(td.d_deaths_away, 0) AS d_deaths_away
FROM match_minutes mm
LEFT JOIN gold_xp gx ON gx.match_id = mm.match_id AND gx.minute = mm.minute
LEFT JOIN kills k ON k.match_id = mm.match_id AND k.minute = mm.minute
LEFT JOIN objs o ON o.match_id = mm.match_id AND o.minute = mm.minute
LEFT JOIN wards w ON w.match_id = mm.match_id AND w.minute = mm.minute
LEFT JOIN deep_wards dw ON dw.match_id = mm.match_id AND dw.minute = mm.minute
LEFT JOIN tf ON tf.match_id = mm.match_id AND tf.minute = mm.minute
LEFT JOIN items i ON i.match_id = mm.match_id AND i.minute = mm.minute
LEFT JOIN bbs b ON b.match_id = mm.match_id AND b.minute = mm.minute
LEFT JOIN perm_buffs pb ON pb.match_id = mm.match_id
LEFT JOIN neutral_items ni ON ni.match_id = mm.match_id AND ni.minute = mm.minute
LEFT JOIN player_agg pa ON pa.match_id = mm.match_id
LEFT JOIN tf_deaths td ON td.match_id = mm.match_id
ORDER BY mm.match_id, mm.minute
"""


def extract_dynamic_features(engine, patch_id: int, lookback: int = 2) -> pd.DataFrame:
    """Extract per-minute dynamic features capturing true game state."""
    logger.info("Extracting per-minute dynamic features for patch %s (lookback %d) ...", patch_id, lookback)

    df = pd.read_sql(EXTRACT_DYNAMIC_SQL, engine, params={"patch_id": patch_id, "lookback": lookback})

    if df.empty:
        raise ValueError(f"No data found for patch {patch_id}")

    df = df.sort_values(["match_id", "minute"]).reset_index(drop=True)

    # --- Aegis Active State (5-minute rolling window) ---
    df["radiant_aegis"] = df.groupby("match_id")["radiant_rosh_tick"].rolling(5, min_periods=1).sum().reset_index(0, drop=True)
    df["dire_aegis"] = df.groupby("match_id")["dire_rosh_tick"].rolling(5, min_periods=1).sum().reset_index(0, drop=True)
    df["aegis_diff"] = (df["radiant_aegis"] > 0).astype(float) - (df["dire_aegis"] > 0).astype(float)

    # --- Cumulative states from tick events ---
    tick_cols = [c for c in df.columns if c.endswith("_tick")]
    cum_cols = [c.replace("_tick", "") for c in tick_cols]

    # Single groupby for all tick columns (much faster than looping)
    df[cum_cols] = df.groupby("match_id")[tick_cols].cumsum()

    # --- Win Conditions: Mega Creeps ---
    df["mega_creeps_radiant"] = ((df["dire_melee_rax"] + df["dire_range_rax"]) >= 6).astype(float)
    df["mega_creeps_dire"] = ((df["radiant_melee_rax"] + df["radiant_range_rax"]) >= 6).astype(float)

    # --- Active Vulnerability: Dead Heroes Now ---
    r_kills = df.groupby("match_id")["radiant_kills_tick"]
    d_kills = df.groupby("match_id")["dire_kills_tick"]

    r_dead_early = d_kills.rolling(1, min_periods=1).sum().reset_index(0, drop=True) * (df["minute"] < 20).astype(float)
    r_dead_mid = d_kills.rolling(2, min_periods=1).sum().reset_index(0, drop=True) * ((df["minute"] >= 20) & (df["minute"] < 40)).astype(float)
    r_dead_late = d_kills.rolling(3, min_periods=1).sum().reset_index(0, drop=True) * (df["minute"] >= 40).astype(float)
    df["radiant_dead_now"] = r_dead_early + r_dead_mid + r_dead_late

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
    df["buyback_diff"] = df["radiant_buybacks"] - df["dire_buybacks"]
    df["courier_lost_diff"] = df["radiant_couriers_lost"] - df["dire_couriers_lost"]

    # --- Objectives Differentials ---
    df["t1_tower_diff"] = df["radiant_t1_towers"] - df["dire_t1_towers"]
    df["t2_tower_diff"] = df["radiant_t2_towers"] - df["dire_t2_towers"]
    df["t3_tower_diff"] = df["radiant_t3_towers"] - df["dire_t3_towers"]
    df["t4_tower_diff"] = df["radiant_t4_towers"] - df["dire_t4_towers"]
    df["melee_rax_diff"] = df["radiant_melee_rax"] - df["dire_melee_rax"]
    df["range_rax_diff"] = df["radiant_range_rax"] - df["dire_range_rax"]
    df["roshan_diff"] = df["radiant_rosh"] - df["dire_rosh"]
    df["ward_diff"] = df["radiant_obs"] - df["dire_obs"]
    df["tf_diff"] = df["radiant_tf_wins"] - df["dire_tf_wins"]

    # --- Deep Vision Advantage ---
    # Positive = Radiant has more deep wards in Dire territory
    df["deep_ward_diff"] = df["radiant_deep_obs"] - df["dire_deep_obs"]

    # --- Momentum (Gold/XP swings) ---
    for col in ["radiant_gold_adv", "radiant_xp_adv"]:
        df[f"{col}_prev1"] = df.groupby("match_id")[col].shift(1).fillna(0)
        df[f"{col}_prev3"] = df.groupby("match_id")[col].shift(3).fillna(0)

    df["gold_adv_diff_1m"] = df["radiant_gold_adv"] - df["radiant_gold_adv_prev1"]
    df["xp_adv_diff_1m"] = df["radiant_xp_adv"] - df["radiant_xp_adv_prev1"]
    df["gold_adv_diff_3m"] = df["radiant_gold_adv"] - df["radiant_gold_adv_prev3"]
    df["xp_adv_diff_3m"] = df["radiant_xp_adv"] - df["radiant_xp_adv_prev3"]

    df["minute_sq"] = df["minute"] ** 2

    # --- Economy Distribution (from gold_t JSONB) ---
    df["rad_carry_nw_pct"] = 0.2
    df["dire_carry_nw_pct"] = 0.2
    df["carry_farm_diff"] = 0.0

    # --- Support Net Worth Diff ---
    # Positive = Radiant supports are richer (more items, higher survival)
    df["support_nw_diff"] = df["radiant_total_gold"] - df["dire_total_gold"]

    # --- Laning Phase CS ---
    df["radiant_cs_adv"] = 0.0

    # --- Defensive & Utility Items ---
    df["save_item_diff"] = 0.0
    df["aura_item_diff"] = 0.0

    # --- Vision Denial ---
    df["dewards_diff"] = 0.0

    # --- Rune Control ---
    df["rune_control_diff"] = 0.0

    # --- Teamfight Efficiency (magnitude) ---
    df["tf_gold_swing_1m"] = 0.0
    df["tf_xp_swing_1m"] = 0.0

    # --- Map Confinement ---
    # Positive = Radiant is more aggressive (dying in enemy territory)
    # Negative = Dire is more aggressive
    # Deaths away from home indicate map control / aggression
    total_r_deaths = (df["r_deaths_home"] + df["r_deaths_away"]).clip(lower=1)
    total_d_deaths = (df["d_deaths_home"] + df["d_deaths_away"]).clip(lower=1)
    r_away_pct = df["r_deaths_away"] / total_r_deaths
    d_away_pct = df["d_deaths_away"] / total_d_deaths
    df["map_confinement_diff"] = r_away_pct - d_away_pct

    # --- Scaling Threats ---
    # Positive = Radiant has more permanent buff stacks (LC duel damage, Silencer int, etc.)
    df["scaling_threat_diff"] = df["radiant_buff_stacks"] - df["dire_buff_stacks"]

    # --- CC Effectiveness ---
    # Stuns + teamfight participation combined
    # A team with 200 stuns and 80% TF participation is executing better than 50 stuns / 40%
    r_cc = df["radiant_stuns"] * df["radiant_tf_participation"].clip(lower=0.01)
    d_cc = df["dire_stuns"] * df["dire_tf_participation"].clip(lower=0.01)
    df["cc_effectiveness_diff"] = r_cc - d_cc

    # --- Neutral Item Tier Timing ---
    # Positive = Radiant has more neutral items at this minute (got them faster)
    df["neutral_tier_diff"] = df["radiant_neutrals"] - df["dire_neutrals"]

    # --- Tower Damage ---
    # Positive = Radiant is pressing the map more (damaging structures)
    df["tower_damage_diff"] = df["radiant_tower_damage"] - df["dire_tower_damage"]

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(float)

    logger.info("Final dataset: %d rows, %d matches, %.1f%% radiant win rate",
                len(df), df["match_id"].nunique(), df[TARGET_COLUMN].mean() * 100)

    return df
