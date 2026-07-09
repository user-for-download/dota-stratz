"""Dynamic feature columns for live match prediction.

Shared between trainer (training) and API (inference).
39 dynamic features capturing the 7 pillars of live Dota 2.
"""

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
    # Laning Phase (CS dominance)
    "radiant_cs_adv",
    # Defensive & Utility Power Spikes
    "save_item_diff",
    "aura_item_diff",
    # Vision Denial (de-warding)
    "dewards_diff",
    # Rune Control
    "rune_control_diff",
    # Teamfight Efficiency (magnitude of swings)
    "tf_gold_swing_1m",
    "tf_xp_swing_1m",
]
