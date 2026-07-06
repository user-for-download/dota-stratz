"""Dynamic feature columns for live match prediction.

Shared between trainer (training) and API (inference).
24 dynamic features capturing the 5 true pillars of live Dota 2.
"""

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
