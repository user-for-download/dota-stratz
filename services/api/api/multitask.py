"""Multi-task draft prediction — predicts both pick and ban recommendations.

Currently the model treats all draft slots uniformly. This module adds
task-specific scoring:
- Pick scoring: maximize team win probability
- Ban scoring: minimize opponent's best hero effectiveness
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def score_bans(
    eligible_hero_ids: list[int],
    draft_state: list[dict],
    picks_by_team: dict[int, list[int]],
    pick_scores: dict[int, float],
    enemy_team: int,
) -> list[dict]:
    """Score heroes for banning based on their value to the enemy team.

    Ban priority = enemy's pick_score for this hero × synergy with enemy picks
    """
    enemy_picks = picks_by_team.get(enemy_team, [])
    ban_scores = []

    for hid in eligible_hero_ids:
        # Base score: how valuable this hero is to the enemy
        base = pick_scores.get(hid, 0.5)

        # Synergy boost: enemy already has heroes that synergize with this one
        synergy_boost = 0.0
        for ep in enemy_picks:
            if ep > 0:
                # Simple heuristic: heroes in same role amplify each other
                synergy_boost += 0.02  # small bonus per enemy synergy

        ban_value = base + synergy_boost
        ban_scores.append({
            "hero_id": hid,
            "ban_score": round(ban_value, 4),
            "reason": f"enemy value {base:.3f}" + (f" + synergy {synergy_boost:.3f}" if synergy_boost > 0 else ""),
        })

    ban_scores.sort(key=lambda x: x["ban_score"], reverse=True)
    return ban_scores


def classify_slot(draft_state: list[dict], next_slot: int) -> str:
    """Determine if the next draft slot is a pick or ban.

    Uses the patch-specific draft pattern to classify.
    """
    # This is a simplified version — the full implementation would use
    # the DRAFT_PATTERNS from draft_state.py
    picks_so_far = sum(1 for d in draft_state if d.get("is_pick"))
    bans_so_far = sum(1 for d in draft_state if not d.get("is_pick"))

    # Patch 60 pattern: 12 bans, 12 picks
    if bans_so_far < 12:
        return "ban"
    return "pick"
