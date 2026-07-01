"""Look-ahead minimax search for draft optimization.

Instead of greedily picking the best hero for the current slot, this module
simulates the opponent's best response and evaluates the final draft quality.
This produces stronger draft recommendations by considering future turns.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Maximum depth for minimax search (limited by API latency budget)
MAX_LOOKAHEAD = 2


def minimax_score(
    draft_so_far: list[dict],
    candidate_hero: int,
    is_pick: bool,
    team: int,
    patch_id: int,
    eligible_hero_ids: list[int],
    model_fn,
    batch_ctx: Any,
    schema: dict,
    depth: int = 0,
    alpha: float = -1.0,
    beta: float = 1.0,
) -> float:
    """Evaluate a draft move using minimax with alpha-beta pruning.

    Parameters
    ----------
    draft_so_far : list of draft slot dicts
    candidate_hero : hero_id to evaluate
    is_pick : True if this is a pick, False if ban
    team : 0 or 1
    patch_id : current patch
    eligible_hero_ids : heroes not yet taken
    model_fn : callable(hero_id, draft_state) -> float (calibrated score)
    batch_ctx : pre-fetched batch context
    schema : feature schema
    depth : current search depth
    alpha, beta : alpha-beta bounds

    Returns
    -------
    float : estimated final draft quality (higher = better for team 0)
    """
    if depth >= MAX_LOOKAHEAD or len(eligible_hero_ids) <= 1:
        # Leaf node or no more heroes — return the direct score
        return model_fn(candidate_hero, draft_so_far)

    # Simulate this move
    next_draft = draft_so_far + [{
        "hero_id": candidate_hero,
        "is_pick": is_pick,
        "team": team,
        "order": len(draft_so_far) + 1,
    }]
    remaining = [h for h in eligible_hero_ids if h != candidate_hero]

    if team == 0:
        # Maximizing team's turn — we pick best, opponent picks worst for us
        best = -1.0
        for i, opp_hero in enumerate(remaining[:3]):  # limit branching
            score = minimax_score(
                next_draft, opp_hero, is_pick, 1 - team,
                patch_id, [h for h in remaining if h != opp_hero],
                model_fn, batch_ctx, schema, depth + 1, alpha, beta
            )
            best = max(best, score)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    else:
        # Minimizing team's turn — opponent picks best for them
        best = 1.0
        for i, opp_hero in enumerate(remaining[:3]):
            score = minimax_score(
                next_draft, opp_hero, is_pick, 1 - team,
                patch_id, [h for h in remaining if h != opp_hero],
                model_fn, batch_ctx, schema, depth + 1, alpha, beta
            )
            best = min(best, score)
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best


def ranked_with_lookahead(
    candidates: list[dict],
    draft_state: list[dict],
    patch_id: int,
    eligible_hero_ids: list[int],
    model_fn,
    batch_ctx: Any,
    schema: dict,
    top_k: int = 10,
) -> list[dict]:
    """Re-rank candidates using 1-ply look-ahead minimax.

    For each candidate hero, simulates the opponent's best response and
    returns the worst-case score. Heroes that survive opponent counter-picks
    rank higher.
    """
    if len(eligible_hero_ids) <= 3 or top_k <= 0:
        return candidates[:top_k]

    ranked = []
    for cand in candidates[:min(top_k * 2, len(candidates))]:
        hero_id = cand["hero_id"]
        # 1-ply look-ahead: simulate opponent's best pick after this hero
        opp_best = -1.0
        opp_remaining = [h for h in eligible_hero_ids if h != hero_id]

        for opp_hero in opp_remaining[:5]:
            opp_score = model_fn(opp_hero, draft_state + [
                {"hero_id": hero_id, "is_pick": True, "team": 0, "order": len(draft_state) + 1},
                {"hero_id": opp_hero, "is_pick": True, "team": 1, "order": len(draft_state) + 2},
            ])
            opp_best = max(opp_best, opp_score)

        # Blend direct score with look-ahead score
        blended = cand.get("pick_probability", cand["score"]) * 0.6 + opp_best * 0.4
        ranked.append({**cand, "lookahead_score": round(blended, 4)})

    ranked.sort(key=lambda x: x["lookahead_score"], reverse=True)
    return ranked[:top_k]
