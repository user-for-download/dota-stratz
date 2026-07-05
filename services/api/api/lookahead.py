"""Monte Carlo Draft Simulation for DraftBERT.

Instead of 1-ply minimax, runs random rollouts to the end of the draft
for each candidate hero and batch-evaluates the 5v5 compositions using
the TorchScript model. This provides much deeper strategic lookahead.

Supports an optional progress callback for WebSocket streaming of MCTS
iterations to the frontend.
"""

import logging
import random
from typing import Callable

import numpy as np
import torch

from .draft_state import DraftContext
from .features import build_feature_vector, pre_fetch_batch

logger = logging.getLogger(__name__)


def run_monte_carlo_rollouts(
    predictor,
    patch_id: int,
    ctx: DraftContext,
    candidates: list[dict],
    eligible_hero_ids: list[int],
    radiant_team_id: int | None,
    dire_team_id: int | None,
    num_simulations: int = 40,
    progress_cb: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Simulate random draft completions and batch-evaluate 5v5 outcomes.

    For each candidate hero, simulates `num_simulations` random completions
    of the remaining draft slots, then uses the TorchScript model to evaluate
    all simulated 5v5 compositions in a single batched forward pass.

    Parameters
    ----------
    progress_cb : callable, optional
        Called with a dict after each candidate is evaluated:
        ``{"iteration": N, "total": M, "hero_id": H, "win_rate": W}``
        Useful for streaming progress to a WebSocket client.
    """
    if not candidates:
        return candidates

    with predictor._lock:
        model = predictor._models[patch_id]
        schema = predictor._schemas[patch_id]

    num_continuous = len(schema["aggregate_columns"])
    max_seq_len = schema.get("max_seq_len", 50)

    # Pre-fetch tabular features for team comparison (static across simulations)
    dummy_ctx = DraftContext(
        turn=11, recommending_team=-1, is_pick_turn=False,
        radiant_picks=list(ctx.radiant_picks), dire_picks=list(ctx.dire_picks),
    )
    batch_data = pre_fetch_batch(patch_id, [1], radiant_team_id, dire_team_id, dummy_ctx)
    base_fv = build_feature_vector(
        1, dummy_ctx, patch_id, batch_data, schema, {}, schema["max_hero_id"],
    )
    base_tabular = base_fv[:num_continuous]

    batch_h, batch_a, batch_f, batch_cid = [], [], [], []
    total = len(candidates)

    for idx, cand in enumerate(candidates):
        cid = cand["hero_id"]
        local_eligible = [h for h in eligible_hero_ids if h != cid]

        is_rad = ctx.recommending_team == 0
        c_rad = list(ctx.radiant_picks) + ([cid] if is_rad else [])
        c_dire = list(ctx.dire_picks) + ([cid] if not is_rad else [])

        r_need = max(0, 5 - len(c_rad))
        d_need = max(0, 5 - len(c_dire))

        for _ in range(num_simulations):
            sampled = random.sample(local_eligible, min(r_need + d_need, len(local_eligible)))
            sim_rad = c_rad + sampled[:r_need]
            sim_dire = c_dire + sampled[r_need:]

            # Build 5v5 sequence
            seq_h, seq_a = [], []
            for j in range(5):
                seq_h.append(sim_rad[j] if j < len(sim_rad) else 0)
                seq_a.append(3)
                seq_h.append(sim_dire[j] if j < len(sim_dire) else 0)
                seq_a.append(4)

            pad = max_seq_len - len(seq_h)
            batch_h.append(seq_h + [0] * pad)
            batch_a.append(seq_a + [0] * pad)
            batch_f.append(base_tabular)
            batch_cid.append(cid)

    # Batched forward pass through TorchScript
    t_h = torch.tensor(batch_h, dtype=torch.long)
    t_a = torch.tensor(batch_a, dtype=torch.long)
    t_f = torch.tensor(np.array(batch_f), dtype=torch.float32)

    with torch.no_grad():
        logits = model(t_h, t_a, t_f)
        probs = torch.sigmoid(logits).numpy()

    # Aggregate per-candidate
    cand_scores: dict[int, list[float]] = {c["hero_id"]: [] for c in candidates}
    for i, prob in enumerate(probs):
        cid = batch_cid[i]
        score = float(prob) if ctx.recommending_team == 0 else 1.0 - float(prob)
        cand_scores[cid].append(score)

    # Blend base policy with Monte Carlo score and emit progress
    for idx, cand in enumerate(candidates):
        mc = cand_scores[cand["hero_id"]]
        mc_avg = sum(mc) / len(mc) if mc else 0.5
        blended = cand["win_probability"] * 0.5 + mc_avg * 0.5
        cand["lookahead_score"] = round(blended, 4)
        cand["mc_win_probability"] = round(mc_avg, 4)

        if progress_cb:
            progress_cb({
                "iteration": idx + 1,
                "total": total,
                "hero_id": cand["hero_id"],
                "win_rate": mc_avg,
                "lookahead_score": blended,
            })

    candidates.sort(key=lambda x: x["lookahead_score"], reverse=True)
    return candidates
