"""Adversarial Monte Carlo Draft Simulation for DraftBERT.

Runs random rollouts for average-case analysis, PLUS an Adversarial Minimax
(Paranoia) pass to detect devastating last-pick counters ("Cheese" heroes).
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
    num_simulations: int = 30,
    progress_cb: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Simulate average completions AND the absolute worst-case enemy counter."""

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
    base_fv = build_feature_vector(1, dummy_ctx, patch_id, batch_data, schema, {}, schema["max_hero_id"])
    base_tabular = base_fv[:num_continuous]

    batch_h, batch_a, batch_f, batch_cid = [], [], [], []
    is_worst_case_flag = []

    total = len(candidates)

    for idx, cand in enumerate(candidates):
        cid = cand["hero_id"]
        local_eligible = [h for h in eligible_hero_ids if h != cid]

        is_rad = ctx.recommending_team == 0
        c_rad = list(ctx.radiant_picks) + ([cid] if is_rad else [])
        c_dire = list(ctx.dire_picks) + ([cid] if not is_rad else [])

        r_need = max(0, 5 - len(c_rad))
        d_need = max(0, 5 - len(c_dire))

        # 1. AVERAGE CASE: Random Rollouts
        for _ in range(num_simulations):
            sampled = random.sample(local_eligible, min(r_need + d_need, len(local_eligible)))
            sim_rad = c_rad + sampled[:r_need]
            sim_dire = c_dire + sampled[r_need:]

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
            is_worst_case_flag.append(False)

        # 2. ADVERSARIAL WORST-CASE (The "Naughty Option" Checker)
        enemy_need = d_need if is_rad else r_need
        if enemy_need > 0:
            for enemy_nemesis in local_eligible:
                nemesis_rad = c_rad + ([enemy_nemesis] if not is_rad else [])
                nemesis_dire = c_dire + ([enemy_nemesis] if is_rad else [])

                seq_h, seq_a = [], []
                for j in range(max(len(nemesis_rad), len(nemesis_dire))):
                    seq_h.append(nemesis_rad[j] if j < len(nemesis_rad) else 0)
                    seq_a.append(3)
                    seq_h.append(nemesis_dire[j] if j < len(nemesis_dire) else 0)
                    seq_a.append(4)

                pad = max_seq_len - len(seq_h)
                batch_h.append(seq_h + [0] * pad)
                batch_a.append(seq_a + [0] * pad)
                batch_f.append(base_tabular)
                batch_cid.append(cid)
                is_worst_case_flag.append(True)

    # Batched forward pass through TorchScript
    t_h = torch.tensor(batch_h, dtype=torch.long)
    t_a = torch.tensor(batch_a, dtype=torch.long)
    t_f = torch.tensor(np.array(batch_f), dtype=torch.float32)

    with torch.no_grad():
        logits = model(t_h, t_a, t_f)
        probs = torch.sigmoid(logits).numpy()

    # Aggregate per-candidate
    mc_scores = {c["hero_id"]: [] for c in candidates}
    worst_case_scores = {c["hero_id"]: [] for c in candidates}

    for i, prob in enumerate(probs):
        cid = batch_cid[i]
        score = float(prob) if ctx.recommending_team == 0 else 1.0 - float(prob)

        if is_worst_case_flag[i]:
            worst_case_scores[cid].append(score)
        else:
            mc_scores[cid].append(score)

    # 3. BLEND SCORES (Base Policy + Avg MC + Worst Case Penalty)
    for idx, cand in enumerate(candidates):
        cid = cand["hero_id"]

        mc = mc_scores[cid]
        mc_avg = sum(mc) / len(mc) if mc else 0.5

        wc = worst_case_scores[cid]
        worst_case_win_rate = min(wc) if wc else mc_avg

        # 40% Base + 40% Average Rollouts + 20% Worst-Case Paranoia
        blended = (cand["win_probability"] * 0.40) + (mc_avg * 0.40) + (worst_case_win_rate * 0.20)

        # Hard-veto if worst-case enemy counter drops WR below 35%
        if worst_case_win_rate < 0.35:
            blended -= 0.15

        cand["lookahead_score"] = round(blended, 4)
        cand["mc_win_probability"] = round(mc_avg, 4)
        cand["worst_case_nemesis_wr"] = round(worst_case_win_rate, 4)

        if progress_cb:
            progress_cb({
                "iteration": idx + 1,
                "total": total,
                "hero_id": cid,
                "win_rate": mc_avg,
                "lookahead_score": blended,
            })

    candidates.sort(key=lambda x: x["lookahead_score"], reverse=True)
    return candidates
