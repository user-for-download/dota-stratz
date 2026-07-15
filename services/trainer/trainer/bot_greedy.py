"""Greedy single-step lookahead drafting bot.

Evaluates all available heroes in a single batched PyTorch forward pass
to find the pick/ban that maximizes (or minimizes) the predicted win
probability for the acting team.

Usage:
    bot = GreedyDraftBot(model, state_builder)
    suggestions = bot.suggest_next_action(
        current_heroes=[14, 53],
        current_actions=[3, 4],
        is_radiant_turn=True,
        is_pick=True,
        radiant_picks=[14],
        dire_picks=[53],
    )
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from .draft_state import DraftStateBuilder

logger = logging.getLogger(__name__)


class GreedyDraftBot:
    """Single-step lookahead bot using batched PyTorch inference.

    Evaluates all ~120 available heroes simultaneously in one forward pass.
    Because DraftBERT predicts win probability for the ACTING team, the bot
    always maximizes the model's output — whether it's Radiant or Dire.
    """

    def __init__(
        self,
        model: torch.jit.ScriptModule | torch.nn.Module,
        state_builder: DraftStateBuilder,
        max_hero_id: int = 160,
        max_seq_len: int = 25,
        device: str = "cpu",
    ):
        self.model = model
        self.model.to(device)
        self.model.eval()
        self.state_builder = state_builder
        self.max_hero_id = max_hero_id
        self.max_seq_len = max_seq_len
        self.device = device

        # Valid heroes: those with >0 games in baseline cache
        self.valid_heroes = state_builder.cache.valid_hero_ids

    def _filter_valid_composition(
        self, available_heroes: list[int], team_picks: list[int]
    ) -> list[int]:
        """Prevent the bot from evaluating comps that exceed 3 cores or 2 supports."""
        if len(team_picks) == 0:
            return available_heroes

        current_cores = sum(
            1 for h in team_picks
            if self.state_builder.cache.get_baseline(h).get("avg_gpm", 0.0) > 420.0
        )
        current_supports = len(team_picks) - current_cores

        filtered = []
        for h in available_heroes:
            is_core = self.state_builder.cache.get_baseline(h).get("avg_gpm", 0.0) > 420.0
            if is_core and current_cores >= 3:
                continue   # no 4th carry
            if not is_core and current_supports >= 2:
                continue   # no 3rd support
            filtered.append(h)

        return filtered if filtered else available_heroes

    @torch.no_grad()
    def suggest_next_action(
        self,
        current_heroes: list[int],
        current_actions: list[int],
        is_radiant_turn: bool,
        is_pick: bool,
        radiant_picks: list[int],
        dire_picks: list[int],
        top_k: int = 5,
        team_id: int | None = None,
        account_id: int | None = None,
        enemy_team_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Evaluate all available heroes and return top-k suggestions.

        Parameters
        ----------
        current_heroes : list[int]
            Hero IDs in draft order so far (picks and bans).
        current_actions : list[int]
            Action tokens (1=RadBan, 2=DireBan, 3=RadPick, 4=DirePick).
        is_radiant_turn : bool
            True if it's Radiant's turn.
        is_pick : bool
            True if this is a pick (not a ban).
        radiant_picks : list[int]
            Hero IDs already picked by Radiant.
        dire_picks : list[int]
            Hero IDs already picked by Dire.
        top_k : int
            Number of top suggestions to return.
        team_id : int, optional
            Pro team ID for team-hero lookup.
        account_id : int, optional
            Player account ID for player-hero lookup.
        enemy_team_id : int, optional
            Enemy team ID for H2H lookup.

        Returns
        -------
        list[dict]
            Top-k suggestions with hero_id and win_probability.
        """
        # 1. Determine available heroes
        available_heroes = [h for h in self.valid_heroes if h not in current_heroes]

        # 1b. Composition constraint: prevent 5-carry OOD drafts on pick turns
        if is_pick:
            team_picks = radiant_picks if is_radiant_turn else dire_picks
            available_heroes = self._filter_valid_composition(available_heroes, team_picks)

        batch_size = len(available_heroes)
        seq_len = len(current_heroes) + 1

        if batch_size == 0:
            logger.warning("No available heroes to suggest!")
            return []

        if seq_len > self.max_seq_len:
            raise ValueError(f"Draft exceeded max sequence length ({self.max_seq_len})")

        # 2. Determine action token
        if is_radiant_turn:
            action_token = 3 if is_pick else 1
        else:
            action_token = 4 if is_pick else 2

        # 3. Pre-allocate batched tensors
        batch_heroes = torch.zeros((batch_size, self.max_seq_len), dtype=torch.long)
        batch_actions = torch.zeros((batch_size, self.max_seq_len), dtype=torch.long)
        batch_tabular = torch.zeros(
            (batch_size, self.state_builder.num_features), dtype=torch.float32
        )
        patch_id = self.state_builder.cache.patch_id
        batch_patches = torch.full((batch_size,), patch_id, dtype=torch.long)

        # 4. Populate the batch with hypothetical states
        for i, hero_id in enumerate(available_heroes):
            # Sequence data
            hyp_heroes = current_heroes + [hero_id]
            hyp_actions = current_actions + [action_token]
            batch_heroes[i, :seq_len] = torch.tensor(hyp_heroes, dtype=torch.long)
            batch_actions[i, :seq_len] = torch.tensor(hyp_actions, dtype=torch.long)

            # Tabular features (fast in-memory update)
            # Calculate pick ordinal for this hypothetical pick
            pick_ordinal = None
            if is_pick:
                if is_radiant_turn:
                    pick_ordinal = len(radiant_picks) + 1
                else:
                    pick_ordinal = len(dire_picks) + 1

            tabular_array = self.state_builder.build_tabular_features(
                hypothetical_hero_id=hero_id,
                is_radiant_turn=is_radiant_turn,
                is_pick=is_pick,
                radiant_picks=radiant_picks,
                dire_picks=dire_picks,
                team_id=team_id,
                account_id=account_id,
                enemy_team_id=enemy_team_id,
                team_pick_ordinal=pick_ordinal,
            )
            batch_tabular[i] = torch.from_numpy(tabular_array)

        # 5. Move to device and run single forward pass
        batch_heroes = batch_heroes.to(self.device)
        batch_actions = batch_actions.to(self.device)
        batch_tabular = batch_tabular.to(self.device)
        batch_patches = batch_patches.to(self.device)

        logits = self.model(batch_heroes, batch_actions, batch_tabular, batch_patches)
        win_probs = torch.sigmoid(logits).cpu().numpy()

        # 6. Sort and get Top K results
        # Because target is relative to acting team, higher prob = better
        best_indices = np.argsort(win_probs)[::-1][:top_k]

        suggestions = []
        for idx in best_indices:
            suggestions.append({
                "hero_id": available_heroes[idx],
                "win_probability": float(win_probs[idx]),
            })

        logger.info(
            "GreedyBot: evaluated %d heroes, top pick = %s (%.2f%%)",
            batch_size,
            suggestions[0]["hero_id"] if suggestions else "N/A",
            suggestions[0]["win_probability"] * 100 if suggestions else 0,
        )

        return suggestions
