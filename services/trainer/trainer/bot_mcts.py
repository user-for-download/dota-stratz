"""Monte Carlo Tree Search (MCTS) drafting bot.

Explores thousands of hypothetical draft completions to find the move
that maximizes the final win probability. Uses DraftBERT as a value
network (like AlphaZero for Go) — no random rollouts needed.

Usage:
    bot = MCTSDraftBot(model, state_builder)
    best_hero, win_prob = bot.search(
        current_heroes=[14, 53, 2],
        current_actions=[3, 4, 3],
        radiant_picks=[14, 2],
        dire_picks=[53],
        turn_idx=3,
        iterations=2000,
    )
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any

import numpy as np
import torch

from .draft_state import DraftStateBuilder

logger = logging.getLogger(__name__)

# Standard Captain's Mode Sequence (24 steps: 14 bans, 10 picks)
# (is_radiant_turn, is_pick)
CAPTAINS_MODE_FORMAT: list[tuple[bool, bool]] = [
    # Phase 1 Bans (8 bans: R D R D R D D R)
    (True, False), (False, False), (True, False), (False, False),
    (True, False), (False, False), (False, False), (True, False),
    # Phase 1 Picks (4 picks: R D D R)
    (True, True), (False, True), (False, True), (True, True),
    # Phase 2 Bans (6 bans: R D R D R D)
    (True, False), (False, False), (True, False), (False, False),
    (True, False), (False, False),
    # Phase 2 Picks (4 picks: D R D R)
    (False, True), (True, True), (False, True), (True, True),
    # Phase 3 Bans (2 bans: R D)
    (True, False), (False, False),
    # Phase 3 Picks (2 picks: R D)
    (True, True), (False, True),
]


class MCTSNode:
    """A node in the MCTS game tree."""

    def __init__(
        self,
        parent: MCTSNode | None,
        hero_action: int | None,
        turn_idx: int,
        draft_format: list[tuple[bool, bool]],
    ):
        self.parent = parent
        self.hero_action = hero_action  # Hero ID picked/banned to reach this node
        self.turn_idx = turn_idx  # Which step in the draft (0 to len-1)
        self.draft_format = draft_format

        self.children: dict[int, MCTSNode] = {}
        self.radiant_wins: float = 0.0  # Sum of Radiant win probs (absolute)
        self.visits: int = 0
        self.untried_heroes: list[int] = []

        # Terminal check
        self.is_terminal = self.turn_idx >= len(self.draft_format)
        if not self.is_terminal:
            self.is_radiant_turn, self.is_pick = self.draft_format[self.turn_idx]
        else:
            self.is_radiant_turn, self.is_pick = False, False

    def ucb1(self, c: float = 1.414) -> float:
        """Upper Confidence Bound 1 (UCB1) formula.

        Balances exploitation (high win rate) with exploration (few visits).
        """
        if self.visits == 0:
            return float("inf")

        # Win rate from the perspective of the PARENT's acting team
        # If parent was Radiant, it wants to maximize Radiant wins
        # If parent was Dire, it wants to maximize Dire wins (1 - Radiant wins)
        radiant_wr = self.radiant_wins / self.visits
        if self.parent and not self.parent.is_radiant_turn:
            exploitation = 1.0 - radiant_wr
        else:
            exploitation = radiant_wr

        # Exploration: prioritize nodes with fewer visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)

        return exploitation + exploration


class MCTSDraftBot:
    """Monte Carlo Tree Search bot for optimal drafting.

    Uses DraftBERT as a value network — no random rollouts needed.
    When MCTS reaches an unexpanded node, it queries DraftBERT for the
    win probability of that partial draft and backpropagates immediately.
    """

    def __init__(
        self,
        model: torch.jit.ScriptModule | torch.nn.Module,
        state_builder: DraftStateBuilder,
        draft_format: list[tuple[bool, bool]] | None = None,
        max_seq_len: int = 50,
        device: str = "cpu",
    ):
        self.model = model
        self.model.to(device)
        self.model.eval()
        self.state_builder = state_builder
        self.draft_format = draft_format or CAPTAINS_MODE_FORMAT
        self.max_seq_len = max_seq_len
        self.device = device

        # Valid heroes: those with >0 games in baseline cache
        self.valid_heroes = state_builder.cache.valid_hero_ids

    def _get_state_from_node(
        self,
        node: MCTSNode,
        base_heroes: list[int],
        base_actions: list[int],
        base_rad_picks: list[int],
        base_dire_picks: list[int],
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        """Walk up the tree to construct the full current state lists."""
        heroes, actions, rad_picks, dire_picks = [], [], [], []
        curr = node

        # Trace path back to root
        while curr.parent is not None:
            heroes.append(curr.hero_action)
            # Parent's turn type defines the action token
            is_rad, is_pick = curr.parent.is_radiant_turn, curr.parent.is_pick
            if is_rad and is_pick:
                token = 3
            elif not is_rad and is_pick:
                token = 4
            elif is_rad:
                token = 1
            else:
                token = 2
            actions.append(token)

            if is_pick:
                if is_rad:
                    rad_picks.append(curr.hero_action)
                else:
                    dire_picks.append(curr.hero_action)
            curr = curr.parent

        # Reverse because we traversed from leaf to root
        heroes.reverse()
        actions.reverse()
        rad_picks.reverse()
        dire_picks.reverse()

        return (
            base_heroes + heroes,
            base_actions + actions,
            base_rad_picks + rad_picks,
            base_dire_picks + dire_picks,
        )

    @torch.no_grad()
    def _evaluate_state(
        self,
        heroes: list[int],
        actions: list[int],
        rad_picks: list[int],
        dire_picks: list[int],
        node: MCTSNode,
    ) -> float:
        """Run DraftBERT once to get the Radiant win probability."""
        seq_len = len(heroes)

        batch_heroes = torch.zeros((1, self.max_seq_len), dtype=torch.long)
        batch_actions = torch.zeros((1, self.max_seq_len), dtype=torch.long)
        batch_heroes[0, :seq_len] = torch.tensor(heroes, dtype=torch.long)
        batch_actions[0, :seq_len] = torch.tensor(actions, dtype=torch.long)

        # Build tabular features for the last hero in the sequence
        # This represents the state AFTER that hero was picked/banned
        if seq_len > 0:
            last_hero = heroes[-1]
            # Determine if the last action was a pick or ban
            last_is_pick = actions[-1] in (3, 4)
            last_is_radiant = actions[-1] in (1, 3)
        else:
            last_hero = 0
            last_is_pick = node.is_pick
            last_is_radiant = node.is_radiant_turn

        tabular_array = self.state_builder.build_tabular_features(
            hypothetical_hero_id=last_hero,
            is_radiant_turn=last_is_radiant,
            is_pick=last_is_pick,
            radiant_picks=rad_picks,
            dire_picks=dire_picks,
        )

        batch_tabular = torch.from_numpy(tabular_array).unsqueeze(0)

        logits = self.model(
            batch_heroes.to(self.device),
            batch_actions.to(self.device),
            batch_tabular.to(self.device),
        )
        # DraftBERT predicts win prob for the ACTING team
        acting_team_win_prob = torch.sigmoid(logits).item()

        # Convert to absolute Radiant win probability for consistent backprop
        # The node represents the state where it's the NODE's turn
        if not node.is_terminal and not node.is_radiant_turn:
            return 1.0 - acting_team_win_prob
        return acting_team_win_prob

    def search(
        self,
        current_heroes: list[int],
        current_actions: list[int],
        radiant_picks: list[int],
        dire_picks: list[int],
        turn_idx: int,
        iterations: int = 1000,
        ucb_c: float = 1.414,
    ) -> tuple[int, float]:
        """Execute MCTS to find the best next move.

        Parameters
        ----------
        current_heroes : list[int]
            Hero IDs in draft order so far.
        current_actions : list[int]
            Action tokens so far.
        radiant_picks : list[int]
            Hero IDs already picked by Radiant.
        dire_picks : list[int]
            Hero IDs already picked by Dire.
        turn_idx : int
            Current turn index in the draft format.
        iterations : int
            Number of MCTS iterations to run.
        ucb_c : float
            Exploration constant for UCB1.

        Returns
        -------
        tuple[int, float]
            (best_hero_id, expected_win_probability)
        """
        root = MCTSNode(
            parent=None,
            hero_action=None,
            turn_idx=turn_idx,
            draft_format=self.draft_format,
        )
        root.untried_heroes = [h for h in self.valid_heroes if h not in current_heroes]
        random.shuffle(root.untried_heroes)

        logger.info(
            "MCTS: starting %d iterations from turn %d with %d available heroes",
            iterations, turn_idx, len(root.untried_heroes),
        )

        for _ in range(iterations):
            node = root

            # 1. SELECTION (traverse down fully expanded nodes using UCB1)
            while not node.is_terminal and len(node.untried_heroes) == 0:
                node = max(node.children.values(), key=lambda c: c.ucb1(ucb_c))

            # 2. EXPANSION (add a new node to the tree)
            if not node.is_terminal and len(node.untried_heroes) > 0:
                hero_to_try = node.untried_heroes.pop()
                child_node = MCTSNode(
                    parent=node,
                    hero_action=hero_to_try,
                    turn_idx=node.turn_idx + 1,
                    draft_format=self.draft_format,
                )

                # Setup untried heroes for the child
                path_heroes, _, _, _ = self._get_state_from_node(
                    child_node, current_heroes, current_actions, radiant_picks, dire_picks
                )
                child_node.untried_heroes = [
                    h for h in self.valid_heroes if h not in path_heroes
                ]
                random.shuffle(child_node.untried_heroes)

                node.children[hero_to_try] = child_node
                node = child_node

            # 3. EVALUATION (ask DraftBERT who is winning this partial draft)
            sim_heroes, sim_actions, sim_rad, sim_dire = self._get_state_from_node(
                node, current_heroes, current_actions, radiant_picks, dire_picks
            )
            radiant_win_prob = self._evaluate_state(
                sim_heroes, sim_actions, sim_rad, sim_dire, node
            )

            # 4. BACKPROPAGATION (update values up to the root)
            curr = node
            while curr is not None:
                curr.visits += 1
                curr.radiant_wins += radiant_win_prob
                curr = curr.parent

        # 5. Return the most visited child (most robust move)
        if not root.children:
            logger.warning("MCTS: no children expanded!")
            return -1, 0.5

        best_child = max(root.children.values(), key=lambda c: c.visits)

        # Calculate win prob from acting team's perspective
        rad_wr = best_child.radiant_wins / best_child.visits
        win_prob = rad_wr if root.is_radiant_turn else (1.0 - rad_wr)

        logger.info(
            "MCTS: best move = hero %d (%.2f%% win prob, %d visits)",
            best_child.hero_action,
            win_prob * 100,
            best_child.visits,
        )

        return best_child.hero_action, win_prob

    def get_top_k(
        self,
        current_heroes: list[int],
        current_actions: list[int],
        radiant_picks: list[int],
        dire_picks: list[int],
        turn_idx: int,
        iterations: int = 1000,
        top_k: int = 5,
        ucb_c: float = 1.414,
    ) -> list[dict[str, Any]]:
        """Run MCTS and return top-k suggestions with visit counts.

        Returns
        -------
        list[dict]
            Top-k suggestions with hero_id, win_probability, and visits.
        """
        root = MCTSNode(
            parent=None,
            hero_action=None,
            turn_idx=turn_idx,
            draft_format=self.draft_format,
        )
        root.untried_heroes = [h for h in self.valid_heroes if h not in current_heroes]
        random.shuffle(root.untried_heroes)

        for _ in range(iterations):
            node = root

            # Selection
            while not node.is_terminal and len(node.untried_heroes) == 0:
                node = max(node.children.values(), key=lambda c: c.ucb1(ucb_c))

            # Expansion
            if not node.is_terminal and len(node.untried_heroes) > 0:
                hero_to_try = node.untried_heroes.pop()
                child_node = MCTSNode(
                    parent=node,
                    hero_action=hero_to_try,
                    turn_idx=node.turn_idx + 1,
                    draft_format=self.draft_format,
                )
                path_heroes, _, _, _ = self._get_state_from_node(
                    child_node, current_heroes, current_actions, radiant_picks, dire_picks
                )
                child_node.untried_heroes = [
                    h for h in self.valid_heroes if h not in path_heroes
                ]
                random.shuffle(child_node.untried_heroes)
                node.children[hero_to_try] = child_node
                node = child_node

            # Evaluation
            sim_heroes, sim_actions, sim_rad, sim_dire = self._get_state_from_node(
                node, current_heroes, current_actions, radiant_picks, dire_picks
            )
            radiant_win_prob = self._evaluate_state(
                sim_heroes, sim_actions, sim_rad, sim_dire, node
            )

            # Backpropagation
            curr = node
            while curr is not None:
                curr.visits += 1
                curr.radiant_wins += radiant_win_prob
                curr = curr.parent

        # Sort children by visits
        sorted_children = sorted(
            root.children.values(), key=lambda c: c.visits, reverse=True
        )[:top_k]

        suggestions = []
        for child in sorted_children:
            rad_wr = child.radiant_wins / child.visits
            win_prob = rad_wr if root.is_radiant_turn else (1.0 - rad_wr)
            suggestions.append({
                "hero_id": child.hero_action,
                "win_probability": win_prob,
                "visits": child.visits,
            })

        return suggestions
