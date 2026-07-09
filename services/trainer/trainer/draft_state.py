"""Fast in-memory feature builder for hypothetical draft picks.

Generates the exact 59-dim continuous feature array that DraftBERT expects,
without querying PostgreSQL. Used by the Greedy/MCTS bots to evaluate
~120 hypothetical picks in a single batched forward pass.

The feature ordering matches trainer/features.py::feature_column_names(include_onehot=False).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .features import feature_column_names
from .inference_cache import InferenceCache

logger = logging.getLogger(__name__)


class DraftStateBuilder:
    """Builds 59-dim tabular feature vectors for hypothetical picks.

    Mirrors the SQL logic in trainer/features.py but operates entirely
    in RAM using the InferenceCache.
    """

    def __init__(self, cache: InferenceCache):
        self.cache = cache
        # The exact order of columns the model was trained on (59 features)
        self.feature_names = feature_column_names(include_onehot=False)
        self.num_features = len(self.feature_names)

        # Build an index map for O(1) array updates
        # e.g. self.col_idx["th_win_rate"] -> 4
        self.col_idx: dict[str, int] = {
            name: i for i, name in enumerate(self.feature_names)
        }

    def build_tabular_features(
        self,
        hypothetical_hero_id: int,
        is_radiant_turn: bool,
        is_pick: bool,
        radiant_picks: list[int],
        dire_picks: list[int],
        team_id: int | None = None,
        account_id: int | None = None,
        enemy_team_id: int | None = None,
        team_pick_ordinal: int | None = None,
    ) -> np.ndarray:
        """Build the 59-dim array for a single hypothetical pick/ban.

        Parameters
        ----------
        hypothetical_hero_id : int
            The hero we are considering drafting.
        is_radiant_turn : bool
            True if Radiant is picking/banning, False for Dire.
        is_pick : bool
            True if it's a pick, False if it's a ban.
        radiant_picks : list[int]
            Hero IDs already PICKED by Radiant.
        dire_picks : list[int]
            Hero IDs already PICKED by Dire.
        team_id : int, optional
            The Pro Team ID making the pick.
        account_id : int, optional
            The Player ID making the pick.
        enemy_team_id : int, optional
            The opposing team ID (for H2H lookup).
        team_pick_ordinal : int, optional
            The pick position (1-5) within the team.
        """
        feat = np.zeros(self.num_features, dtype=np.float32)
        idx = self.col_idx

        # --- 1. Draft Context ---
        feat[idx["is_pick"]] = 1.0 if is_pick else 0.0
        feat[idx["team"]] = 0.0 if is_radiant_turn else 1.0

        # --- 2. Hero Baseline ---
        bl = self.cache.get_baseline(hypothetical_hero_id)
        feat[idx["bl_total_picks"]] = bl.get("total_picks", 0)
        feat[idx["bl_total_wins"]] = bl.get("total_wins", 0)
        feat[idx["bl_total_bans"]] = bl.get("total_bans", 0)
        feat[idx["bl_win_rate"]] = bl.get("win_rate", 0.5)
        feat[idx["bl_pick_rate"]] = bl.get("pick_rate", 0.0)
        feat[idx["bl_ban_rate"]] = bl.get("ban_rate", 0.0)
        feat[idx["bl_avg_gpm"]] = bl.get("avg_gpm", 0.0)
        feat[idx["bl_avg_xpm"]] = bl.get("avg_xpm", 0.0)
        feat[idx["bl_avg_kills"]] = bl.get("avg_kills", 0.0)
        feat[idx["bl_avg_deaths"]] = bl.get("avg_deaths", 0.0)
        feat[idx["bl_avg_assists"]] = bl.get("avg_assists", 0.0)
        feat[idx["bl_avg_gold_10"]] = bl.get("avg_gold_10", 0.0)
        feat[idx["bl_avg_xp_10"]] = bl.get("avg_xp_10", 0.0)

        # --- 3. Synergies (Only applies to PICKS) ---
        if is_pick:
            ally_picks = radiant_picks if is_radiant_turn else dire_picks
            sy_wins = 0.0
            sy_games = 0

            for ally_id in ally_picks:
                sy_data = self.cache.get_synergy(hypothetical_hero_id, ally_id)
                # Weight the average by games played to match SQL logic
                sy_wins += sy_data["win_rate"] * sy_data["games"]
                sy_games += sy_data["games"]

            feat[idx["sy_n_teammates"]] = len(ally_picks)
            if sy_games > 0:
                feat[idx["sy_avg_win_rate"]] = sy_wins / sy_games
            else:
                feat[idx["sy_avg_win_rate"]] = 0.5

        # --- 4. Counters (Only applies to PICKS) ---
        if is_pick:
            enemy_picks = dire_picks if is_radiant_turn else radiant_picks
            co_wins = 0.0
            co_kd_diff = 0.0
            co_games = 0

            for enemy_id in enemy_picks:
                co_data = self.cache.get_counter(hypothetical_hero_id, enemy_id)
                co_wins += co_data["win_rate"] * co_data["games"]
                co_kd_diff += co_data["avg_kd_diff"] * co_data["games"]
                co_games += co_data["games"]

            feat[idx["co_n_enemies"]] = len(enemy_picks)
            if co_games > 0:
                feat[idx["co_avg_win_rate"]] = co_wins / co_games
                feat[idx["co_avg_kd_diff"]] = co_kd_diff / co_games
            else:
                feat[idx["co_avg_win_rate"]] = 0.5

        # --- 5. Team-Hero (If Pro Match) ---
        if team_id:
            th = self.cache.get_team_hero(team_id, hypothetical_hero_id)
            feat[idx["th_games"]] = th.get("games", 0)
            feat[idx["th_wins"]] = th.get("wins", 0)
            feat[idx["th_win_rate"]] = th.get("win_rate", 0.5)
            feat[idx["th_bans"]] = th.get("bans", 0)
            feat[idx["th_avg_gpm"]] = th.get("avg_gpm", 0.0)
            feat[idx["th_avg_xpm"]] = th.get("avg_xpm", 0.0)
            feat[idx["th_avg_kills"]] = th.get("avg_kills", 0.0)
            feat[idx["th_avg_deaths"]] = th.get("avg_deaths", 0.0)
            feat[idx["th_avg_assists"]] = th.get("avg_assists", 0.0)
            feat[idx["th_firstblood_rate"]] = th.get("firstblood_rate", 0.0)
            feat[idx["th_avg_camps_stacked"]] = th.get("avg_camps_stacked", 0.0)
            feat[idx["th_avg_vision_placed"]] = th.get("avg_vision_placed", 0.0)
            feat[idx["th_avg_gold_10"]] = th.get("avg_gold_10", 0.0)
            feat[idx["th_avg_xp_10"]] = th.get("avg_xp_10", 0.0)

            feat[idx["th_is_new_team_hero"]] = 1.0 if th.get("games", 0) < 5 else 0.0
            feat[idx["rel_th_win_rate"]] = feat[idx["th_win_rate"]] - feat[idx["bl_win_rate"]]
        else:
            feat[idx["th_win_rate"]] = 0.5
            feat[idx["th_is_new_team_hero"]] = 1.0
            feat[idx["rel_th_win_rate"]] = 0.0

        # --- 6. Player-Hero (If player known) ---
        if account_id:
            ph = self.cache.get_player_hero(account_id, hypothetical_hero_id)
            feat[idx["ph_games"]] = ph.get("games", 0)
            feat[idx["ph_wins"]] = ph.get("wins", 0)
            feat[idx["ph_win_rate"]] = ph.get("win_rate", 0.5)
            feat[idx["ph_avg_gpm"]] = ph.get("avg_gpm", 0.0)
            feat[idx["ph_avg_xpm"]] = ph.get("avg_xpm", 0.0)
            feat[idx["ph_avg_kills"]] = ph.get("avg_kills", 0.0)
            feat[idx["ph_avg_deaths"]] = ph.get("avg_deaths", 0.0)
            feat[idx["ph_avg_assists"]] = ph.get("avg_assists", 0.0)
            feat[idx["ph_avg_kda"]] = ph.get("avg_kda", 0.0)
            feat[idx["ph_lane_role"]] = ph.get("lane_role", 0)
            feat[idx["ph_firstblood_rate"]] = ph.get("firstblood_rate", 0.0)
            feat[idx["ph_avg_camps_stacked"]] = ph.get("avg_camps_stacked", 0.0)
            feat[idx["ph_avg_vision_placed"]] = ph.get("avg_vision_placed", 0.0)
            feat[idx["ph_avg_gold_10"]] = ph.get("avg_gold_10", 0.0)
            feat[idx["ph_avg_xp_10"]] = ph.get("avg_xp_10", 0.0)

            feat[idx["ph_is_new_player"]] = 1.0 if ph.get("games", 0) < 5 else 0.0
            feat[idx["rel_ph_win_rate"]] = feat[idx["ph_win_rate"]] - feat[idx["bl_win_rate"]]

            # Role interactions
            lr = ph.get("lane_role", 0)
            feat[idx["ph_vision_support_score"]] = ph.get("avg_vision_placed", 0.0) if lr == 5 else 0.0
            feat[idx["ph_gpm_carry_score"]] = ph.get("avg_gpm", 0.0) if lr == 1 else 0.0
        else:
            feat[idx["ph_win_rate"]] = 0.5
            feat[idx["ph_is_new_player"]] = 1.0
            feat[idx["rel_ph_win_rate"]] = 0.0

        # --- 7. Team H2H (If both teams known) ---
        if team_id and enemy_team_id:
            h2h = self.cache.get_h2h(team_id, enemy_team_id)
            feat[idx["h2h_win_rate"]] = h2h.get("win_rate", 0.5)
            feat[idx["h2h_games"]] = h2h.get("games", 0)
        else:
            feat[idx["h2h_win_rate"]] = 0.5
            feat[idx["h2h_games"]] = 0

        # --- 8. Hero Draft Slot (If pick ordinal known) ---
        if team_pick_ordinal is not None and is_pick:
            hds = self.cache.get_draft_slot(hypothetical_hero_id, team_pick_ordinal)
            feat[idx["hds_win_rate"]] = hds.get("win_rate", 0.5)
            feat[idx["hds_games"]] = hds.get("games", 0)
        else:
            feat[idx["hds_win_rate"]] = 0.5
            feat[idx["hds_games"]] = 0

        # --- 9. Macro Composition & Pick Propensity ---
        ally_picks = radiant_picks if is_radiant_turn else dire_picks
        team_gpm = bl.get("avg_gpm", 0.0)
        team_xpm = bl.get("avg_xpm", 0.0)
        for ally_id in ally_picks:
            ally_bl = self.cache.get_baseline(ally_id)
            team_gpm += ally_bl.get("avg_gpm", 0.0)
            team_xpm += ally_bl.get("avg_xpm", 0.0)
        feat[idx["team_gpm_budget"]] = team_gpm
        feat[idx["team_xpm_budget"]] = team_xpm
        bl_picks = bl.get("total_picks", 0)
        th_games = th.get("games", 0) if team_id else 0
        feat[idx["team_pick_propensity"]] = th_games / bl_picks if bl_picks > 0 else 0.0

        return feat
