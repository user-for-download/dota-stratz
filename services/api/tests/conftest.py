"""Shared fixtures for the inference API test suite."""

from __future__ import annotations

from typing import Any

import pytest

from api.draft_state import DraftContext
from api.features import BatchContext


# ---------------------------------------------------------------------------
# Feature schema fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def feature_schema() -> dict[str, Any]:
    """Return a minimal feature_schema dict matching what the trainer writes.

    The ``aggregate_columns`` list mirrors the column order used in
    ``build_feature_vector``.  This minimal set is sufficient for any test
    that needs a schema for shape/order verification; full integration tests
    against a trained model would load the real ``feature_schema.json``.
    """
    return {
        "feature_version": 2,
        "aggregate_columns": [
            "is_pick",
            "team",
            # -- team-hero --
            "th_games",
            "th_wins",
            "th_win_rate",
            "th_bans",
            "th_avg_gpm",
            "th_avg_xpm",
            "th_avg_kills",
            "th_avg_deaths",
            "th_avg_assists",
            "th_firstblood_rate",
            "th_avg_camps_stacked",
            "th_avg_vision_placed",
            "th_avg_gold_10",
            "th_avg_xp_10",
            # -- player-hero --
            "ph_games",
            "ph_wins",
            "ph_win_rate",
            "ph_avg_gpm",
            "ph_avg_xpm",
            "ph_avg_kills",
            "ph_avg_deaths",
            "ph_avg_assists",
            "ph_avg_kda",
            "ph_lane_role",
            "ph_firstblood_rate",
            "ph_avg_camps_stacked",
            "ph_avg_vision_placed",
            "ph_avg_gold_10",
            "ph_avg_xp_10",
            # -- synergy --
            "sy_avg_win_rate",
            "sy_n_teammates",
            # -- counter --
            "co_avg_win_rate",
            "co_n_enemies",
            # -- head-to-head --
            "h2h_win_rate",
            "h2h_games",
            # -- hero draft-slot --
            "hds_win_rate",
            "hds_games",
            # -- hero baseline --
            "bl_total_picks",
            "bl_total_wins",
            "bl_total_bans",
            "bl_win_rate",
            "bl_pick_rate",
            "bl_ban_rate",
            "bl_avg_gpm",
            "bl_avg_xpm",
            "bl_avg_kills",
            "bl_avg_deaths",
            "bl_avg_assists",
            "bl_avg_gold_10",
            "bl_avg_xp_10",
            # -- derived / meta --
            "ph_is_new_player",
            "th_is_new_team_hero",
            "rel_th_win_rate",
            "rel_ph_win_rate",
            "ph_vision_support_score",
            "ph_gpm_carry_score",
        ],
        "num_aggregates": 56,
    }


# ---------------------------------------------------------------------------
# BatchContext factory fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def batch_context_factory():
    """Return a factory that builds a ``BatchContext`` with sensible defaults.

    Usage::

        ctx = batch_context_factory()
        ctx = batch_context_factory(baselines={1: {"win_rate": 0.55, ...}})
    """

    def _build(**overrides: Any) -> BatchContext:
        defaults: dict[str, Any] = {
            "baselines": {},
            "team_hero_agg": {},
            "player_hero_agg": {},
            "synergy": {},
            "counter": {},
            "h2h_row": None,
            "hero_draft_slot": {},
        }
        defaults.update(overrides)
        return BatchContext(**defaults)

    return _build


# ---------------------------------------------------------------------------
# DraftContext factory fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def draft_context_factory():
    """Return a factory that builds a ``DraftContext`` with sensible defaults.

    The default context represents Radiant (team 0) about to make a pick
    on turn 7 with one hero already taken per section::

        radiant_picks=[1], dire_picks=[2],
        radiant_bans=[3], dire_bans=[4],

    Usage::

        ctx = draft_context_factory()
        ctx = draft_context_factory(radiant_picks=[1, 5], recommending_team=1)
    """

    def _build(**overrides: Any) -> DraftContext:
        defaults: dict[str, Any] = {
            "turn": 7,
            "recommending_team": 0,
            "is_pick_turn": True,
            "radiant_picks": [1],
            "dire_picks": [2],
            "radiant_bans": [3],
            "dire_bans": [4],
        }
        defaults.update(overrides)
        return DraftContext(**defaults)

    return _build


# ---------------------------------------------------------------------------
# Integration-test marker
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """Register the ``integration`` marker to avoid ``PytestUnknownMarkWarning``."""
    config.addinivalue_line("markers", "integration: marks tests that require a real Postgres database.")
