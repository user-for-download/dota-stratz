"""Tests for ``api/features.py`` — feature computation, batch pre-fetching,
and NULL-safe helpers.

Regression bugs covered:
    - BUG-013: hds_games is integer-valued float, not fractional
    - BUG-N03: team_id=0 treated as None (skips DB query)
    - BUG-007: account_id=0 should not be skipped (falsy but valid)
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from api.draft_state import DraftContext
from api.features import _float, _int, build_feature_vector, pre_fetch_batch


# ===================================================================
# _float helper
# ===================================================================


class TestFloat:
    """``_float`` safely converts values to float, returning sensible
    defaults for None / invalid inputs — prevents ``float(None)`` crashes
    when aggregate tables have NULL values.
    """

    def test_none_known_key(self):
        """✅ None + known key → correct default from _FLOAT_DEFAULTS."""
        assert _float(None, "win_rate") == 0.5
        assert _float(None, "hds_win_rate") == 0.5

    def test_none_unknown_key(self):
        """✅ None + unknown key → 0.0."""
        assert _float(None, "nonexistent") == 0.0

    def test_valid_numeric(self):
        """✅ Valid numeric input → same value as float."""
        assert _float(42) == 42.0
        assert _float(3.14) == 3.14
        assert _float("3.14") == 3.14

    def test_non_castable_input_abc(self):
        """❌ Non-castable string ``"abc"`` → default."""
        assert _float("abc", "win_rate") == 0.5
        assert _float("abc", "unknown_key") == 0.0


# ===================================================================
# _int helper
# ===================================================================


class TestInt:
    """``_int`` safely converts values to int, returning sensible
    defaults for None / invalid inputs.
    """

    def test_none_known_key(self):
        """✅ None + known key → correct default from _INT_DEFAULTS."""
        assert _int(None, "games") == 0
        assert _int(None, "total_picks") == 0

    def test_none_unknown_key(self):
        """✅ None + unknown key → 0."""
        assert _int(None, "nonexistent") == 0

    def test_valid_numeric(self):
        """✅ Valid numeric input → same value as int."""
        assert _int(42) == 42
        assert _int(3.14) == 3
        assert _int("42") == 42

    def test_non_castable_input_abc(self):
        """❌ Non-castable string ``"abc"`` → default."""
        assert _int("abc", "games") == 0
        assert _int("abc", "unknown_key") == 0


# ===================================================================
# build_feature_vector
# ===================================================================


class TestBuildFeatureVector:
    """``build_feature_vector`` constructs the full feature vector:
    numeric aggregate columns + one-hot hero encoding.

    The output shape must be ``(len(aggregate_columns) + max_hero_id,)``
    and individual column values must match the contract established by
    the trainer's ``extract_features``.
    """

    # -- shape & one-hot -------------------------------------------------

    def test_output_shape(self, feature_schema, batch_context_factory,
                          draft_context_factory):
        """✅ Output shape = (len(agg_cols) + max_hero_id,)."""
        schema = {**feature_schema, "max_hero_id": 160}
        batch = batch_context_factory()
        ctx = draft_context_factory()
        fv = build_feature_vector(
            hero_id=1, ctx=ctx, patch_id=1, batch=batch, schema=schema,
        )
        expected = len(schema["aggregate_columns"]) + 160
        assert fv.shape == (expected,), (
            f"Expected shape ({expected},), got {fv.shape}"
        )

    def test_one_hot_correct_position(self, feature_schema,
                                      batch_context_factory,
                                      draft_context_factory):
        """✅ One-hot at output[len(agg_cols) + hero_id - 1] == 1.0,
        all others == 0.0.
        """
        schema = {**feature_schema, "max_hero_id": 160}
        batch = batch_context_factory()
        ctx = draft_context_factory()
        hero_id = 42
        fv = build_feature_vector(
            hero_id=hero_id, ctx=ctx, patch_id=1, batch=batch, schema=schema,
        )
        n_agg = len(schema["aggregate_columns"])
        onehot = fv[n_agg:]
        assert onehot[hero_id - 1] == 1.0
        assert float(onehot.sum()) == 1.0, "Only one one-hot position should be 1"

    def test_hero_id_0_all_zero_onehot(self, feature_schema,
                                       batch_context_factory,
                                       draft_context_factory):
        """❌ hero_id=0 → all-zero one-hot (out of range)."""
        schema = {**feature_schema, "max_hero_id": 160}
        batch = batch_context_factory()
        ctx = draft_context_factory()
        fv = build_feature_vector(
            hero_id=0, ctx=ctx, patch_id=1, batch=batch, schema=schema,
        )
        n_agg = len(schema["aggregate_columns"])
        onehot = fv[n_agg:]
        assert float(onehot.sum()) == 0.0

    def test_hero_id_exceeds_max_all_zero_onehot(self, feature_schema,
                                                  batch_context_factory,
                                                  draft_context_factory):
        """❌ hero_id=161 (exceeds max_hero_id) → all-zero one-hot."""
        schema = {**feature_schema, "max_hero_id": 160}
        batch = batch_context_factory()
        ctx = draft_context_factory()
        fv = build_feature_vector(
            hero_id=161, ctx=ctx, patch_id=1, batch=batch, schema=schema,
        )
        n_agg = len(schema["aggregate_columns"])
        onehot = fv[n_agg:]
        assert float(onehot.sum()) == 0.0

    # -- hds_games integer-valued (BUG-013) -----------------------------

    def test_hds_games_is_integer_valued_float(self, feature_schema,
                                               batch_context_factory,
                                               draft_context_factory):
        """✅ hds_games is integer-valued (3.0, not 3.14) — REGRESSION
        BUG-013.
        """
        schema = {**feature_schema, "max_hero_id": 160}
        batch = batch_context_factory(
            hero_draft_slot={1: (0.55, 3)},
        )
        ctx = draft_context_factory()
        fv = build_feature_vector(
            hero_id=1, ctx=ctx, patch_id=1, batch=batch, schema=schema,
        )
        idx = schema["aggregate_columns"].index("hds_games")
        val = fv[idx]
        assert val == 3.0, f"Expected hds_games=3.0, got {val}"
        assert val == float(int(val)), "hds_games must not be fractional"

    def test_hds_games_default_zero(self, feature_schema,
                                    batch_context_factory,
                                    draft_context_factory):
        """✅ hds_games defaults to 0.0 when no data available."""
        schema = {**feature_schema, "max_hero_id": 160}
        batch = batch_context_factory()  # empty hero_draft_slot
        ctx = draft_context_factory()
        fv = build_feature_vector(
            hero_id=1, ctx=ctx, patch_id=1, batch=batch, schema=schema,
        )
        idx = schema["aggregate_columns"].index("hds_games")
        assert fv[idx] == 0.0

    # -- is_pick --------------------------------------------------------

    def test_is_pick_always_1(self, feature_schema, batch_context_factory,
                              draft_context_factory):
        """✅ ``is_pick`` is always 1.0 during inference."""
        schema = {**feature_schema, "max_hero_id": 160}
        batch = batch_context_factory()
        ctx = draft_context_factory()
        fv = build_feature_vector(
            hero_id=1, ctx=ctx, patch_id=1, batch=batch, schema=schema,
        )
        idx = schema["aggregate_columns"].index("is_pick")
        assert fv[idx] == 1.0

    # -- empty batch context defaults -----------------------------------

    def test_empty_batch_context_win_rate_defaults_to_05(
            self, feature_schema, batch_context_factory,
            draft_context_factory):
        """✅ Empty BatchContext → win-rate fields default to 0.5."""
        schema = {**feature_schema, "max_hero_id": 160}
        batch = batch_context_factory()
        ctx = draft_context_factory()
        fv = build_feature_vector(
            hero_id=1, ctx=ctx, patch_id=1, batch=batch, schema=schema,
        )
        for col in ("th_win_rate", "ph_win_rate", "bl_win_rate",
                    "hds_win_rate", "h2h_win_rate", "sy_avg_win_rate",
                    "co_avg_win_rate"):
            idx = schema["aggregate_columns"].index(col)
            assert fv[idx] == 0.5, f"Expected {col}=0.5, got {fv[idx]}"

    def test_empty_batch_context_counts_default_to_0(
            self, feature_schema, batch_context_factory,
            draft_context_factory):
        """✅ Empty BatchContext → count fields default to 0.0."""
        schema = {**feature_schema, "max_hero_id": 160}
        batch = batch_context_factory()
        ctx = draft_context_factory()
        fv = build_feature_vector(
            hero_id=1, ctx=ctx, patch_id=1, batch=batch, schema=schema,
        )
        for col in ("sy_n_teammates", "co_n_enemies", "hds_games",
                    "ph_games", "th_games", "h2h_games"):
            idx = schema["aggregate_columns"].index(col)
            assert fv[idx] == 0.0, f"Expected {col}=0.0, got {fv[idx]}"

    # -- ph_is_new_player / th_is_new_team_hero -------------------------

    def test_ph_is_new_player_when_games_below_5(
            self, feature_schema, batch_context_factory,
            draft_context_factory):
        """✅ ph_is_new_player=1.0 when ph_games < 5, else 0.0."""
        schema = {**feature_schema, "max_hero_id": 160}
        ctx = draft_context_factory()
        idx = schema["aggregate_columns"].index("ph_is_new_player")

        # ph_games = 3 (< 5) → 1.0
        batch = batch_context_factory(
            player_hero_agg={1: {"games": 3}},
        )
        fv = build_feature_vector(
            hero_id=1, ctx=ctx, patch_id=1, batch=batch, schema=schema,
        )
        assert fv[idx] == 1.0

        # ph_games = 5 (= threshold) → 0.0
        batch = batch_context_factory(
            player_hero_agg={1: {"games": 5}},
        )
        fv = build_feature_vector(
            hero_id=1, ctx=ctx, patch_id=1, batch=batch, schema=schema,
        )
        assert fv[idx] == 0.0

    def test_th_is_new_team_hero_when_games_below_5(
            self, feature_schema, batch_context_factory,
            draft_context_factory):
        """✅ th_is_new_team_hero=1.0 when th_games < 5, else 0.0."""
        schema = {**feature_schema, "max_hero_id": 160}
        ctx = draft_context_factory()
        idx = schema["aggregate_columns"].index("th_is_new_team_hero")

        batch = batch_context_factory(
            team_hero_agg={1: {"games": 2}},
        )
        fv = build_feature_vector(
            hero_id=1, ctx=ctx, patch_id=1, batch=batch, schema=schema,
        )
        assert fv[idx] == 1.0

        batch = batch_context_factory(
            team_hero_agg={1: {"games": 5}},
        )
        fv = build_feature_vector(
            hero_id=1, ctx=ctx, patch_id=1, batch=batch, schema=schema,
        )
        assert fv[idx] == 0.0

    # -- relative win rates ---------------------------------------------

    def test_rel_th_win_rate_difference(self, feature_schema,
                                        batch_context_factory,
                                        draft_context_factory):
        """✅ rel_th_win_rate = th_win_rate - bl_win_rate."""
        schema = {**feature_schema, "max_hero_id": 160}
        batch = batch_context_factory(
            team_hero_agg={1: {"win_rate": 0.62}},
            baselines={1: {"win_rate": 0.50}},
        )
        ctx = draft_context_factory()
        fv = build_feature_vector(
            hero_id=1, ctx=ctx, patch_id=1, batch=batch, schema=schema,
        )
        idx = schema["aggregate_columns"].index("rel_th_win_rate")
        assert fv[idx] == pytest.approx(0.12)

    def test_rel_ph_win_rate_difference(self, feature_schema,
                                        batch_context_factory,
                                        draft_context_factory):
        """✅ rel_ph_win_rate = ph_win_rate - bl_win_rate."""
        schema = {**feature_schema, "max_hero_id": 160}
        batch = batch_context_factory(
            player_hero_agg={1: {"win_rate": 0.58}},
            baselines={1: {"win_rate": 0.50}},
        )
        ctx = draft_context_factory()
        fv = build_feature_vector(
            hero_id=1, ctx=ctx, patch_id=1, batch=batch, schema=schema,
        )
        idx = schema["aggregate_columns"].index("rel_ph_win_rate")
        assert fv[idx] == pytest.approx(0.08)

    # -- role interaction features --------------------------------------

    def test_ph_vision_support_score_non_zero_only_when_lane_role_5(
            self, feature_schema, batch_context_factory,
            draft_context_factory):
        """✅ ph_vision_support_score non-zero only when
        ph_lane_role == 5 (support).
        """
        schema = {**feature_schema, "max_hero_id": 160}
        ctx = draft_context_factory()
        idx = schema["aggregate_columns"].index("ph_vision_support_score")

        # lane_role=5 → non-zero
        batch = batch_context_factory(
            player_hero_agg={1: {"lane_role": 5, "avg_vision_placed": 42.0}},
        )
        fv = build_feature_vector(
            hero_id=1, ctx=ctx, patch_id=1, batch=batch, schema=schema,
        )
        assert fv[idx] == 42.0

        # lane_role=1 (carry) → zero
        batch = batch_context_factory(
            player_hero_agg={1: {"lane_role": 1, "avg_vision_placed": 42.0}},
        )
        fv = build_feature_vector(
            hero_id=1, ctx=ctx, patch_id=1, batch=batch, schema=schema,
        )
        assert fv[idx] == 0.0

    def test_ph_gpm_carry_score_non_zero_only_when_lane_role_1(
            self, feature_schema, batch_context_factory,
            draft_context_factory):
        """✅ ph_gpm_carry_score non-zero only when
        ph_lane_role == 1 (carry).
        """
        schema = {**feature_schema, "max_hero_id": 160}
        ctx = draft_context_factory()
        idx = schema["aggregate_columns"].index("ph_gpm_carry_score")

        # lane_role=1 → non-zero
        batch = batch_context_factory(
            player_hero_agg={1: {"lane_role": 1, "avg_gpm": 650.0}},
        )
        fv = build_feature_vector(
            hero_id=1, ctx=ctx, patch_id=1, batch=batch, schema=schema,
        )
        assert fv[idx] == 650.0

        # lane_role=5 (support) → zero
        batch = batch_context_factory(
            player_hero_agg={1: {"lane_role": 5, "avg_gpm": 650.0}},
        )
        fv = build_feature_vector(
            hero_id=1, ctx=ctx, patch_id=1, batch=batch, schema=schema,
        )
        assert fv[idx] == 0.0


# ===================================================================
# pre_fetch_batch
# ===================================================================


class TestPreFetchBatch:
    """``pre_fetch_batch`` bulk-fetches aggregate data in 7 batched
    queries and returns a ``BatchContext``.

    Edge cases around falsy-but-valid identifiers are covered to prevent
    regressions (BUG-N03, BUG-007).
    """

    def test_team_id_none_returns_empty_team_agg(self):
        """✅ team_id=None → team_hero_agg={}, no db call."""
        with mock.patch("api.features.db_") as mock_db:
            mock_db.fetch_baselines_batch.return_value = {}
            result = pre_fetch_batch(
                patch_id=1,
                hero_ids=[1, 2],
                team_id=None,
                enemy_team_id=None,
                ctx=None,
                account_id=None,
            )
            assert result.team_hero_agg == {}
            mock_db.fetch_team_hero_agg_batch.assert_not_called()

    def test_team_id_zero_no_db_call(self):
        """❌ team_id=0 → team_hero_agg={}, no db call —
        REGRESSION BUG-N03 (0 is falsy but not None, should be treated
        as "not available").
        """
        with mock.patch("api.features.db_") as mock_db:
            mock_db.fetch_baselines_batch.return_value = {}
            result = pre_fetch_batch(
                patch_id=1,
                hero_ids=[1, 2],
                team_id=0,
                enemy_team_id=None,
                ctx=None,
                account_id=None,
            )
            assert result.team_hero_agg == {}
            mock_db.fetch_team_hero_agg_batch.assert_not_called()

    def test_account_id_none_returns_empty_player_agg(self):
        """✅ account_id=None → player_hero_agg={}, no db call."""
        with mock.patch("api.features.db_") as mock_db:
            mock_db.fetch_baselines_batch.return_value = {}
            result = pre_fetch_batch(
                patch_id=1,
                hero_ids=[1, 2],
                team_id=None,
                enemy_team_id=None,
                ctx=None,
                account_id=None,
            )
            assert result.player_hero_agg == {}
            mock_db.fetch_player_hero_agg_batch.assert_not_called()

    def test_account_id_zero_executes_query(self):
        """❌ account_id=0 → query IS executed —
        REGRESSION BUG-007 (0 is falsy but a valid account_id).
        """
        with mock.patch("api.features.db_") as mock_db:
            mock_db.fetch_baselines_batch.return_value = {}
            mock_db.fetch_player_hero_agg_batch.return_value = {}
            result = pre_fetch_batch(
                patch_id=1,
                hero_ids=[1, 2],
                team_id=None,
                enemy_team_id=None,
                ctx=None,
                account_id=0,
            )
            mock_db.fetch_player_hero_agg_batch.assert_called_once_with(
                1, 0, [1, 2],
            )

    def test_ctx_none_returns_empty_synergy_and_counter(self):
        """✅ ctx=None → synergy={}, counter={}, no db calls."""
        with mock.patch("api.features.db_") as mock_db:
            mock_db.fetch_baselines_batch.return_value = {}
            result = pre_fetch_batch(
                patch_id=1,
                hero_ids=[1, 2],
                team_id=None,
                enemy_team_id=None,
                ctx=None,
                account_id=None,
            )
            assert result.synergy == {}
            assert result.counter == {}
            mock_db.fetch_synergy_batch.assert_not_called()
            mock_db.fetch_counter_batch.assert_not_called()

    def test_team_pick_ordinal_from_ally_picks(self):
        """✅ team_pick_ordinal = len(ctx.ally_picks) + 1."""
        ctx = DraftContext(
            turn=3,
            recommending_team=0,
            is_pick_turn=True,
            radiant_picks=[1, 5],  # ally_picks=[1, 5] for recommending_team=0
        )
        with mock.patch("api.features.db_") as mock_db:
            mock_db.fetch_baselines_batch.return_value = {}
            mock_db.fetch_hero_draft_slot_batch.return_value = {}
            result = pre_fetch_batch(
                patch_id=1,
                hero_ids=[1, 2],
                team_id=None,
                enemy_team_id=None,
                ctx=ctx,
                account_id=None,
            )
            # ally_picks = [1, 5] → len=2 → ordinal=3
            mock_db.fetch_hero_draft_slot_batch.assert_called_once_with(
                1, [1, 2], 3,
            )

    def test_no_hero_ids_returns_empty_baselines(self):
        """✅ Empty hero_ids list → baselines={}, no fatal error."""
        with mock.patch("api.features.db_") as mock_db:
            mock_db.fetch_baselines_batch.return_value = {}
            result = pre_fetch_batch(
                patch_id=1,
                hero_ids=[],
                team_id=None,
                enemy_team_id=None,
                ctx=None,
                account_id=None,
            )
            assert result.baselines == {}
            mock_db.fetch_baselines_batch.assert_not_called()
