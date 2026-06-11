"""Tests for feature computation functions — ``make_target``,
``extract_features``, and ``feature_column_names``.

All tests use the shared fixtures from ``conftest.py`` (``sample_df``,
``mock_engine``, ``trainer_config``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trainer.features import (
    extract_features,
    feature_column_names,
    make_target,
)


# ===========================================================================
# make_target
# ===========================================================================

class TestMakeTarget:
    """``make_target`` — ``(df["radiant_win"] == (df["team"] == 0)).astype(int).values``

    The target is relative to the *picker's* team, not absolute radiant_win:

        team=0 (Radiant): target = radiant_win
        team=1 (Dire):    target = 1 - radiant_win
    """

    @pytest.mark.parametrize(
        ("team", "radiant_win", "expected"),
        [
            # team=0, radiant_win=True  → (True == True)  → True  → 1
            (0, True, 1),
            # team=0, radiant_win=False → (False == True) → False → 0
            (0, False, 0),
            # team=1, radiant_win=True  → (True == False) → False → 0
            (1, True, 0),
            # team=1, radiant_win=False → (False == False) → True  → 1
            (1, False, 1),
        ],
    )
    def test_target_from_team_and_radiant_win(
        self,
        team: int,
        radiant_win: bool,
        expected: int,
    ) -> None:
        """Positive + negative: target correctly reflects picker's win."""
        df = pd.DataFrame({"team": [team], "radiant_win": [radiant_win]})
        result = make_target(df)
        assert result[0] == expected
        assert result.dtype == np.int64


# ===========================================================================
# extract_features
# ===========================================================================

class TestExtractFeatures:
    """``extract_features`` — builds the full feature matrix including
    one-hot hero encoding from the raw SQL result DataFrame.
    """

    def test_shape_matches_agg_plus_onehot(self, sample_df: pd.DataFrame) -> None:
        """✅ Positive: output shape = (n_rows, len(agg_cols) + max_hero_id)."""
        agg_cols = feature_column_names(include_onehot=False)
        max_hero_id = 160
        result = extract_features(sample_df, agg_cols, max_hero_id)
        expected_n_cols = len(agg_cols) + max_hero_id
        assert result.shape == (len(sample_df), expected_n_cols), (
            f"Expected ({len(sample_df)}, {expected_n_cols}), "
            f"got {result.shape}"
        )

    def test_onehot_is_binary_and_sums_to_one(self, sample_df: pd.DataFrame) -> None:
        """✅ Positive: one-hot columns contain only 0/1 and sum to 1 per row."""
        agg_cols = feature_column_names(include_onehot=False)
        max_hero_id = 160
        result = extract_features(sample_df, agg_cols, max_hero_id)
        onehot = result[:, len(agg_cols):]

        assert np.all((onehot == 0) | (onehot == 1)), (
            "One-hot section contains non-binary values"
        )
        row_sums = onehot.sum(axis=1)
        assert np.all(row_sums == 1), (
            f"Each row must sum to 1 in one-hot section; got sums {row_sums}"
        )

    def test_fillna_replaces_nan_with_zero(self, sample_df: pd.DataFrame) -> None:
        """✅ fillna(0): NaN in feature columns becomes 0 after extraction.

        ``sample_df`` has a NaN injected at ``[0, AGG_COLUMNS[0]]``.
        """
        agg_cols = feature_column_names(include_onehot=False)
        result = extract_features(sample_df, agg_cols, max_hero_id=160)
        # The first column of the numeric block corresponds to agg_cols[0],
        # which was NaN in row 0 of sample_df → must be 0 after fillna.
        assert result[0, 0] == 0.0, (
            "NaN was not replaced with 0.0 by fillna"
        )


# ===========================================================================
# feature_column_names
# ===========================================================================

class TestFeatureColumnNames:
    """``feature_column_names`` — column-ordering contract between
    training and inference.
    """

    N_AGG_COLS: int = len(feature_column_names(include_onehot=False))
    """Number of aggregate (non-one-hot) feature columns."""

    def test_includes_onehot_when_requested(self) -> None:
        """✅ Positive ``include_onehot=True``: exactly N_AGG + 160 names,
        with ``oh_hero_1`` .. ``oh_hero_160`` at the end.
        """
        max_hero_id = 160
        names = feature_column_names(include_onehot=True, max_hero_id=max_hero_id)
        assert len(names) == self.N_AGG_COLS + max_hero_id
        assert names[self.N_AGG_COLS] == "oh_hero_1"
        assert names[-1] == f"oh_hero_{max_hero_id}"

    def test_excludes_onehot_when_requested(self) -> None:
        """❌ Negative ``include_onehot=False``: only aggregate names, no
        ``oh_`` prefix.
        """
        names = feature_column_names(include_onehot=False)
        assert len(names) == self.N_AGG_COLS
        assert all(not n.startswith("oh_") for n in names), (
            "One-hot columns present when include_onehot=False"
        )

    def test_deterministic_order(self) -> None:
        """✅ Deterministic: two calls produce identical lists (order + content)."""
        names1 = feature_column_names(include_onehot=True, max_hero_id=160)
        names2 = feature_column_names(include_onehot=True, max_hero_id=160)
        assert names1 == names2
