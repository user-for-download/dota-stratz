"""Tests for dataset construction — ``load_dataset``.

All tests mock ``pandas.read_sql`` to avoid live database connections.
Synthetic DataFrames are provided by fixtures in ``conftest.py``.
"""

from __future__ import annotations

from unittest import mock

import lightgbm as lgb
import pandas as pd
import pytest

from trainer.dataset import load_dataset
from trainer.features import feature_column_names

# Column count used to verify metadata["n_features"]
_N_AGG_COLS: int = len(feature_column_names(include_onehot=False))
_MAX_HERO_ID: int = 160
_EXPECTED_N_FEATURES: int = _N_AGG_COLS + _MAX_HERO_ID


# ===========================================================================
# load_dataset — chronological split (REGRESSION BUG-020)
# ===========================================================================

class TestChronologicalSplit:
    """Chronological train/val split — oldest matches → train, newest → val.

    Regression test for BUG-020: the split must use ``val_mask`` directly,
    *not* the complement of ``train_mask``, because rows not belonging to
    any match group would silently distort counts.
    """

    def test_n_val_matches_newest_20_percent(
        self,
        trainer_config: object,
        mock_engine: object,
        synthetic_match_df: pd.DataFrame,
    ) -> None:
        """✅ Positive: with 20 matches × 5 slots and val_ratio=0.2,
        ``metadata["n_val"]`` equals the rows in the 4 newest matches.

        4 matches × 5 slots = 20 validation rows.
        """
        with mock.patch("pandas.read_sql", return_value=synthetic_match_df):
            train_data, val_data, metadata = load_dataset(
                trainer_config, mock_engine,
            )

        # 20 matches, val_ratio=0.2 → 16 train (oldest) + 4 val (newest)
        assert metadata["n_val"] == 4 * 5, (
            f"Expected 20 val rows (4 newest matches × 5 slots), "
            f"got {metadata['n_val']}"
        )
        assert metadata["n_train"] == 16 * 5, (
            f"Expected 80 train rows (16 oldest matches × 5 slots), "
            f"got {metadata['n_train']}"
        )
        assert metadata["n_features"] == _EXPECTED_N_FEATURES, (
            f"Expected {_EXPECTED_N_FEATURES} features, "
            f"got {metadata['n_features']}"
        )

    def test_both_datasets_are_returned(
        self,
        trainer_config: object,
        mock_engine: object,
        synthetic_match_df: pd.DataFrame,
    ) -> None:
        """✅ Positive: both train and validation Datasets are created."""
        with mock.patch("pandas.read_sql", return_value=synthetic_match_df):
            train_data, val_data, metadata = load_dataset(
                trainer_config, mock_engine,
            )

        assert isinstance(train_data, lgb.Dataset)
        assert val_data is not None
        assert isinstance(val_data, lgb.Dataset)

    def test_n_val_equals_val_mask_sum(
        self,
        trainer_config: object,
        mock_engine: object,
        synthetic_match_df: pd.DataFrame,
    ) -> None:
        """✅ Regression (BUG-020): ``n_val`` is computed from ``val_mask``,
        not from ``(~train_mask).sum()``.

        Although these are equal for a clean partition, relying on the
        complement would silently hide off-by-one errors or unassigned
        rows.  We verify the internal computation matches by manually
        reproducing the split.
        """
        # Reproduce the chronological split externally
        start_times = (
            synthetic_match_df
            .groupby("match_id")["start_time"]
            .first()
            .sort_values()
        )
        match_ids_sorted = start_times.index
        n_train = int(len(match_ids_sorted) * (1 - trainer_config.val_ratio))
        val_matches = match_ids_sorted[n_train:]

        n_val_expected = synthetic_match_df[
            synthetic_match_df["match_id"].isin(val_matches)
        ].shape[0]

        with mock.patch("pandas.read_sql", return_value=synthetic_match_df):
            _, _, metadata = load_dataset(trainer_config, mock_engine)

        assert metadata["n_val"] == n_val_expected, (
            f"metadata['n_val'] ({metadata['n_val']}) does not match "
            f"expected val rows ({n_val_expected})"
        )


# ===========================================================================
# load_dataset — min_matches_per_group filter
# ===========================================================================

class TestMinMatchesPerGroup:
    """Matches with fewer draft slots than ``cfg.min_matches_per_group``
    are filtered out before the train/val split.
    """

    def test_small_matches_are_filtered_out(
        self,
        trainer_config: object,
        mock_engine: object,
        small_match_df: pd.DataFrame,
    ) -> None:
        """✅ Positive: only 3 large matches (≥5 slots) survive the filter.

        Input: 3 matches × 20 slots + 2 matches × 2 slots.
        Filter (min_matches_per_group=5): 2 small matches removed.
        Survival: 3 matches × 20 slots = 60 rows.
        """
        with mock.patch("pandas.read_sql", return_value=small_match_df):
            train_data, val_data, metadata = load_dataset(
                trainer_config, mock_engine,
            )

        # 3 matches survive, val_ratio=0.2 → 2 train, 1 val
        # train: 2 matches × 20 slots = 40, val: 1 match × 20 slots = 20
        assert metadata["n_train"] == 2 * 20, (
            f"Expected 40 train rows, got {metadata['n_train']}"
        )
        assert metadata["n_val"] == 1 * 20, (
            f"Expected 20 val rows, got {metadata['n_val']}"
        )
        assert metadata["n_features"] == _EXPECTED_N_FEATURES

    def test_val_data_is_returned(
        self,
        trainer_config: object,
        mock_engine: object,
        small_match_df: pd.DataFrame,
    ) -> None:
        """✅ Positive: validation set exists after filtering."""
        with mock.patch("pandas.read_sql", return_value=small_match_df):
            _, val_data, metadata = load_dataset(
                trainer_config, mock_engine,
            )

        assert val_data is not None
        assert isinstance(val_data, lgb.Dataset)
        # Sanity-check the label count matches metadata
        assert val_data.get_label().shape[0] == metadata["n_val"]

    def test_filtered_match_ids_not_in_output(
        self,
        trainer_config: object,
        mock_engine: object,
        small_match_df: pd.DataFrame,
    ) -> None:
        """❌ Negative: the 2 small matches (match_ids 4, 5) are completely
        absent from the resulting datasets.
        """
        with mock.patch("pandas.read_sql", return_value=small_match_df):
            train_data, val_data, metadata = load_dataset(
                trainer_config, mock_engine,
            )

        # Collect all labels (one per row) — we cannot directly inspect
        # match_ids from lgb.Dataset. Instead, check that the total row
        # count equals only the 3 large matches (60 rows), not 64.
        n_total = train_data.get_label().shape[0] + val_data.get_label().shape[0]
        assert n_total == 60, (
            f"Expected 60 total rows after filter, got {n_total}"
        )
