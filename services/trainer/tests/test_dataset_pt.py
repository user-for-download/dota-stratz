"""Tests for dataset construction — ``load_sequence_dataset`` (PyTorch path).

Regression bugs covered:
    - BUG-020: chronological train/val split must use ``val_mask`` directly
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest import mock

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from trainer.dataset_pt import DraftSequenceDataset, load_sequence_dataset
from trainer.features import feature_column_names


AGG_COLUMNS = feature_column_names(include_onehot=False)


def _setup_mock_engine(engine, df):
    """Configure mock_engine to return df when execute() is called."""
    mock_result = mock.MagicMock()
    mock_result.keys.return_value = df.columns.tolist()
    mock_result.fetchall.return_value = df.values.tolist()

    mock_conn = mock.MagicMock()
    mock_conn.execute.return_value = mock_result

    mock_cm = mock.MagicMock()
    mock_cm.__enter__ = mock.MagicMock(return_value=mock_conn)
    mock_cm.__exit__ = mock.MagicMock(return_value=False)
    engine.connect.return_value = mock_cm


# ===========================================================================
# BUG-020: Chronological split regression
# ===========================================================================

class TestChronologicalSplit:
    """Chronological train/val split — oldest matches → train, newest → val.

    Regression test for BUG-020: the split must use val_mask directly,
    not the complement of train_mask.
    """

    def test_split_ratios(
        self, trainer_config, mock_engine, synthetic_match_df,
    ):
        """20 matches × 5 slots with val_ratio=0.2:
        16 train matches (oldest) + 4 val matches (newest)."""
        _setup_mock_engine(mock_engine, synthetic_match_df)
        train_ds, val_ds, metadata = load_sequence_dataset(
            trainer_config, mock_engine, max_len=25,
        )

        n_val_matches = metadata["n_val_matches"]
        n_train_matches = metadata["n_train_matches"]
        assert n_val_matches == 4, f"Expected 4 val matches, got {n_val_matches}"
        assert n_train_matches == 16, f"Expected 16 train matches, got {n_train_matches}"

        # 16 matches × 5 slots × 2 (symmetry augmentation) = 160 prefix sequences
        assert metadata["n_train_sequences"] == 16 * 5 * 2
        # 4 matches × 5 slots × 2 = 40 prefix sequences
        assert metadata["n_val_sequences"] == 4 * 5 * 2

    def test_datasets_are_tensorized(self, trainer_config, mock_engine, synthetic_match_df):
        """Train and val are pre-tensorized DraftSequenceDataset instances."""
        _setup_mock_engine(mock_engine, synthetic_match_df)
        train_ds, val_ds, metadata = load_sequence_dataset(
            trainer_config, mock_engine, max_len=25,
        )

        assert isinstance(train_ds, DraftSequenceDataset)
        assert isinstance(val_ds, DraftSequenceDataset)
        assert len(train_ds) == metadata["n_train_sequences"]
        assert len(val_ds) == metadata["n_val_sequences"]

    def test_tensor_shapes(self, trainer_config, mock_engine, synthetic_match_df):
        """Each sample returns (heroes, actions, tabular, patches, label) tensors."""
        _setup_mock_engine(mock_engine, synthetic_match_df)
        train_ds, val_ds, _ = load_sequence_dataset(
            trainer_config, mock_engine, max_len=25,
        )

        h, a, t, p, l = train_ds[0]
        assert h.shape == (25,), f"heroes shape: {h.shape}"
        assert a.shape == (25,), f"actions shape: {a.shape}"
        assert t.dtype == torch.float32
        assert p.dtype == torch.long
        assert l.dtype == torch.float32

    def test_val_mask_not_complement(
        self, trainer_config, mock_engine, synthetic_match_df,
    ):
        """BUG-020: n_val_sequences must equal val_ds length,
        not len(train_ds) complement."""
        _setup_mock_engine(mock_engine, synthetic_match_df)
        train_ds, val_ds, metadata = load_sequence_dataset(
            trainer_config, mock_engine, max_len=25,
        )

        assert metadata["n_val_sequences"] == len(val_ds)
        assert metadata["n_train_sequences"] == len(train_ds)
        assert metadata["n_train_sequences"] + metadata["n_val_sequences"] == len(train_ds) + len(val_ds)


# ===========================================================================
# DraftSequenceDataset unit tests
# ===========================================================================

class TestDraftSequenceDataset:
    def test_empty_dataset(self):
        ds = DraftSequenceDataset([], [], [], [], [], max_len=10)
        assert len(ds) == 0

    def test_single_sample(self):
        heroes = [[1, 2, 3]]
        actions = [[3, 4, 1]]
        tabular = [[0.5] * 3]
        patches = [60]
        labels = [1.0]
        ds = DraftSequenceDataset(heroes, actions, tabular, patches, labels, max_len=5)
        assert len(ds) == 1
        h, a, t, p, l = ds[0]
        assert h.tolist() == [1, 2, 3, 0, 0]
        assert a.tolist() == [3, 4, 1, 0, 0]
        assert float(l) == 1.0

    def test_truncation_at_max_len(self):
        heroes = [[1, 2, 3, 4, 5, 6]]
        ds = DraftSequenceDataset(heroes, heroes, [[0.0]], [60], [0.0], max_len=3)
        h, _, _, _, _ = ds[0]
        assert h.tolist() == [1, 2, 3]
