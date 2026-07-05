"""Tests for feature computation functions — ``make_target``
and ``feature_column_names``.

All tests use the shared fixtures from ``conftest.py`` (``sample_df``,
``mock_engine``, ``trainer_config``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trainer.features import (
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
            (0, True, 1),
            (0, False, 0),
            (1, True, 0),
            (1, False, 1),
        ],
    )
    def test_target_from_team_and_radiant_win(
        self,
        team: int,
        radiant_win: bool,
        expected: int,
    ) -> None:
        df = pd.DataFrame({"team": [team], "radiant_win": [radiant_win]})
        result = make_target(df)
        assert result[0] == expected
        assert result.dtype == np.int64


# ===========================================================================
# feature_column_names
# ===========================================================================

class TestFeatureColumnNames:
    """``feature_column_names`` — column-ordering contract between
    training and inference.
    """

    N_AGG_COLS: int = len(feature_column_names(include_onehot=False))

    def test_includes_embeddings_when_requested(self) -> None:
        n_emb = 32
        names = feature_column_names(include_onehot=True, n_embeddings=n_emb)
        assert len(names) == self.N_AGG_COLS + 1 + n_emb
        assert names[self.N_AGG_COLS] == "hero_id"
        assert names[self.N_AGG_COLS + 1] == "emb_0"
        assert names[-1] == f"emb_{n_emb - 1}"

    def test_excludes_embeddings_when_requested(self) -> None:
        names = feature_column_names(include_onehot=False)
        assert len(names) == self.N_AGG_COLS
        assert all(not n.startswith("oh_") and not n.startswith("emb_") for n in names)

    def test_deterministic_order(self) -> None:
        names1 = feature_column_names(include_onehot=True, n_embeddings=32)
        names2 = feature_column_names(include_onehot=True, n_embeddings=32)
        assert names1 == names2
