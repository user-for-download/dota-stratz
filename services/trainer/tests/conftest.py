"""Shared fixtures and constants for the trainer test suite.

All tests avoid live database connections — external dependencies (DB,
network) are always mocked.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from trainer.config import TrainerConfig
from trainer.features import feature_column_names


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

AGG_COLUMNS: list[str] = feature_column_names(include_onehot=False)
"""The 59 aggregate feature columns used by feature functions (no one-hot)."""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small synthetic DataFrame for unit-testing feature functions.

    Contains ``team``, ``radiant_win``, ``hero_id`` and all aggregate columns
    with a NaN injected at position ``[0, AGG_COLUMNS[0]]``.
    """
    rng = np.random.default_rng(42)
    n = 5
    data: dict[str, object] = {
        "team": [0, 0, 1, 1, 0],
        "radiant_win": [True, False, True, False, True],
        "hero_id": [1, 3, 5, 7, 2],
    }
    for col in AGG_COLUMNS:
        data[col] = rng.random(n).tolist()

    df = pd.DataFrame(data)
    df.loc[0, AGG_COLUMNS[0]] = None
    return df


@pytest.fixture
def mock_engine() -> MagicMock:
    """A mock SQLAlchemy engine — never actually connects to any database."""
    return MagicMock()


@pytest.fixture
def trainer_config() -> TrainerConfig:
    """A ``TrainerConfig`` pre-configured for deterministic test runs."""
    return TrainerConfig(
        patch_id=7,
        max_hero_id=160,
        val_ratio=0.2,
        min_matches_per_group=5,
    )


# ---------------------------------------------------------------------------
# Synthetic DataFrame builders — used by test_dataset_pt.py
# ---------------------------------------------------------------------------

def _fill_agg_columns(row: dict[str, object]) -> dict[str, object]:
    """Add all aggregate columns with a constant 0.5 value."""
    for col in AGG_COLUMNS:
        row[col] = 0.5
    return row


def _make_draft_row(match_id: int, slot: int) -> dict[str, object]:
    """Return a single draft-slot row with basic columns filled."""
    return {
        "match_id": match_id,
        "hero_id": 1,
        "is_pick": 1,
        "team": slot % 2,
        "order": slot,
        "radiant_team_id": 100,
        "dire_team_id": 200,
        "radiant_win": bool(match_id % 2),
        "start_time": match_id,
        "patch_id": 7,
    }


@pytest.fixture
def synthetic_match_df() -> pd.DataFrame:
    """20 matches × 5 draft slots, start_time spread across 20 days.

    Used for the chronological split regression test (BUG-020).
    Each match has exactly 5 draft slots so that the default
    ``min_matches_per_group=5`` filter keeps all matches.
    """
    n_matches = 20
    slots_per_match = 5
    rows: list[dict[str, object]] = []
    for match_id in range(1, n_matches + 1):
        for slot in range(slots_per_match):
            row = _fill_agg_columns(_make_draft_row(match_id, slot))
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def small_match_df() -> pd.DataFrame:
    """3 matches × 20 slots + 2 matches × 2 slots.

    Only the 3 large matches (≥5 slots) survive the
    ``min_matches_per_group`` filter.
    """
    rows: list[dict[str, object]] = []
    for match_id in range(1, 4):
        for slot in range(20):
            row = _fill_agg_columns(_make_draft_row(match_id, slot))
            rows.append(row)
    for match_id in range(4, 6):
        for slot in range(2):
            row = _fill_agg_columns(_make_draft_row(match_id, slot))
            rows.append(row)
    return pd.DataFrame(rows)
