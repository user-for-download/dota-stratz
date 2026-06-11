"""Shared fixtures and constants for the trainer test suite.

All tests avoid live database connections — external dependencies (DB,
network) are always mocked.
"""

from collections.abc import Generator
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
"""The 58 aggregate feature columns used by extract_features (no one-hot)."""

RAW_SQL_COLUMNS: list[str] = [
    # Draft-slot identification & context (provided by the SQL CTE)
    "match_id",
    "hero_id",
    "is_pick",
    "team",
    "order",
    "radiant_team_id",
    "dire_team_id",
    "radiant_win",
    # All aggregate feature columns
    *AGG_COLUMNS,
]
"""
Columns expected from ``TRAINING_FEATURES_SQL``.

Note: ``start_time`` is **not** projected by the SQL SELECT but is used by
``load_dataset`` for the chronological split.  Test data that exercises the
split must include ``start_time`` separately.
"""


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
    # Inject NaN in the first feature cell to exercise fillna(0)
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
# Synthetic DataFrame builders — used by test_dataset.py
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
    ``min_matches_per_group`` filter.  Used to verify the
    group-size filtering logic in ``load_dataset``.
    """
    rows: list[dict[str, object]] = []
    # 3 large matches (each 20 draft slots)
    for match_id in range(1, 4):
        for slot in range(20):
            row = _fill_agg_columns(_make_draft_row(match_id, slot))
            rows.append(row)
    # 2 small matches (each only 2 draft slots — below threshold)
    for match_id in range(4, 6):
        for slot in range(2):
            row = _fill_agg_columns(_make_draft_row(match_id, slot))
            rows.append(row)
    return pd.DataFrame(rows)
