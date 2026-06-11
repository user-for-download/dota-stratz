"""Dataset construction: load raw training data from the DB, build feature
vectors, and split into train/validation sets.

Uses a **chronological** train/val split (oldest matches → train, newest → val)
rather than random, because the pre-computed ml.*_agg tables include ALL
matches in a patch — a random split would leak future information into
training features (PIT leakage issue #2).

Uses binary classification (not lambdarank) — see config.py for rationale.
"""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import TrainerConfig
from .features import (
    TRAINING_FEATURES_SQL,
    extract_features,
    feature_column_names,
    make_target,
)

logger = logging.getLogger(__name__)


def load_dataset(
    cfg: TrainerConfig,
    engine: Any,
) -> tuple[lgb.Dataset, lgb.Dataset | None, dict[str, Any]]:
    """Load training data from the database, build feature matrix, and
    split into train/validation LightGBM Datasets.

    Returns
    -------
    train_data : lgb.Dataset
    val_data : lgb.Dataset or None (if insufficient data)
    metadata : dict with keys ``n_train``, ``n_val``, ``n_features``
    """
    logger.info("Loading training data from DB for patch %s ...", cfg.patch_id)

    df = pd.read_sql(
        TRAINING_FEATURES_SQL,
        engine,
        params={"patch_id": cfg.patch_id},
    )

    if df.empty:
        raise ValueError(
            f"No training data found for patch {cfg.patch_id}. "
            "Has the aggregate population been run?"
        )

    logger.info("Loaded %d draft slots from %d matches", len(df), df["match_id"].nunique())

    # Filter out matches with too few draft slots (abandoned lobbies, etc.)
    # The TRAINER_MIN_MATCHES_PER_GROUP config (default 5) was previously
    # defined but never enforced (issue #15).
    match_sizes = df.groupby("match_id").size()
    valid_matches = match_sizes[match_sizes >= cfg.min_matches_per_group].index
    df = df[df["match_id"].isin(valid_matches)]
    logger.info(
        "After min_matches_per_group=%d filter: %d slots from %d matches",
        cfg.min_matches_per_group, len(df), df["match_id"].nunique(),
    )

    if df.empty:
        raise ValueError(
            f"No matches meet min_matches_per_group={cfg.min_matches_per_group} "
            f"for patch {cfg.patch_id}."
        )

    # Sort by (match_id, order) for deterministic row order.
    df = df.sort_values(["match_id", "order"]).reset_index(drop=True)

    # Build feature matrix
    agg_cols = feature_column_names(include_onehot=False)
    X = extract_features(df, agg_cols, max_hero_id=cfg.max_hero_id)
    y = make_target(df)

    logger.info(
        "Feature matrix shape: %s, mean target: %.3f",
        X.shape,
        float(y.mean()),
    )

    # Train / val split at match level (not row level)
    # Chronological split: oldest matches → train, newest → val.
    # This is critical because the ml.*_agg tables contain ALL matches in the
    # patch (not PIT-filtered), so a random split would let training rows
    # "see" future matches via aggregate features (PIT leakage issue #2).
    # A chronological split at least makes the validation metric a more
    # honest estimate of out-of-time performance.
    match_start_times = df.groupby("match_id")["start_time"].first().sort_values()
    match_ids_sorted = match_start_times.index
    n_train = int(len(match_ids_sorted) * (1 - cfg.val_ratio))
    train_matches = match_ids_sorted[:n_train]
    val_matches = match_ids_sorted[n_train:]

    train_mask = df["match_id"].isin(train_matches)
    val_mask = df["match_id"].isin(val_matches)

    X_train = X[train_mask]
    y_train = y[train_mask]

    metadata: dict[str, Any] = {
        "n_train": len(X_train),
        "n_val": (~train_mask).sum(),
        "n_features": X.shape[1],
    }

    train_data = lgb.Dataset(X_train, y_train)

    val_data: lgb.Dataset | None = None
    if val_mask.sum() > 0:
        X_val = X[val_mask]
        y_val = y[val_mask]
        val_data = lgb.Dataset(X_val, y_val, reference=train_data)

    return train_data, val_data, metadata
