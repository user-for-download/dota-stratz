"""Dataset construction: load raw training data from the DB, build feature
vectors, and split into train/validation sets.

LightGBM lambdarank requires a ``group`` array that specifies the number of
rows in each query group (each match = one group). The group array MUST be
sorted by match_id and draft_order, then the group counts reflect contiguous
blocks of rows per match.
"""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import TrainerConfig
from .features import (
    TRAINING_FEATURES_SQL,
    extract_features,
    feature_column_names,
    make_group,
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
    metadata : dict with keys ``n_train``, ``n_val``, ``n_features``,
               ``n_groups_train``, ``n_groups_val``
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

    # -------------------------------------------------------------------
    # Sort by (match_id, order) — critical for LightGBM group alignment.
    # The group array must reflect *contiguous* blocks of rows, each
    # block belonging to one match, in the same order as the rows in X.
    # -------------------------------------------------------------------
    df = df.sort_values(["match_id", "order"]).reset_index(drop=True)

    # Build feature matrix
    agg_cols = feature_column_names(include_onehot=False)
    X = extract_features(df, agg_cols, max_hero_id=cfg.max_hero_id)
    y = make_target(df)
    groups = make_group(df)

    logger.info(
        "Feature matrix shape: %s, groups: %d (min %d, max %d)",
        X.shape,
        len(groups),
        groups.min(),
        groups.max(),
    )

    # Train / val split at match level (not row level)
    match_ids = df["match_id"].unique()
    train_matches, val_matches = train_test_split(
        match_ids,
        test_size=cfg.val_ratio,
        random_state=42,
    )

    train_mask = df["match_id"].isin(train_matches)
    val_mask = df["match_id"].isin(val_matches)

    X_train = X[train_mask]
    y_train = y[train_mask]
    groups_train = make_group(df[train_mask])

    metadata: dict[str, Any] = {
        "n_train": len(X_train),
        "n_val": (~train_mask).sum(),
        "n_features": X.shape[1],
        "n_groups_train": len(groups_train),
    }

    train_data = lgb.Dataset(X_train, y_train, group=groups_train)

    val_data: lgb.Dataset | None = None
    if val_mask.sum() > 0:
        X_val = X[val_mask]
        y_val = y[val_mask]
        groups_val = make_group(df[val_mask])
        val_data = lgb.Dataset(X_val, y_val, group=groups_val, reference=train_data)
        metadata["n_groups_val"] = len(groups_val)

    return train_data, val_data, metadata
