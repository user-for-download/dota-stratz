"""LightGBM lambdarank model training.

Returns the trained Booster and the best NDCG score.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .config import TrainerConfig
from .dataset import load_dataset
from .features import write_schema

logger = logging.getLogger(__name__)


def train_model(
    cfg: TrainerConfig,
    engine,
) -> tuple[lgb.Booster, float]:
    """Train a LightGBM lambdarank model for *cfg.patch_id*.

    Returns
    -------
    model : lgb.Booster
        Trained LightGBM model.
    best_ndcg : float
        Best NDCG@5 on the validation set (or training set if no val split).
    """
    train_data, val_data, metadata = load_dataset(cfg, engine)

    logger.info(
        "Training set: %d rows, %d groups | Validation set: %d rows",
        metadata["n_train"],
        metadata["n_groups_train"],
        metadata.get("n_val", 0),
    )

    callbacks = [
        lgb.log_evaluation(period=50),
        lgb.early_stopping(stopping_rounds=50, verbose=True),
    ]

    model = lgb.train(
        cfg.lgbm_params,
        train_data,
        valid_sets=[val_data] if val_data else [train_data],
        valid_names=["val"] if val_data else ["train"],
        num_boost_round=1000,
        callbacks=callbacks,
    )

    # Extract best NDCG score
    evals_result = {}
    if val_data:
        best_ndcg = float(model.best_score["val"]["ndcg@5"])
    else:
        best_ndcg = float(model.best_score["train"]["ndcg@5"])

    logger.info("Training complete. Best NDCG@5: %.4f", best_ndcg)

    # Save model
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"model_patch_{cfg.patch_id}.txt"
    model.save_model(str(model_path))
    logger.info("Model saved to %s", model_path)

    # Save metadata
    meta = {
        "patch_id": cfg.patch_id,
        "best_ndcg_5": best_ndcg,
        "n_train": metadata["n_train"],
        "n_val": metadata.get("n_val", 0),
        "n_features": metadata["n_features"],
        "n_groups_train": metadata["n_groups_train"],
        "params": cfg.lgbm_params,
    }
    meta_path = model_dir / f"model_patch_{cfg.patch_id}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Metadata saved to %s", meta_path)

    # Write the authoritative column-order schema
    write_schema(model_dir, max_hero_id=cfg.max_hero_id)

    return model, best_ndcg
