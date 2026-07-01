"""LightGBM binary classification model training with calibration.

Returns the trained Booster, the best validation loss, and a calibration model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression

from .config import TrainerConfig
from .dataset import load_dataset
from .features import write_schema

logger = logging.getLogger(__name__)


def train_model(
    cfg: TrainerConfig,
    engine,
) -> tuple[lgb.Booster, float]:
    """Train a LightGBM binary classifier for *cfg.patch_id*.

    Uses binary (not lambdarank) because every draft slot in a match shares
    the same radiant_win target — lambdarank requires varied relevance within
    each group and would produce zero-gradient trees (issue #7).

    Returns
    -------
    model : lgb.Booster
        Trained LightGBM model.
    best_loss : float
        Best binary_logloss on the validation set (or training set).
    """
    train_data, val_data, metadata, X_train, y_train = load_dataset(cfg, engine)

    logger.info(
        "Training set: %d rows | Validation set: %d rows",
        metadata["n_train"],
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

    # Extract best binary_logloss
    metric_name = "binary_logloss"
    if val_data:
        best_loss = float(model.best_score["val"][metric_name])
    else:
        best_loss = float(model.best_score["train"][metric_name])

    logger.info("Training complete. Best %s: %.4f", metric_name, best_loss)

    # === Task 11: Confidence Calibration (Platt scaling) ===
    # Train a logistic regression on model predictions to calibrate probabilities
    if val_data:
        X_val = val_data.get_data()
        val_labels = val_data.get_label()
        val_preds = model.predict(X_val)
    else:
        val_preds = model.predict(X_train)
        val_labels = y_train

    calibrator = LogisticRegression(C=1.0, solver='lbfgs')
    calibrator.fit(val_preds.reshape(-1, 1), val_labels)

    cal_train_preds = model.predict(X_train)
    calibrator.fit(cal_train_preds.reshape(-1, 1), y_train)

    logger.info("Calibration model trained (Platt scaling)")

    # Save model
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"model_patch_{cfg.patch_id}.txt"
    model.save_model(str(model_path))
    logger.info("Model saved to %s", model_path)

    # Save calibrator
    cal_path = model_dir / f"calibrator_patch_{cfg.patch_id}.json"
    cal_data = {
        "coef": calibrator.coef_[0].tolist(),
        "intercept": calibrator.intercept_[0].tolist(),
    }
    cal_path.write_text(json.dumps(cal_data))
    logger.info("Calibrator saved to %s", cal_path)

    # Save metadata (cast numpy types to native Python for JSON)
    meta = {
        "patch_id": int(cfg.patch_id),
        "best_binary_logloss": best_loss,
        "n_train": metadata["n_train"],
        "n_val": metadata.get("n_val", 0),
        "n_features": metadata["n_features"],
        "params": cfg.lgbm_params,
        "has_calibrator": True,
    }
    meta_path = model_dir / f"model_patch_{cfg.patch_id}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Metadata saved to %s", meta_path)

    # Write the authoritative column-order schema (patch-specific filename
    # to prevent overwrites across multiple patch training runs — Bug #5).
    write_schema(model_dir, patch_id=cfg.patch_id, max_hero_id=cfg.max_hero_id)

    return model, best_loss
