"""Ensemble model support — combines multiple LightGBM models for robust predictions.

Supports:
- Multiple patch models (ensemble across patches)
- Snapshot ensemble (multiple training runs with different seeds)
- Weighted averaging based on model validation performance
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Combines predictions from multiple LightGBM models."""

    def __init__(self):
        self._models: list[tuple[lgb.Booster, float]] = []  # (model, weight)
        self._schemas: list[dict] = []

    def add_model(self, model: lgb.Booster, schema: dict, weight: float = 1.0):
        """Add a model to the ensemble with a weight (inverse of logloss)."""
        self._models.append((model, weight))
        self._schemas.append(schema)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted average prediction across all ensemble members."""
        if not self._models:
            raise ValueError("No models in ensemble")

        if len(self._models) == 1:
            return self._models[0][0].predict(X)

        total_weight = sum(w for _, w in self._models)
        weighted_preds = np.zeros(len(X))

        for model, weight in self._models:
            preds = model.predict(X)
            weighted_preds += preds * (weight / total_weight)

        return weighted_preds

    @property
    def n_models(self) -> int:
        return len(self._models)


def load_ensemble(
    model_dir: Path,
    patch_id: int,
    max_models: int = 3,
) -> EnsemblePredictor | None:
    """Load multiple model variants for ensemble prediction.

    Looks for model_patch_{patch_id}.txt and optionally
    model_patch_{patch_id}_v{N}.txt variants.
    """
    ensemble = EnsemblePredictor()

    # Primary model
    primary_path = model_dir / f"model_patch_{patch_id}.txt"
    if not primary_path.exists():
        return None

    model = lgb.Booster(model_file=str(primary_path))
    ensemble.add_model(model, {}, weight=1.0)

    # Look for variant models
    for v in range(2, max_models + 1):
        variant_path = model_dir / f"model_patch_{patch_id}_v{v}.txt"
        if variant_path.exists():
            try:
                variant = lgb.Booster(model_file=str(variant_path))
                ensemble.add_model(variant, {}, weight=0.8)
            except Exception:
                logger.warning("Failed to load ensemble variant %s", variant_path)

    if ensemble.n_models > 1:
        logger.info("Loaded ensemble with %d models for patch %d", ensemble.n_models, patch_id)

    return ensemble if ensemble.n_models > 1 else None
