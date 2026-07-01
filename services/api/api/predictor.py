"""Model loading and prediction orchestration.

Models are lazy-loaded per patch ID. The ``Predictor`` class manages
a cache of loaded LightGBM Boosters and their associated feature schemas.

Thread safety: ``_models`` and ``_schemas`` are guarded by ``_lock``
(``threading.RLock``) so a concurrent ``/reload`` call cannot race with
an in-flight ``/predict`` (BUG-001).
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from . import db as db_
from .config import APIConfig
from .draft_state import DraftContext
from .features import BatchContext, build_feature_vector, load_schema, pre_fetch_batch
from .lookahead import ranked_with_lookahead
from .reasoning import generate_reasoning

logger = logging.getLogger(__name__)


class Predictor:
    """Manages per-patch model loading and prediction."""

    def __init__(self, cfg: APIConfig):
        self._cfg = cfg
        self._models: dict[int, lgb.Booster] = {}
        self._schemas: dict[int, dict[str, Any]] = {}
        self._calibrators: dict[int, Any] = {}
        self._model_dir = Path(cfg.model_dir)
        self._max_hero_id = cfg.max_hero_id
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def _model_path(self, patch_id: int) -> Path:
        return self._model_dir / f"model_patch_{patch_id}.txt"

    def is_loaded(self, patch_id: int) -> bool:
        with self._lock:
            return patch_id in self._models

    def loaded_patches(self) -> list[int]:
        with self._lock:
            return sorted(self._models.keys())

    def load_model(self, patch_id: int) -> bool:
        """Load model + schema + calibrator for *patch_id*. Returns True on success."""
        model_path = self._model_path(patch_id)
        if not model_path.exists():
            logger.warning("Model file not found for patch %s: %s", patch_id, model_path)
            return False

        try:
            model = lgb.Booster(model_file=str(model_path))
            schema = load_schema(self._model_dir, patch_id)

            # Load calibrator if available
            calibrator = None
            cal_path = self._model_dir / f"calibrator_patch_{patch_id}.json"
            if cal_path.exists():
                import json
                cal_data = json.loads(cal_path.read_text())
                calibrator = {
                    "coef": cal_data["coef"],
                    "intercept": cal_data["intercept"],
                }

            with self._lock:
                self._models[patch_id] = model
                self._schemas[patch_id] = schema
                if calibrator:
                    self._calibrators[patch_id] = calibrator
            logger.info(
                "Loaded model for patch %s (%d features, calibrator=%s)",
                patch_id, schema["n_features"], "yes" if calibrator else "no",
            )
            return True
        except Exception:
            logger.exception("Failed to load model for patch %s", patch_id)
            return False

    def unload_model(self, patch_id: int):
        """Remove a model from the cache."""
        with self._lock:
            self._models.pop(patch_id, None)
            self._schemas.pop(patch_id, None)
            self._calibrators.pop(patch_id, None)
        logger.info("Unloaded model for patch %s", patch_id)

    def reload_all(self):
        """Scan the model directory and reload all available models."""
        new_models: dict[int, lgb.Booster] = {}
        new_schemas: dict[int, dict[str, Any]] = {}
        new_calibrators: dict[int, Any] = {}
        count = 0
        for fpath in sorted(self._model_dir.glob("model_patch_*.txt")):
            try:
                pid = int(fpath.stem.replace("model_patch_", ""))
                model_path = self._model_path(pid)
                if not model_path.exists():
                    continue
                model = lgb.Booster(model_file=str(model_path))
                schema = load_schema(self._model_dir, pid)
                new_models[pid] = model
                new_schemas[pid] = schema

                # Load calibrator
                cal_path = self._model_dir / f"calibrator_patch_{pid}.json"
                if cal_path.exists():
                    import json
                    cal_data = json.loads(cal_path.read_text())
                    new_calibrators[pid] = {
                        "coef": cal_data["coef"],
                        "intercept": cal_data["intercept"],
                    }

                count += 1
            except (ValueError, IndexError, Exception):
                logger.exception("Failed to load model during reload_all for patch")
                continue
        with self._lock:
            self._models = new_models
            self._schemas = new_schemas
            self._calibrators = new_calibrators
        logger.info("Loaded %d models from %s", count, self._model_dir)
        return count

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        patch_id: int,
        ctx: DraftContext,
        radiant_team_id: int | None,
        dire_team_id: int | None,
        num_recommendations: int = 5,
        account_id: int | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Score all eligible heroes and return the top recommendations.

        Uses calibrated probabilities and look-ahead minimax search.
        """
        # Atomically load (if missing) and capture references under the lock
        with self._lock:
            if patch_id not in self._models:
                loaded = self.load_model(patch_id)
                if not loaded:
                    raise ValueError(
                        f"No model available for patch {patch_id}. "
                        "Run the trainer first (`make train PATCH=<id>`)."
                    )
            model = self._models[patch_id]
            schema = self._schemas[patch_id]
            calibrator = self._calibrators.get(patch_id)

        taken_heroes = ctx.all_taken

        max_id = schema["max_hero_id"]
        eligible_hero_ids = [
            hid for hid in range(1, max_id + 1)
            if hid not in taken_heroes
        ]
        if not eligible_hero_ids:
            return [], None

        team_id = radiant_team_id if ctx.recommending_team == 0 else dire_team_id
        enemy_team_id = dire_team_id if ctx.recommending_team == 0 else radiant_team_id

        try:
            batch = pre_fetch_batch(
                patch_id=patch_id,
                hero_ids=eligible_hero_ids,
                team_id=team_id,
                enemy_team_id=enemy_team_id,
                ctx=ctx,
                account_id=account_id,
            )
        except (RuntimeError, TimeoutError) as e:
            raise ValueError(
                f"Database pool is not available: {e}. "
                "The service may still be starting up."
            )

        feature_vectors: list[np.ndarray] = []
        for hid in eligible_hero_ids:
            fv = build_feature_vector(
                hero_id=hid,
                ctx=ctx,
                patch_id=patch_id,
                batch=batch,
                schema=schema,
                max_hero_id=schema["max_hero_id"],
            )
            feature_vectors.append(fv)

        if not feature_vectors:
            return [], None

        X = np.stack(feature_vectors, axis=0)
        raw_scores = model.predict(X)

        # === Task 11: Apply calibration ===
        if calibrator:
            coef = np.array(calibrator["coef"])
            intercept = calibrator["intercept"]
            logits = raw_scores * coef[0] + intercept
            scores = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
        else:
            scores = raw_scores

        # Sort by calibrated score descending
        indices = np.argsort(scores)[::-1][:num_recommendations * 2]

        recommendations: list[dict[str, Any]] = []
        for idx in indices:
            hid = eligible_hero_ids[idx]
            sc = float(scores[idx])
            raw = float(raw_scores[idx])
            recommendations.append({
                "hero_id": hid,
                "score": raw,
                "pick_probability": round(sc, 4),
                "win_probability": round(sc, 4),
            })

        # === Look-ahead minimax re-ranking ===
        try:
            def model_fn(hero_id, draft_state):
                # Simplified: return the calibrated score for this hero
                for i, h in enumerate(eligible_hero_ids):
                    if h == hero_id and i < len(scores):
                        return float(scores[i])
                return 0.5

            recommendations = ranked_with_lookahead(
                candidates=recommendations,
                draft_state=[],
                patch_id=patch_id,
                eligible_hero_ids=eligible_hero_ids,
                model_fn=model_fn,
                batch_ctx=batch,
                schema=schema,
                top_k=num_recommendations,
            )
        except Exception as e:
            logger.warning("Look-ahead failed, falling back to greedy: %s", e)
            recommendations = recommendations[:num_recommendations]

        # Generate reasoning for the top recommendation
        reasoning: str | None = None
        if recommendations:
            top = recommendations[0]
            reasoning = self._build_reasoning(
                top["hero_id"], top["score"], ctx, patch_id,
                radiant_team_id, dire_team_id,
                batch=batch,
            )

        return recommendations, reasoning

    def _build_reasoning(
        self,
        hero_id: int,
        score: float,
        ctx: DraftContext,
        patch_id: int,
        radiant_team_id: int | None = None,
        dire_team_id: int | None = None,
        batch: BatchContext | None = None,
    ) -> str | None:
        """Gather explanation data for the top hero.

        Reads from the pre-fetched ``BatchContext`` (passed from ``predict``)
        instead of making 4 independent DB round-trips (BUG-002, BUG-003).
        Falls back to database queries if ``batch`` is not provided.
        """
        if batch is not None:
            # Read from pre-fetched batch — zero extra queries (BUG-002).
            # Convert DB values (Decimal from psycopg2) to float for
            # format-string safety in generate_reasoning (BUG-003).
            bl = batch.baselines.get(hero_id)
            bl_wr = float(bl["win_rate"]) if bl else None

            th = batch.team_hero_agg.get(hero_id)
            th_wr = float(th["win_rate"]) if th else None

            sy = batch.synergy.get(hero_id)
            sy_wr = float(sy[0]) if sy else None

            co = batch.counter.get(hero_id)
            co_wr = float(co[0]) if co else None

            # BUG-003: h2h_row was fetched in pre_fetch_batch but never read.
            h2h = batch.h2h_row
            h2h_wr = float(h2h["win_rate"]) if h2h else None
        else:
            # Fallback (should not happen in production).
            team_id = radiant_team_id if ctx.recommending_team == 0 else dire_team_id
            bl = db_.fetch_baseline(patch_id, hero_id)
            bl_wr = bl["win_rate"] if bl else None
            th = db_.fetch_team_hero_agg(patch_id, team_id, hero_id) if team_id else None
            th_wr = th["win_rate"] if th else None
            sy_wr, _ = db_.fetch_synergy_avg(patch_id, hero_id, ctx.ally_picks)
            co_wr, _, _ = db_.fetch_counter_avg(patch_id, hero_id, ctx.enemy_picks)
            h2h_wr = None

        return generate_reasoning(
            hero_id=hero_id,
            score=score,
            ctx=ctx,
            baseline_win_rate=bl_wr,
            team_hero_win_rate=th_wr,
            synergy_win_rate=sy_wr if sy_wr is not None and sy_wr != 0.5 else None,
            counter_win_rate=co_wr if co_wr is not None and co_wr != 0.5 else None,
            h2h_win_rate=h2h_wr,
        )
