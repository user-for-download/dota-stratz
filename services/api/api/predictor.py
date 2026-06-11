"""Model loading and prediction orchestration.

Models are lazy-loaded per patch ID. The ``Predictor`` class manages
a cache of loaded LightGBM Boosters and their associated feature schemas.

Thread safety: ``_models`` and ``_schemas`` are guarded by ``_lock``
(``threading.RLock``) so a concurrent ``/reload`` call cannot race with
an in-flight ``/predict`` (BUG-001).
"""

from __future__ import annotations

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
from .reasoning import generate_reasoning

logger = logging.getLogger(__name__)


class Predictor:
    """Manages per-patch model loading and prediction."""

    def __init__(self, cfg: APIConfig):
        self._cfg = cfg
        self._models: dict[int, lgb.Booster] = {}
        self._schemas: dict[int, dict[str, Any]] = {}
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
        """Load model + schema for *patch_id*. Returns True on success."""
        model_path = self._model_path(patch_id)
        if not model_path.exists():
            logger.warning("Model file not found for patch %s: %s", patch_id, model_path)
            return False

        try:
            model = lgb.Booster(model_file=str(model_path))
            schema = load_schema(self._model_dir)
            with self._lock:
                self._models[patch_id] = model
                self._schemas[patch_id] = schema
            logger.info(
                "Loaded model for patch %s (%d features)",
                patch_id, schema["n_features"],
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
        logger.info("Unloaded model for patch %s", patch_id)

    def reload_all(self):
        """Scan the model directory and reload all available models.

        Uses a copy-on-write pattern (BUG-N05): builds the new model/schema
        dicts in local variables, then atomically swaps them in under the
        lock.  This avoids exposing an empty ``_models`` dict between the
        old ``clear()`` and the first ``load_model()``, which would cause
        live ``/predict`` requests to fail with ``ValueError`` during a
        full reload.
        """
        new_models: dict[int, lgb.Booster] = {}
        new_schemas: dict[int, dict[str, Any]] = {}
        count = 0
        for fpath in sorted(self._model_dir.glob("model_patch_*.txt")):
            try:
                pid = int(fpath.stem.replace("model_patch_", ""))
                model_path = self._model_path(pid)
                if not model_path.exists():
                    continue
                model = lgb.Booster(model_file=str(model_path))
                schema = load_schema(self._model_dir)
                new_models[pid] = model
                new_schemas[pid] = schema
                count += 1
            except (ValueError, IndexError, Exception):
                continue
        with self._lock:
            self._models = new_models
            self._schemas = new_schemas
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

        Returns
        -------
        recommendations : list[dict]
            Each dict has ``hero_id``, ``score``, (one-hot based, not calibrated),
            ``pick_probability``, ``win_probability`` (currently None).
        reasoning : str or None
            Human-readable explanation for the top hero.
        """
        # Atomically load (if missing) and capture references under the lock
        # so a concurrent /reload cannot produce a None between check and use.
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

        # model and schema are now local references — the lock is released.
        # LightGBM Booster.predict() is read-only after load, so concurrent
        # threads can use the same Booster safely.

        taken_heroes = ctx.all_taken

        # Score all heroes (1..max_hero_id from the schema) that aren't
        # already taken.  Uses schema["max_hero_id"] instead of the API's
        # own config so the one-hot encoding dimension always matches the
        # trained model — prevents a dimension-mismatch crash when a new
        # hero ships mid-deployment (issue #8).
        max_id = schema["max_hero_id"]
        eligible_hero_ids = [
            hid for hid in range(1, max_id + 1)
            if hid not in taken_heroes
        ]
        if not eligible_hero_ids:
            return [], None

        team_id = radiant_team_id if ctx.recommending_team == 0 else dire_team_id
        enemy_team_id = dire_team_id if ctx.recommending_team == 0 else radiant_team_id

        # Pre-fetch all per-hero aggregate data in 2-3 batch queries,
        # replacing the previous N+1 pattern (~500 queries per request).
        # Gracefully handle an uninitialised DB pool (e.g. during startup)
        # by converting RuntimeError → ValueError so the HTTP layer returns
        # a clean error response instead of a 500.
        try:
            batch = pre_fetch_batch(
                patch_id=patch_id,
                hero_ids=eligible_hero_ids,
                team_id=team_id,
                enemy_team_id=enemy_team_id,
                ctx=ctx,
                account_id=account_id,
            )
        except RuntimeError as e:
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
        scores = model.predict(X)

        # Sort by score descending
        indices = np.argsort(scores)[::-1][:num_recommendations]

        recommendations: list[dict[str, Any]] = []
        top_hero_id = None
        top_score = None

        for idx in indices:
            hid = eligible_hero_ids[idx]
            sc = float(scores[idx])
            if top_hero_id is None:
                top_hero_id = hid
                top_score = sc
            recommendations.append({
                "hero_id": hid,
                "score": sc,
                "pick_probability": None,
                "win_probability": None,
            })

        # Generate reasoning for the top recommendation
        reasoning: str | None = None
        if top_hero_id is not None:
            reasoning = self._build_reasoning(
                top_hero_id, top_score, ctx, patch_id,
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
            bl = batch.baselines.get(hero_id)
            bl_wr = bl["win_rate"] if bl else None

            th = batch.team_hero_agg.get(hero_id)
            th_wr = th["win_rate"] if th else None

            sy = batch.synergy.get(hero_id)
            sy_wr = sy[0] if sy else None

            co = batch.counter.get(hero_id)
            co_wr = co[0] if co else None

            # BUG-003: h2h_row was fetched in pre_fetch_batch but never read.
            h2h = batch.h2h_row
            h2h_wr = h2h["win_rate"] if h2h else None
        else:
            # Fallback (should not happen in production).
            team_id = radiant_team_id if ctx.recommending_team == 0 else dire_team_id
            bl = db_.fetch_baseline(patch_id, hero_id)
            bl_wr = bl["win_rate"] if bl else None
            th = db_.fetch_team_hero_agg(patch_id, team_id, hero_id) if team_id else None
            th_wr = th["win_rate"] if th else None
            sy_wr, _ = db_.fetch_synergy_avg(patch_id, hero_id, ctx.ally_picks)
            co_wr, _ = db_.fetch_counter_avg(patch_id, hero_id, ctx.enemy_picks)
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
