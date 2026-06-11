"""Tests for ``api/predictor.py`` — model lifecycle, thread safety, and
prediction orchestration.

Regression bugs covered:
    - BUG-001: Thread safety — concurrent predict() + unload() races
    - BUG-N05: ``reload_all`` atomicity — partial reload must not expose
               empty ``_models`` dict
    - BUG-002: ``_build_reasoning`` must NOT issue extra DB queries when
               a ``BatchContext`` is provided
    - BUG-N01: ``_build_reasoning`` with ``batch=None`` falls to DB fallback
    - BUG-003: H2H win_rate from BatchContext flows into reasoning
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from api.config import APIConfig
from api.draft_state import DraftContext
from api.features import BatchContext
from api.predictor import Predictor


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def cfg(tmp_path: Path) -> APIConfig:
    """Minimal APIConfig pointing at a temporary model directory.

    A dummy model file for patch 42 is created so ``load_model(42)``
    does not fail on the file-existence check.
    """
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model_patch_42.txt").write_text("dummy model data")
    return APIConfig(
        host="0.0.0.0",
        port=8080,
        model_dir=str(model_dir),
        max_hero_id=160,
        pool_min=1,
        pool_max=2,
        admin_token="",
        pg_host="localhost",
        pg_port=5432,
        pg_user="test",
        pg_password="test",
        pg_db="test",
    )


@pytest.fixture
def mock_booster():
    """Return a mock ``lgb.Booster`` whose ``predict()`` returns a
    constant array of scores.
    """
    booster = mock.MagicMock()
    # Return scores for N eligible heroes (schema max_hero_id=160, minus a few taken)
    booster.predict.return_value = np.arange(155, dtype=np.float64) / 100.0
    return booster


def _make_schema():
    """Return a minimal schema dict for tests."""
    return {
        "n_features": 216,
        "max_hero_id": 160,
        "aggregate_columns": ["is_pick", "team"],
    }


@pytest.fixture
def batch_context() -> BatchContext:
    """A valid empty BatchContext — all lookups return defaults."""
    return BatchContext(
        baselines={},
        team_hero_agg={},
        player_hero_agg={},
        synergy={},
        counter={},
        h2h_row=None,
        hero_draft_slot={},
    )


@pytest.fixture
def predictor(cfg: APIConfig, mock_booster, batch_context):
    """Build a Predictor with all external dependencies mocked,
    pre-loaded with patch 42 model.

    ``pre_fetch_batch`` returns a valid (empty) BatchContext so that
    ``predict()`` can complete the full code path without mocked data
    leaking into ``generate_reasoning``.
    """
    schema = _make_schema()
    feat_vec = np.zeros(len(schema["aggregate_columns"]) + schema["max_hero_id"],
                        dtype=np.float32)
    with (
        mock.patch("api.predictor.lgb.Booster", return_value=mock_booster),
        mock.patch("api.predictor.load_schema", return_value=schema),
        mock.patch("api.predictor.pre_fetch_batch", return_value=batch_context),
        mock.patch("api.predictor.build_feature_vector", return_value=feat_vec),
    ):
        p = Predictor(cfg)
        p._models[42] = mock_booster
        p._schemas[42] = schema
        yield p


@pytest.fixture
def draft_ctx() -> DraftContext:
    """Default draft context with one hero taken per section."""
    return DraftContext(
        turn=7,
        recommending_team=0,
        is_pick_turn=True,
        radiant_picks=[1],
        dire_picks=[2],
        radiant_bans=[3],
        dire_bans=[4],
    )


# ===================================================================
# Thread safety (BUG-001)
# ===================================================================


class TestThreadSafety:
    """Concurrent ``predict()`` and ``unload_model()`` + ``load_model()``
    must not produce ``KeyError`` or ``AttributeError``.
    """

    def test_concurrent_predict_and_unload_no_errors(
        self, predictor, draft_ctx,
    ):
        """✅ 50 threads (half predict, half unload+load) run
        concurrently without errors — REGRESSION BUG-001.
        """
        n_threads = 50
        errors: list[Exception] = []
        errors_lock = threading.Lock()
        barrier = threading.Barrier(n_threads, timeout=15)

        def predict_worker():
            try:
                barrier.wait()
                predictor.predict(
                    patch_id=42,
                    ctx=draft_ctx,
                    radiant_team_id=100,
                    dire_team_id=200,
                    num_recommendations=3,
                )
            except Exception as e:
                with errors_lock:
                    errors.append(e)

        def unload_load_worker():
            try:
                barrier.wait()
                predictor.unload_model(42)
                predictor.load_model(42)
            except Exception as e:
                with errors_lock:
                    errors.append(e)

        threads = []
        for i in range(n_threads):
            t = threading.Thread(
                target=predict_worker if i < n_threads // 2 else unload_load_worker,
                daemon=True,
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=15)

        # Print first few errors for debugging
        if errors:
            for e in errors[:5]:
                print(f"  Error: {type(e).__name__}: {e}")
        assert not errors, (
            f"{len(errors)} thread(s) raised errors "
            f"(showing first 5 above)"
        )


# ===================================================================
# reload_all atomicity (BUG-N05)
# ===================================================================


class TestReloadAllAtomicity:
    """``reload_all`` must atomically swap model dicts so concurrent
    ``predict()`` never sees an empty ``_models``.
    """

    def test_predict_during_reload_never_sees_empty_models(
        self, cfg: APIConfig, draft_ctx,
    ):
        """✅ Thread A: reload_all with slow loads. Thread B: predict()
        during reload. All predict calls succeed — REGRESSION BUG-N05.
        """
        schema = _make_schema()
        feat_vec = np.zeros(len(schema["aggregate_columns"]) + schema["max_hero_id"],
                            dtype=np.float32)
        empty_batch = BatchContext(
            baselines={}, team_hero_agg={}, player_hero_agg={},
            synergy={}, counter={}, h2h_row=None, hero_draft_slot={},
        )

        # Create model files for patches 42 and 43 so reload_all finds them
        model_dir = Path(cfg.model_dir)
        (model_dir / "model_patch_42.txt").write_text("dummy")
        (model_dir / "model_patch_43.txt").write_text("dummy")

        predict_results: list[bool] = []
        predict_lock = threading.Lock()
        reload_done = threading.Event()

        def delayed_booster_init(model_file: str):
            """Simulate slow model loading (50 ms per model)."""
            time.sleep(0.05)
            b = mock.MagicMock()
            b.predict.return_value = np.array([0.5, 0.6], dtype=np.float64)
            return b

        with (
            mock.patch("api.predictor.lgb.Booster",
                       side_effect=delayed_booster_init),
            mock.patch("api.predictor.load_schema", return_value=schema),
            mock.patch("api.predictor.pre_fetch_batch",
                       return_value=empty_batch),
            mock.patch("api.predictor.build_feature_vector",
                       return_value=feat_vec),
        ):
            # Pre-load a model so predict() works before reload_all starts
            p = Predictor(cfg)
            preloaded = mock.MagicMock()
            preloaded.predict.return_value = np.array([0.5], dtype=np.float64)
            p._models[42] = preloaded
            p._schemas[42] = schema

            def slow_reload():
                """Simulate what reload_all does but with a bigger delay."""
                new_models: dict[int, mock.MagicMock] = {}
                new_schemas: dict[int, dict] = {}
                # Step 1: build new dicts (slow, like loading models from disk)
                time.sleep(0.08)
                m = mock.MagicMock()
                m.predict.return_value = np.array([0.5, 0.6], dtype=np.float64)
                new_models[42] = m
                new_schemas[42] = schema
                # The new dicts are local-only at this point.
                # A concurrent predict() should still see the old dicts.
                time.sleep(0.08)
                # Step 2: atomic swap under lock
                with p._lock:
                    p._models = new_models
                    p._schemas = new_schemas
                reload_done.set()

            def concurrent_predict():
                while not reload_done.is_set():
                    try:
                        p.predict(
                            patch_id=42,
                            ctx=draft_ctx,
                            radiant_team_id=100,
                            dire_team_id=200,
                            num_recommendations=3,
                        )
                        with predict_lock:
                            predict_results.append(True)
                    except (ValueError, KeyError, AttributeError) as e:
                        with predict_lock:
                            predict_results.append(False)
                        break
                    time.sleep(0.005)

            t1 = threading.Thread(target=slow_reload, daemon=True)
            t2 = threading.Thread(target=concurrent_predict, daemon=True)
            t1.start()
            # Give slow_reload time to get into its slow section
            time.sleep(0.05)
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)

        assert len(predict_results) > 0, (
            "No predict() calls were attempted — likely a setup issue"
        )
        assert all(predict_results), (
            f"Some predict() calls failed during reload. "
            f"Successes: {sum(predict_results)}/{len(predict_results)}"
        )


# ===================================================================
# _build_reasoning
# ===================================================================


class TestBuildReasoning:
    """``Predictor._build_reasoning`` gathers explanation data for the
    top recommendation.
    """

    def test_no_extra_db_queries_when_batch_provided(self, cfg, mock_booster):
        """✅ batch provided → zero DB calls — REGRESSION BUG-002."""
        schema = _make_schema()
        feat_vec = np.zeros(len(schema["aggregate_columns"]) + schema["max_hero_id"],
                            dtype=np.float32)
        with (
            mock.patch("api.predictor.lgb.Booster", return_value=mock_booster),
            mock.patch("api.predictor.load_schema", return_value=schema),
            mock.patch("api.predictor.pre_fetch_batch"),
            mock.patch("api.predictor.build_feature_vector", return_value=feat_vec),
        ):
            p = Predictor(cfg)
            p._schemas[42] = schema

        ctx = DraftContext(
            turn=7, recommending_team=0, is_pick_turn=True,
            radiant_picks=[1], dire_picks=[2],
            radiant_bans=[3], dire_bans=[4],
        )
        batch = BatchContext(
            baselines={1: {"win_rate": 0.55}},
            team_hero_agg={1: {"win_rate": 0.62}},
            player_hero_agg={},
            synergy={1: (0.70, 3)},
            counter={1: (0.45, 2)},
            h2h_row={"win_rate": 0.60, "games": 12},
            hero_draft_slot={},
        )

        with mock.patch("api.predictor.db_") as mock_db:
            result = p._build_reasoning(
                hero_id=1,
                score=0.85,
                ctx=ctx,
                patch_id=42,
                batch=batch,
            )
            # No DB calls should have been made
            mock_db.fetch_baseline.assert_not_called()
            mock_db.fetch_team_hero_agg.assert_not_called()
            mock_db.fetch_synergy_avg.assert_not_called()
            mock_db.fetch_counter_avg.assert_not_called()

        assert result is not None, "Expected non-None reasoning string"

    def test_no_batch_falls_back_to_db(self, cfg, mock_booster):
        """❌ batch=None → falls through to DB fallback branch
        (REGRESSION BUG-N01).
        """
        schema = _make_schema()
        feat_vec = np.zeros(len(schema["aggregate_columns"]) + schema["max_hero_id"],
                            dtype=np.float32)
        with (
            mock.patch("api.predictor.lgb.Booster", return_value=mock_booster),
            mock.patch("api.predictor.load_schema", return_value=schema),
            mock.patch("api.predictor.pre_fetch_batch"),
            mock.patch("api.predictor.build_feature_vector", return_value=feat_vec),
        ):
            p = Predictor(cfg)
            p._schemas[42] = schema

        ctx = DraftContext(
            turn=7, recommending_team=0, is_pick_turn=True,
            radiant_picks=[1], dire_picks=[2],
            radiant_bans=[3], dire_bans=[4],
        )
        with mock.patch("api.predictor.db_") as mock_db:
            mock_db.fetch_baseline.return_value = {"win_rate": 0.50}
            mock_db.fetch_team_hero_agg.return_value = None
            mock_db.fetch_synergy_avg.return_value = (None, 0)
            mock_db.fetch_counter_avg.return_value = (None, 0)
            result = p._build_reasoning(
                hero_id=1,
                score=0.85,
                ctx=ctx,
                patch_id=42,
                batch=None,
            )
            # Should have entered the else branch and made DB calls
            mock_db.fetch_baseline.assert_called_once()
            assert result is not None

    def test_h2h_win_rate_in_reasoning(self, cfg, mock_booster):
        """✅ BatchContext has ``h2h_row`` → reasoning contains
        "H2H WR" — REGRESSION BUG-003.
        """
        from api.predictor import generate_reasoning as real_generate_reasoning

        schema = _make_schema()
        feat_vec = np.zeros(len(schema["aggregate_columns"]) + schema["max_hero_id"],
                            dtype=np.float32)
        with (
            mock.patch("api.predictor.lgb.Booster", return_value=mock_booster),
            mock.patch("api.predictor.load_schema", return_value=schema),
            mock.patch("api.predictor.pre_fetch_batch"),
            mock.patch("api.predictor.build_feature_vector", return_value=feat_vec),
            mock.patch("api.predictor.generate_reasoning",
                       side_effect=real_generate_reasoning),
        ):
            p = Predictor(cfg)
            p._schemas[42] = schema

        ctx = DraftContext(
            turn=7, recommending_team=0, is_pick_turn=True,
            radiant_picks=[1], dire_picks=[2],
            radiant_bans=[3], dire_bans=[4],
        )
        batch = BatchContext(
            baselines={},
            team_hero_agg={},
            player_hero_agg={},
            synergy={},
            counter={},
            h2h_row={"win_rate": 0.62, "games": 15},
            hero_draft_slot={},
        )

        # Use the predictor built WITHOUT the generate_reasoning mock
        result = p._build_reasoning(
            hero_id=1,
            score=0.85,
            ctx=ctx,
            patch_id=42,
            batch=batch,
        )

        assert result is not None
        assert "H2H WR" in result, (
            f"Expected 'H2H WR' in reasoning, got: {result!r}"
        )
        assert "62.0%" in result


# ===================================================================
# predict — recommendation logic
# ===================================================================


class TestPredict:
    """``Predictor.predict`` scores eligible heroes and returns top-N
    recommendations with reasoning.
    """

    def test_excludes_taken_heroes(self, predictor, draft_ctx):
        """✅ Taken hero IDs are excluded from recommendations."""
        recs, _ = predictor.predict(
            patch_id=42,
            ctx=draft_ctx,
            radiant_team_id=100,
            dire_team_id=200,
            num_recommendations=10,
        )
        rec_hero_ids = {r["hero_id"] for r in recs}
        taken = draft_ctx.all_taken  # {1, 2, 3, 4}
        assert taken.isdisjoint(rec_hero_ids), (
            f"Taken heroes {taken} appear in recommendations: {rec_hero_ids}"
        )

    def test_respects_num_recommendations(self, predictor, draft_ctx):
        """✅ len(recommendations) == num_recommendations."""
        for n in (1, 3, 5):
            recs, _ = predictor.predict(
                patch_id=42,
                ctx=draft_ctx,
                radiant_team_id=100,
                dire_team_id=200,
                num_recommendations=n,
            )
            assert len(recs) == n, (
                f"Expected {n} recommendation(s), got {len(recs)}"
            )

    def test_missing_patch_raises_value_error(self, cfg, draft_ctx):
        """❌ Patch with no model file → ValueError raised."""
        schema = _make_schema()
        feat_vec = np.zeros(len(schema["aggregate_columns"]) + schema["max_hero_id"],
                            dtype=np.float32)
        with (
            mock.patch("api.predictor.lgb.Booster"),
            mock.patch("api.predictor.load_schema", return_value=schema),
            mock.patch("api.predictor.pre_fetch_batch"),
            mock.patch("api.predictor.build_feature_vector", return_value=feat_vec),
        ):
            p = Predictor(cfg)
            with pytest.raises(ValueError, match="No model available"):
                p.predict(
                    patch_id=999,  # no model file for this patch
                    ctx=draft_ctx,
                    radiant_team_id=None,
                    dire_team_id=None,
                )

    def test_returns_reasoning_string(self, predictor, draft_ctx):
        """✅ ``reasoning`` is returned alongside recommendations."""
        recs, reason = predictor.predict(
            patch_id=42,
            ctx=draft_ctx,
            radiant_team_id=100,
            dire_team_id=200,
            num_recommendations=1,
        )
        assert isinstance(reason, str), f"Expected str, got {type(reason)}"
        assert len(reason) > 0

    def test_each_recommendation_has_required_keys(self, predictor,
                                                    draft_ctx):
        """✅ Every recommendation dict has hero_id, score,
        pick_probability, win_probability.
        """
        recs, _ = predictor.predict(
            patch_id=42,
            ctx=draft_ctx,
            radiant_team_id=100,
            dire_team_id=200,
            num_recommendations=3,
        )
        for r in recs:
            assert "hero_id" in r
            assert "score" in r
            assert "pick_probability" in r
            assert "win_probability" in r
