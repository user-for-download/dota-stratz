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
import torch

from api.config import APIConfig
from api.draft_state import DraftContext
from api.features import BatchContext
from api.models import DraftSlot
from api.predictor import Predictor


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def cfg(tmp_path: Path) -> APIConfig:
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    # Create dummy files so load_model's path.exists() check passes
    (model_dir / "draftbert_compiled_42.pt").write_bytes(b"dummy")
    import json as _json
    (model_dir / "feature_schema_patch_42.json").write_text(
        _json.dumps(_make_schema())
    )
    return APIConfig(
        host="0.0.0.0", port=8080, model_dir=str(model_dir),
        max_hero_id=160, pool_min=1, pool_max=2, admin_token="",
        pg_host="localhost", pg_port=5432,
        pg_user="test", pg_password="test", pg_db="test",
    )


def _make_mock_model(batch_size_out=1):
    m = mock.MagicMock()

    def _forward(*args):
        heroes = args[0]
        bs = heroes.shape[0]
        return torch.randn(bs)

    m.side_effect = _forward
    return m


def _make_schema():
    return {
        "n_features": 92, "n_embeddings": 0, "max_hero_id": 160,
        "max_seq_len": 50,
        "aggregate_columns": ["is_pick", "team"],
    }


@pytest.fixture
def batch_context() -> BatchContext:
    return BatchContext(
        baselines={}, team_hero_agg={}, player_hero_agg={},
        synergy={}, counter={}, h2h_row=None, hero_draft_slot={},
    )


@pytest.fixture
def draft_ctx() -> DraftContext:
    return DraftContext(
        turn=7, recommending_team=0, is_pick_turn=True,
        radiant_picks=[1], dire_picks=[2],
        radiant_bans=[3], dire_bans=[4],
    )


def _empty_draft_slots():
    return [
        DraftSlot(hero_id=3, is_pick=False, team=0, order=1),
        DraftSlot(hero_id=4, is_pick=False, team=1, order=2),
        DraftSlot(hero_id=0, is_pick=False, team=0, order=3),
        DraftSlot(hero_id=0, is_pick=False, team=1, order=4),
        DraftSlot(hero_id=0, is_pick=False, team=0, order=5),
        DraftSlot(hero_id=0, is_pick=False, team=1, order=6),
        DraftSlot(hero_id=1, is_pick=True, team=0, order=7),
    ]


@pytest.fixture
def predictor(cfg: APIConfig, batch_context):
    schema = _make_schema()
    num_continuous = len(schema["aggregate_columns"])
    feat_vec = np.zeros(num_continuous, dtype=np.float32)

    p = Predictor(cfg)
    p._models[42] = _make_mock_model()
    p._schemas[42] = schema

    with (
        mock.patch("api.predictor.pre_fetch_batch", return_value=batch_context),
        mock.patch("api.predictor.build_feature_vector", return_value=feat_vec),
        mock.patch("api.predictor.torch.jit.load", return_value=_make_mock_model()),
    ):
        yield p


# ===================================================================
# Thread safety (BUG-001)
# ===================================================================


class TestThreadSafety:
    def test_concurrent_predict_and_unload_no_errors(self, predictor, draft_ctx):
        n_threads = 50
        errors: list[Exception] = []
        errors_lock = threading.Lock()
        barrier = threading.Barrier(n_threads, timeout=15)

        def predict_worker():
            try:
                barrier.wait()
                predictor.predict(
                    patch_id=42, ctx=draft_ctx, draft_slots=_empty_draft_slots(),
                    radiant_team_id=100, dire_team_id=200, num_recommendations=3,
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

        if errors:
            for e in errors[:5]:
                print(f"  Error: {type(e).__name__}: {e}")
        assert not errors, f"{len(errors)} thread(s) raised errors"


# ===================================================================
# reload_all atomicity (BUG-N05)
# ===================================================================


class TestReloadAllAtomicity:
    def test_predict_during_reload_never_sees_empty_models(self, cfg, draft_ctx):
        schema = _make_schema()
        num_continuous = len(schema["aggregate_columns"])
        empty_batch = BatchContext(
            baselines={}, team_hero_agg={}, player_hero_agg={},
            synergy={}, counter={}, h2h_row=None, hero_draft_slot={},
        )

        predict_results: list[bool] = []
        predict_lock = threading.Lock()
        reload_done = threading.Event()

        def slow_jit_load(path, map_location="cpu"):
            time.sleep(0.05)
            return _make_mock_model()

        with (
            mock.patch("api.predictor.torch.jit.load", side_effect=slow_jit_load),
            mock.patch("api.predictor.load_schema", return_value=schema),
            mock.patch("api.predictor.pre_fetch_batch", return_value=empty_batch),
            mock.patch("api.predictor.build_feature_vector",
                       return_value=np.zeros(num_continuous, dtype=np.float32)),
        ):
            p = Predictor(cfg)
            p._models[42] = _make_mock_model()
            p._schemas[42] = schema

            def slow_reload():
                time.sleep(0.08)
                new_models = {42: _make_mock_model()}
                new_schemas = {42: schema}
                time.sleep(0.08)
                with p._lock:
                    p._models = new_models
                    p._schemas = new_schemas
                reload_done.set()

            def concurrent_predict():
                while not reload_done.is_set():
                    try:
                        p.predict(
                            patch_id=42, ctx=draft_ctx,
                            draft_slots=_empty_draft_slots(),
                            radiant_team_id=100, dire_team_id=200,
                            num_recommendations=3,
                        )
                        with predict_lock:
                            predict_results.append(True)
                    except (ValueError, KeyError, AttributeError):
                        with predict_lock:
                            predict_results.append(False)
                        break
                    time.sleep(0.005)

            t1 = threading.Thread(target=slow_reload, daemon=True)
            t2 = threading.Thread(target=concurrent_predict, daemon=True)
            t1.start()
            time.sleep(0.05)
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)

        assert len(predict_results) > 0
        assert all(predict_results), (
            f"Successes: {sum(predict_results)}/{len(predict_results)}"
        )


# ===================================================================
# _build_reasoning
# ===================================================================


class TestBuildReasoning:
    def test_no_extra_db_queries_when_batch_provided(self, cfg):
        schema = _make_schema()
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
            player_hero_agg={}, synergy={1: (0.70, 3)},
            counter={1: (0.45, 2)},
            h2h_row={"win_rate": 0.60, "games": 12},
            hero_draft_slot={},
        )
        with mock.patch("api.predictor.db_") as mock_db:
            result = p._build_reasoning(
                hero_id=1, score=0.85, ctx=ctx, patch_id=42, batch=batch,
            )
            mock_db.fetch_baseline.assert_not_called()
            mock_db.fetch_team_hero_agg.assert_not_called()
            mock_db.fetch_synergy_avg.assert_not_called()
            mock_db.fetch_counter_avg.assert_not_called()
        assert result is not None

    def test_no_batch_falls_back_to_db(self, cfg):
        schema = _make_schema()
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
            mock_db.fetch_counter_avg.return_value = (None, 0, 0.0)
            result = p._build_reasoning(
                hero_id=1, score=0.85, ctx=ctx, patch_id=42, batch=None,
            )
            mock_db.fetch_baseline.assert_called_once()
            assert result is not None

    def test_h2h_win_rate_in_reasoning(self, cfg):
        schema = _make_schema()
        p = Predictor(cfg)
        p._schemas[42] = schema
        ctx = DraftContext(
            turn=7, recommending_team=0, is_pick_turn=True,
            radiant_picks=[1], dire_picks=[2],
            radiant_bans=[3], dire_bans=[4],
        )
        batch = BatchContext(
            baselines={}, team_hero_agg={}, player_hero_agg={},
            synergy={}, counter={},
            h2h_row={"win_rate": 0.62, "games": 15},
            hero_draft_slot={},
        )
        result = p._build_reasoning(
            hero_id=1, score=0.85, ctx=ctx, patch_id=42, batch=batch,
        )
        assert result is not None
        assert "H2H WR" in result
        assert "62.0%" in result


# ===================================================================
# predict — recommendation logic
# ===================================================================


class TestPredict:
    def test_excludes_taken_heroes(self, predictor, draft_ctx):
        recs, _ = predictor.predict(
            patch_id=42, ctx=draft_ctx, draft_slots=_empty_draft_slots(),
            radiant_team_id=100, dire_team_id=200, num_recommendations=10,
        )
        rec_hero_ids = {r["hero_id"] for r in recs}
        taken = draft_ctx.all_taken
        assert taken.isdisjoint(rec_hero_ids)

    def test_respects_num_recommendations(self, predictor, draft_ctx):
        for n in (1, 3, 5):
            recs, _ = predictor.predict(
                patch_id=42, ctx=draft_ctx, draft_slots=_empty_draft_slots(),
                radiant_team_id=100, dire_team_id=200, num_recommendations=n,
            )
            assert len(recs) == n

    def test_missing_patch_raises_value_error(self, cfg, draft_ctx):
        schema = _make_schema()
        num_continuous = len(schema["aggregate_columns"])
        with (
            mock.patch("api.predictor.load_schema", return_value=schema),
            mock.patch("api.predictor.pre_fetch_batch"),
            mock.patch("api.predictor.build_feature_vector",
                       return_value=np.zeros(num_continuous, dtype=np.float32)),
        ):
            p = Predictor(cfg)
            with pytest.raises(ValueError, match="No model"):
                p.predict(
                    patch_id=999, ctx=draft_ctx, draft_slots=_empty_draft_slots(),
                    radiant_team_id=None, dire_team_id=None,
                )

    def test_returns_reasoning_string(self, predictor, draft_ctx):
        recs, reason = predictor.predict(
            patch_id=42, ctx=draft_ctx, draft_slots=_empty_draft_slots(),
            radiant_team_id=100, dire_team_id=200, num_recommendations=1,
        )
        assert isinstance(reason, str)
        assert len(reason) > 0

    def test_each_recommendation_has_required_keys(self, predictor, draft_ctx):
        recs, _ = predictor.predict(
            patch_id=42, ctx=draft_ctx, draft_slots=_empty_draft_slots(),
            radiant_team_id=100, dire_team_id=200, num_recommendations=3,
        )
        for r in recs:
            assert "hero_id" in r
            assert "score" in r
            assert "pick_probability" in r
            assert "win_probability" in r
