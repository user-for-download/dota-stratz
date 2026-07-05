"""FastAPI endpoint tests for app.py.

Exercises request validation, error handling, and response serialisation
for all three endpoints (GET /health, POST /predict, POST /reload/{patch_id})
by injecting a mocked Predictor and APIConfig onto ``app.state`` and using a
no-op lifespan so no real database or PyTorch models are involved.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest import mock
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.app import app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_DRAFT: list[dict[str, Any]] = [
    {"hero_id": 1, "is_pick": False, "team": 0, "order": 1},
    {"hero_id": 2, "is_pick": False, "team": 1, "order": 2},
    {"hero_id": 3, "is_pick": False, "team": 0, "order": 3},
    {"hero_id": 4, "is_pick": False, "team": 1, "order": 4},
    {"hero_id": 5, "is_pick": True,  "team": 0, "order": 5},
]

_DEFAULT_RECOMMENDATIONS: list[dict[str, Any]] = [
    {"hero_id": 1, "score": 0.95, "pick_probability": None, "win_probability": None},
]
_DEFAULT_REASONING = "Hero 1: baseline win rate 52.0% | score 0.95"


def _make_predict_body(**overrides: Any) -> dict[str, Any]:
    """Build a valid /predict request body."""
    body: dict[str, Any] = {
        "patch_id": 8,
        "first_pick_team": 0,
        "draft": _VALID_DRAFT,
        "num_recommendations": 5,
        "account_id": None,
        "radiant_team_id": None,
        "dire_team_id": None,
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _noop_lifespan():
    """Replace the real lifespan so startup/shutdown are no-ops.

    TestClient does not run lifespan by default when used outside a
    ``with`` block, but this fixture ensures safety even if callers
    use ``with TestClient(app)`` context-manager style.
    """
    original = app.router.lifespan_context

    @asynccontextmanager
    async def _noop(_app):
        yield

    app.router.lifespan_context = _noop
    yield
    app.router.lifespan_context = original


@pytest.fixture
def mock_predictor() -> MagicMock:
    """Predictor mock with sensible defaults for happy-path tests.

    Test methods that need custom behaviour can reassign the mock's
    return_value / side_effect on the fixture returned by the
    ``_setup_app_state`` fixture.
    """
    p = MagicMock()
    p.loaded_patches.return_value = [1, 2, 3]
    p.predict.return_value = (_DEFAULT_RECOMMENDATIONS, _DEFAULT_REASONING)
    p.load_model.return_value = True
    p.unload_model.return_value = None
    return p


@pytest.fixture(autouse=True)
def _setup_app_state(mock_predictor: MagicMock):
    """Attach mocked Predictor and APIConfig to app.state."""
    cfg = MagicMock()
    cfg.admin_token = "test-token"
    app.state.cfg = cfg
    app.state.predictor = mock_predictor
    yield


@pytest.fixture
def client() -> TestClient:
    """Return a TestClient bound to the real (but lifespan-noop'd) app."""
    return TestClient(app)


# ===================================================================
# GET /health
# ===================================================================


class TestHealth:
    """``GET /health`` — service health check."""

    def test_ok(self, client: TestClient, mock_predictor: MagicMock):
        """Returns 200 with status=ok and loaded patches when DB is available."""
        mock_predictor.loaded_patches.return_value = [1, 2, 3]
        with mock.patch("api.app.get_conn") as mock_get_conn, \
             mock.patch("api.app.put_conn"):
            mock_conn = mock.MagicMock()
            mock_get_conn.return_value = mock_conn
            resp = client.get("/health")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "ok"
            assert body["patch_models_loaded"] == [1, 2, 3]

    def test_no_models(self, client: TestClient, mock_predictor: MagicMock):
        """No models loaded → empty list."""
        mock_predictor.loaded_patches.return_value = []
        with mock.patch("api.app.get_conn") as mock_get_conn, \
             mock.patch("api.app.put_conn"):
            mock_get_conn.return_value = mock.MagicMock()
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["patch_models_loaded"] == []

    def test_degraded_when_db_down(self, client: TestClient, mock_predictor: MagicMock):
        """DB connection failure → status=degraded."""
        mock_predictor.loaded_patches.return_value = [1]
        with mock.patch("api.app.get_conn", side_effect=RuntimeError("DB down")):
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "degraded"


# ===================================================================
# POST /predict
# ===================================================================


class TestPredict:
    """``POST /predict`` — score heroes for the current draft state."""

    def test_basic_predict(self, client: TestClient, mock_predictor: MagicMock):
        """✅ Valid draft → 200 with recommendations and reasoning."""
        resp = client.post("/predict", json=_make_predict_body())
        assert resp.status_code == 200
        body = resp.json()
        assert body["patch_id"] == 8
        assert body["turn"] == 6  # 5 slots filled → turn 6
        assert body["recommending_team"] == 1  # slot 6 = P1 (Dire pick)
        assert len(body["recommendations"]) == 1
        assert body["recommendations"][0]["hero_id"] == 1
        assert body["recommendations"][0]["score"] == 0.95
        assert body["reasoning"] == _DEFAULT_REASONING

        # Verify the predictor was called with the expected args
        mock_predictor.predict.assert_called_once()
        _call_kwargs = mock_predictor.predict.call_args.kwargs
        assert _call_kwargs["patch_id"] == 8
        assert _call_kwargs["num_recommendations"] == 5
        assert _call_kwargs["account_id"] is None

    def test_num_recommendations(self, client: TestClient, mock_predictor: MagicMock):
        """✅ Custom ``num_recommendations`` is passed through."""
        client.post("/predict", json=_make_predict_body(num_recommendations=3))
        assert mock_predictor.predict.call_args.kwargs["num_recommendations"] == 3

    def test_with_account_id(self, client: TestClient, mock_predictor: MagicMock):
        """✅ ``account_id`` is forwarded to predictor."""
        body = _make_predict_body(account_id=12345)
        client.post("/predict", json=body)
        assert mock_predictor.predict.call_args.kwargs["account_id"] == 12345

    def test_with_team_ids(self, client: TestClient, mock_predictor: MagicMock):
        """✅ Team IDs are forwarded to predictor."""
        body = _make_predict_body(radiant_team_id=100, dire_team_id=200)
        client.post("/predict", json=body)
        kwargs = mock_predictor.predict.call_args.kwargs
        assert kwargs["radiant_team_id"] == 100
        assert kwargs["dire_team_id"] == 200

    def test_draft_already_complete(self, client: TestClient):
        """Full draft (recommending_team == -1) defaults to team 0 and returns 200."""
        full_draft = [
            {"hero_id": 1, "is_pick": False, "team": 0, "order": 1},
            {"hero_id": 2, "is_pick": False, "team": 1, "order": 2},
            {"hero_id": 3, "is_pick": False, "team": 0, "order": 3},
            {"hero_id": 4, "is_pick": False, "team": 1, "order": 4},
            {"hero_id": 5, "is_pick": True,  "team": 0, "order": 5},
            {"hero_id": 6, "is_pick": True,  "team": 1, "order": 6},
            {"hero_id": 7, "is_pick": True,  "team": 1, "order": 7},
            {"hero_id": 8, "is_pick": True,  "team": 0, "order": 8},
            {"hero_id": 9, "is_pick": True,  "team": 0, "order": 9},
            {"hero_id": 10, "is_pick": True,  "team": 1, "order": 10},
            {"hero_id": 11, "is_pick": False, "team": 0, "order": 11},
            {"hero_id": 12, "is_pick": False, "team": 1, "order": 12},
            {"hero_id": 13, "is_pick": False, "team": 0, "order": 13},
            {"hero_id": 14, "is_pick": False, "team": 1, "order": 14},
            {"hero_id": 15, "is_pick": False, "team": 0, "order": 15},
            {"hero_id": 16, "is_pick": False, "team": 1, "order": 16},
            {"hero_id": 17, "is_pick": True,  "team": 0, "order": 17},
            {"hero_id": 18, "is_pick": True,  "team": 1, "order": 18},
            {"hero_id": 19, "is_pick": True,  "team": 0, "order": 19},
            {"hero_id": 20, "is_pick": True,  "team": 1, "order": 20},
        ]
        resp = client.post("/predict", json=_make_predict_body(draft=full_draft))
        assert resp.status_code == 200
        body = resp.json()
        assert body["recommending_team"] == 0
        assert len(body["recommendations"]) == 1

    def test_invalid_draft(self, client: TestClient):
        """❌ Invalid draft (wrong team for slot) → 422."""
        bad_draft = [
            {"hero_id": 1, "is_pick": False, "team": 1, "order": 1},  # Slot 1 must be team 0
        ]
        resp = client.post("/predict", json=_make_predict_body(draft=bad_draft))
        assert resp.status_code == 422
        assert "team" in resp.json()["detail"].lower()

    def test_model_not_found(self, client: TestClient, mock_predictor: MagicMock):
        """❌ Predictor raises ValueError → 404."""
        mock_predictor.predict.side_effect = ValueError("No model for patch 8")
        resp = client.post("/predict", json=_make_predict_body(patch_id=8))
        assert resp.status_code == 404
        assert "No model" in resp.json()["detail"]

    def test_db_pool_not_ready(self, client: TestClient, mock_predictor: MagicMock):
        """❌ RuntimeError from pre_fetch_batch → 404 with pool message."""
        mock_predictor.predict.side_effect = ValueError(
            "Database pool is not available: pool not initialised. "
            "The service may still be starting up."
        )
        resp = client.post("/predict", json=_make_predict_body())
        assert resp.status_code == 404
        assert "Database pool" in resp.json()["detail"]

    def test_draft_validation_errors(self, client: TestClient):
        """❌ Duplicate hero IDs → 422."""
        dup_draft = [
            {"hero_id": 1, "is_pick": False, "team": 0, "order": 1},
            {"hero_id": 1, "is_pick": False, "team": 1, "order": 2},  # same hero_id
        ]
        resp = client.post("/predict", json=_make_predict_body(draft=dup_draft))
        assert resp.status_code == 422
        assert "Duplicate" in resp.json()["detail"]


# ===================================================================
# POST /reload/{patch_id}
# ===================================================================


class TestReload:
    """``POST /reload/{patch_id}`` — hot-reload a model (admin, requires token)."""

    def test_success_with_valid_token(self, client: TestClient, mock_predictor: MagicMock):
        """✅ Valid Bearer token → 200 and model reloaded."""
        resp = client.post("/reload/42", headers={"Authorization": "Bearer test-token"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["patch_id"] == 42
        assert "reloaded" in body["message"].lower()

        mock_predictor.load_model.assert_called_once_with(42)

    def test_no_token_sent(self, client: TestClient):
        """❌ No Authorization header → 403."""
        resp = client.post("/reload/42")
        assert resp.status_code == 403
        assert "token" in resp.json()["detail"].lower()

    def test_wrong_token(self, client: TestClient):
        """❌ Wrong token → 403."""
        resp = client.post(
            "/reload/42",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 403

    def test_bad_auth_scheme(self, client: TestClient):
        """❌ Non-Bearer scheme → 403."""
        resp = client.post(
            "/reload/42",
            headers={"Authorization": "Basic dGVzdDp0ZXN0"},
        )
        assert resp.status_code == 403

    def test_no_auth_when_token_blank(self, client: TestClient, _setup_app_state):
        """admin_token="" → reload rejected (token not configured)."""
        cfg = MagicMock()
        cfg.admin_token = ""
        app.state.cfg = cfg

        resp = client.post("/reload/42")
        assert resp.status_code == 403

    def test_model_file_not_found(self, client: TestClient, mock_predictor: MagicMock):
        """❌ load_model returns False → 404."""
        mock_predictor.load_model.return_value = False
        resp = client.post("/reload/42", headers={"Authorization": "Bearer test-token"})
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_reload_different_patches(self, client: TestClient, mock_predictor: MagicMock):
        """Reload multiple patch IDs independently."""
        client.post("/reload/1", headers={"Authorization": "Bearer test-token"})
        client.post("/reload/2", headers={"Authorization": "Bearer test-token"})

        assert mock_predictor.load_model.call_count == 2
        mock_predictor.load_model.assert_any_call(1)
        mock_predictor.load_model.assert_any_call(2)
