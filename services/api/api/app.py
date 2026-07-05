"""FastAPI application entry point for the ML Inference API.

Endpoints:
  POST /predict       — Score heroes for the current draft state.
  POST /predict-match — Evaluate a completed 5v5 composition.
  GET  /health        — Health check with loaded patches.
  POST /reload/{patch_id} — Hot-reload a model (admin, requires token).
"""

from __future__ import annotations

import asyncio
import hmac
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Header, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Use ORJSONResponse for faster serialization if orjson is installed
try:
    import orjson  # noqa: F401
    from fastapi.responses import ORJSONResponse
    _default_response_class = ORJSONResponse
except (ImportError, ModuleNotFoundError):
    _default_response_class = None

from .config import APIConfig
from .db import close_pool, get_conn, init_pool, put_conn
from .draft_state import build_draft_context
from .models import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    HeroScore,
    ReloadResponse,
)
from .predictor import Predictor

# Logging setup at module level so it applies regardless of ASGI server
# behaviour — basicConfig is a no-op after the root logger has handlers.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown lifecycle."""
    # Startup — construct config and predictor INSIDE lifespan so they are
    # not created at module import time (BUG-011). Store on app.state for
    # endpoint access via request.app.state.
    cfg = APIConfig()
    predictor = Predictor(cfg)
    app.state.cfg = cfg
    app.state.predictor = predictor

    init_pool(cfg)
    n_loaded = predictor.reload_all()
    logger.info(
        "API started on %s:%s | %d models loaded",
        cfg.host, cfg.port, n_loaded,
    )
    yield
    # Shutdown
    close_pool()
    logger.info("API shut down.")


app_kwargs = dict(
    title="dota-stratz ML Inference API",
    version="1.0.0",
    lifespan=lifespan,
)
if _default_response_class is not None:
    app_kwargs["default_response_class"] = _default_response_class

app = FastAPI(**app_kwargs)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:80", "http://127.0.0.1:80"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    predictor: Predictor = request.app.state.predictor
    # Verify DB connection is alive
    db_ok = False
    conn = None
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        db_ok = True
    except Exception:
        logger.warning("Health check: DB connection failed")
    finally:
        if conn is not None:
            put_conn(conn)
    status = "ok" if db_ok else "degraded"
    return HealthResponse(
        status=status,
        patch_models_loaded=predictor.loaded_patches(),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    predictor: Predictor = request.app.state.predictor
    try:
        ctx = build_draft_context(req.draft, patch_id=req.patch_id, first_pick_team=req.first_pick_team)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Allow overriding which team to recommend for
    if req.for_team is not None:
        ctx.recommending_team = req.for_team
    elif ctx.recommending_team == -1:
        # Default to Radiant if the draft is complete and for_team is not specified
        ctx.recommending_team = 0

    try:
        recommendations, reasoning = predictor.predict(
            patch_id=req.patch_id,
            ctx=ctx,
            draft_slots=req.draft,
            radiant_team_id=req.radiant_team_id,
            dire_team_id=req.dire_team_id,
            num_recommendations=req.num_recommendations,
            account_id=req.account_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return PredictResponse(
        patch_id=req.patch_id,
        turn=ctx.turn,
        recommending_team=ctx.recommending_team,
        recommendations=[HeroScore(**r) for r in recommendations],
        reasoning=reasoning,
    )


@app.post("/reload/{patch_id}", response_model=ReloadResponse)
def reload_model(
    patch_id: int,
    request: Request,
    authorization: str = Header(None),
):
    predictor: Predictor = request.app.state.predictor
    cfg: APIConfig = request.app.state.cfg

    # Validate admin token
    if not cfg.admin_token:
        logger.warning(
            "STRATZ_ADMIN_TOKEN not set — rejecting /reload requests for security"
        )
        raise HTTPException(
            status_code=403,
            detail="STRATZ_ADMIN_TOKEN not configured",
        )
    token = ""
    if authorization and authorization.startswith("Bearer "):
        token = authorization[len("Bearer "):]
    if not hmac.compare_digest(token, cfg.admin_token):
        raise HTTPException(status_code=403, detail="Invalid admin token")

    # load_model overwrites the model in-place under the thread-safe lock.
    # No need to call unload_model — it would immediately delete the new model.
    loaded = predictor.load_model(patch_id)
    if not loaded:
        raise HTTPException(
            status_code=404,
            detail=f"Model file not found for patch {patch_id}",
        )
    return ReloadResponse(
        status="ok",
        patch_id=patch_id,
        message=f"Model for patch {patch_id} reloaded",
    )


# ---------------------------------------------------------------------------
# 5v5 Match Forecast
# ---------------------------------------------------------------------------

class MatchForecastRequest(BaseModel):
    patch_id: int
    radiant_heroes: list[int]
    dire_heroes: list[int]
    radiant_team_id: int | None = None
    dire_team_id: int | None = None


@app.post("/predict-match")
async def predict_match(req: MatchForecastRequest, request: Request):
    """Evaluate a completed 5v5 composition and return win probability."""
    if len(req.radiant_heroes) != 5 or len(req.dire_heroes) != 5:
        raise HTTPException(status_code=400, detail="Must provide exactly 5 heroes per team.")

    predictor: Predictor = request.app.state.predictor
    loop = asyncio.get_running_loop()

    try:
        rad_win_prob = await loop.run_in_executor(
            None,
            predictor.predict_match_outcome,
            req.patch_id, req.radiant_heroes, req.dire_heroes,
            req.radiant_team_id, req.dire_team_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "radiant_win_probability": round(rad_win_prob, 4),
        "dire_win_probability": round(1.0 - rad_win_prob, 4),
    }


# ---------------------------------------------------------------------------
# WebSocket: Real-time MCTS draft simulation
# ---------------------------------------------------------------------------

@app.websocket("/ws/draft")
async def ws_draft(websocket: WebSocket):
    """Stream MCTS simulation progress in real-time over WebSocket.

    Uses an asyncio.Queue to bridge the sync executor thread (MCTS) and
    the async WebSocket send loop. Progress packets are queued by the
    executor callback and drained by a concurrent sender coroutine.
    """
    await websocket.accept()
    predictor: Predictor = websocket.app.state.predictor
    import queue as _queue

    try:
        while True:
            raw = await websocket.receive_json()

            patch_id = raw.get("patch_id", 60)
            draft_list = raw.get("draft", [])
            for_team = raw.get("for_team", 0)
            turn_id = raw.get("turn_id", 0)
            first_pick_team = raw.get("first_pick_team", 0)
            radiant_team_id = raw.get("radiant_team_id")
            dire_team_id = raw.get("dire_team_id")

            from .models import DraftSlot
            draft_slots = [DraftSlot(**s) for s in draft_list]

            try:
                ctx = build_draft_context(draft_slots, patch_id=patch_id, first_pick_team=first_pick_team)
            except ValueError as e:
                await websocket.send_json({"type": "error", "detail": str(e)})
                continue

            if for_team is not None:
                ctx.recommending_team = for_team
            elif ctx.recommending_team == -1:
                ctx.recommending_team = 0

            # Run base prediction in executor
            try:
                import asyncio
                loop = asyncio.get_running_loop()
                recommendations, reasoning = await loop.run_in_executor(
                    None,
                    lambda: predictor.predict(
                        patch_id=patch_id, ctx=ctx, draft_slots=draft_slots,
                        radiant_team_id=radiant_team_id,
                        dire_team_id=dire_team_id,
                        num_recommendations=10,
                    ),
                )
            except Exception as e:
                await websocket.send_json({"type": "error", "detail": str(e)})
                continue

            # Send base results immediately
            await websocket.send_json({
                "type": "mcts_progress",
                "turn_id": turn_id,
                "iteration": 0,
                "total": 15,
                "hero_id": 0,
                "win_rate": 0.0,
                "top_picks": [
                    {"hero_id": r["hero_id"], "win_rate": r.get("win_probability", 0),
                     "visits": 0, "action": "Pick"}
                    for r in recommendations[:5]
                ],
                "reasoning_snippet": reasoning or "",
            })

            # Queue-based progress streaming: executor thread pushes, async loop drains
            from .lookahead import run_monte_carlo_rollouts
            top_candidates = recommendations[:15]
            eligible = [h for h in range(1, 156) if h not in ctx.all_taken]
            progress_queue: _queue.Queue = _queue.Queue()
            SENTINEL = object()

            def on_progress(data):
                progress_queue.put(data)

            async def drain_queue():
                """Read from the thread-safe queue and send WebSocket messages."""
                while True:
                    data = await loop.run_in_executor(None, progress_queue.get)
                    if data is SENTINEL:
                        break
                    await websocket.send_json({
                        "type": "mcts_progress",
                        "turn_id": turn_id,
                        "action": "Pick",
                        **data,
                    })

            # Run MCTS in executor and drain queue concurrently
            async def run_mcts_async():
                return await loop.run_in_executor(
                    None,
                    lambda: run_monte_carlo_rollouts(
                        predictor, patch_id, ctx, top_candidates, eligible,
                        radiant_team_id, dire_team_id,
                        num_simulations=40, progress_cb=on_progress,
                    ),
                )

            try:
                mcts_task = asyncio.create_task(run_mcts_async())
                drain_task = asyncio.create_task(drain_queue())

                final_recs = await mcts_task
                progress_queue.put(SENTINEL)  # Signal drain to stop
                await drain_task

                final_recs = final_recs[:10]

                if final_recs:
                    top = final_recs[0]
                    final_reasoning = await loop.run_in_executor(
                        None,
                        lambda: predictor._build_reasoning(
                            top["hero_id"], top.get("lookahead_score", top.get("score", 0)),
                            ctx, patch_id, radiant_team_id, dire_team_id,
                        ),
                    )
                    mc_prob = top.get("mc_win_probability")
                    if mc_prob:
                        final_reasoning = (final_reasoning or "") + f" | MCTS Rollout WR: {mc_prob*100:.1f}%"
                else:
                    final_reasoning = reasoning

                await websocket.send_json({
                    "type": "mcts_complete",
                    "turn_id": turn_id,
                    "recommendations": [
                        {
                            "hero_id": r["hero_id"],
                            "score": r.get("lookahead_score", r.get("score", 0)),
                            "win_probability": r.get("win_probability", 0),
                            "mc_win_probability": r.get("mc_win_probability"),
                            "team_games": r.get("team_games", 0),
                            "team_win_rate": r.get("team_win_rate"),
                            "boosted": r.get("boosted", False),
                        }
                        for r in final_recs
                    ],
                    "reasoning": final_reasoning,
                })

            except Exception as e:
                logger.warning("MCTS streaming failed: %s", e)
                progress_queue.put(SENTINEL)
                await websocket.send_json({
                    "type": "mcts_complete",
                    "turn_id": turn_id,
                    "recommendations": [
                        {
                            "hero_id": r["hero_id"],
                            "score": r.get("score", 0),
                            "win_probability": r.get("win_probability", 0),
                            "team_games": r.get("team_games", 0),
                            "team_win_rate": r.get("team_win_rate"),
                            "boosted": r.get("boosted", False),
                        }
                        for r in recommendations
                    ],
                    "reasoning": reasoning,
                })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.warning("WebSocket error: %s", e)
