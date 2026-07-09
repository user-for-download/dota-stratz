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
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Header, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import APIConfig
from .db import close_pool, get_conn, init_pool, put_conn
from .draft_state import build_draft_context, DraftContext
from .models import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    HeroScore,
    ReloadResponse,
)
from .predictor import Predictor
from .live_predict import LivePredictor, fetch_live_matches, fetch_match_state, compute_dynamic_features
from .live_features import DYNAMIC_FEATURE_COLUMNS

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
    live_predictor = LivePredictor(cfg.model_dir)
    app.state.cfg = cfg
    app.state.predictor = predictor
    app.state.live_predictor = live_predictor

    init_pool(cfg)
    n_loaded = predictor.reload_all()
    logger.info(
        "API started on %s:%s | %d DraftBERT models loaded",
        cfg.host, cfg.port, n_loaded,
    )
    yield
    # Shutdown
    close_pool()
    logger.info("API shut down.")


app = FastAPI(
    title="dota-stratz ML Inference API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS origins from environment variable (comma-separated)
_cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "API_CORS_ORIGINS",
        "http://localhost,http://localhost:80,http://localhost:3000,http://localhost:5173,http://127.0.0.1:80"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
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
async def predict(req: PredictRequest, request: Request):
    predictor: Predictor = request.app.state.predictor
    try:
        ctx = build_draft_context(req.draft, patch_id=req.patch_id, first_pick_team=req.first_pick_team)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Allow overriding which team to recommend for
    if req.for_team is not None:
        ctx.recommending_team = req.for_team
    elif ctx.recommending_team == -1:
        ctx.recommending_team = 0

    loop = asyncio.get_running_loop()
    try:
        recommendations, reasoning = await loop.run_in_executor(
            None,
            lambda: predictor.predict(
                patch_id=req.patch_id,
                ctx=ctx,
                draft_slots=req.draft,
                radiant_team_id=req.radiant_team_id,
                dire_team_id=req.dire_team_id,
                num_recommendations=req.num_recommendations,
                account_id=req.account_id,
                run_mcts=req.run_mcts,
            ),
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
# Live Match Prediction
# ---------------------------------------------------------------------------

class LiveMatchListItem(BaseModel):
    match_id: int
    duration: int
    game_mode: int | None = None
    radiant_team: str
    dire_team: str
    league_id: int | None = None
    spectators: int = 0


class LivePredictRequest(BaseModel):
    match_id: int
    minute: int = 0


@app.get("/api/live/matches")
def list_live_matches():
    """List currently live matches from OpenDota."""
    raw = fetch_live_matches()
    matches = []
    for m in raw:
        matches.append({
            "match_id": m.get("match_id"),
            "duration": m.get("duration", 0),
            "game_mode": m.get("game_mode"),
            "radiant_team": m.get("radiant_name", "Radiant"),
            "dire_team": m.get("dire_name", "Dire"),
            "league_id": m.get("league_id"),
            "spectators": m.get("spectators", 0),
        })
    return {"matches": matches}


@app.post("/api/live/predict")
def live_predict(req: LivePredictRequest, request: Request):
    """Predict win probability for a live match at a given minute."""
    live_pred: LivePredictor = request.app.state.live_predictor

    match_data = fetch_match_state(req.match_id)
    if not match_data:
        raise HTTPException(status_code=404, detail="Match not found or not available")

    # Compute dynamic features
    features = compute_dynamic_features(match_data, req.minute)

    # Build draft sequence from match data
    picks_bans = match_data.get("picks_bans", [])
    if not picks_bans:
        # Fallback: use player hero picks
        heroes = []
        actions = []
        for player in match_data.get("players", []):
            hid = player.get("hero_id", 0)
            slot = player.get("player_slot", 0)
            team = 1 if slot >= 128 else 0
            heroes.append(hid)
            actions.append(team * 1 + 3)  # Pick action
    else:
        heroes = [pb["hero_id"] for pb in picks_bans]
        actions = [pb["team"] * 1 + (1 if pb.get("is_pick", True) else 0) * 2 + 1 for pb in picks_bans]

    # Static features: dynamically sized based on loaded model schema
    patch_id = match_data.get("patch", 60)
    n_static = live_pred._schemas.get(patch_id, {}).get("n_static_features", 61)
    static_feats = [0.0] * n_static

    try:
        result = live_pred.predict(
            patch_id=patch_id,
            match_id=req.match_id,
            heroes=heroes,
            actions=actions,
            static_feats=static_feats,
            dynamic_feats=[features[col] for col in DYNAMIC_FEATURE_COLUMNS],
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "match_id": req.match_id,
        "minute": req.minute,
        **result,
        "features": features,
    }


@app.get("/api/live/predictions/{match_id}")
def get_prediction_timeline(match_id: int, request: Request):
    """Get win probability at every minute for a match (historical or live)."""
    live_pred: LivePredictor = request.app.state.live_predictor

    match_data = fetch_match_state(match_id)
    if not match_data:
        raise HTTPException(status_code=404, detail="Match not found")

    duration = match_data.get("duration", 0)
    max_minute = min(duration // 60, 60)
    patch_id = match_data.get("patch", 60)

    # Build draft sequence
    picks_bans = match_data.get("picks_bans", [])
    if picks_bans:
        heroes = [pb["hero_id"] for pb in picks_bans]
        actions = [pb["team"] * 1 + (1 if pb.get("is_pick", True) else 0) * 2 + 1 for pb in picks_bans]
    else:
        heroes = []
        actions = []

    n_static = live_pred._schemas.get(patch_id, {}).get("n_static_features", 61)
    static_feats = [0.0] * n_static

    timeline = []
    for minute in range(1, max_minute + 1):
        features = compute_dynamic_features(match_data, minute)
        try:
            pred = live_pred.predict(
                patch_id=patch_id,
                match_id=match_id,
                heroes=heroes,
                actions=actions,
                static_feats=static_feats,
                dynamic_feats=[features[col] for col in DYNAMIC_FEATURE_COLUMNS],
            )
            timeline.append({
                "minute": minute,
                "radiant_win_probability": pred["radiant_win_probability"],
                "tower_diff": sum(features.get(f"t{i}_tower_diff", 0) for i in range(1, 5)),
                **features,
            })
        except Exception:
            continue

    return {
        "match_id": match_id,
        "duration": duration,
        "timeline": timeline,
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
    _ws_semaphore = asyncio.Semaphore(2)  # Limit concurrent evaluations

    async def handle_message(raw: dict):
        async with _ws_semaphore:
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
                await websocket.send_json({"type": "error", "detail": str(e), "for_team": for_team})
                return

            if for_team is not None:
                ctx.recommending_team = for_team
            elif ctx.recommending_team == -1:
                ctx.recommending_team = 0

            # Run base prediction in executor without triggering internal MCTS
            try:
                loop = asyncio.get_running_loop()
                recommendations, reasoning = await loop.run_in_executor(
                    None,
                    lambda: predictor.predict(
                        patch_id=patch_id, ctx=ctx, draft_slots=draft_slots,
                        radiant_team_id=radiant_team_id,
                        dire_team_id=dire_team_id,
                        num_recommendations=10,
                        run_mcts=False,
                    ),
                )
            except Exception as e:
                await websocket.send_json({"type": "error", "detail": str(e), "for_team": for_team})
                return

            action_str = "Pick" if ctx.is_pick_turn else "Ban"

            # Send base results immediately
            await websocket.send_json({
                "type": "mcts_progress",
                "turn_id": turn_id,
                "for_team": for_team,
                "iteration": 0,
                "total": 15,
                "hero_id": 0,
                "win_rate": 0.0,
                "top_picks": [
                    {"hero_id": r["hero_id"], "win_rate": r.get("win_probability", 0),
                     "visits": 0, "action": action_str}
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
                        "for_team": for_team,
                        "action": action_str,
                        **data,
                    })

            # Run MCTS in executor and drain queue concurrently
            async def run_mcts_async():
                eval_team = ctx.recommending_team if ctx.is_pick_turn else (1 - ctx.recommending_team)
                eval_ctx = DraftContext(
                    turn=ctx.turn,
                    recommending_team=eval_team,
                    is_pick_turn=True,
                    radiant_picks=list(ctx.radiant_picks),
                    dire_picks=list(ctx.dire_picks),
                    radiant_bans=list(ctx.radiant_bans),
                    dire_bans=list(ctx.dire_bans),
                )
                return await loop.run_in_executor(
                    None,
                    lambda: run_monte_carlo_rollouts(
                        predictor, patch_id, eval_ctx, top_candidates, eligible,
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

                # Sort MCTS output by lookahead score
                final_recs.sort(key=lambda r: r.get("lookahead_score", r.get("score", 0)), reverse=True)
                final_recs = final_recs[:10]

                # Convert enemy WR back to acting team WR for display if banning
                if not ctx.is_pick_turn:
                    for r in final_recs:
                        if "mc_win_probability" in r:
                            r["mc_win_probability"] = 1.0 - r["mc_win_probability"]
                        if "worst_case_nemesis_wr" in r:
                            r["worst_case_nemesis_wr"] = 1.0 - r["worst_case_nemesis_wr"]

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
                    if top.get("comp_penalty"):
                        final_reasoning = (final_reasoning or "") + f" | WARNING: {top['comp_penalty']}"
                else:
                    final_reasoning = reasoning

                await websocket.send_json({
                    "type": "mcts_complete",
                    "turn_id": turn_id,
                    "for_team": for_team,
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
                    "for_team": for_team,
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

    try:
        while True:
            raw = await websocket.receive_json()
            asyncio.create_task(handle_message(raw))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.warning("WebSocket error: %s", e)


# ---------------------------------------------------------------------------
# WebSocket: Live match prediction streaming
# ---------------------------------------------------------------------------

@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    """Stream live match predictions in real-time over WebSocket.

    Client sends: {"match_id": 12345, "interval": 10}
    Server streams: {"type": "prediction", "minute": N, "radiant_win_probability": 0.65, ...}

    Caches transformer + static embeddings per match for fast per-tick updates.
    Client can send a new message at any time to switch matches.
    """
    await websocket.accept()
    live_pred: LivePredictor = websocket.app.state.live_predictor

    try:
        while True:
            raw = await websocket.receive_json()
            match_id = raw.get("match_id")
            interval = raw.get("interval", 10)

            if not match_id:
                await websocket.send_json({"type": "error", "detail": "match_id required"})
                continue

            loop = asyncio.get_running_loop()
            cached_seq = None
            cached_static = None
            cached_match = None
            cached_patch = None

            # Poll loop — breaks on match end or new client message
            while True:
                try:
                    # Use wait_for to allow client messages between polls
                    try:
                        raw = await asyncio.wait_for(
                            websocket.receive_json(), timeout=interval,
                        )
                        # Client sent a new message — switch matches
                        match_id = raw.get("match_id", match_id)
                        interval = raw.get("interval", interval)
                        cached_seq = None
                        cached_static = None
                        break
                    except asyncio.TimeoutError:
                        pass  # No new message, continue polling

                    match_data = await loop.run_in_executor(
                        None, fetch_match_state, match_id
                    )
                    if not match_data:
                        await websocket.send_json({"type": "error", "detail": "Match not available"})
                        break

                    duration = match_data.get("duration", 0)
                    minute = duration // 60
                    patch_id = match_data.get("patch", 60)

                    # Build draft sequence
                    picks_bans = match_data.get("picks_bans", [])
                    if picks_bans:
                        heroes = [pb["hero_id"] for pb in picks_bans]
                        actions_list = [pb["team"] * 1 + (1 if pb.get("is_pick", True) else 0) * 2 + 1 for pb in picks_bans]
                    else:
                        heroes = []
                        actions_list = []

                    n_static = live_pred._schemas.get(patch_id, {}).get("n_static_features", 61)
                    static_feats = [0.0] * n_static

                    # Compute dynamic features first (needed for dyn_feats below)
                    features = await loop.run_in_executor(
                        None, compute_dynamic_features, match_data, minute
                    )
                    dyn_feats = [features[col] for col in DYNAMIC_FEATURE_COLUMNS]

                    # Encode draft once, cache transformer + static embeddings
                    if cached_seq is None or cached_match != match_id or cached_patch != patch_id:
                        def _encode():
                            return live_pred.encode_draft(patch_id, match_id, heroes, actions_list, static_feats)
                        cached_seq, cached_static = await loop.run_in_executor(None, _encode)
                        cached_match = match_id
                        cached_patch = patch_id

                    # Fast path: only Dynamic MLP + Fusion runs
                    pred = await loop.run_in_executor(
                        None,
                        lambda: live_pred.predict_with_cache(
                            patch_id=patch_id,
                            match_id=match_id,
                            seq_repr=cached_seq,
                            static_repr=cached_static,
                            dynamic_feats=dyn_feats,
                        ),
                    )

                    is_live = duration > 0 and duration < 3600

                    await websocket.send_json({
                        "type": "prediction",
                        "match_id": match_id,
                        "minute": minute,
                        "duration": duration,
                        "is_live": is_live,
                        **pred,
                        "features": {
                            **features,
                            "tower_diff": sum(features.get(f"t{i}_tower_diff", 0) for i in range(1, 5)),
                        },
                    })

                    if not is_live:
                        break

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    await websocket.send_json({"type": "error", "detail": str(e)})
                    await asyncio.sleep(interval)

    except WebSocketDisconnect:
        logger.info("Live prediction WS client disconnected")
    except Exception as e:
        logger.warning("Live prediction WS error: %s", e)
