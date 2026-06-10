"""FastAPI application entry point for the ML Inference API.

Endpoints:
  POST /predict  — Score heroes for the current draft state.
  GET  /health   — Health check with loaded patches.
  POST /reload/{patch_id}  — Hot-reload a model (admin, requires token).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Header

from .config import APIConfig
from .db import close_pool, init_pool
from .draft_state import build_draft_context
from .models import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    HeroScore,
    ReloadResponse,
)
from .predictor import Predictor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals (set during lifespan)
# ---------------------------------------------------------------------------

cfg = APIConfig()
predictor = Predictor(cfg)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown lifecycle."""
    # Startup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
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


app = FastAPI(
    title="dota-stratz ML Inference API",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        patch_models_loaded=predictor.loaded_patches(),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        ctx = build_draft_context(req.draft, patch_id=req.patch_id, first_pick_team=req.first_pick_team)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if ctx.recommending_team == -1:
        raise HTTPException(status_code=400, detail="Draft is already complete")

    try:
        recommendations, reasoning = predictor.predict(
            patch_id=req.patch_id,
            ctx=ctx,
            radiant_team_id=req.radiant_team_id,
            dire_team_id=req.dire_team_id,
            num_recommendations=req.num_recommendations,
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
    authorization: str = Header(None),
):
    # Validate admin token
    if cfg.admin_token:
        token = ""
        if authorization and authorization.startswith("Bearer "):
            token = authorization[len("Bearer "):]
        if token != cfg.admin_token:
            raise HTTPException(status_code=403, detail="Invalid admin token")

    predictor.unload_model(patch_id)
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
