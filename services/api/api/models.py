"""Pydantic models for the inference API request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DraftSlot(BaseModel):
    """A single pick or ban in the draft so far."""

    hero_id: int = Field(..., ge=0, description="Hero ID (0 = not yet decided)")
    is_pick: bool = Field(..., description="True if pick, False if ban")
    team: int = Field(..., ge=0, le=1, description="0 = radiant, 1 = dire")
    order: int = Field(..., ge=1, le=30, description="Draft order (1-indexed; max varies by patch)")


class PredictRequest(BaseModel):
    """Request body for the /predict endpoint."""

    radiant_team_id: int | None = Field(None, description="Steam team ID for radiant")
    dire_team_id: int | None = Field(None, description="Steam team ID for dire")
    patch_id: int = Field(..., ge=0, description="Dota 2 patch ID")
    first_pick_team: int = Field(
        0, ge=0, le=1,
        description="0 = Radiant first pick (standard), 1 = Dire first pick",
    )
    draft: list[DraftSlot] = Field(
        ...,
        min_length=1,
        max_length=30,
        description="Current draft state (1-30 slots filled; max varies by patch)",
    )
    num_recommendations: int = Field(
        5, ge=1, le=20,
        description="Number of hero recommendations to return",
    )


class HeroScore(BaseModel):
    """A single hero recommendation with its score."""

    hero_id: int = Field(..., description="Hero ID")
    score: float = Field(..., description="Model score (higher = better for the recommending team)")
    pick_probability: float | None = Field(None, description="Estimated pick probability (0-1)")
    win_probability: float | None = Field(None, description="Estimated win probability (0-1)")


class PredictResponse(BaseModel):
    """Response from the /predict endpoint."""

    patch_id: int
    turn: int = Field(..., description="Which draft slot this prediction is for")
    recommending_team: int = Field(..., description="0 = radiant, 1 = dire")
    recommendations: list[HeroScore] = Field(
        ..., description="Sorted by score descending",
    )
    reasoning: str | None = Field(
        None,
        description="Human-readable reasoning for the top recommendation",
    )


class HealthResponse(BaseModel):
    status: str
    patch_models_loaded: list[int]


class ReloadResponse(BaseModel):
    status: str
    patch_id: int
    message: str
