"""Pydantic models for the inference API request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DraftSlot(BaseModel):
    """A single pick or ban in the draft so far."""

    hero_id: int = Field(..., ge=0, description="Hero ID (0 = not yet decided)")
    is_pick: bool = Field(..., description="True if pick, False if ban")
    team: int = Field(..., ge=0, le=1, description="0 = radiant, 1 = dire")
    order: int = Field(..., ge=1, le=50, description="Draft order (1-indexed; max varies by patch; raised from 30 for future patches)")
    account_id: int | None = Field(None, description="Steam account ID of the player (for player-hero agg lookup)")


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
        min_length=0,
        max_length=50,
        description="Current draft state (0-50 slots filled; 0 = before first action)",
    )
    account_id: int | None = Field(
        None,
        description="Steam account ID of the player who will make the next pick. "
                    "When provided, real player-hero aggregates are used instead of "
                    "hardcoded defaults, reducing train-serving skew.",
    )
    num_recommendations: int = Field(
        5, ge=1, le=50,
        description="Number of hero recommendations to return",
    )
    for_team: int | None = Field(
        None, ge=0, le=1,
        description="Override which team to recommend for (0=radiant, 1=dire). "
                    "When set, recommendations are computed for this team regardless "
                    "of whose turn it is in the draft order.",
    )
    run_mcts: bool = Field(
        True,
        description="Whether to run Monte Carlo Tree Search rollouts. "
                    "Set to False for instant tooltip predictions.",
    )


class HeroScore(BaseModel):
    """A single hero recommendation with its score."""

    hero_id: int = Field(..., description="Hero ID")
    score: float = Field(..., description="Model score (higher = better for the recommending team)")
    pick_probability: float | None = Field(None, description="Estimated pick probability (0-1)")
    win_probability: float | None = Field(None, description="Estimated win probability (0-1)")
    team_games: int | None = Field(None, description="Team's games on this hero")
    team_win_rate: float | None = Field(None, description="Team's win rate on this hero (0-1)")
    boosted: bool | None = Field(None, description="True if score was boosted by team proficiency")


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
