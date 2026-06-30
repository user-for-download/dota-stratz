"""Configuration for the LightGBM training pipeline.

All values are read from environment variables with sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class TrainerConfig:
    # ── Database ──────────────────────────────────────────────────────────
    pg_host: str = os.getenv("POSTGRES_HOST", "localhost")
    pg_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    pg_user: str = os.getenv("POSTGRES_USER", "dota2")
    pg_password: str = os.getenv("POSTGRES_PASSWORD", "dota2")
    pg_db: str = os.getenv("POSTGRES_DB", "dota2")

    # ── Training ──────────────────────────────────────────────────────────
    patch_id: int = int(os.getenv("TRAINER_PATCH_ID", "0"))
    """Target Dota 2 patch ID. 0 means auto-detect (most recent with data)."""

    model_dir: str = os.getenv("TRAINER_MODEL_DIR", "/models")
    """Directory where model files (TXT + JSON) are written."""

    # LightGBM hyper-parameters
    # Uses binary classification (not lambdarank) because the training data
    # groups rows by match_id with uniform target values (radiant_win is the
    # same for every draft slot in a match). Lambdarank requires varied
    # relevance within each group — with uniform targets the NDCG gradient
    # is zero and the model outputs constant 0.0 (issue #7).
    lgbm_params: dict = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "binary_logloss",
        "first_metric_only": True,
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "num_threads": 4,
        "seed": 42,
    })

    # ── Data ──────────────────────────────────────────────────────────────
    min_matches_per_group: int = int(os.getenv("TRAINER_MIN_MATCHES_PER_GROUP", "5"))
    """Skip match groups (draft sequences) with fewer than this many picks + bans."""

    max_hero_id: int = int(os.getenv("TRAINER_MAX_HERO_ID", "160"))
    """Upper bound for hero IDs in one-hot encoding. Adjust as new heroes ship."""

    val_ratio: float = float(os.getenv("TRAINER_VAL_RATIO", "0.15"))
    """Fraction of matches held out for validation."""

    # ── Bayesian shrinkage priors ─────────────────────────────────────────
    prior_games: float = float(os.getenv("TRAINER_PRIOR_GAMES", "3.0"))
    """Prior pseudo-count for Bayesian shrinkage of win rates."""

    prior_win_rate: float = float(os.getenv("TRAINER_PRIOR_WR", "0.5"))
    """Prior win rate toward which sparse observations are shrunk."""

    # ── Aggregates ────────────────────────────────────────────────────────
    agg_batch_size: int = int(os.getenv("TRAINER_AGG_BATCH_SIZE", "500"))
    """Rows per chunk when populating aggregate tables via INSERT."""

    # ── Cross-patch lookback (sparse snapshot tables) ─────────────────────
    lookback_patches: int = int(os.getenv("TRAINER_LOOKBACK_PATCHES", "2"))
    """Number of prior patches to include when computing sparse combo-keyed
    snapshot tables (team_hero, player_hero, synergy, counter). 0 = single-patch
    (original behavior)."""

    prior_patch_weight: float = float(os.getenv("TRAINER_PRIOR_PATCH_WEIGHT", "0.5"))
    """Relative weight (0.0–1.0) for games from prior patches vs. the current
    patch. 1.0 = treat equally to current-patch data; 0.5 = prior-patch
    games count half as much toward sample size and win-rate estimates."""

    # ── Match filtering (pro/league/game-mode) ────────────────────────────
    league_only: bool = os.getenv("TRAINER_LEAGUE_ONLY", "false").lower() == "true"
    """If true, only aggregate matches with leagueid > 0 (pro matches)."""

    lobby_types: str = os.getenv("TRAINER_LOBBY_TYPES", "")
    """Comma-separated lobby_type IDs to whitelist (e.g. '7,8' for ranked).
    Empty string means no lobby_type filter."""

    @property
    def pg_dsn(self) -> str:
        return (
            f"host={self.pg_host} port={self.pg_port} "
            f"dbname={self.pg_db} user={self.pg_user} password={self.pg_password}"
        )

    @property
    def sqlalchemy_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        )
