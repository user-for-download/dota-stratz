"""Configuration for the PyTorch DraftBERT training pipeline.

All values are read from environment variables with sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


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
    """Directory where model files are written."""

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

    # ── PyTorch Deep Learning Config ──────────────────────────────────────
    num_threads: int = int(os.getenv("TRAINER_NUM_THREADS", "12"))
    batch_size: int = int(os.getenv("TRAINER_BATCH_SIZE", "256"))
    epochs: int = int(os.getenv("TRAINER_EPOCHS", "15"))
    lr: float = float(os.getenv("TRAINER_LR", "5e-4"))
    weight_decay: float = float(os.getenv("TRAINER_WEIGHT_DECAY", "1e-3"))
    max_seq_len: int = int(os.getenv("TRAINER_MAX_SEQ_LEN", "25"))
    d_model: int = int(os.getenv("TRAINER_D_MODEL", "128"))
    nhead: int = int(os.getenv("TRAINER_NHEAD", "4"))
    num_layers: int = int(os.getenv("TRAINER_NUM_LAYERS", "3"))

    def __post_init__(self):
        if self.d_model % self.nhead != 0:
            raise ValueError(
                f"TRAINER_D_MODEL ({self.d_model}) must be divisible by "
                f"TRAINER_NHEAD ({self.nhead})"
            )

    # ── Model Architecture ───────────────────────────────────────────────
    dropout: float = float(os.getenv("TRAINER_DROPOUT", "0.3"))
    """Dropout rate for embedding, MLP, and fusion layers."""
    transformer_dropout: float = float(os.getenv("TRAINER_TRANSFORMER_DROPOUT", "0.1"))
    """Dropout rate inside TransformerEncoderLayer."""
    static_hidden: int = int(os.getenv("TRAINER_STATIC_HIDDEN", "64"))
    """Hidden dim for the static MLP branch. Only used by LiveDraftBERT."""
    dynamic_hidden: int = int(os.getenv("TRAINER_DYNAMIC_HIDDEN", "32"))
    """Hidden dim for the dynamic MLP branch. Only used by LiveDraftBERT."""
    fusion_hidden: int = int(os.getenv("TRAINER_FUSION_HIDDEN", "64"))
    """Hidden dim for the fusion head."""

    # ── Training Loop ────────────────────────────────────────────────────
    grad_clip: float = float(os.getenv("TRAINER_GRAD_CLIP", "1.0"))
    """Max gradient norm for clipping."""
    early_stop_patience: int = int(os.getenv("TRAINER_EARLY_STOP_PATIENCE", "5"))
    """Epochs to wait before early stopping."""
    lr_scheduler_patience: int = int(os.getenv("TRAINER_LR_SCHEDULER_PATIENCE", "2"))
    """Epochs to wait before reducing LR."""
    lr_scheduler_factor: float = float(os.getenv("TRAINER_LR_SCHEDULER_FACTOR", "0.5"))
    """Factor by which LR is reduced."""
    log_interval: int = int(os.getenv("TRAINER_LOG_INTERVAL", "50"))
    """Log training progress every N batches."""

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
