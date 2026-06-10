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
    lgbm_params: dict = field(default_factory=lambda: {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5, 10],
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

    # ── Aggregates ────────────────────────────────────────────────────────
    agg_batch_size: int = int(os.getenv("TRAINER_AGG_BATCH_SIZE", "500"))
    """Rows per chunk when populating aggregate tables via INSERT."""

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
