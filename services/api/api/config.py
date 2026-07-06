"""Configuration for the FastAPI inference service."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class APIConfig:
    # ── Server ────────────────────────────────────────────────────────────
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8080"))

    # ── Database ──────────────────────────────────────────────────────────
    pg_host: str = os.getenv("POSTGRES_HOST", "localhost")
    pg_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    pg_user: str = os.getenv("POSTGRES_USER", "dota2")
    pg_password: str = os.getenv("POSTGRES_PASSWORD", "dota2")
    pg_db: str = os.getenv("POSTGRES_DB", "dota2")

    # ── Model / Feature ───────────────────────────────────────────────────
    model_dir: str = os.getenv("API_MODEL_DIR", "/models")
    """Directory containing model_patch_N.txt and feature_schema.json."""

    max_hero_id: int = int(os.getenv("API_MAX_HERO_ID", "160"))

    # ── CORS ────────────────────────────────────────────────────────────────
    cors_origins: list[str] = field(default_factory=lambda: [
        origin.strip()
        for origin in os.getenv(
            "API_CORS_ORIGINS",
            "http://localhost,http://localhost:80,http://localhost:3000,http://localhost:5173,http://127.0.0.1:80"
        ).split(",")
        if origin.strip()
    ])

    # ── Admin ─────────────────────────────────────────────────────────────
    admin_token: str = os.getenv("STRATZ_ADMIN_TOKEN", "")

    # ── Connection Pool ───────────────────────────────────────────────────
    pool_min: int = int(os.getenv("API_POOL_MIN", "1"))
    pool_max: int = int(os.getenv("API_POOL_MAX", "8"))

    @property
    def pg_dsn(self) -> str:
        return (
            f"host={self.pg_host} port={self.pg_port} "
            f"dbname={self.pg_db} user={self.pg_user} password={self.pg_password}"
        )
