"""Database helpers for the training pipeline.

The trainer uses:
  - Raw psycopg2 connection for aggregate INSERT/UPDATE (batch writes).
  - SQLAlchemy engine for pandas read_sql (feature extraction).

This is a batch / CLI process — a single connection is sufficient.
"""

from __future__ import annotations

import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine

from .config import TrainerConfig


def connect(cfg: TrainerConfig):
    """Return a raw psycopg2 connection for writes."""
    return psycopg2.connect(cfg.pg_dsn)


def engine(cfg: TrainerConfig):
    """Return a SQLAlchemy engine for pandas read_sql."""
    return create_engine(cfg.sqlalchemy_url, pool_pre_ping=True)


def fetch_patch_id(cfg: TrainerConfig, conn) -> int:
    """Auto-detect the most recent patch ID with match data if not explicitly set."""
    if cfg.patch_id != 0:
        return cfg.patch_id
    with conn.cursor() as cur:
        cur.execute("""
            SELECT patch_id
            FROM patches
            WHERE patch_id IN (SELECT DISTINCT patch_id FROM matches WHERE patch_id IS NOT NULL)
            ORDER BY patch_id DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("No patches found with match data in the database.")
        return row[0]


def load_heroes(conn) -> dict[int, str]:
    """Return {hero_id: localized_name} for all heroes in the constants table."""
    with conn.cursor() as cur:
        cur.execute("SELECT hero_id, localized_name FROM heroes ORDER BY hero_id")
        return dict(cur.fetchall())
