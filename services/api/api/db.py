"""Database connection pool for the inference API.

Uses psycopg2 ``ThreadedConnectionPool`` for thread-safe concurrent access.
"""

from __future__ import annotations

import logging
from typing import Any

import psycopg2
from psycopg2 import pool as pg_pool

from .config import APIConfig

logger = logging.getLogger(__name__)

_pool: pg_pool.ThreadedConnectionPool | None = None


def init_pool(cfg: APIConfig):
    """Create the global connection pool."""
    global _pool
    _pool = pg_pool.ThreadedConnectionPool(
        minconn=cfg.pool_min,
        maxconn=cfg.pool_max,
        dsn=cfg.pg_dsn,
    )
    logger.info(
        "DB pool initialised (min=%d, max=%d)", cfg.pool_min, cfg.pool_max,
    )


def close_pool():
    """Close all connections in the pool."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None
        logger.info("DB pool closed")


def get_conn() -> Any:
    """Get a connection from the pool."""
    if _pool is None:
        raise RuntimeError("DB pool not initialised")
    return _pool.getconn()


def put_conn(conn):
    """Return a connection to the pool."""
    if _pool is not None:
        _pool.putconn(conn)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def fetch_team_hero_agg(patch_id: int, team_id: int, hero_id: int) -> dict | None:
    """Look up a single row in ml.team_hero_agg."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT games, wins, win_rate, bans, avg_gpm, avg_xpm,
                          avg_kills, avg_deaths, avg_assists
                   FROM ml.team_hero_agg
                   WHERE patch_id = %s AND team_id = %s AND hero_id = %s""",
                (patch_id, team_id, hero_id),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return {
                "games": row[0], "wins": row[1], "win_rate": row[2],
                "bans": row[3], "avg_gpm": row[4], "avg_xpm": row[5],
                "avg_kills": row[6], "avg_deaths": row[7], "avg_assists": row[8],
            }
    finally:
        put_conn(conn)


def fetch_player_hero_agg(patch_id: int, account_id: int, hero_id: int) -> dict | None:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT games, wins, win_rate, avg_gpm, avg_xpm,
                          avg_kills, avg_deaths, avg_assists, avg_kda, lane_role
                   FROM ml.player_hero_agg
                   WHERE patch_id = %s AND account_id = %s AND hero_id = %s""",
                (patch_id, account_id, hero_id),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return {
                "games": row[0], "wins": row[1], "win_rate": row[2],
                "avg_gpm": row[3], "avg_xpm": row[4],
                "avg_kills": row[5], "avg_deaths": row[6], "avg_assists": row[7],
                "avg_kda": row[8], "lane_role": row[9],
            }
    finally:
        put_conn(conn)


def fetch_synergy_avg(patch_id: int, hero_id: int, teammate_hero_ids: list[int]) -> tuple[float, int]:
    """Average synergy win_rate between *hero_id* and a list of teammates."""
    if not teammate_hero_ids:
        return 0.5, 0
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT AVG(s.win_rate)::FLOAT, COUNT(*)::INT
                   FROM ml.hero_synergy_agg s
                   WHERE s.patch_id = %s
                     AND ((s.hero_a = %s AND s.hero_b = ANY(%s))
                       OR (s.hero_b = %s AND s.hero_a = ANY(%s)))""",
                (patch_id, hero_id, teammate_hero_ids, hero_id, teammate_hero_ids),
            )
            row = cur.fetchone()
            if row is None or row[0] is None:
                return 0.5, 0
            return float(row[0]), int(row[1])
    finally:
        put_conn(conn)


def fetch_counter_avg(patch_id: int, hero_id: int, enemy_hero_ids: list[int]) -> tuple[float, int]:
    """Average counter win_rate for *hero_id* vs a list of enemy heroes."""
    if not enemy_hero_ids:
        return 0.5, 0
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT AVG(c.win_rate)::FLOAT, COUNT(*)::INT
                   FROM ml.hero_counter_agg c
                   WHERE c.patch_id = %s
                     AND c.hero_id = %s
                     AND c.enemy_hero_id = ANY(%s)""",
                (patch_id, hero_id, enemy_hero_ids),
            )
            row = cur.fetchone()
            if row is None or row[0] is None:
                return 0.5, 0
            return float(row[0]), int(row[1])
    finally:
        put_conn(conn)


def fetch_h2h(patch_id: int, team_id: int, enemy_team_id: int) -> dict | None:
    """Look up head-to-head aggregate for a team pair."""
    if team_id is None or enemy_team_id is None:
        return None
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT games, wins, win_rate
                   FROM ml.team_h2h_agg
                   WHERE patch_id = %s AND team_id = %s AND enemy_team_id = %s""",
                (patch_id, team_id, enemy_team_id),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return {"games": row[0], "wins": row[1], "win_rate": row[2]}
    finally:
        put_conn(conn)


def fetch_baseline(patch_id: int, hero_id: int) -> dict | None:
    """Look up hero baseline for a hero."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT total_picks, total_wins, total_bans, win_rate,
                          pick_rate, ban_rate, avg_gpm, avg_xpm,
                          avg_kills, avg_deaths, avg_assists
                   FROM ml.hero_baseline_agg
                   WHERE patch_id = %s AND hero_id = %s""",
                (patch_id, hero_id),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return {
                "total_picks": row[0], "total_wins": row[1], "total_bans": row[2],
                "win_rate": row[3], "pick_rate": row[4], "ban_rate": row[5],
                "avg_gpm": row[6], "avg_xpm": row[7],
                "avg_kills": row[8], "avg_deaths": row[9], "avg_assists": row[10],
            }
    finally:
        put_conn(conn)


def fetch_pick_ban_hero_ids(
    patch_id: int,
    match_ids: list[int],
    order_cutoff: int,
) -> list[int]:
    """Fetch hero_ids from picks_bans for given matches before order_cutoff."""
    if not match_ids:
        return []
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT DISTINCT pb.hero_id
                   FROM picks_bans pb
                   INNER JOIN matches m ON pb.match_id = m.match_id
                   WHERE m.patch_id = %s
                     AND pb.match_id = ANY(%s)
                     AND pb.order < %s
                   ORDER BY pb.hero_id""",
                (patch_id, match_ids, order_cutoff),
            )
            return [r[0] for r in cur.fetchall()]
    finally:
        put_conn(conn)
