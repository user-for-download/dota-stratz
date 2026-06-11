"""Database connection pool for the inference API.

Uses psycopg2 ``ThreadedConnectionPool`` guarded by a ``threading.Lock``
for safe concurrent access across FastAPI worker threads.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import psycopg2
from psycopg2 import pool as pg_pool

from .config import APIConfig

logger = logging.getLogger(__name__)

_pool: pg_pool.ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()


def init_pool(cfg: APIConfig):
    """Create the global connection pool."""
    global _pool
    with _pool_lock:
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
    with _pool_lock:
        if _pool is not None:
            _pool.closeall()
            _pool = None
    logger.info("DB pool closed")


def get_conn() -> Any:
    """Get a connection from the pool.

    The lock is only held for the pool reference read, NOT during
    ``getconn()`` (which may block if all connections are in use).
    Holding the lock during a blocking call would prevent any other
    thread from returning a connection via ``put_conn``, causing a
    deadlock when the pool is exhausted (BUG-004).
    """
    if _pool is None:
        raise RuntimeError("DB pool not initialised")
    with _pool_lock:
        pool = _pool
    return pool.getconn()


def put_conn(conn):
    """Return a connection to the pool.

    Rolls back any pending transaction first so the connection is returned
    in a clean state. Without this, psycopg2's implicit transactions accumulate
    on the connection, poisoning the pool with stale MVCC snapshots — every
    subsequent query using that connection sees frozen feature data and blocks
    autovacuum (issue #10).

    If the rollback fails (broken socket, DB restart), the connection is
    discarded via ``close=True`` instead of being returned to the pool.
    Without this, dead connections accumulate in the pool and every future
    ``/predict`` request that draws one gets an HTTP 500 until the API is
    restarted (issue #34).
    """
    if _pool is not None:
        try:
            conn.rollback()
        except Exception:
            # Connection is broken — discard it instead of poisoning the pool.
            with _pool_lock:
                _pool.putconn(conn, close=True)
            return
        with _pool_lock:
            _pool.putconn(conn)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def fetch_team_hero_agg_batch(patch_id: int, team_id: int, hero_ids: list[int]) -> dict[int, dict]:
    """Batch fetch ml.team_hero_agg for multiple heroes in one query.

    Returns a dict mapping hero_id → row dict (or empty dict for missing).
    """
    if not hero_ids:
        return {}
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT hero_id, games, wins, win_rate, bans, avg_gpm, avg_xpm,
                          avg_kills, avg_deaths, avg_assists,
                          firstblood_rate, avg_camps_stacked, avg_vision_placed,
                          avg_gold_10, avg_xp_10
                   FROM ml.team_hero_agg
                   WHERE patch_id = %s AND team_id = %s AND hero_id = ANY(%s)""",
                (patch_id, team_id, hero_ids),
            )
            result: dict[int, dict] = {}
            for row in cur.fetchall():
                result[row[0]] = {
                    "games": row[1], "wins": row[2], "win_rate": row[3],
                    "bans": row[4], "avg_gpm": row[5], "avg_xpm": row[6],
                    "avg_kills": row[7], "avg_deaths": row[8], "avg_assists": row[9],
                    "firstblood_rate": row[10], "avg_camps_stacked": row[11], "avg_vision_placed": row[12],
                    "avg_gold_10": row[13], "avg_xp_10": row[14],
                }
            return result
    finally:
        put_conn(conn)


def fetch_team_hero_agg(patch_id: int, team_id: int, hero_id: int) -> dict | None:
    """Look up a single row in ml.team_hero_agg."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT games, wins, win_rate, bans, avg_gpm, avg_xpm,
                          avg_kills, avg_deaths, avg_assists,
                          firstblood_rate, avg_camps_stacked, avg_vision_placed,
                          avg_gold_10, avg_xp_10
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
                "firstblood_rate": row[9], "avg_camps_stacked": row[10], "avg_vision_placed": row[11],
                "avg_gold_10": row[12], "avg_xp_10": row[13],
            }
    finally:
        put_conn(conn)


def fetch_player_hero_agg_batch(
    patch_id: int,
    account_id: int,
    hero_ids: list[int],
) -> dict[int, dict]:
    """Batch fetch ml.player_hero_agg for a single player across multiple heroes.

    Returns ``{hero_id: row_dict}`` (or empty dict for missing/unknown account).
    """
    if not hero_ids or account_id is None:
        return {}
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT hero_id, games, wins, win_rate, avg_gpm, avg_xpm,
                          avg_kills, avg_deaths, avg_assists, avg_kda, lane_role,
                          firstblood_rate, avg_camps_stacked, avg_vision_placed,
                          avg_gold_10, avg_xp_10
                   FROM ml.player_hero_agg
                   WHERE patch_id = %s AND account_id = %s AND hero_id = ANY(%s)""",
                (patch_id, account_id, hero_ids),
            )
            result: dict[int, dict] = {}
            for row in cur.fetchall():
                result[row[0]] = {
                    "games": row[1], "wins": row[2], "win_rate": row[3],
                    "avg_gpm": row[4], "avg_xpm": row[5],
                    "avg_kills": row[6], "avg_deaths": row[7], "avg_assists": row[8],
                    "avg_kda": row[9], "lane_role": row[10],
                    "firstblood_rate": row[11], "avg_camps_stacked": row[12], "avg_vision_placed": row[13],
                    "avg_gold_10": row[14], "avg_xp_10": row[15],
                }
            return result
    finally:
        put_conn(conn)


def fetch_player_hero_agg(patch_id: int, account_id: int, hero_id: int) -> dict | None:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT games, wins, win_rate, avg_gpm, avg_xpm,
                          avg_kills, avg_deaths, avg_assists, avg_kda, lane_role,
                          firstblood_rate, avg_camps_stacked, avg_vision_placed,
                          avg_gold_10, avg_xp_10
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
                "firstblood_rate": row[10], "avg_camps_stacked": row[11], "avg_vision_placed": row[12],
                "avg_gold_10": row[13], "avg_xp_10": row[14],
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


def fetch_baselines_batch(patch_id: int, hero_ids: list[int]) -> dict[int, dict]:
    """Batch fetch ml.hero_baseline_agg for multiple heroes in one query.

    Returns a dict mapping hero_id → row dict (or empty dict for missing).
    """
    if not hero_ids:
        return {}
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT hero_id, total_picks, total_wins, total_bans, win_rate,
                          pick_rate, ban_rate, avg_gpm, avg_xpm,
                          avg_kills, avg_deaths, avg_assists,
                          avg_gold_10, avg_xp_10
                   FROM ml.hero_baseline_agg
                   WHERE patch_id = %s AND hero_id = ANY(%s)""",
                (patch_id, hero_ids),
            )
            result: dict[int, dict] = {}
            for row in cur.fetchall():
                result[row[0]] = {
                    "total_picks": row[1], "total_wins": row[2], "total_bans": row[3],
                    "win_rate": row[4], "pick_rate": row[5], "ban_rate": row[6],
                    "avg_gpm": row[7], "avg_xpm": row[8],
                    "avg_kills": row[9], "avg_deaths": row[10], "avg_assists": row[11],
                    "avg_gold_10": row[12], "avg_xp_10": row[13],
                }
            return result
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
                          avg_kills, avg_deaths, avg_assists,
                          avg_gold_10, avg_xp_10
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
                "avg_gold_10": row[11], "avg_xp_10": row[12],
            }
    finally:
        put_conn(conn)


def fetch_synergy_batch(
    patch_id: int,
    hero_ids: list[int],
    ally_picks: list[int],
) -> dict[int, tuple[float, int]]:
    """Batch fetch synergy win_rate for *hero_ids* vs *ally_picks* in one query.

    Returns ``{hero_id: (avg_win_rate, count)}`` — missing heroes get a
    default of ``(0.5, 0)`` from the caller.
    """
    if not hero_ids or not ally_picks:
        return {}
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT
                       CASE
                           WHEN s.hero_a = ANY(%s) THEN s.hero_a
                           ELSE s.hero_b
                       END AS hero_id,
                       s.win_rate
                   FROM ml.hero_synergy_agg s
                   WHERE s.patch_id = %s
                     AND (
                         (s.hero_a = ANY(%s) AND s.hero_b = ANY(%s))
                         OR (s.hero_b = ANY(%s) AND s.hero_a = ANY(%s))
                     )""",
                (hero_ids, patch_id, hero_ids, ally_picks, hero_ids, ally_picks),
            )
            agg: dict[int, list[float]] = {}
            for row in cur.fetchall():
                hid = row[0]
                wr = row[1]
                agg.setdefault(hid, []).append(wr)
            return {
                hid: (float(sum(wrs)) / len(wrs), len(wrs))
                for hid, wrs in agg.items()
            }
    finally:
        put_conn(conn)


def fetch_counter_batch(
    patch_id: int,
    hero_ids: list[int],
    enemy_picks: list[int],
) -> dict[int, tuple[float, int]]:
    """Batch fetch counter win_rate for *hero_ids* vs *enemy_picks* in one query.

    Returns ``{hero_id: (avg_win_rate, count)}`` — missing heroes get a
    default of ``(0.5, 0)`` from the caller.
    """
    if not hero_ids or not enemy_picks:
        return {}
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT
                       c.hero_id,
                       c.win_rate
                   FROM ml.hero_counter_agg c
                   WHERE c.patch_id = %s
                     AND c.hero_id = ANY(%s)
                     AND c.enemy_hero_id = ANY(%s)""",
                (patch_id, hero_ids, enemy_picks),
            )
            agg: dict[int, list[float]] = {}
            for row in cur.fetchall():
                hid = row[0]
                wr = row[1]
                agg.setdefault(hid, []).append(wr)
            return {
                hid: (float(sum(wrs)) / len(wrs), len(wrs))
                for hid, wrs in agg.items()
            }
    finally:
        put_conn(conn)


def fetch_hero_draft_slot_batch(
    patch_id: int,
    hero_ids: list[int],
    team_pick_ordinal: int,
) -> dict[int, tuple[float, int]]:
    """Batch fetch hero draft-slot aggregates for the given *team_pick_ordinal*.

    Returns ``{hero_id: (win_rate, games)}`` — missing heroes get defaults
    of ``(0.5, 0)`` from the caller.
    """
    if not hero_ids:
        return {}
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT hero_id, win_rate, games
                   FROM ml.hero_draft_slot_agg
                   WHERE patch_id = %s
                     AND team_pick_ordinal = %s
                     AND hero_id = ANY(%s)""",
                (patch_id, team_pick_ordinal, hero_ids),
            )
            result: dict[int, tuple[float, int]] = {}
            for row in cur.fetchall():
                hid, wr, games = row
                result[hid] = (float(wr), int(games))
            return result
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
                   WHERE m.patch = %s
                     AND pb.match_id = ANY(%s)
                     AND pb.order < %s
                   ORDER BY pb.hero_id""",
                (patch_id, match_ids, order_cutoff),
            )
            return [r[0] for r in cur.fetchall()]
    finally:
        put_conn(conn)
