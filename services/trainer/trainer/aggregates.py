"""Populate the six ML aggregate tables.

Each function reads from the core tables (matches, players, picks_bans, teams,
team_games, etc.) and writes into the corresponding ml.*_agg table, filtered
to a single patch_id. These are UNLOGGED tables — fast for batch writes but
volatile on crash. The inference API reads from them at prediction time.

**NOTE**: These aggregate tables are used for LIVE INFERENCE and contain ALL
historical matches. The training pipeline uses point-in-time (PIT) aggregates
computed inline via LATERAL subqueries with ``start_time < ds.start_time``
filters to prevent feature leakage. See ``features.py`` for the PIT variants.
"""

from __future__ import annotations

import logging
from typing import Any

import psycopg2
import psycopg2.extras

from .config import TrainerConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shrunk_wr(wins: int, games: int, prior_games: float = 3.0, prior_wr: float = 0.5) -> float:
    """Bayesian-shrunken win rate: (wins + prior_games * prior_wr) / (games + prior_games)."""
    return (wins + prior_games * prior_wr) / (games + prior_games) if games > 0 else prior_wr


def _batched(rows: list[tuple], batch_size: int):
    """Yield successive chunks of *rows*."""
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


# ---------------------------------------------------------------------------
# 1. ml.team_hero_agg
# ---------------------------------------------------------------------------

POPULATE_TEAM_HERO = """
    INSERT INTO ml.team_hero_agg (patch_id, team_id, hero_id, games, wins, bans, win_rate, avg_gpm, avg_xpm, avg_kills, avg_deaths, avg_assists, firstblood_rate, avg_camps_stacked, avg_vision_placed, avg_gold_10, avg_xp_10, last_played)
    VALUES %s
    ON CONFLICT (patch_id, team_id, hero_id) DO UPDATE SET
        games      = EXCLUDED.games,
        wins       = EXCLUDED.wins,
        bans       = EXCLUDED.bans,
        win_rate   = EXCLUDED.win_rate,
        avg_gpm    = EXCLUDED.avg_gpm,
        avg_xpm    = EXCLUDED.avg_xpm,
        avg_kills  = EXCLUDED.avg_kills,
        avg_deaths = EXCLUDED.avg_deaths,
        avg_assists= EXCLUDED.avg_assists,
        firstblood_rate   = EXCLUDED.firstblood_rate,
        avg_camps_stacked = EXCLUDED.avg_camps_stacked,
        avg_vision_placed = EXCLUDED.avg_vision_placed,
        avg_gold_10       = EXCLUDED.avg_gold_10,
        avg_xp_10         = EXCLUDED.avg_xp_10,
        last_played= EXCLUDED.last_played;
"""


def populate_team_hero(cfg: TrainerConfig, conn) -> int:
    """Populate ml.team_hero_agg for *patch_id*."""
    patch_id = cfg.patch_id
    pg = cfg.prior_games
    pw = cfg.prior_win_rate
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                CASE WHEN p.is_radiant THEN m.radiant_team_id ELSE m.dire_team_id END AS team_id,
                p.hero_id,
                COUNT(*)                                           AS games,
                SUM(CASE WHEN p.win = 1 THEN 1 ELSE 0 END)         AS wins,
                0::INT                                              AS bans,
                AVG(p.gold_per_min)::FLOAT                          AS avg_gpm,
                AVG(p.xp_per_min)::FLOAT                            AS avg_xpm,
                AVG(p.kills)::FLOAT                                 AS avg_kills,
                AVG(p.deaths)::FLOAT                                AS avg_deaths,
                AVG(p.assists)::FLOAT                               AS avg_assists,
                AVG(p.firstblood_claimed)::FLOAT                    AS firstblood_rate,
                AVG(p.camps_stacked)::FLOAT                         AS avg_camps_stacked,
                AVG(p.obs_placed + p.sen_placed)::FLOAT             AS avg_vision_placed,
                COALESCE(AVG(gold10.avg_gold_10)::FLOAT, 0)        AS avg_gold_10,
                COALESCE(AVG(xp10.avg_xp_10)::FLOAT, 0)            AS avg_xp_10,
                MAX(m.start_time)                                   AS last_played
            FROM matches m
            INNER JOIN players p ON p.match_id = m.match_id
            LEFT JOIN LATERAL (
                SELECT AVG(arr.elem::numeric) AS avg_gold_10
                FROM player_minute_stats pms,
                LATERAL jsonb_array_elements_text(pms.gold_t) WITH ORDINALITY AS arr(elem, pos)
                WHERE pms.match_id = m.match_id
                  AND pms.player_slot = p.player_slot
                  AND pms.minute = 0
                  AND pos <= 10
            ) gold10 ON TRUE
            LEFT JOIN LATERAL (
                SELECT AVG(arr.elem::numeric) AS avg_xp_10
                FROM player_minute_stats pms,
                LATERAL jsonb_array_elements_text(pms.xp_t) WITH ORDINALITY AS arr(elem, pos)
                WHERE pms.match_id = m.match_id
                  AND pms.player_slot = p.player_slot
                  AND pms.minute = 0
                  AND pos <= 10
            ) xp10 ON TRUE
            WHERE m.patch = %s
              AND m.radiant_win IS NOT NULL
              AND CASE WHEN p.is_radiant THEN m.radiant_team_id ELSE m.dire_team_id END IS NOT NULL
            GROUP BY team_id, p.hero_id
            ORDER BY team_id, p.hero_id
        """, (patch_id,))
        rows: list[tuple[Any, ...]] = []
        for r in cur.fetchall():
            team_id, hero_id, games, wins, bans_, ag, ax, ak, ad, aa, fbr, acs, avp, ag10, ax10, lp = r
            rows.append((
                patch_id, team_id, hero_id, games, wins, 0,
                _shrunk_wr(wins, games, pg, pw), ag, ax, ak, ad, aa,
                fbr, acs, avp, ag10, ax10, lp,
            ))

    # Write in batches
    total = 0
    with conn.cursor() as cur:
        for batch in _batched(rows, cfg.agg_batch_size):
            psycopg2.extras.execute_values(cur, POPULATE_TEAM_HERO, batch, template=None)
            total += len(batch)
    conn.commit()
    logger.info("populate_team_hero: %s rows for patch %s", total, patch_id)
    return total


# ---------------------------------------------------------------------------
# 2. ml.player_hero_agg
# ---------------------------------------------------------------------------

POPULATE_PLAYER_HERO = """
    INSERT INTO ml.player_hero_agg (patch_id, account_id, hero_id, games, wins, win_rate, avg_gpm, avg_xpm, avg_kills, avg_deaths, avg_assists, avg_kda, lane_role, firstblood_rate, avg_camps_stacked, avg_vision_placed, avg_gold_10, avg_xp_10, last_played)
    VALUES %s
    ON CONFLICT (patch_id, account_id, hero_id) DO UPDATE SET
        games       = EXCLUDED.games,
        wins        = EXCLUDED.wins,
        win_rate    = EXCLUDED.win_rate,
        avg_gpm     = EXCLUDED.avg_gpm,
        avg_xpm     = EXCLUDED.avg_xpm,
        avg_kills   = EXCLUDED.avg_kills,
        avg_deaths  = EXCLUDED.avg_deaths,
        avg_assists = EXCLUDED.avg_assists,
        avg_kda     = EXCLUDED.avg_kda,
        lane_role   = EXCLUDED.lane_role,
        firstblood_rate   = EXCLUDED.firstblood_rate,
        avg_camps_stacked = EXCLUDED.avg_camps_stacked,
        avg_vision_placed = EXCLUDED.avg_vision_placed,
        avg_gold_10       = EXCLUDED.avg_gold_10,
        avg_xp_10         = EXCLUDED.avg_xp_10,
        last_played = EXCLUDED.last_played;
"""


def populate_player_hero(cfg: TrainerConfig, conn) -> int:
    patch_id = cfg.patch_id
    pg = cfg.prior_games
    pw = cfg.prior_win_rate
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                p.account_id,
                p.hero_id,
                COUNT(*)                                    AS games,
                SUM(CASE WHEN p.win = 1 THEN 1 ELSE 0 END) AS wins,
                AVG(p.gold_per_min)::FLOAT                  AS avg_gpm,
                AVG(p.xp_per_min)::FLOAT                    AS avg_xpm,
                AVG(p.kills)::FLOAT                         AS avg_kills,
                AVG(p.deaths)::FLOAT                        AS avg_deaths,
                AVG(p.assists)::FLOAT                       AS avg_assists,
                AVG(p.kda)::FLOAT                           AS avg_kda,
                MODE() WITHIN GROUP (ORDER BY p.lane_role)  AS lane_role,
                AVG(p.firstblood_claimed)::FLOAT            AS firstblood_rate,
                AVG(p.camps_stacked)::FLOAT                 AS avg_camps_stacked,
                AVG(p.obs_placed + p.sen_placed)::FLOAT     AS avg_vision_placed,
                COALESCE(AVG(gold10.avg_gold_10)::FLOAT, 0) AS avg_gold_10,
                COALESCE(AVG(xp10.avg_xp_10)::FLOAT, 0)     AS avg_xp_10,
                MAX(m.start_time)                           AS last_played
            FROM matches m
            INNER JOIN players p ON p.match_id = m.match_id
            LEFT JOIN LATERAL (
                SELECT AVG(arr.elem::numeric) AS avg_gold_10
                FROM player_minute_stats pms,
                LATERAL jsonb_array_elements_text(pms.gold_t) WITH ORDINALITY AS arr(elem, pos)
                WHERE pms.match_id = m.match_id
                  AND pms.player_slot = p.player_slot
                  AND pms.minute = 0
                  AND pos <= 10
            ) gold10 ON TRUE
            LEFT JOIN LATERAL (
                SELECT AVG(arr.elem::numeric) AS avg_xp_10
                FROM player_minute_stats pms,
                LATERAL jsonb_array_elements_text(pms.xp_t) WITH ORDINALITY AS arr(elem, pos)
                WHERE pms.match_id = m.match_id
                  AND pms.player_slot = p.player_slot
                  AND pms.minute = 0
                  AND pos <= 10
            ) xp10 ON TRUE
            WHERE m.patch = %s
              AND m.radiant_win IS NOT NULL
              AND p.account_id IS NOT NULL
            GROUP BY p.account_id, p.hero_id
            ORDER BY p.account_id, p.hero_id
        """, (patch_id,))
        rows = []
        for r in cur.fetchall():
            aid, hid, games, wins, ag, ax, ak, ad, aa, akda, lr, fbr, acs, avp, ag10, ax10, lp = r
            rows.append((
                patch_id, aid, hid, games, wins, _shrunk_wr(wins, games, pg, pw),
                ag, ax, ak, ad, aa, akda, lr, fbr, acs, avp, ag10, ax10, lp,
            ))

    total = 0
    with conn.cursor() as cur:
        for batch in _batched(rows, cfg.agg_batch_size):
            psycopg2.extras.execute_values(cur, POPULATE_PLAYER_HERO, batch, template=None)
            total += len(batch)
    conn.commit()
    logger.info("populate_player_hero: %s rows for patch %s", total, patch_id)
    return total


# ---------------------------------------------------------------------------
# 3. ml.hero_synergy_agg
# ---------------------------------------------------------------------------

POPULATE_SYNERGY = """
    INSERT INTO ml.hero_synergy_agg (patch_id, hero_a, hero_b, games, wins, win_rate)
    VALUES %s
    ON CONFLICT (patch_id, hero_a, hero_b) DO UPDATE SET
        games    = EXCLUDED.games,
        wins     = EXCLUDED.wins,
        win_rate = EXCLUDED.win_rate;
"""


def populate_synergy(cfg: TrainerConfig, conn) -> int:
    patch_id = cfg.patch_id
    pg = cfg.prior_games
    pw = cfg.prior_win_rate
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                p1.hero_id AS hero_a,
                p2.hero_id AS hero_b,
                COUNT(*)                                           AS games,
                SUM(CASE WHEN CASE WHEN p1.is_radiant THEN m.radiant_win ELSE NOT m.radiant_win END THEN 1 ELSE 0 END) AS wins
            FROM matches m
            INNER JOIN players p1 ON p1.match_id = m.match_id
            INNER JOIN players p2 ON p2.match_id = m.match_id
                AND p2.is_radiant = p1.is_radiant
                AND p2.hero_id > p1.hero_id
            WHERE m.patch = %s
              AND m.radiant_win IS NOT NULL
            GROUP BY p1.hero_id, p2.hero_id
            HAVING COUNT(*) >= 3
            ORDER BY p1.hero_id, p2.hero_id
        """, (patch_id,))
        rows = []
        for r in cur.fetchall():
            ha, hb, games, wins = r
            rows.append((patch_id, ha, hb, games, wins, _shrunk_wr(wins, games, pg, pw)))

    total = 0
    with conn.cursor() as cur:
        for batch in _batched(rows, cfg.agg_batch_size):
            psycopg2.extras.execute_values(cur, POPULATE_SYNERGY, batch, template=None)
            total += len(batch)
    conn.commit()
    logger.info("populate_synergy: %s rows for patch %s", total, patch_id)
    return total


# ---------------------------------------------------------------------------
# 4. ml.hero_counter_agg
# ---------------------------------------------------------------------------

POPULATE_COUNTER = """
    INSERT INTO ml.hero_counter_agg (patch_id, hero_id, enemy_hero_id, games, wins, win_rate, avg_kd_diff)
    VALUES %s
    ON CONFLICT (patch_id, hero_id, enemy_hero_id) DO UPDATE SET
        games       = EXCLUDED.games,
        wins        = EXCLUDED.wins,
        win_rate    = EXCLUDED.win_rate,
        avg_kd_diff = EXCLUDED.avg_kd_diff;
"""


def populate_counter(cfg: TrainerConfig, conn) -> int:
    patch_id = cfg.patch_id
    pg = cfg.prior_games
    pw = cfg.prior_win_rate
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                p1.hero_id,
                p2.hero_id AS enemy_hero_id,
                COUNT(*)                                           AS games,
                SUM(CASE WHEN CASE WHEN p1.is_radiant THEN m.radiant_win ELSE NOT m.radiant_win END THEN 1 ELSE 0 END) AS wins,
                AVG(p1.kills - p1.deaths)::FLOAT                   AS avg_kd_diff
            FROM matches m
            INNER JOIN players p1 ON p1.match_id = m.match_id
            INNER JOIN players p2 ON p2.match_id = m.match_id
                AND p2.is_radiant != p1.is_radiant
            WHERE m.patch = %s
              AND m.radiant_win IS NOT NULL
            GROUP BY p1.hero_id, p2.hero_id
            HAVING COUNT(*) >= 3
            ORDER BY p1.hero_id, p2.hero_id
        """, (patch_id,))
        rows = []
        for r in cur.fetchall():
            hid, ehid, games, wins, akd = r
            rows.append((patch_id, hid, ehid, games, wins, _shrunk_wr(wins, games, pg, pw), akd))

    total = 0
    with conn.cursor() as cur:
        for batch in _batched(rows, cfg.agg_batch_size):
            psycopg2.extras.execute_values(cur, POPULATE_COUNTER, batch, template=None)
            total += len(batch)
    conn.commit()
    logger.info("populate_counter: %s rows for patch %s", total, patch_id)
    return total


# ---------------------------------------------------------------------------
# 5. ml.team_h2h_agg
# ---------------------------------------------------------------------------

POPULATE_H2H = """
    INSERT INTO ml.team_h2h_agg (patch_id, team_id, enemy_team_id, games, wins, win_rate)
    VALUES %s
    ON CONFLICT (patch_id, team_id, enemy_team_id) DO UPDATE SET
        games    = EXCLUDED.games,
        wins     = EXCLUDED.wins,
        win_rate = EXCLUDED.win_rate;
"""


def populate_h2h(cfg: TrainerConfig, conn) -> int:
    patch_id = cfg.patch_id
    pg = cfg.prior_games
    pw = cfg.prior_win_rate
    with conn.cursor() as cur:
        cur.execute("""
            WITH valid_matches AS (
                SELECT match_id, radiant_team_id, dire_team_id, radiant_win
                FROM matches
                WHERE patch = %s
                  AND radiant_win IS NOT NULL
                  AND radiant_team_id IS NOT NULL
                  AND dire_team_id IS NOT NULL
                  AND leagueid > 0
            ),
            h2h AS (
                -- Radiant perspective
                SELECT radiant_team_id AS team_id, dire_team_id AS enemy_team_id, radiant_win AS won
                FROM valid_matches
                UNION ALL
                -- Dire perspective
                SELECT dire_team_id AS team_id, radiant_team_id AS enemy_team_id, NOT radiant_win AS won
                FROM valid_matches
            )
            SELECT
                team_id,
                enemy_team_id,
                COUNT(*) AS games,
                SUM(CASE WHEN won THEN 1 ELSE 0 END) AS wins
            FROM h2h
            GROUP BY team_id, enemy_team_id
            HAVING COUNT(*) >= 2
            ORDER BY team_id, enemy_team_id
        """, (patch_id,))
        rows = []
        for r in cur.fetchall():
            tid, etid, games, wins = r
            rows.append((patch_id, tid, etid, games, wins, _shrunk_wr(wins, games, pg, pw)))

    total = 0
    with conn.cursor() as cur:
        for batch in _batched(rows, cfg.agg_batch_size):
            psycopg2.extras.execute_values(cur, POPULATE_H2H, batch, template=None)
            total += len(batch)
    conn.commit()
    logger.info("populate_h2h: %s rows for patch %s", total, patch_id)
    return total


# ---------------------------------------------------------------------------
# 6. ml.hero_baseline_agg
# ---------------------------------------------------------------------------

POPULATE_BASELINE = """
    INSERT INTO ml.hero_baseline_agg (patch_id, hero_id, total_picks, total_wins, total_bans, win_rate, pick_rate, ban_rate, avg_gpm, avg_xpm, avg_kills, avg_deaths, avg_assists, avg_gold_10, avg_xp_10)
    VALUES %s
    ON CONFLICT (patch_id, hero_id) DO UPDATE SET
        total_picks  = EXCLUDED.total_picks,
        total_wins   = EXCLUDED.total_wins,
        total_bans   = EXCLUDED.total_bans,
        win_rate     = EXCLUDED.win_rate,
        pick_rate    = EXCLUDED.pick_rate,
        ban_rate     = EXCLUDED.ban_rate,
        avg_gpm      = EXCLUDED.avg_gpm,
        avg_xpm      = EXCLUDED.avg_xpm,
        avg_kills    = EXCLUDED.avg_kills,
        avg_deaths   = EXCLUDED.avg_deaths,
        avg_assists  = EXCLUDED.avg_assists,
        avg_gold_10  = EXCLUDED.avg_gold_10,
        avg_xp_10    = EXCLUDED.avg_xp_10;
"""


def populate_baseline(cfg: TrainerConfig, conn) -> int:
    patch_id = cfg.patch_id
    pg = cfg.prior_games
    pw = cfg.prior_win_rate
    with conn.cursor() as cur:
        cur.execute("""
            WITH hero_picks AS (
                SELECT
                    p.hero_id,
                    COUNT(*)                                    AS total_picks,
                    SUM(CASE WHEN p.win = 1 THEN 1 ELSE 0 END) AS total_wins,
                    AVG(p.gold_per_min)::FLOAT                  AS avg_gpm,
                    AVG(p.xp_per_min)::FLOAT                    AS avg_xpm,
                    AVG(p.kills)::FLOAT                         AS avg_kills,
                    AVG(p.deaths)::FLOAT                        AS avg_deaths,
                    AVG(p.assists)::FLOAT                       AS avg_assists,
                    COALESCE(AVG(gold10.avg_gold_10)::FLOAT, 0) AS avg_gold_10,
                    COALESCE(AVG(xp10.avg_xp_10)::FLOAT, 0)     AS avg_xp_10
                FROM matches m
                INNER JOIN players p ON p.match_id = m.match_id
                LEFT JOIN LATERAL (
                    SELECT AVG(arr.elem::numeric) AS avg_gold_10
                    FROM player_minute_stats pms,
                    LATERAL jsonb_array_elements_text(pms.gold_t) WITH ORDINALITY AS arr(elem, pos)
                    WHERE pms.match_id = m.match_id
                      AND pms.player_slot = p.player_slot
                      AND pms.minute = 0
                      AND pos <= 10
                ) gold10 ON TRUE
                LEFT JOIN LATERAL (
                    SELECT AVG(arr.elem::numeric) AS avg_xp_10
                    FROM player_minute_stats pms,
                    LATERAL jsonb_array_elements_text(pms.xp_t) WITH ORDINALITY AS arr(elem, pos)
                    WHERE pms.match_id = m.match_id
                      AND pms.player_slot = p.player_slot
                      AND pms.minute = 0
                      AND pos <= 10
                ) xp10 ON TRUE
                WHERE m.patch = %s
                  AND m.radiant_win IS NOT NULL
                GROUP BY p.hero_id
            ),
            hero_bans AS (
                SELECT pb.hero_id, COUNT(*) AS total_bans
                FROM matches m
                INNER JOIN picks_bans pb ON pb.match_id = m.match_id AND pb.is_pick = FALSE
                WHERE m.patch = %s
                  AND m.radiant_win IS NOT NULL
                GROUP BY pb.hero_id
            ),
            total_matches AS (
                SELECT COUNT(DISTINCT match_id) AS total FROM matches WHERE patch = %s AND radiant_win IS NOT NULL
            )
            SELECT
                COALESCE(p.hero_id, b.hero_id) AS hero_id,
                COALESCE(p.total_picks, 0)     AS total_picks,
                COALESCE(p.total_wins, 0)      AS total_wins,
                COALESCE(b.total_bans, 0)      AS total_bans,
                p.avg_gpm, p.avg_xpm, p.avg_kills, p.avg_deaths, p.avg_assists,
                p.avg_gold_10, p.avg_xp_10,
                tm.total
            FROM hero_picks p
            FULL OUTER JOIN hero_bans b ON b.hero_id = p.hero_id
            CROSS JOIN total_matches tm
            ORDER BY hero_id
        """, (patch_id, patch_id, patch_id))
        rows = []
        for r in cur.fetchall():
            hid, picks, wins, bans, ag, ax, ak, ad, aa, ag10, ax10, tot = r
            pick_rate = picks / tot if tot > 0 else 0.0
            ban_rate  = bans / tot if tot > 0 else 0.0
            rows.append((
                patch_id, hid, picks, wins, bans,
                _shrunk_wr(wins, picks, pg, pw),
                pick_rate, ban_rate, ag, ax, ak, ad, aa, ag10, ax10,
            ))

    total = 0
    with conn.cursor() as cur:
        for batch in _batched(rows, cfg.agg_batch_size):
            psycopg2.extras.execute_values(cur, POPULATE_BASELINE, batch, template=None)
            total += len(batch)
    conn.commit()
    logger.info("populate_baseline: %s rows for patch %s", total, patch_id)
    return total


# ---------------------------------------------------------------------------
# 7. ml.hero_draft_slot_agg
# ---------------------------------------------------------------------------

POPULATE_HERO_DRAFT_SLOT = """
    INSERT INTO ml.hero_draft_slot_agg (patch_id, hero_id, team_pick_ordinal, games, wins, win_rate)
    VALUES %s
    ON CONFLICT (patch_id, hero_id, team_pick_ordinal) DO UPDATE SET
        games    = EXCLUDED.games,
        wins     = EXCLUDED.wins,
        win_rate = EXCLUDED.win_rate;
"""


def populate_hero_draft_slot(cfg: TrainerConfig, conn) -> int:
    """Populate ml.hero_draft_slot_agg for *patch_id*.

    Computes the pick position (1st/2nd/3rd/4th/5th) within each team and
    aggregates win/loss outcome for each (hero, team_pick_ordinal) bucket.
    """
    patch_id = cfg.patch_id
    pg = cfg.prior_games
    pw = cfg.prior_win_rate
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                ds.hero_id,
                ds.team_pick_ordinal,
                COUNT(*) AS games,
                SUM(CASE WHEN ds.won THEN 1 ELSE 0 END) AS wins
            FROM (
                SELECT
                    pb.match_id,
                    pb.hero_id,
                    pb.team,
                    pb."order",
                    pb.is_pick,
                    ROW_NUMBER() OVER (
                        PARTITION BY pb.match_id, pb.team, pb.is_pick
                        ORDER BY pb."order"
                    ) AS team_pick_ordinal,
                    CASE
                        WHEN (pb.team = 0 AND m.radiant_win) OR (pb.team = 1 AND NOT m.radiant_win)
                        THEN TRUE ELSE FALSE
                    END AS won
                FROM picks_bans pb
                INNER JOIN matches m ON m.match_id = pb.match_id
                WHERE m.patch = %s
                  AND m.radiant_win IS NOT NULL
                  AND pb.is_pick = TRUE
            ) ds
            GROUP BY ds.hero_id, ds.team_pick_ordinal
            ORDER BY ds.hero_id, ds.team_pick_ordinal
        """, (patch_id,))
        rows = []
        for r in cur.fetchall():
            hid, tpo, games, wins = r
            rows.append((patch_id, hid, tpo, games, wins, _shrunk_wr(wins, games, pg, pw)))

    total = 0
    with conn.cursor() as cur:
        for batch in _batched(rows, cfg.agg_batch_size):
            psycopg2.extras.execute_values(cur, POPULATE_HERO_DRAFT_SLOT, batch, template=None)
            total += len(batch)
    conn.commit()
    logger.info("populate_hero_draft_slot: %s rows for patch %s", total, patch_id)
    return total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

ALL_POPULATORS = [
    ("ml.team_hero_agg",        populate_team_hero),
    ("ml.player_hero_agg",      populate_player_hero),
    ("ml.hero_synergy_agg",     populate_synergy),
    ("ml.hero_counter_agg",     populate_counter),
    ("ml.team_h2h_agg",         populate_h2h),
    ("ml.hero_baseline_agg",    populate_baseline),
    ("ml.hero_draft_slot_agg",  populate_hero_draft_slot),
]


def _analyze_ml_tables(conn) -> None:
    """Refresh statistics on all seven ML aggregate tables after population."""
    tables = [
        "ml.team_hero_agg",
        "ml.player_hero_agg",
        "ml.hero_synergy_agg",
        "ml.hero_counter_agg",
        "ml.team_h2h_agg",
        "ml.hero_baseline_agg",
        "ml.hero_draft_slot_agg",
    ]
    with conn.cursor() as cur:
        for tbl in tables:
            cur.execute(f"ANALYZE {tbl}")
    conn.commit()
    logger.info("Analyzed %d ML tables", len(tables))


def populate_all(cfg: TrainerConfig, conn) -> dict[str, int]:
    """Run all seven populator functions, then ANALYZE for fresh stats.

    Returns a dict of ``{table_name: row_count}``.
    """
    counts: dict[str, int] = {}
    for name, fn in ALL_POPULATORS:
        logger.info("Populating %s ...", name)
        cnt = fn(cfg, conn)
        counts[name] = cnt
    _analyze_ml_tables(conn)
    return counts
