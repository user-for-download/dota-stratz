"""Populate the ML aggregate tables.

Each function reads from the core tables (matches, players, picks_bans, teams,
team_games, etc.) and writes into the corresponding ml.*_agg or ml.*_snapshot
table, filtered to a single patch_id. The base tables are LOGGED (crash-safe).
The snapshot tables are also LOGGED. The inference API reads from the base
tables at prediction time (PIT safety is only needed during training).

**Stale row protection**: Every populator ``DELETE``s rows for the current
patch_id before re-inserting, so rows that disappear from source queries
(e.g. after data corrections) do not persist in aggregate tables.

**Consistent match filtering**: All seven populators now apply the same
config-driven match filter (``TRAINER_LEAGUE_ONLY`` / ``TRAINER_LOBBY_TYPES``)
instead of having a hardcoded ``leagueid > 0`` only in ``populate_h2h``.
Set these env vars to restrict training to pro/ranked matches only.

**PIT-safe snapshots**: The ``*_snapshot`` tables (migration 014) provide
point-in-time-correct aggregates bucketed by ``as_of_date`` so that training
features reference only matches that occurred *before* the draft start time.
The base ``ml.*_agg`` tables are retained for API inference (prediction-time
feature lookup uses the same flat joins — the API does not need PIT safety
since it evaluates the live draft state).
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

def _shrunk_wr(wins: float, games: float, prior_games: float = 3.0, prior_wr: float = 0.5) -> float:
    """Bayesian-shrunken win rate: (wins + prior_games * prior_wr) / (games + prior_games).
    Accepts float *wins/games* (fractional counts from cross-patch weighting)."""
    return (wins + prior_games * prior_wr) / (games + prior_games) if games > 0 else prior_wr


def _batched(rows: list[tuple], batch_size: int):
    """Yield successive chunks of *rows*."""
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


def _wavg(c_val: float, c_w: float, p_val: float, p_w: float, total_g: float) -> float:
    """Weighted average of current and prior values."""
    return (c_val * c_w + p_val * p_w) / total_g


_VALID_TABLES = frozenset({
    "ml.team_hero_agg", "ml.player_hero_agg",
    "ml.hero_synergy_agg", "ml.hero_counter_agg",
    "ml.team_h2h_agg", "ml.hero_baseline_agg",
    "ml.hero_draft_slot_agg",
    "ml.team_hero_snapshot", "ml.player_hero_snapshot",
    "ml.hero_synergy_snapshot", "ml.hero_counter_snapshot",
    "ml.team_h2h_snapshot", "ml.hero_baseline_snapshot",
    "ml.hero_draft_slot_snapshot",
})


def _clean_patch_rows(conn, table: str, patch_id: int) -> None:
    """Delete stale rows for *patch_id* before re-populating *table*.

    Without this, rows that disappear from the source query (e.g. due to
    data corrections or filter changes) remain in the aggregate table and
    are served at inference time as if they are current.

    Does NOT commit — the caller owns the transaction so DELETE and INSERT
    are atomic (if the insert fails, the DELETE rolls back too).
    """
    if table not in _VALID_TABLES:
        raise ValueError(f"Invalid aggregate table: {table}")
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {table} WHERE patch_id = %s", (patch_id,))


def _match_extra_where(  # noqa: N802 (matches public SQL function naming)
    cfg: TrainerConfig, alias: str = "m",
) -> str:
    """Return extra ``AND ...`` conditions for match filtering.

    Constructed from ``cfg.league_only`` and ``cfg.lobby_types`` so that
    all aggregate populators apply the **same** filter — previously only
    ``populate_h2h`` filtered by ``leagueid > 0``, causing feature
    distribution mismatch.
    """
    parts: list[str] = []
    if cfg.league_only:
        parts.append(f"{alias}.leagueid > 0")
    if cfg.lobby_types:
        lobby_ids = [int(x.strip()) for x in cfg.lobby_types.split(",") if x.strip()]
        if lobby_ids:
            parts.append(f"{alias}.lobby_type IN ({','.join(map(str, lobby_ids))})")
    if not parts:
        return ""
    return " AND " + " AND ".join(parts)


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
    _clean_patch_rows(conn, "ml.team_hero_agg", patch_id)
    extra = _match_extra_where(cfg, "m")
    with conn.cursor() as cur:
        cur.execute(f"""
            WITH team_hero_picks AS (
                SELECT
                    CASE WHEN p.is_radiant THEN m.radiant_team_id ELSE m.dire_team_id END AS team_id,
                    p.hero_id,
                    COUNT(*)                                           AS games,
                    SUM(CASE WHEN p.win = 1 THEN 1 ELSE 0 END)         AS wins,
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
                    SELECT AVG((pta.gold_t ->> 10)::numeric) AS avg_gold_10
                    FROM player_time_series_arrays pta
                    WHERE pta.match_id = m.match_id
                      AND pta.player_slot = p.player_slot
                ) gold10 ON TRUE
                LEFT JOIN LATERAL (
                    SELECT AVG((pta.xp_t ->> 10)::numeric) AS avg_xp_10
                    FROM player_time_series_arrays pta
                    WHERE pta.match_id = m.match_id
                      AND pta.player_slot = p.player_slot
                ) xp10 ON TRUE
                WHERE m.patch = %s
                  AND m.radiant_win IS NOT NULL{extra}
                  AND CASE WHEN p.is_radiant THEN m.radiant_team_id ELSE m.dire_team_id END IS NOT NULL
                GROUP BY team_id, p.hero_id
            ),
            team_hero_bans AS (
                SELECT
                    CASE WHEN pb.team = 0 THEN m.radiant_team_id ELSE m.dire_team_id END AS team_id,
                    pb.hero_id,
                    COUNT(*) AS bans
                FROM matches m
                INNER JOIN picks_bans pb ON pb.match_id = m.match_id
                WHERE m.patch = %s
                  AND m.radiant_win IS NOT NULL{extra}
                  AND pb.is_pick = FALSE
                  AND pb.team IN (0, 1)
                  AND pb.hero_id IS NOT NULL
                  AND CASE WHEN pb.team = 0 THEN m.radiant_team_id ELSE m.dire_team_id END IS NOT NULL
                GROUP BY team_id, pb.hero_id
            )
            SELECT
                p.team_id,
                p.hero_id,
                p.games,
                p.wins,
                COALESCE(b.bans, 0) AS bans,
                p.avg_gpm,
                p.avg_xpm,
                p.avg_kills,
                p.avg_deaths,
                p.avg_assists,
                p.firstblood_rate,
                p.avg_camps_stacked,
                p.avg_vision_placed,
                p.avg_gold_10,
                p.avg_xp_10,
                p.last_played
            FROM team_hero_picks p
            LEFT JOIN team_hero_bans b ON b.team_id = p.team_id AND b.hero_id = p.hero_id
            ORDER BY p.team_id, p.hero_id
        """, (patch_id, patch_id))
        rows: list[tuple[Any, ...]] = []
        for r in cur.fetchall():
            team_id, hero_id, games, wins, bans_v, ag, ax, ak, ad, aa, fbr, acs, avp, ag10, ax10, lp = r
            rows.append((
                patch_id, team_id, hero_id, games, wins, bans_v,
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
    _clean_patch_rows(conn, "ml.player_hero_agg", patch_id)
    extra = _match_extra_where(cfg, "m")
    with conn.cursor() as cur:
        cur.execute(f"""
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
                    SELECT AVG((pta.gold_t ->> 10)::numeric) AS avg_gold_10
                    FROM player_time_series_arrays pta
                    WHERE pta.match_id = m.match_id
                      AND pta.player_slot = p.player_slot
                ) gold10 ON TRUE
            LEFT JOIN LATERAL (
                    SELECT AVG((pta.xp_t ->> 10)::numeric) AS avg_xp_10
                    FROM player_time_series_arrays pta
                    WHERE pta.match_id = m.match_id
                      AND pta.player_slot = p.player_slot
                ) xp10 ON TRUE
            WHERE m.patch = %s
              AND m.radiant_win IS NOT NULL{extra}
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
    _clean_patch_rows(conn, "ml.hero_synergy_agg", patch_id)
    extra = _match_extra_where(cfg, "m")
    with conn.cursor() as cur:
        cur.execute(f"""
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
              AND m.radiant_win IS NOT NULL{extra}
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
    _clean_patch_rows(conn, "ml.hero_counter_agg", patch_id)
    extra = _match_extra_where(cfg, "m")
    with conn.cursor() as cur:
        cur.execute(f"""
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
              AND m.radiant_win IS NOT NULL{extra}
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
    _clean_patch_rows(conn, "ml.team_h2h_agg", patch_id)
    extra = _match_extra_where(cfg, "m")
    with conn.cursor() as cur:
        # NOTE: _match_extra_where builds the league/lobby filter that was
        # previously a hardcoded ``AND leagueid > 0`` only here — now all
        # seven populators use the same config-driven filter.
        cur.execute(f"""
            WITH valid_matches AS (
                SELECT match_id, radiant_team_id, dire_team_id, radiant_win
                FROM matches
                WHERE patch = %s
                  AND radiant_win IS NOT NULL
                  AND radiant_team_id IS NOT NULL
                  AND dire_team_id IS NOT NULL{extra}
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
    _clean_patch_rows(conn, "ml.hero_baseline_agg", patch_id)
    extra = _match_extra_where(cfg, "m")
    with conn.cursor() as cur:
        cur.execute(f"""
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
                    SELECT AVG((pta.gold_t ->> 10)::numeric) AS avg_gold_10
                    FROM player_time_series_arrays pta
                    WHERE pta.match_id = m.match_id
                      AND pta.player_slot = p.player_slot
                ) gold10 ON TRUE
                LEFT JOIN LATERAL (
                    SELECT AVG((pta.xp_t ->> 10)::numeric) AS avg_xp_10
                    FROM player_time_series_arrays pta
                    WHERE pta.match_id = m.match_id
                      AND pta.player_slot = p.player_slot
                ) xp10 ON TRUE
                WHERE m.patch = %s
                  AND m.radiant_win IS NOT NULL{extra}
                GROUP BY p.hero_id
            ),
            hero_bans AS (
                SELECT pb.hero_id, COUNT(*) AS total_bans
                FROM matches m
                INNER JOIN picks_bans pb ON pb.match_id = m.match_id AND pb.is_pick = FALSE
                WHERE m.patch = %s
                  AND m.radiant_win IS NOT NULL{extra}
                GROUP BY pb.hero_id
            ),
            total_matches AS (
                SELECT COUNT(DISTINCT match_id) AS total FROM matches WHERE patch = %s AND radiant_win IS NOT NULL{extra}
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
    _clean_patch_rows(conn, "ml.hero_draft_slot_agg", patch_id)
    extra = _match_extra_where(cfg, "m")
    with conn.cursor() as cur:
        cur.execute(f"""
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
                  AND m.radiant_win IS NOT NULL{extra}
                  AND pb.is_pick = TRUE
            ) ds
            WHERE ds.team_pick_ordinal <= 5
            GROUP BY ds.hero_id, ds.team_pick_ordinal
            HAVING COUNT(*) >= 3  -- FIX: minimum games filter (matches synergy/counter)
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
# PIT-safe snapshot populators — cross-patch lookback (sparse tables)
# ---------------------------------------------------------------------------
# Tables that key on combos (team+hero, player+hero, hero+hero pairs)
# suffer from sparse data within a single patch (~3.5 games/combo).
# These four tables include an optional cross-patch lookback
# (cfg.lookback_patches, default 2) that pools matches from N prior
# patches to densify the snapshot buckets.
#
# Prior-patch matches are weighted by cfg.prior_patch_weight (default 0.5)
# so they contribute to sample size without being treated as equally
# trustworthy as current-patch data (hero balance shifts at patch
# boundaries).
#
# The remaining 3 tables (hero_baseline, hero_draft_slot, team_h2h)
# stay single-patch — their hero-level/team-level aggregation already
# achieves >98% lookup hit rates.
#
# SQL strategy:
#   - patch_days CTE:  as_of_date spine from the TARGET patch only.
#   - eligible_matches: current-patch matches (PIT-cut per as_of_date)
#     UNION prior-patch matches (fully eligible, no per-day cutoff).
#   - Main aggregation then JOINs through eligible_matches.
# ---------------------------------------------------------------------------

POPULATE_TEAM_HERO_SNAPSHOT = """
    INSERT INTO ml.team_hero_snapshot
        (patch_id, as_of_date, team_id, hero_id,
         games, wins, bans, win_rate,
         avg_gpm, avg_xpm, avg_kills, avg_deaths, avg_assists,
         firstblood_rate, avg_camps_stacked, avg_vision_placed,
         avg_gold_10, avg_xp_10, last_played)
    VALUES %s
    ON CONFLICT (patch_id, as_of_date, team_id, hero_id) DO UPDATE SET
        games = EXCLUDED.games, wins = EXCLUDED.wins, bans = EXCLUDED.bans,
        win_rate = EXCLUDED.win_rate,
        avg_gpm = EXCLUDED.avg_gpm, avg_xpm = EXCLUDED.avg_xpm,
        avg_kills = EXCLUDED.avg_kills, avg_deaths = EXCLUDED.avg_deaths,
        avg_assists = EXCLUDED.avg_assists,
        firstblood_rate = EXCLUDED.firstblood_rate,
        avg_camps_stacked = EXCLUDED.avg_camps_stacked,
        avg_vision_placed = EXCLUDED.avg_vision_placed,
        avg_gold_10 = EXCLUDED.avg_gold_10, avg_xp_10 = EXCLUDED.avg_xp_10,
        last_played = EXCLUDED.last_played;
"""


def _team_hero_prior_agg(cfg: TrainerConfig, conn, extra: str, min_patch: int, prior_weight: float):
    """Pre-compute the prior-patch (weighted) aggregate — no date dimension.

    Returns a list of dicts keyed by (team_id, hero_id).
    Prior-patch data is same for every as_of_date (no PIT cutoff).

    All averages are *weighted* by prior_weight so they compose correctly
    with the per-date current-patch aggregates.
    """
    patch_id = cfg.patch_id
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT
                CASE WHEN p.is_radiant THEN m.radiant_team_id ELSE m.dire_team_id END AS team_id,
                p.hero_id,
                SUM(%s)::FLOAT                                                    AS games,
                SUM(CASE WHEN p.win = 1 THEN %s ELSE 0 END)::FLOAT                AS wins,
                SUM(p.gold_per_min * %s) / NULLIF(SUM(%s), 0)::FLOAT              AS avg_gpm,
                SUM(p.xp_per_min * %s) / NULLIF(SUM(%s), 0)::FLOAT                AS avg_xpm,
                SUM(p.kills * %s) / NULLIF(SUM(%s), 0)::FLOAT                     AS avg_kills,
                SUM(p.deaths * %s) / NULLIF(SUM(%s), 0)::FLOAT                    AS avg_deaths,
                SUM(p.assists * %s) / NULLIF(SUM(%s), 0)::FLOAT                   AS avg_assists,
                SUM(p.firstblood_claimed * %s) / NULLIF(SUM(%s), 0)::FLOAT        AS firstblood_rate,
                SUM(p.camps_stacked * %s) / NULLIF(SUM(%s), 0)::FLOAT             AS avg_camps_stacked,
                SUM((p.obs_placed + p.sen_placed) * %s) / NULLIF(SUM(%s), 0)::FLOAT AS avg_vision_placed,
                COALESCE(SUM(g10.avg_gold_10 * %s) / NULLIF(SUM(%s), 0), 0)::FLOAT AS avg_gold_10,
                COALESCE(SUM(x10.avg_xp_10 * %s) / NULLIF(SUM(%s), 0), 0)::FLOAT   AS avg_xp_10,
                MAX(m.start_time)                                                AS last_played
            FROM matches m
            JOIN players p ON p.match_id = m.match_id
            LEFT JOIN LATERAL (
                SELECT AVG((pta.gold_t ->> 10)::numeric) AS avg_gold_10
                    FROM player_time_series_arrays pta
                    WHERE pta.match_id = m.match_id AND pta.player_slot = p.player_slot
            ) g10 ON TRUE
            LEFT JOIN LATERAL (
                SELECT AVG((pta.xp_t ->> 10)::numeric) AS avg_xp_10
                    FROM player_time_series_arrays pta
                    WHERE pta.match_id = m.match_id AND pta.player_slot = p.player_slot
            ) x10 ON TRUE
            WHERE m.radiant_win IS NOT NULL{extra}
              AND m.patch >= %s AND m.patch < %s
              AND CASE WHEN p.is_radiant THEN m.radiant_team_id ELSE m.dire_team_id END IS NOT NULL
            GROUP BY team_id, p.hero_id
        """, [prior_weight] * 22 + [min_patch, patch_id])
        prior: dict[tuple, dict] = {}
        for r in cur.fetchall():
            tid, hid, g, w, agpm, axpm, ak_, ad_, aa_, fbr, acs, avp, ag10, ax10, lp = r
            prior[(tid, hid)] = {
                "games": g, "wins": w,
                "avg_gpm": agpm, "avg_xpm": axpm,
                "avg_kills": ak_, "avg_deaths": ad_, "avg_assists": aa_,
                "firstblood_rate": fbr, "avg_camps_stacked": acs, "avg_vision_placed": avp,
                "avg_gold_10": ag10, "avg_xp_10": ax10,
                "last_played": lp,
            }
        logger.info("  prior_agg: %d team_hero combos from patches %d-%d",
                     len(prior), min_patch, patch_id - 1)
        return prior


def _team_hero_bans_prior(cfg: TrainerConfig, conn, extra: str, min_patch: int):
    """Pre-compute prior-patch bans — no date dimension."""
    patch_id = cfg.patch_id
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT
                CASE WHEN pb.team = 0 THEN m.radiant_team_id ELSE m.dire_team_id END AS team_id,
                pb.hero_id,
                COUNT(*) AS bans
            FROM matches m
            JOIN picks_bans pb ON pb.match_id = m.match_id
            WHERE m.radiant_win IS NOT NULL{extra}
              AND m.patch >= %s AND m.patch < %s
              AND pb.is_pick = FALSE AND pb.team IN (0, 1) AND pb.hero_id IS NOT NULL
              AND CASE WHEN pb.team = 0 THEN m.radiant_team_id ELSE m.dire_team_id END IS NOT NULL
            GROUP BY team_id, pb.hero_id
        """, (min_patch, patch_id))
        bans_prior: dict[tuple, int] = {}
        for r in cur.fetchall():
            tid, hid, cnt = r
            bans_prior[(tid, hid)] = cnt
        logger.info("  bans_prior: %d team_hero combos from patches %d-%d",
                     len(bans_prior), min_patch, patch_id - 1)
        return bans_prior


def populate_team_hero_snapshot(cfg: TrainerConfig, conn) -> int:
    """Populate ml.team_hero_snapshot for *patch_id* (Tier 1 — daily).

    Two-phase cross-patch lookback to avoid materialising the full cross-product
    of dates × patches:

      1. Pre-compute prior-patch aggregate (same for every as_of_date).
      2. For each date, compute current-patch PIT aggregate and merge with prior.
         Prior-only combos are preserved by using prior as the base.
    """
    patch_id = cfg.patch_id
    pg = cfg.prior_games
    pw = cfg.prior_win_rate
    lookback = cfg.lookback_patches
    prior_weight = cfg.prior_patch_weight
    min_patch = patch_id - lookback
    extra = _match_extra_where(cfg, "m")

    _clean_patch_rows(conn, "ml.team_hero_snapshot", patch_id)

    # Phase 1: pre-compute prior-patch aggregates (no date dimension).
    prior = _team_hero_prior_agg(cfg, conn, extra, min_patch, prior_weight)
    bans_prior = _team_hero_bans_prior(cfg, conn, extra, min_patch)

    # Phase 2: for each date, compute current-patch PIT and merge with prior.
    # Use prior as base so prior-only combos are preserved.
    rows = []
    with conn.cursor() as cur:
        # Get all dates for the current patch
        cur.execute(f"""
            SELECT generate_series(
                date_trunc('day', to_timestamp(MIN(start_time)))::date,
                date_trunc('day', to_timestamp(MAX(start_time)))::date,
                '1 day'::interval
            )::date AS as_of_date
            FROM matches
            WHERE patch = %s AND radiant_win IS NOT NULL{extra}
        """, (patch_id,))
        dates = [r[0] for r in cur.fetchall()]

        # For each date, compute current-patch aggregates
        for as_of in dates:
            cur.execute(f"""
                SELECT
                    CASE WHEN p.is_radiant THEN m.radiant_team_id ELSE m.dire_team_id END AS team_id,
                    p.hero_id,
                    COUNT(*)::FLOAT                                  AS games,
                    SUM(CASE WHEN p.win = 1 THEN 1 ELSE 0 END)::FLOAT AS wins,
                    AVG(p.gold_per_min)::FLOAT         AS avg_gpm,
                    AVG(p.xp_per_min)::FLOAT           AS avg_xpm,
                    AVG(p.kills)::FLOAT                AS avg_kills,
                    AVG(p.deaths)::FLOAT               AS avg_deaths,
                    AVG(p.assists)::FLOAT              AS avg_assists,
                    AVG(p.firstblood_claimed)::FLOAT   AS firstblood_rate,
                    AVG(p.camps_stacked)::FLOAT        AS avg_camps_stacked,
                    AVG(p.obs_placed + p.sen_placed)::FLOAT AS avg_vision_placed,
                    COALESCE(AVG(g10.avg_gold_10)::FLOAT, 0) AS avg_gold_10,
                    COALESCE(AVG(x10.avg_xp_10)::FLOAT, 0)   AS avg_xp_10,
                    MAX(m.start_time)                  AS last_played
                FROM matches m
                JOIN players p ON p.match_id = m.match_id
                LEFT JOIN LATERAL (
                    SELECT AVG((pta.gold_t ->> 10)::numeric) AS avg_gold_10
                    FROM player_time_series_arrays pta
                    WHERE pta.match_id = m.match_id AND pta.player_slot = p.player_slot
                ) g10 ON TRUE
                LEFT JOIN LATERAL (
                    SELECT AVG((pta.xp_t ->> 10)::numeric) AS avg_xp_10
                    FROM player_time_series_arrays pta
                    WHERE pta.match_id = m.match_id AND pta.player_slot = p.player_slot
                ) x10 ON TRUE
                WHERE m.radiant_win IS NOT NULL{extra}
                  AND m.patch = %s
                  AND to_timestamp(m.start_time) < %s
                  AND CASE WHEN p.is_radiant THEN m.radiant_team_id ELSE m.dire_team_id END IS NOT NULL
                GROUP BY team_id, p.hero_id
            """, (patch_id, as_of))

            # Build current-patch data for this date
            current: dict[tuple, dict] = {}
            for r in cur.fetchall():
                tid, hid, c_games, c_wins, agpm, axpm, ak_, ad_, aa_, fbr, acs, avp, ag10, ax10, lp = r
                current[(tid, hid)] = {
                    "games": c_games, "wins": c_wins,
                    "avg_gpm": agpm, "avg_xpm": axpm,
                    "avg_kills": ak_, "avg_deaths": ad_, "avg_assists": aa_,
                    "firstblood_rate": fbr, "avg_camps_stacked": acs, "avg_vision_placed": avp,
                    "avg_gold_10": ag10, "avg_xp_10": ax10,
                    "last_played": lp,
                }

            # Merge: start with prior, overlay current
            all_combos = set(prior.keys()) | set(current.keys())
            for combo in all_combos:
                p = prior.get(combo)
                c = current.get(combo)

                if p is not None and c is not None:
                    # Both prior and current exist — merge
                    tid, hid = combo
                    games = c["games"] + p["games"]
                    wins = c["wins"] + p["wins"]
                    total_g = games if games > 0 else 1.0
                    ag   = _wavg(c["avg_gpm"], c["games"], p["avg_gpm"], p["games"], total_g)
                    ax   = _wavg(c["avg_xpm"], c["games"], p["avg_xpm"], p["games"], total_g)
                    ak_  = _wavg(c["avg_kills"], c["games"], p["avg_kills"], p["games"], total_g)
                    ad_  = _wavg(c["avg_deaths"], c["games"], p["avg_deaths"], p["games"], total_g)
                    aa_  = _wavg(c["avg_assists"], c["games"], p["avg_assists"], p["games"], total_g)
                    fbr  = _wavg(c["firstblood_rate"], c["games"], p["firstblood_rate"], p["games"], total_g)
                    acs  = _wavg(c["avg_camps_stacked"], c["games"], p["avg_camps_stacked"], p["games"], total_g)
                    avp  = _wavg(c["avg_vision_placed"], c["games"], p["avg_vision_placed"], p["games"], total_g)
                    ag10 = _wavg(c["avg_gold_10"], c["games"], p["avg_gold_10"], p["games"], total_g)
                    ax10 = _wavg(c["avg_xp_10"], c["games"], p["avg_xp_10"], p["games"], total_g)
                    lp   = max(c["last_played"], p["last_played"])
                elif c is not None:
                    # Current-only (no prior)
                    tid, hid = combo
                    games, wins = c["games"], c["wins"]
                    ag, ax = c["avg_gpm"], c["avg_xpm"]
                    ak_, ad_, aa_ = c["avg_kills"], c["avg_deaths"], c["avg_assists"]
                    fbr, acs, avp = c["firstblood_rate"], c["avg_camps_stacked"], c["avg_vision_placed"]
                    ag10, ax10 = c["avg_gold_10"], c["avg_xp_10"]
                    lp = c["last_played"]
                else:
                    # Prior-only (no current matches for this date)
                    tid, hid = combo
                    games, wins = p["games"], p["wins"]
                    ag, ax = p["avg_gpm"], p["avg_xpm"]
                    ak_, ad_, aa_ = p["avg_kills"], p["avg_deaths"], p["avg_assists"]
                    fbr, acs, avp = p["firstblood_rate"], p["avg_camps_stacked"], p["avg_vision_placed"]
                    ag10, ax10 = p["avg_gold_10"], p["avg_xp_10"]
                    lp = p["last_played"]

                rows.append((
                    patch_id, as_of, tid, hid, games, wins,
                    bans_prior.get(combo, 0),
                    _shrunk_wr(wins, games, pg, pw),
                    ag, ax, ak_, ad_, aa_, fbr, acs, avp, ag10, ax10, lp,
                ))

    total = 0
    with conn.cursor() as cur:
        for batch in _batched(rows, cfg.agg_batch_size):
            psycopg2.extras.execute_values(cur, POPULATE_TEAM_HERO_SNAPSHOT, batch, template=None)
            total += len(batch)
    conn.commit()
    logger.info(
        "populate_team_hero_snapshot: %s rows for patch %s (lookback=%d, prior_w=%.2f)",
        total, patch_id, lookback, prior_weight,
    )
    return total


# ---------------------------------------------------------------------------
# 2b. ml.player_hero_snapshot (Tier 2 — weekly, cross-patch lookback)
# ---------------------------------------------------------------------------

POPULATE_PLAYER_HERO_SNAPSHOT = """
    INSERT INTO ml.player_hero_snapshot
        (patch_id, as_of_date, account_id, hero_id,
         games, wins, win_rate,
         avg_gpm, avg_xpm, avg_kills, avg_deaths, avg_assists,
         avg_kda, lane_role,
         firstblood_rate, avg_camps_stacked, avg_vision_placed,
         avg_gold_10, avg_xp_10, last_played)
    VALUES %s
    ON CONFLICT (patch_id, as_of_date, account_id, hero_id) DO UPDATE SET
        games = EXCLUDED.games, wins = EXCLUDED.wins, win_rate = EXCLUDED.win_rate,
        avg_gpm = EXCLUDED.avg_gpm, avg_xpm = EXCLUDED.avg_xpm,
        avg_kills = EXCLUDED.avg_kills, avg_deaths = EXCLUDED.avg_deaths,
        avg_assists = EXCLUDED.avg_assists,
        avg_kda = EXCLUDED.avg_kda, lane_role = EXCLUDED.lane_role,
        firstblood_rate = EXCLUDED.firstblood_rate,
        avg_camps_stacked = EXCLUDED.avg_camps_stacked,
        avg_vision_placed = EXCLUDED.avg_vision_placed,
        avg_gold_10 = EXCLUDED.avg_gold_10, avg_xp_10 = EXCLUDED.avg_xp_10,
        last_played = EXCLUDED.last_played;
"""


def populate_player_hero_snapshot(cfg: TrainerConfig, conn) -> int:
    """Populate ml.player_hero_snapshot for *patch_id* (Tier 2 — weekly).

    Includes cross-patch lookback (cfg.lookback_patches) to combat sparsity.
    """
    patch_id = cfg.patch_id
    pg = cfg.prior_games
    pw = cfg.prior_win_rate
    lookback = cfg.lookback_patches
    prior_weight = cfg.prior_patch_weight
    min_patch = patch_id - lookback
    _clean_patch_rows(conn, "ml.player_hero_snapshot", patch_id)
    extra = _match_extra_where(cfg, "m")
    with conn.cursor() as cur:
        cur.execute(f"""
            WITH patch_days AS (
                SELECT generate_series(
                    date_trunc('day', to_timestamp(MIN(start_time)))::date,
                    date_trunc('day', to_timestamp(MAX(start_time)))::date,
                    '7 days'::interval
                )::date AS as_of_date
                FROM matches
                WHERE patch = %s AND radiant_win IS NOT NULL{extra}
            ),
            eligible_matches AS (
                SELECT
                    d.as_of_date,
                    m.match_id,
                    CASE WHEN m.patch = %s THEN 1.0 ELSE %s END AS patch_weight
                FROM patch_days d
                JOIN matches m
                    ON m.radiant_win IS NOT NULL{extra}
                   AND m.patch BETWEEN %s AND %s
                   AND (
                        (m.patch = %s AND to_timestamp(m.start_time) < d.as_of_date)
                        OR m.patch < %s
                   )
            )
            SELECT
                em.as_of_date,
                p.account_id, p.hero_id,
                SUM(em.patch_weight)::FLOAT                                          AS games,
                SUM(CASE WHEN p.win = 1 THEN em.patch_weight ELSE 0 END)::FLOAT      AS wins,
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
            FROM eligible_matches em
            JOIN matches m ON m.match_id = em.match_id
            JOIN players p ON p.match_id = m.match_id
            LEFT JOIN LATERAL (
                SELECT AVG((pta.gold_t ->> 10)::numeric) AS avg_gold_10
                    FROM player_time_series_arrays pta
                    WHERE pta.match_id = m.match_id AND pta.player_slot = p.player_slot
            ) gold10 ON TRUE
            LEFT JOIN LATERAL (
                SELECT AVG((pta.xp_t ->> 10)::numeric) AS avg_xp_10
                    FROM player_time_series_arrays pta
                    WHERE pta.match_id = m.match_id AND pta.player_slot = p.player_slot
            ) xp10 ON TRUE
            WHERE p.account_id IS NOT NULL
            GROUP BY em.as_of_date, p.account_id, p.hero_id
            ORDER BY em.as_of_date, p.account_id, p.hero_id
        """, (patch_id, patch_id, prior_weight, min_patch, patch_id, patch_id, patch_id))
        rows = []
        for r in cur.fetchall():
            (as_of, aid, hid, games, wins,
             ag, ax, ak, ad, aa, akda, lr, fbr, acs, avp, ag10, ax10, lp) = r
            rows.append((
                patch_id, as_of, aid, hid,
                games, wins, _shrunk_wr(wins, games, pg, pw),
                ag, ax, ak, ad, aa, akda, lr, fbr, acs, avp, ag10, ax10, lp,
            ))

    total = 0
    with conn.cursor() as cur:
        for batch in _batched(rows, cfg.agg_batch_size):
            psycopg2.extras.execute_values(cur, POPULATE_PLAYER_HERO_SNAPSHOT, batch, template=None)
            total += len(batch)
    conn.commit()
    logger.info(
        "populate_player_hero_snapshot: %s rows for patch %s (lookback=%d, prior_w=%.2f)",
        total, patch_id, lookback, prior_weight,
    )
    return total


# ---------------------------------------------------------------------------
# 3b. ml.hero_synergy_snapshot (Tier 2 — weekly, cross-patch lookback)
# ---------------------------------------------------------------------------

POPULATE_SYNERGY_SNAPSHOT = """
    INSERT INTO ml.hero_synergy_snapshot
        (patch_id, as_of_date, hero_a, hero_b, games, wins, win_rate)
    VALUES %s
    ON CONFLICT (patch_id, as_of_date, hero_a, hero_b) DO UPDATE SET
        games = EXCLUDED.games, wins = EXCLUDED.wins, win_rate = EXCLUDED.win_rate;
"""


def populate_synergy_snapshot(cfg: TrainerConfig, conn) -> int:
    """Populate ml.hero_synergy_snapshot for *patch_id* (Tier 2 — weekly).

    Includes cross-patch lookback (cfg.lookback_patches) to combat sparsity.
    """
    patch_id = cfg.patch_id
    pg = cfg.prior_games
    pw = cfg.prior_win_rate
    lookback = cfg.lookback_patches
    prior_weight = cfg.prior_patch_weight
    min_patch = patch_id - lookback
    _clean_patch_rows(conn, "ml.hero_synergy_snapshot", patch_id)
    extra = _match_extra_where(cfg, "m")
    with conn.cursor() as cur:
        cur.execute(f"""
            WITH patch_days AS (
                SELECT generate_series(
                    date_trunc('day', to_timestamp(MIN(start_time)))::date,
                    date_trunc('day', to_timestamp(MAX(start_time)))::date,
                    '7 days'::interval
                )::date AS as_of_date
                FROM matches
                WHERE patch = %s AND radiant_win IS NOT NULL{extra}
            ),
            eligible_matches AS (
                SELECT
                    d.as_of_date,
                    m.match_id,
                    CASE WHEN m.patch = %s THEN 1.0 ELSE %s END AS patch_weight
                FROM patch_days d
                JOIN matches m
                    ON m.radiant_win IS NOT NULL{extra}
                   AND m.patch BETWEEN %s AND %s
                   AND (
                        (m.patch = %s AND to_timestamp(m.start_time) < d.as_of_date)
                        OR m.patch < %s
                   )
            )
            SELECT
                em.as_of_date,
                p1.hero_id AS hero_a,
                p2.hero_id AS hero_b,
                SUM(em.patch_weight)::FLOAT                                                       AS games,
                SUM(CASE WHEN CASE WHEN p1.is_radiant THEN m.radiant_win ELSE NOT m.radiant_win END
                    THEN em.patch_weight ELSE 0 END)::FLOAT                                       AS wins
            FROM eligible_matches em
            JOIN matches m ON m.match_id = em.match_id
            JOIN players p1 ON p1.match_id = m.match_id
            JOIN players p2 ON p2.match_id = m.match_id
                AND p2.is_radiant = p1.is_radiant
                AND p2.hero_id > p1.hero_id
            GROUP BY em.as_of_date, p1.hero_id, p2.hero_id
            HAVING SUM(em.patch_weight) >= 3
            ORDER BY em.as_of_date, p1.hero_id, p2.hero_id
        """, (patch_id, patch_id, prior_weight, min_patch, patch_id, patch_id, patch_id))
        rows = []
        for r in cur.fetchall():
            as_of, ha, hb, games, wins = r
            rows.append((patch_id, as_of, ha, hb, games, wins, _shrunk_wr(wins, games, pg, pw)))

    total = 0
    with conn.cursor() as cur:
        for batch in _batched(rows, cfg.agg_batch_size):
            psycopg2.extras.execute_values(cur, POPULATE_SYNERGY_SNAPSHOT, batch, template=None)
            total += len(batch)
    conn.commit()
    logger.info(
        "populate_synergy_snapshot: %s rows for patch %s (lookback=%d, prior_w=%.2f)",
        total, patch_id, lookback, prior_weight,
    )
    return total


# ---------------------------------------------------------------------------
# 4b. ml.hero_counter_snapshot (Tier 2 — weekly, cross-patch lookback)
# ---------------------------------------------------------------------------

POPULATE_COUNTER_SNAPSHOT = """
    INSERT INTO ml.hero_counter_snapshot
        (patch_id, as_of_date, hero_id, enemy_hero_id, games, wins, win_rate, avg_kd_diff)
    VALUES %s
    ON CONFLICT (patch_id, as_of_date, hero_id, enemy_hero_id) DO UPDATE SET
        games = EXCLUDED.games, wins = EXCLUDED.wins,
        win_rate = EXCLUDED.win_rate, avg_kd_diff = EXCLUDED.avg_kd_diff;
"""


def populate_counter_snapshot(cfg: TrainerConfig, conn) -> int:
    """Populate ml.hero_counter_snapshot for *patch_id* (Tier 2 — weekly).

    Includes cross-patch lookback (cfg.lookback_patches) to combat sparsity.
    """
    patch_id = cfg.patch_id
    pg = cfg.prior_games
    pw = cfg.prior_win_rate
    lookback = cfg.lookback_patches
    prior_weight = cfg.prior_patch_weight
    min_patch = patch_id - lookback
    _clean_patch_rows(conn, "ml.hero_counter_snapshot", patch_id)
    extra = _match_extra_where(cfg, "m")
    with conn.cursor() as cur:
        cur.execute(f"""
            WITH patch_days AS (
                SELECT generate_series(
                    date_trunc('day', to_timestamp(MIN(start_time)))::date,
                    date_trunc('day', to_timestamp(MAX(start_time)))::date,
                    '7 days'::interval
                )::date AS as_of_date
                FROM matches
                WHERE patch = %s AND radiant_win IS NOT NULL{extra}
            ),
            eligible_matches AS (
                SELECT
                    d.as_of_date,
                    m.match_id,
                    CASE WHEN m.patch = %s THEN 1.0 ELSE %s END AS patch_weight
                FROM patch_days d
                JOIN matches m
                    ON m.radiant_win IS NOT NULL{extra}
                   AND m.patch BETWEEN %s AND %s
                   AND (
                        (m.patch = %s AND to_timestamp(m.start_time) < d.as_of_date)
                        OR m.patch < %s
                   )
            )
            SELECT
                em.as_of_date,
                p1.hero_id,
                p2.hero_id AS enemy_hero_id,
                SUM(em.patch_weight)::FLOAT                                                       AS games,
                SUM(CASE WHEN CASE WHEN p1.is_radiant THEN m.radiant_win ELSE NOT m.radiant_win END
                    THEN em.patch_weight ELSE 0 END)::FLOAT                                       AS wins,
                AVG(p1.kills - p1.deaths)::FLOAT                    AS avg_kd_diff
            FROM eligible_matches em
            JOIN matches m ON m.match_id = em.match_id
            JOIN players p1 ON p1.match_id = m.match_id
            JOIN players p2 ON p2.match_id = m.match_id
                AND p2.is_radiant != p1.is_radiant
            GROUP BY em.as_of_date, p1.hero_id, p2.hero_id
            HAVING SUM(em.patch_weight) >= 3
            ORDER BY em.as_of_date, p1.hero_id, p2.hero_id
        """, (patch_id, patch_id, prior_weight, min_patch, patch_id, patch_id, patch_id))
        rows = []
        for r in cur.fetchall():
            as_of, hid, ehid, games, wins, akd = r
            rows.append((patch_id, as_of, hid, ehid, games, wins, _shrunk_wr(wins, games, pg, pw), akd))

    total = 0
    with conn.cursor() as cur:
        for batch in _batched(rows, cfg.agg_batch_size):
            psycopg2.extras.execute_values(cur, POPULATE_COUNTER_SNAPSHOT, batch, template=None)
            total += len(batch)
    conn.commit()
    logger.info(
        "populate_counter_snapshot: %s rows for patch %s (lookback=%d, prior_w=%.2f)",
        total, patch_id, lookback, prior_weight,
    )
    return total


# ---------------------------------------------------------------------------
# 5b. ml.team_h2h_snapshot (Tier 1 — daily)
# ---------------------------------------------------------------------------

POPULATE_H2H_SNAPSHOT = """
    INSERT INTO ml.team_h2h_snapshot
        (as_of_date, snapshot_tier, patch_id, team_id, enemy_team_id, games, wins, win_rate)
    VALUES %s
    ON CONFLICT (patch_id, as_of_date, team_id, enemy_team_id) DO UPDATE SET
        games = EXCLUDED.games, wins = EXCLUDED.wins, win_rate = EXCLUDED.win_rate;
"""


def populate_h2h_snapshot(cfg: TrainerConfig, conn) -> int:
    """Populate ml.team_h2h_snapshot for *patch_id* (Tier 1 — daily)."""
    patch_id = cfg.patch_id
    pg = cfg.prior_games
    pw = cfg.prior_win_rate
    _clean_patch_rows(conn, "ml.team_h2h_snapshot", patch_id)
    extra = _match_extra_where(cfg, "m")
    with conn.cursor() as cur:
        cur.execute(f"""
            WITH dates AS (
                SELECT DISTINCT d::DATE AS as_of_date
                FROM generate_series(
                    (SELECT MIN(to_timestamp(m.start_time))::DATE FROM matches m WHERE m.patch = %s),
                    (SELECT MAX(to_timestamp(m.start_time))::DATE FROM matches m WHERE m.patch = %s),
                    '1 day'::INTERVAL
                ) d
            )
            SELECT
                d.as_of_date, sub.team_id, sub.enemy_team_id,
                sub.games, sub.wins
            FROM dates d
            CROSS JOIN LATERAL (
                WITH valid_matches AS (
                    SELECT match_id, radiant_team_id, dire_team_id, radiant_win
                    FROM matches
                    WHERE patch = %s
                       AND to_timestamp(start_time) < d.as_of_date
                       AND radiant_win IS NOT NULL
                       AND radiant_team_id IS NOT NULL
                       AND dire_team_id IS NOT NULL{extra}
                ),
                h2h AS (
                    SELECT radiant_team_id AS team_id, dire_team_id AS enemy_team_id, radiant_win AS won
                    FROM valid_matches
                    UNION ALL
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
            ) sub
            ORDER BY d.as_of_date, sub.team_id, sub.enemy_team_id
        """, (patch_id, patch_id, patch_id))
        rows = []
        for r in cur.fetchall():
            as_of, tid, etid, games, wins = r
            rows.append((as_of, "daily", patch_id, tid, etid, games, wins, _shrunk_wr(wins, games, pg, pw)))

    total = 0
    with conn.cursor() as cur:
        for batch in _batched(rows, cfg.agg_batch_size):
            psycopg2.extras.execute_values(cur, POPULATE_H2H_SNAPSHOT, batch, template=None)
            total += len(batch)
    conn.commit()
    logger.info("populate_h2h_snapshot: %s rows for patch %s", total, patch_id)
    return total


# ---------------------------------------------------------------------------
# 6b. ml.hero_baseline_snapshot (Tier 1 — daily)
# ---------------------------------------------------------------------------

POPULATE_BASELINE_SNAPSHOT = """
    INSERT INTO ml.hero_baseline_snapshot
        (as_of_date, snapshot_tier, patch_id, hero_id,
         total_picks, total_wins, total_bans, win_rate,
         pick_rate, ban_rate,
         avg_gpm, avg_xpm, avg_kills, avg_deaths, avg_assists,
         avg_gold_10, avg_xp_10)
    VALUES %s
    ON CONFLICT (patch_id, as_of_date, hero_id) DO UPDATE SET
        total_picks = EXCLUDED.total_picks, total_wins = EXCLUDED.total_wins,
        total_bans = EXCLUDED.total_bans, win_rate = EXCLUDED.win_rate,
        pick_rate = EXCLUDED.pick_rate, ban_rate = EXCLUDED.ban_rate,
        avg_gpm = EXCLUDED.avg_gpm, avg_xpm = EXCLUDED.avg_xpm,
        avg_kills = EXCLUDED.avg_kills, avg_deaths = EXCLUDED.avg_deaths,
        avg_assists = EXCLUDED.avg_assists,
        avg_gold_10 = EXCLUDED.avg_gold_10, avg_xp_10 = EXCLUDED.avg_xp_10;
"""


def populate_baseline_snapshot(cfg: TrainerConfig, conn) -> int:
    """Populate ml.hero_baseline_snapshot for *patch_id* (Tier 1 — daily)."""
    patch_id = cfg.patch_id
    pg = cfg.prior_games
    pw = cfg.prior_win_rate
    _clean_patch_rows(conn, "ml.hero_baseline_snapshot", patch_id)
    extra = _match_extra_where(cfg, "m")
    with conn.cursor() as cur:
        cur.execute(f"""
            WITH dates AS (
                SELECT DISTINCT d::DATE AS as_of_date
                FROM generate_series(
                    (SELECT MIN(to_timestamp(m.start_time))::DATE FROM matches m WHERE m.patch = %s),
                    (SELECT MAX(to_timestamp(m.start_time))::DATE FROM matches m WHERE m.patch = %s),
                    '1 day'::INTERVAL
                ) d
            ),
            picks_pit AS (
                SELECT d.as_of_date,
                       sub.hero_id, sub.total_picks, sub.total_wins,
                       sub.avg_gpm, sub.avg_xpm, sub.avg_kills, sub.avg_deaths, sub.avg_assists,
                       sub.avg_gold_10, sub.avg_xp_10
                FROM dates d
                CROSS JOIN LATERAL (
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
                        SELECT AVG((pta.gold_t ->> 10)::numeric) AS avg_gold_10
                        FROM player_time_series_arrays pta
                        WHERE pta.match_id = m.match_id AND pta.player_slot = p.player_slot
                    ) gold10 ON TRUE
                    LEFT JOIN LATERAL (
                        SELECT AVG((pta.xp_t ->> 10)::numeric) AS avg_xp_10
                        FROM player_time_series_arrays pta
                        WHERE pta.match_id = m.match_id AND pta.player_slot = p.player_slot
                    ) xp10 ON TRUE
                    WHERE m.patch = %s
                      AND to_timestamp(m.start_time) < d.as_of_date
                      AND m.radiant_win IS NOT NULL{extra}
                    GROUP BY p.hero_id
                ) sub
            ),
            bans_pit AS (
                SELECT d.as_of_date, sub.hero_id, sub.total_bans
                FROM dates d
                CROSS JOIN LATERAL (
                    SELECT pb.hero_id, COUNT(*) AS total_bans
                    FROM matches m
                    INNER JOIN picks_bans pb ON pb.match_id = m.match_id AND pb.is_pick = FALSE
                    WHERE m.patch = %s
                      AND to_timestamp(m.start_time) < d.as_of_date
                      AND m.radiant_win IS NOT NULL{extra}
                    GROUP BY pb.hero_id
                ) sub
            ),
            total_matches_pit AS (
                SELECT d.as_of_date, sub.total
                FROM dates d
                CROSS JOIN LATERAL (
                    SELECT COUNT(DISTINCT match_id) AS total
                    FROM matches
                    WHERE patch = %s
                      AND to_timestamp(start_time) < d.as_of_date
                      AND radiant_win IS NOT NULL{extra}
                ) sub
            )
            SELECT
                COALESCE(p.as_of_date, b.as_of_date) AS as_of_date,
                COALESCE(p.hero_id, b.hero_id)        AS hero_id,
                COALESCE(p.total_picks, 0)             AS total_picks,
                COALESCE(p.total_wins, 0)              AS total_wins,
                COALESCE(b.total_bans, 0)              AS total_bans,
                p.avg_gpm, p.avg_xpm, p.avg_kills, p.avg_deaths, p.avg_assists,
                p.avg_gold_10, p.avg_xp_10,
                tm.total
            FROM picks_pit p
            FULL OUTER JOIN bans_pit b ON b.as_of_date = p.as_of_date AND b.hero_id = p.hero_id
            INNER JOIN total_matches_pit tm ON tm.as_of_date = COALESCE(p.as_of_date, b.as_of_date)
            ORDER BY hero_id, as_of_date
        """, (patch_id, patch_id, patch_id, patch_id, patch_id))
        rows = []
        for r in cur.fetchall():
            as_of, hid, picks, wins, bans, ag, ax, ak, ad, aa, ag10, ax10, tot = r
            pick_rate = picks / tot if tot > 0 else 0.0
            ban_rate = bans / tot if tot > 0 else 0.0
            rows.append((
                as_of, "daily", patch_id, hid,
                picks, wins, bans, _shrunk_wr(wins, picks, pg, pw),
                pick_rate, ban_rate, ag, ax, ak, ad, aa, ag10, ax10,
            ))

    total = 0
    with conn.cursor() as cur:
        for batch in _batched(rows, cfg.agg_batch_size):
            psycopg2.extras.execute_values(cur, POPULATE_BASELINE_SNAPSHOT, batch, template=None)
            total += len(batch)
    conn.commit()
    logger.info("populate_baseline_snapshot: %s rows for patch %s", total, patch_id)
    return total


# ---------------------------------------------------------------------------
# 7b. ml.hero_draft_slot_snapshot (Tier 1 — daily)
# ---------------------------------------------------------------------------

POPULATE_HERO_DRAFT_SLOT_SNAPSHOT = """
    INSERT INTO ml.hero_draft_slot_snapshot
        (as_of_date, snapshot_tier, patch_id, hero_id, team_pick_ordinal, games, wins, win_rate)
    VALUES %s
    ON CONFLICT (patch_id, as_of_date, hero_id, team_pick_ordinal) DO UPDATE SET
        games = EXCLUDED.games, wins = EXCLUDED.wins, win_rate = EXCLUDED.win_rate;
"""


def populate_hero_draft_slot_snapshot(cfg: TrainerConfig, conn) -> int:
    """Populate ml.hero_draft_slot_snapshot for *patch_id* (Tier 1 — daily)."""
    patch_id = cfg.patch_id
    pg = cfg.prior_games
    pw = cfg.prior_win_rate
    _clean_patch_rows(conn, "ml.hero_draft_slot_snapshot", patch_id)
    extra = _match_extra_where(cfg, "m")
    with conn.cursor() as cur:
        cur.execute(f"""
            WITH dates AS (
                SELECT DISTINCT d::DATE AS as_of_date
                FROM generate_series(
                    (SELECT MIN(to_timestamp(m.start_time))::DATE FROM matches m WHERE m.patch = %s),
                    (SELECT MAX(to_timestamp(m.start_time))::DATE FROM matches m WHERE m.patch = %s),
                    '1 day'::INTERVAL
                ) d
            )
            SELECT
                d.as_of_date, sub.hero_id, sub.team_pick_ordinal,
                sub.games, sub.wins
            FROM dates d
            CROSS JOIN LATERAL (
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
                      AND to_timestamp(m.start_time) < d.as_of_date
                      AND m.radiant_win IS NOT NULL{extra}
                      AND pb.is_pick = TRUE
                ) ds
                WHERE ds.team_pick_ordinal <= 5
                GROUP BY ds.hero_id, ds.team_pick_ordinal
                HAVING COUNT(*) >= 3
            ) sub
            ORDER BY d.as_of_date, sub.hero_id, sub.team_pick_ordinal
        """, (patch_id, patch_id, patch_id))
        rows = []
        for r in cur.fetchall():
            as_of, hid, tpo, games, wins = r
            rows.append((as_of, "daily", patch_id, hid, tpo, games, wins, _shrunk_wr(wins, games, pg, pw)))

    total = 0
    with conn.cursor() as cur:
        for batch in _batched(rows, cfg.agg_batch_size):
            psycopg2.extras.execute_values(cur, POPULATE_HERO_DRAFT_SLOT_SNAPSHOT, batch, template=None)
            total += len(batch)
    conn.commit()
    logger.info("populate_hero_draft_slot_snapshot: %s rows for patch %s", total, patch_id)
    return total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

ALL_POPULATORS = [
    ("ml.team_hero_agg",               populate_team_hero),
    ("ml.player_hero_agg",             populate_player_hero),
    ("ml.hero_synergy_agg",            populate_synergy),
    ("ml.hero_counter_agg",            populate_counter),
    ("ml.team_h2h_agg",                populate_h2h),
    ("ml.hero_baseline_agg",           populate_baseline),
    ("ml.hero_draft_slot_agg",         populate_hero_draft_slot),
    # PIT-safe snapshot tables
    ("ml.team_hero_snapshot",          populate_team_hero_snapshot),
    ("ml.player_hero_snapshot",        populate_player_hero_snapshot),
    ("ml.hero_synergy_snapshot",       populate_synergy_snapshot),
    ("ml.hero_counter_snapshot",       populate_counter_snapshot),
    ("ml.team_h2h_snapshot",           populate_h2h_snapshot),
    ("ml.hero_baseline_snapshot",      populate_baseline_snapshot),
    ("ml.hero_draft_slot_snapshot",    populate_hero_draft_slot_snapshot),
]


def _analyze_ml_tables(conn) -> None:
    """Refresh statistics on all ML aggregate tables after population."""
    tables = [
        "ml.team_hero_agg",
        "ml.player_hero_agg",
        "ml.hero_synergy_agg",
        "ml.hero_counter_agg",
        "ml.team_h2h_agg",
        "ml.hero_baseline_agg",
        "ml.hero_draft_slot_agg",
        # PIT-safe snapshot tables
        "ml.team_hero_snapshot",
        "ml.player_hero_snapshot",
        "ml.hero_synergy_snapshot",
        "ml.hero_counter_snapshot",
        "ml.team_h2h_snapshot",
        "ml.hero_baseline_snapshot",
        "ml.hero_draft_slot_snapshot",
    ]
    # VACUUM requires autocommit (Postgres forbids VACUUM inside a transaction block).
    old_autocommit = conn.autocommit
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            for tbl in tables:
                cur.execute(f"VACUUM ANALYZE {tbl}")
    finally:
        conn.autocommit = old_autocommit
    logger.info("Analyzed %d ML tables", len(tables))


def populate_all(cfg: TrainerConfig, conn) -> dict[str, int]:
    """Run all 14 populator functions (7 base + 7 snapshot), then ANALYZE.

    Returns a dict of ``{table_name: row_count}``.
    """
    counts: dict[str, int] = {}
    for name, fn in ALL_POPULATORS:
        logger.info("Populating %s ...", name)
        cnt = fn(cfg, conn)
        counts[name] = cnt
    _analyze_ml_tables(conn)
    return counts
