"""Populate the ML aggregate tables.

Each function reads from the core tables (matches, players, picks_bans, teams,
team_games, etc.) and writes into the corresponding ml.*_agg or ml.*_snapshot
table, filtered to a single patch_id. The base tables are LOGGED (crash-safe).
The snapshot tables are also LOGGED. The inference API reads from the base
tables at prediction time (PIT safety is only needed during training).

**Stale row protection**: Every populator ``DELETE``s rows for the current
patch_id before re-inserting, so rows that disappear from source queries
(e.g. after data corrections or filter changes) remain in aggregate tables and
are served at inference time as if they are current.

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

def _shrunk_wr(wins, games, prior_games: float = 3.0, prior_wr: float = 0.5) -> float:
    """Bayesian-shrunken win rate: (wins + prior_games * prior_wr) / (games + prior_games).
    Accepts float or Decimal *wins/games* (fractional counts from time decay)."""
    w = float(wins or 0)
    g = float(games or 0)
    return (w + prior_games * prior_wr) / (g + prior_games) if g > 0 else prior_wr


def _batched(rows: list[tuple], batch_size: int):
    """Yield successive chunks of *rows*."""
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


def _wavg(c_val, c_w, p_val, p_w, total_g) -> float:
    """Weighted average of current and prior values."""
    return (float(c_val or 0) * float(c_w or 0) + float(p_val or 0) * float(p_w or 0)) / float(total_g or 1)


_VALID_TABLES = frozenset({
    "ml.team_hero_agg", "ml.player_hero_agg",
    "ml.hero_synergy_agg", "ml.hero_counter_agg",
    "ml.team_h2h_agg", "ml.hero_baseline_agg",
    "ml.hero_draft_slot_agg",
    "ml.team_hero_snapshot", "ml.player_hero_snapshot",
    "ml.hero_synergy_snapshot", "ml.hero_counter_snapshot",
    "ml.team_h2h_snapshot", "ml.hero_baseline_snapshot",
    "ml.hero_draft_slot_snapshot",
    "ml.team_elo",
})


def _clean_patch_rows(conn, table: str, patch_id: int) -> None:
    """Delete stale rows for *patch_id* before re-populating *table*."""
    if table not in _VALID_TABLES:
        raise ValueError(f"Invalid aggregate table: {table}")
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {table} WHERE patch_id = %s", (patch_id,))


def _bulk_upsert(conn, table: str, patch_id: int, query: str, rows: list[tuple], batch_size: int) -> int:
    """Handles deletion of stale rows and batch insertion (no commit — caller manages transaction)."""
    _clean_patch_rows(conn, table, patch_id)
    total = 0
    with conn.cursor() as cur:
        for batch in _batched(rows, batch_size):
            psycopg2.extras.execute_values(cur, query, batch, template=None)
            total += len(batch)
    logger.info("populate_%s: %s rows for patch %s", table.split('.')[-1], total, patch_id)
    return total


def _match_extra_where(cfg: TrainerConfig, alias: str = "m") -> str:
    """Return extra ``AND ...`` conditions for match filtering."""
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
# 0. ml.team_elo (chronological Elo, K=32)
# ---------------------------------------------------------------------------

def populate_team_elo(cfg: TrainerConfig, conn) -> int:
    """Compute chronological Elo ratings for all teams from match results."""
    logger.info("Computing Team Elo ratings chronologically...")
    with conn.cursor() as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS ml.team_elo (team_id BIGINT PRIMARY KEY, elo FLOAT)")
        cur.execute("""
            SELECT radiant_team_id, dire_team_id, radiant_win
            FROM matches
            WHERE radiant_team_id IS NOT NULL AND dire_team_id IS NOT NULL
              AND radiant_win IS NOT NULL
            ORDER BY start_time ASC
        """)
        matches = cur.fetchall()

    elos: dict[int, float] = {}
    for rad, dire, r_win in matches:
        r_elo = elos.get(rad, 1500.0)
        d_elo = elos.get(dire, 1500.0)
        e_rad = 1.0 / (1.0 + 10 ** ((d_elo - r_elo) / 400.0))
        s_rad = 1.0 if r_win else 0.0
        elos[rad] = r_elo + 32 * (s_rad - e_rad)
        elos[dire] = d_elo + ((1.0 - s_rad) - (1.0 - e_rad)) * 32

    rows = [(t, e) for t, e in elos.items()]
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE ml.team_elo")
        psycopg2.extras.execute_values(cur, "INSERT INTO ml.team_elo (team_id, elo) VALUES %s", rows)
    logger.info("populate_team_elo: computed Elo for %d teams", len(rows))
    return len(rows)


# ---------------------------------------------------------------------------
# 1. ml.team_hero_agg
# ---------------------------------------------------------------------------

POPULATE_TEAM_HERO = """
    INSERT INTO ml.team_hero_agg (patch_id, team_id, hero_id, games, wins, bans, win_rate, avg_gpm, avg_xpm, avg_kills, avg_deaths, avg_assists, firstblood_rate, avg_camps_stacked, avg_vision_placed, avg_gold_10, avg_xp_10, last_played)
    VALUES %s
    ON CONFLICT (patch_id, team_id, hero_id) DO UPDATE SET
        games=EXCLUDED.games, wins=EXCLUDED.wins, bans=EXCLUDED.bans, win_rate=EXCLUDED.win_rate,
        avg_gpm=EXCLUDED.avg_gpm, avg_xpm=EXCLUDED.avg_xpm, avg_kills=EXCLUDED.avg_kills,
        avg_deaths=EXCLUDED.avg_deaths, avg_assists=EXCLUDED.avg_assists,
        firstblood_rate=EXCLUDED.firstblood_rate, avg_camps_stacked=EXCLUDED.avg_camps_stacked,
        avg_vision_placed=EXCLUDED.avg_vision_placed, avg_gold_10=EXCLUDED.avg_gold_10,
        avg_xp_10=EXCLUDED.avg_xp_10, last_played=EXCLUDED.last_played;
"""

def _decay_expr(cfg: TrainerConfig, alias: str = "m") -> str:
    """Return SQL expression for exponential time decay weight.
    Uses cfg.decay_ref_time as reference (0 = NOW()). Half-life ~14 days."""
    if cfg.decay_ref_time:
        ref = str(cfg.decay_ref_time)
    else:
        ref = "EXTRACT(EPOCH FROM NOW())"
    return f"EXP(-0.05 * ({ref} - {alias}.start_time) / 86400.0)"


def populate_team_hero(cfg: TrainerConfig, conn) -> int:
    patch_id = cfg.patch_id
    pg = cfg.prior_games
    pw = cfg.prior_win_rate
    _clean_patch_rows(conn, "ml.team_hero_agg", patch_id)
    extra = _match_extra_where(cfg, "m")
    decay = _decay_expr(cfg, "m")
    with conn.cursor() as cur:
        cur.execute(f"""
            WITH team_hero_picks AS (
                SELECT
                    CASE WHEN p.is_radiant THEN m.radiant_team_id ELSE m.dire_team_id END AS team_id,
                    p.hero_id,
                    SUM({decay})::FLOAT AS games,
                    SUM(CASE WHEN p.win = 1 THEN {decay} ELSE 0 END)::FLOAT AS wins,
                    SUM(p.gold_per_min * {decay}) / NULLIF(SUM({decay}), 0)::FLOAT AS avg_gpm,
                    SUM(p.xp_per_min * {decay}) / NULLIF(SUM({decay}), 0)::FLOAT AS avg_xpm,
                    SUM(p.kills * {decay}) / NULLIF(SUM({decay}), 0)::FLOAT AS avg_kills,
                    SUM(p.deaths * {decay}) / NULLIF(SUM({decay}), 0)::FLOAT AS avg_deaths,
                    SUM(p.assists * {decay}) / NULLIF(SUM({decay}), 0)::FLOAT AS avg_assists,
                    SUM(p.firstblood_claimed * {decay}) / NULLIF(SUM({decay}), 0)::FLOAT AS firstblood_rate,
                    SUM(p.camps_stacked * {decay}) / NULLIF(SUM({decay}), 0)::FLOAT AS avg_camps_stacked,
                    SUM((p.obs_placed + p.sen_placed) * {decay}) / NULLIF(SUM({decay}), 0)::FLOAT AS avg_vision_placed,
                    COALESCE(SUM(gold10.avg_gold_10 * {decay}) / NULLIF(SUM({decay}), 0), 0)::FLOAT AS avg_gold_10,
                    COALESCE(SUM(xp10.avg_xp_10 * {decay}) / NULLIF(SUM({decay}), 0), 0)::FLOAT AS avg_xp_10,
                    MAX(m.start_time) AS last_played
                FROM matches m INNER JOIN players p ON p.match_id = m.match_id
                LEFT JOIN LATERAL (SELECT AVG((pta.gold_t ->> 10)::numeric) AS avg_gold_10 FROM player_time_series_arrays pta WHERE pta.match_id = m.match_id AND pta.player_slot = p.player_slot) gold10 ON TRUE
                LEFT JOIN LATERAL (SELECT AVG((pta.xp_t ->> 10)::numeric) AS avg_xp_10 FROM player_time_series_arrays pta WHERE pta.match_id = m.match_id AND pta.player_slot = p.player_slot) xp10 ON TRUE
                WHERE m.patch = %s AND m.radiant_win IS NOT NULL{extra}
                  AND CASE WHEN p.is_radiant THEN m.radiant_team_id ELSE m.dire_team_id END IS NOT NULL
                GROUP BY team_id, p.hero_id
                HAVING SUM({decay}) >= 1
            ),
            team_hero_bans AS (
                SELECT CASE WHEN pb.team = 0 THEN m.radiant_team_id ELSE m.dire_team_id END AS team_id,
                       pb.hero_id, SUM({decay})::FLOAT AS bans
                FROM matches m INNER JOIN picks_bans pb ON pb.match_id = m.match_id
                WHERE m.patch = %s AND m.radiant_win IS NOT NULL{extra}
                  AND pb.is_pick = FALSE AND pb.team IN (0, 1) AND pb.hero_id IS NOT NULL
                  AND CASE WHEN pb.team = 0 THEN m.radiant_team_id ELSE m.dire_team_id END IS NOT NULL
                GROUP BY team_id, pb.hero_id
                HAVING SUM({decay}) >= 1
            )
            SELECT p.team_id, p.hero_id, p.games, p.wins, COALESCE(b.bans, 0) AS bans,
                   p.avg_gpm, p.avg_xpm, p.avg_kills, p.avg_deaths, p.avg_assists,
                   p.firstblood_rate, p.avg_camps_stacked, p.avg_vision_placed,
                   p.avg_gold_10, p.avg_xp_10, p.last_played
            FROM team_hero_picks p LEFT JOIN team_hero_bans b ON b.team_id = p.team_id AND b.hero_id = p.hero_id
            ORDER BY p.team_id, p.hero_id
        """, (patch_id, patch_id))
        rows = []
        for r in cur.fetchall():
            tid, hid, games, wins, bans_v, ag, ax, ak, ad, aa, fbr, acs, avp, ag10, ax10, lp = r
            rows.append((patch_id, tid, hid, games, wins, bans_v, _shrunk_wr(wins, games, pg, pw), ag, ax, ak, ad, aa, fbr, acs, avp, ag10, ax10, lp))
    total = 0
    with conn.cursor() as cur:
        for batch in _batched(rows, cfg.agg_batch_size):
            psycopg2.extras.execute_values(cur, POPULATE_TEAM_HERO, batch, template=None)
            total += len(batch)
    logger.info("populate_team_hero: %s rows for patch %s", total, patch_id)
    return total


# ---------------------------------------------------------------------------
# 2. ml.player_hero_agg
# ---------------------------------------------------------------------------

POPULATE_PLAYER_HERO = """
    INSERT INTO ml.player_hero_agg (patch_id, account_id, hero_id, games, wins, win_rate, avg_gpm, avg_xpm, avg_kills, avg_deaths, avg_assists, avg_kda, lane_role, firstblood_rate, avg_camps_stacked, avg_vision_placed, avg_gold_10, avg_xp_10, last_played)
    VALUES %s
    ON CONFLICT (patch_id, account_id, hero_id) DO UPDATE SET
        games=EXCLUDED.games, wins=EXCLUDED.wins, win_rate=EXCLUDED.win_rate,
        avg_gpm=EXCLUDED.avg_gpm, avg_xpm=EXCLUDED.avg_xpm, avg_kills=EXCLUDED.avg_kills,
        avg_deaths=EXCLUDED.avg_deaths, avg_assists=EXCLUDED.avg_assists, avg_kda=EXCLUDED.avg_kda,
        lane_role=EXCLUDED.lane_role, firstblood_rate=EXCLUDED.firstblood_rate,
        avg_camps_stacked=EXCLUDED.avg_camps_stacked, avg_vision_placed=EXCLUDED.avg_vision_placed,
        avg_gold_10=EXCLUDED.avg_gold_10, avg_xp_10=EXCLUDED.avg_xp_10, last_played=EXCLUDED.last_played;
"""

def populate_player_hero(cfg: TrainerConfig, conn) -> int:
    patch_id = cfg.patch_id
    pg = cfg.prior_games
    pw = cfg.prior_win_rate
    _clean_patch_rows(conn, "ml.player_hero_agg", patch_id)
    extra = _match_extra_where(cfg, "m")
    decay = _decay_expr(cfg, "m")
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT p.account_id, p.hero_id,
                   SUM({decay})::FLOAT AS games,
                   SUM(CASE WHEN p.win = 1 THEN {decay} ELSE 0 END)::FLOAT AS wins,
                   SUM(p.gold_per_min * {decay}) / NULLIF(SUM({decay}), 0)::FLOAT AS avg_gpm,
                   SUM(p.xp_per_min * {decay}) / NULLIF(SUM({decay}), 0)::FLOAT AS avg_xpm,
                   SUM(p.kills * {decay}) / NULLIF(SUM({decay}), 0)::FLOAT AS avg_kills,
                   SUM(p.deaths * {decay}) / NULLIF(SUM({decay}), 0)::FLOAT AS avg_deaths,
                   SUM(p.assists * {decay}) / NULLIF(SUM({decay}), 0)::FLOAT AS avg_assists,
                   SUM(p.kda * {decay}) / NULLIF(SUM({decay}), 0)::FLOAT AS avg_kda,
                   MODE() WITHIN GROUP (ORDER BY p.lane_role) AS lane_role,
                   SUM(p.firstblood_claimed * {decay}) / NULLIF(SUM({decay}), 0)::FLOAT AS firstblood_rate,
                   SUM(p.camps_stacked * {decay}) / NULLIF(SUM({decay}), 0)::FLOAT AS avg_camps_stacked,
                   SUM((p.obs_placed + p.sen_placed) * {decay}) / NULLIF(SUM({decay}), 0)::FLOAT AS avg_vision_placed,
                   COALESCE(SUM(gold10.avg_gold_10 * {decay}) / NULLIF(SUM({decay}), 0), 0)::FLOAT AS avg_gold_10,
                   COALESCE(SUM(xp10.avg_xp_10 * {decay}) / NULLIF(SUM({decay}), 0), 0)::FLOAT AS avg_xp_10,
                   MAX(m.start_time) AS last_played
            FROM matches m INNER JOIN players p ON p.match_id = m.match_id
            LEFT JOIN LATERAL (SELECT AVG((pta.gold_t ->> 10)::numeric) AS avg_gold_10 FROM player_time_series_arrays pta WHERE pta.match_id = m.match_id AND pta.player_slot = p.player_slot) gold10 ON TRUE
            LEFT JOIN LATERAL (SELECT AVG((pta.xp_t ->> 10)::numeric) AS avg_xp_10 FROM player_time_series_arrays pta WHERE pta.match_id = m.match_id AND pta.player_slot = p.player_slot) xp10 ON TRUE
            WHERE m.patch = %s AND m.radiant_win IS NOT NULL{extra} AND p.account_id IS NOT NULL
            GROUP BY p.account_id, p.hero_id
            HAVING SUM({decay}) >= 1
            ORDER BY p.account_id, p.hero_id
        """, (patch_id,))
        rows = []
        for r in cur.fetchall():
            aid, hid, games, wins, ag, ax, ak, ad, aa, akda, lr, fbr, acs, avp, ag10, ax10, lp = r
            rows.append((patch_id, aid, hid, games, wins, _shrunk_wr(wins, games, pg, pw), ag, ax, ak, ad, aa, akda, lr, fbr, acs, avp, ag10, ax10, lp))
    total = 0
    with conn.cursor() as cur:
        for batch in _batched(rows, cfg.agg_batch_size):
            psycopg2.extras.execute_values(cur, POPULATE_PLAYER_HERO, batch, template=None)
            total += len(batch)
    logger.info("populate_player_hero: %s rows for patch %s", total, patch_id)
    return total


# ---------------------------------------------------------------------------
# 3. ml.hero_synergy_agg
# ---------------------------------------------------------------------------

POPULATE_SYNERGY = """
    INSERT INTO ml.hero_synergy_agg (patch_id, hero_a, hero_b, games, wins, win_rate)
    VALUES %s ON CONFLICT (patch_id, hero_a, hero_b) DO UPDATE SET games=EXCLUDED.games, wins=EXCLUDED.wins, win_rate=EXCLUDED.win_rate;
"""

def populate_synergy(cfg: TrainerConfig, conn) -> int:
    patch_id = cfg.patch_id
    pg = cfg.prior_games
    pw = cfg.prior_win_rate
    _clean_patch_rows(conn, "ml.hero_synergy_agg", patch_id)
    extra = _match_extra_where(cfg, "m")
    decay = _decay_expr(cfg, "m")
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT p1.hero_id AS hero_a, p2.hero_id AS hero_b,
                   SUM({decay})::FLOAT AS games,
                   SUM(CASE WHEN CASE WHEN p1.is_radiant THEN m.radiant_win ELSE NOT m.radiant_win END THEN {decay} ELSE 0 END)::FLOAT AS wins
            FROM matches m INNER JOIN players p1 ON p1.match_id = m.match_id
            INNER JOIN players p2 ON p2.match_id = m.match_id AND p2.is_radiant = p1.is_radiant AND p2.hero_id > p1.hero_id
            WHERE m.patch = %s AND m.radiant_win IS NOT NULL{extra}
            GROUP BY p1.hero_id, p2.hero_id HAVING SUM({decay}) >= 1 ORDER BY p1.hero_id, p2.hero_id
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
    logger.info("populate_synergy: %s rows for patch %s", total, patch_id)
    return total


# ---------------------------------------------------------------------------
# 4. ml.hero_counter_agg
# ---------------------------------------------------------------------------

POPULATE_COUNTER = """
    INSERT INTO ml.hero_counter_agg (patch_id, hero_id, enemy_hero_id, games, wins, win_rate, avg_kd_diff)
    VALUES %s ON CONFLICT (patch_id, hero_id, enemy_hero_id) DO UPDATE SET games=EXCLUDED.games, wins=EXCLUDED.wins, win_rate=EXCLUDED.win_rate, avg_kd_diff=EXCLUDED.avg_kd_diff;
"""

def populate_counter(cfg: TrainerConfig, conn) -> int:
    patch_id = cfg.patch_id
    pg = cfg.prior_games
    pw = cfg.prior_win_rate
    _clean_patch_rows(conn, "ml.hero_counter_agg", patch_id)
    extra = _match_extra_where(cfg, "m")
    decay = _decay_expr(cfg, "m")
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT p1.hero_id, p2.hero_id AS enemy_hero_id,
                   SUM({decay})::FLOAT AS games,
                   SUM(CASE WHEN CASE WHEN p1.is_radiant THEN m.radiant_win ELSE NOT m.radiant_win END THEN {decay} ELSE 0 END)::FLOAT AS wins,
                   SUM((p1.kills - p1.deaths) * {decay}) / NULLIF(SUM({decay}), 0)::FLOAT AS avg_kd_diff
            FROM matches m INNER JOIN players p1 ON p1.match_id = m.match_id
            INNER JOIN players p2 ON p2.match_id = m.match_id AND p2.is_radiant != p1.is_radiant
            WHERE m.patch = %s AND m.radiant_win IS NOT NULL{extra}
            GROUP BY p1.hero_id, p2.hero_id HAVING SUM({decay}) >= 1 ORDER BY p1.hero_id, p2.hero_id
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
    logger.info("populate_counter: %s rows for patch %s", total, patch_id)
    return total


# ---------------------------------------------------------------------------
# 5. ml.team_h2h_agg
# ---------------------------------------------------------------------------

POPULATE_H2H = """
    INSERT INTO ml.team_h2h_agg (patch_id, team_id, enemy_team_id, games, wins, win_rate)
    VALUES %s ON CONFLICT (patch_id, team_id, enemy_team_id) DO UPDATE SET games=EXCLUDED.games, wins=EXCLUDED.wins, win_rate=EXCLUDED.win_rate;
"""

def populate_h2h(cfg: TrainerConfig, conn) -> int:
    patch_id = cfg.patch_id
    pg = cfg.prior_games
    pw = cfg.prior_win_rate
    _clean_patch_rows(conn, "ml.team_h2h_agg", patch_id)
    extra = _match_extra_where(cfg, "m")
    with conn.cursor() as cur:
        cur.execute(f"""
            WITH valid_matches AS (
                SELECT match_id, radiant_team_id, dire_team_id, radiant_win FROM matches
                WHERE patch = %s AND radiant_win IS NOT NULL AND radiant_team_id IS NOT NULL AND dire_team_id IS NOT NULL{extra}
            ),
            h2h AS (
                SELECT radiant_team_id AS team_id, dire_team_id AS enemy_team_id, radiant_win AS won FROM valid_matches
                UNION ALL
                SELECT dire_team_id AS team_id, radiant_team_id AS enemy_team_id, NOT radiant_win AS won FROM valid_matches
            )
            SELECT team_id, enemy_team_id, COUNT(*) AS games, SUM(CASE WHEN won THEN 1 ELSE 0 END) AS wins
            FROM h2h GROUP BY team_id, enemy_team_id HAVING COUNT(*) >= 2 ORDER BY team_id, enemy_team_id
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
    logger.info("populate_h2h: %s rows for patch %s", total, patch_id)
    return total


# ---------------------------------------------------------------------------
# 6. ml.hero_baseline_agg
# ---------------------------------------------------------------------------

POPULATE_BASELINE = """
    INSERT INTO ml.hero_baseline_agg (patch_id, hero_id, total_picks, total_wins, total_bans, win_rate, pick_rate, ban_rate, avg_gpm, avg_xpm, avg_kills, avg_deaths, avg_assists, avg_gold_10, avg_xp_10)
    VALUES %s ON CONFLICT (patch_id, hero_id) DO UPDATE SET total_picks=EXCLUDED.total_picks, total_wins=EXCLUDED.total_wins, total_bans=EXCLUDED.total_bans, win_rate=EXCLUDED.win_rate, pick_rate=EXCLUDED.pick_rate, ban_rate=EXCLUDED.ban_rate, avg_gpm=EXCLUDED.avg_gpm, avg_xpm=EXCLUDED.avg_xpm, avg_kills=EXCLUDED.avg_kills, avg_deaths=EXCLUDED.avg_deaths, avg_assists=EXCLUDED.avg_assists, avg_gold_10=EXCLUDED.avg_gold_10, avg_xp_10=EXCLUDED.avg_xp_10;
"""

def populate_baseline(cfg: TrainerConfig, conn) -> int:
    patch_id = cfg.patch_id
    pg = cfg.prior_games
    pw = cfg.prior_win_rate
    _clean_patch_rows(conn, "ml.hero_baseline_agg", patch_id)
    extra = _match_extra_where(cfg, "m")
    decay = _decay_expr(cfg, "m")
    with conn.cursor() as cur:
        cur.execute(f"""
            WITH hero_picks AS (
                SELECT p.hero_id,
                       SUM({decay})::FLOAT AS total_picks,
                       SUM(CASE WHEN p.win = 1 THEN {decay} ELSE 0 END)::FLOAT AS total_wins,
                       SUM(p.gold_per_min * {decay}) / NULLIF(SUM({decay}), 0)::FLOAT AS avg_gpm,
                       SUM(p.xp_per_min * {decay}) / NULLIF(SUM({decay}), 0)::FLOAT AS avg_xpm,
                       SUM(p.kills * {decay}) / NULLIF(SUM({decay}), 0)::FLOAT AS avg_kills,
                       SUM(p.deaths * {decay}) / NULLIF(SUM({decay}), 0)::FLOAT AS avg_deaths,
                       SUM(p.assists * {decay}) / NULLIF(SUM({decay}), 0)::FLOAT AS avg_assists,
                       COALESCE(SUM(gold10.avg_gold_10 * {decay}) / NULLIF(SUM({decay}), 0), 0)::FLOAT AS avg_gold_10,
                       COALESCE(SUM(xp10.avg_xp_10 * {decay}) / NULLIF(SUM({decay}), 0), 0)::FLOAT AS avg_xp_10
                FROM matches m INNER JOIN players p ON p.match_id = m.match_id
                LEFT JOIN LATERAL (SELECT AVG((pta.gold_t ->> 10)::numeric) AS avg_gold_10 FROM player_time_series_arrays pta WHERE pta.match_id = m.match_id AND pta.player_slot = p.player_slot) gold10 ON TRUE
                LEFT JOIN LATERAL (SELECT AVG((pta.xp_t ->> 10)::numeric) AS avg_xp_10 FROM player_time_series_arrays pta WHERE pta.match_id = m.match_id AND pta.player_slot = p.player_slot) xp10 ON TRUE
                WHERE m.patch = %s AND m.radiant_win IS NOT NULL{extra}
                GROUP BY p.hero_id
                HAVING SUM({decay}) >= 1
            ),
            hero_bans AS (
                SELECT pb.hero_id, SUM({decay})::FLOAT AS total_bans
                FROM matches m INNER JOIN picks_bans pb ON pb.match_id = m.match_id AND pb.is_pick = FALSE
                WHERE m.patch = %s AND m.radiant_win IS NOT NULL{extra}
                GROUP BY pb.hero_id
                HAVING SUM({decay}) >= 1
            ),
            total_matches AS (
                SELECT SUM({decay})::FLOAT AS total
                FROM matches m WHERE patch = %s AND radiant_win IS NOT NULL{extra}
            )
            SELECT COALESCE(p.hero_id, b.hero_id) AS hero_id, COALESCE(p.total_picks, 0), COALESCE(p.total_wins, 0), COALESCE(b.total_bans, 0),
                   p.avg_gpm, p.avg_xpm, p.avg_kills, p.avg_deaths, p.avg_assists, p.avg_gold_10, p.avg_xp_10, tm.total
            FROM hero_picks p FULL OUTER JOIN hero_bans b ON b.hero_id = p.hero_id CROSS JOIN total_matches tm ORDER BY hero_id
        """, (patch_id, patch_id, patch_id))
        rows = []
        for r in cur.fetchall():
            hid, picks, wins, bans, ag, ax, ak, ad, aa, ag10, ax10, tot = r
            picks = float(picks or 0)
            wins = float(wins or 0)
            bans = float(bans or 0)
            tot = float(tot or 1)
            rows.append((patch_id, hid, picks, wins, bans, _shrunk_wr(wins, picks, pg, pw), picks/tot if tot>0 else 0, bans/tot if tot>0 else 0, ag, ax, ak, ad, aa, ag10, ax10))
    total = 0
    with conn.cursor() as cur:
        for batch in _batched(rows, cfg.agg_batch_size):
            psycopg2.extras.execute_values(cur, POPULATE_BASELINE, batch, template=None)
            total += len(batch)
    logger.info("populate_baseline: %s rows for patch %s", total, patch_id)
    return total


# ---------------------------------------------------------------------------
# 7. ml.hero_draft_slot_agg
# ---------------------------------------------------------------------------

POPULATE_HERO_DRAFT_SLOT = """
    INSERT INTO ml.hero_draft_slot_agg (patch_id, hero_id, team_pick_ordinal, games, wins, win_rate)
    VALUES %s ON CONFLICT (patch_id, hero_id, team_pick_ordinal) DO UPDATE SET games=EXCLUDED.games, wins=EXCLUDED.wins, win_rate=EXCLUDED.win_rate;
"""

def populate_hero_draft_slot(cfg: TrainerConfig, conn) -> int:
    patch_id = cfg.patch_id
    pg = cfg.prior_games
    pw = cfg.prior_win_rate
    _clean_patch_rows(conn, "ml.hero_draft_slot_agg", patch_id)
    extra = _match_extra_where(cfg, "m")
    decay = _decay_expr(cfg, "m")
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT ds.hero_id, ds.team_pick_ordinal, SUM(ds.dw) AS games, SUM(CASE WHEN ds.won THEN ds.dw ELSE 0 END) AS wins
            FROM (
                SELECT pb.match_id, pb.hero_id, pb.team, pb."order", pb.is_pick,
                       {decay} AS dw,
                       ROW_NUMBER() OVER (PARTITION BY pb.match_id, pb.team, pb.is_pick ORDER BY pb."order") AS team_pick_ordinal,
                       CASE WHEN (pb.team = 0 AND m.radiant_win) OR (pb.team = 1 AND NOT m.radiant_win) THEN TRUE ELSE FALSE END AS won
                FROM picks_bans pb INNER JOIN matches m ON m.match_id = pb.match_id
                WHERE m.patch = %s AND m.radiant_win IS NOT NULL{extra} AND pb.is_pick = TRUE
            ) ds WHERE ds.team_pick_ordinal <= 5
            GROUP BY ds.hero_id, ds.team_pick_ordinal HAVING SUM(ds.dw) >= 1 ORDER BY ds.hero_id, ds.team_pick_ordinal
        """, (patch_id,))
        rows = []
        for r in cur.fetchall():
            hid, tpo, games, wins = r
            games = float(games or 0)
            wins = float(wins or 0)
            rows.append((patch_id, hid, tpo, games, wins, _shrunk_wr(wins, games, pg, pw)))
    total = 0
    with conn.cursor() as cur:
        for batch in _batched(rows, cfg.agg_batch_size):
            psycopg2.extras.execute_values(cur, POPULATE_HERO_DRAFT_SLOT, batch, template=None)
            total += len(batch)
    logger.info("populate_hero_draft_slot: %s rows for patch %s", total, patch_id)
    return total


# ---------------------------------------------------------------------------
# PIT-safe snapshot populators — cross-patch lookback (sparse tables)
# ---------------------------------------------------------------------------

POPULATE_TEAM_HERO_SNAPSHOT = """
    INSERT INTO ml.team_hero_snapshot
        (patch_id, as_of_date, team_id, hero_id, games, wins, bans, win_rate,
         avg_gpm, avg_xpm, avg_kills, avg_deaths, avg_assists, firstblood_rate,
         avg_camps_stacked, avg_vision_placed, avg_gold_10, avg_xp_10, last_played)
    VALUES %s
    ON CONFLICT (patch_id, as_of_date, team_id, hero_id) DO UPDATE SET
        games=EXCLUDED.games, wins=EXCLUDED.wins, bans=EXCLUDED.bans, win_rate=EXCLUDED.win_rate,
        avg_gpm=EXCLUDED.avg_gpm, avg_xpm=EXCLUDED.avg_xpm, avg_kills=EXCLUDED.avg_kills,
        avg_deaths=EXCLUDED.avg_deaths, avg_assists=EXCLUDED.avg_assists, firstblood_rate=EXCLUDED.firstblood_rate,
        avg_camps_stacked=EXCLUDED.avg_camps_stacked, avg_vision_placed=EXCLUDED.avg_vision_placed,
        avg_gold_10=EXCLUDED.avg_gold_10, avg_xp_10=EXCLUDED.avg_xp_10, last_played=EXCLUDED.last_played;
"""


def _team_hero_prior_agg(cfg, conn, extra, min_patch, prior_weight):
    patch_id = cfg.patch_id
    with conn.cursor() as cur:
        pw_params = [prior_weight] * 22
        cur.execute(f"""
            SELECT CASE WHEN p.is_radiant THEN m.radiant_team_id ELSE m.dire_team_id END AS team_id, p.hero_id,
                   SUM(%s)::FLOAT AS games, SUM(CASE WHEN p.win = 1 THEN %s ELSE 0 END)::FLOAT AS wins,
                   SUM(p.gold_per_min * %s) / NULLIF(SUM(%s), 0)::FLOAT AS avg_gpm,
                   SUM(p.xp_per_min * %s) / NULLIF(SUM(%s), 0)::FLOAT AS avg_xpm,
                   SUM(p.kills * %s) / NULLIF(SUM(%s), 0)::FLOAT AS avg_kills,
                   SUM(p.deaths * %s) / NULLIF(SUM(%s), 0)::FLOAT AS avg_deaths,
                   SUM(p.assists * %s) / NULLIF(SUM(%s), 0)::FLOAT AS avg_assists,
                   SUM(p.firstblood_claimed * %s) / NULLIF(SUM(%s), 0)::FLOAT AS firstblood_rate,
                   SUM(p.camps_stacked * %s) / NULLIF(SUM(%s), 0)::FLOAT AS avg_camps_stacked,
                   SUM((p.obs_placed + p.sen_placed) * %s) / NULLIF(SUM(%s), 0)::FLOAT AS avg_vision_placed,
                   COALESCE(SUM(g10.avg_gold_10 * %s) / NULLIF(SUM(%s), 0), 0)::FLOAT AS avg_gold_10,
                   COALESCE(SUM(x10.avg_xp_10 * %s) / NULLIF(SUM(%s), 0), 0)::FLOAT AS avg_xp_10,
                   MAX(m.start_time) AS last_played
            FROM matches m JOIN players p ON p.match_id = m.match_id
            LEFT JOIN LATERAL (SELECT AVG((pta.gold_t ->> 10)::numeric) AS avg_gold_10 FROM player_time_series_arrays pta WHERE pta.match_id = m.match_id AND pta.player_slot = p.player_slot) g10 ON TRUE
            LEFT JOIN LATERAL (SELECT AVG((pta.xp_t ->> 10)::numeric) AS avg_xp_10 FROM player_time_series_arrays pta WHERE pta.match_id = m.match_id AND pta.player_slot = p.player_slot) x10 ON TRUE
            WHERE m.radiant_win IS NOT NULL{extra} AND m.patch >= %s AND m.patch < %s
              AND CASE WHEN p.is_radiant THEN m.radiant_team_id ELSE m.dire_team_id END IS NOT NULL
            GROUP BY team_id, p.hero_id
        """, pw_params + [min_patch, patch_id])
        prior = {}
        for r in cur.fetchall():
            tid, hid, g, w, agpm, axpm, ak_, ad_, aa_, fbr, acs, avp, ag10, ax10, lp = r
            prior[(tid, hid)] = {"games":g,"wins":w,"avg_gpm":agpm,"avg_xpm":axpm,"avg_kills":ak_,"avg_deaths":ad_,"avg_assists":aa_,"firstblood_rate":fbr,"avg_camps_stacked":acs,"avg_vision_placed":avp,"avg_gold_10":ag10,"avg_xp_10":ax10,"last_played":lp}
        logger.info("  prior_agg: %d team_hero combos from patches %d-%d", len(prior), min_patch, patch_id - 1)
        return prior


def _team_hero_bans_prior(cfg, conn, extra, min_patch):
    patch_id = cfg.patch_id
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT CASE WHEN pb.team = 0 THEN m.radiant_team_id ELSE m.dire_team_id END AS team_id, pb.hero_id, COUNT(*) AS bans
            FROM matches m JOIN picks_bans pb ON pb.match_id = m.match_id
            WHERE m.radiant_win IS NOT NULL{extra} AND m.patch >= %s AND m.patch < %s
              AND pb.is_pick = FALSE AND pb.team IN (0, 1) AND pb.hero_id IS NOT NULL
              AND CASE WHEN pb.team = 0 THEN m.radiant_team_id ELSE m.dire_team_id END IS NOT NULL
            GROUP BY team_id, pb.hero_id
        """, (min_patch, patch_id))
        bans_prior = {}
        for r in cur.fetchall():
            bans_prior[(r[0], r[1])] = r[2]
        logger.info("  bans_prior: %d team_hero combos from patches %d-%d", len(bans_prior), min_patch, patch_id - 1)
        return bans_prior


def _team_hero_bans_current(cfg, conn, extra):
    patch_id = cfg.patch_id
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT CASE WHEN pb.team = 0 THEN m.radiant_team_id ELSE m.dire_team_id END AS team_id, pb.hero_id, COUNT(*) AS bans
            FROM matches m JOIN picks_bans pb ON pb.match_id = m.match_id
            WHERE m.radiant_win IS NOT NULL{extra} AND m.patch = %s
              AND pb.is_pick = FALSE AND pb.team IN (0, 1) AND pb.hero_id IS NOT NULL
              AND CASE WHEN pb.team = 0 THEN m.radiant_team_id ELSE m.dire_team_id END IS NOT NULL
            GROUP BY team_id, pb.hero_id
        """, (patch_id,))
        bans_current = {}
        for r in cur.fetchall():
            bans_current[(r[0], r[1])] = r[2]
        logger.info("  bans_current: %d team_hero combos for patch %d", len(bans_current), patch_id)
        return bans_current


def populate_team_hero_snapshot(cfg, conn) -> int:
    patch_id = cfg.patch_id
    pg = cfg.prior_games
    pw = cfg.prior_win_rate
    lookback = cfg.lookback_patches
    prior_weight = cfg.prior_patch_weight
    min_patch = patch_id - lookback
    extra = _match_extra_where(cfg, "m")
    _clean_patch_rows(conn, "ml.team_hero_snapshot", patch_id)

    prior = _team_hero_prior_agg(cfg, conn, extra, min_patch, prior_weight)
    bans_prior = _team_hero_bans_prior(cfg, conn, extra, min_patch)
    bans_current = _team_hero_bans_current(cfg, conn, extra)
    all_bans = {**bans_prior, **bans_current}

    rows = []
    last_emitted = {}
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT generate_series(date_trunc('day', to_timestamp(MIN(start_time)))::date,
                date_trunc('day', to_timestamp(MAX(start_time)))::date, '1 day'::interval)::date AS as_of_date
            FROM matches WHERE patch = %s AND radiant_win IS NOT NULL{extra}
        """, (patch_id,))
        dates = [r[0] for r in cur.fetchall()]

        for as_of in sorted(dates):
            cur.execute(f"""
                WITH team_hero_picks AS (
                    SELECT CASE WHEN p.is_radiant THEN m.radiant_team_id ELSE m.dire_team_id END AS team_id, p.hero_id,
                           COUNT(*)::FLOAT AS games, SUM(CASE WHEN p.win = 1 THEN 1 ELSE 0 END)::FLOAT AS wins,
                           AVG(p.gold_per_min)::FLOAT AS avg_gpm, AVG(p.xp_per_min)::FLOAT AS avg_xpm,
                           AVG(p.kills)::FLOAT AS avg_kills, AVG(p.deaths)::FLOAT AS avg_deaths, AVG(p.assists)::FLOAT AS avg_assists,
                           AVG(p.firstblood_claimed)::FLOAT AS firstblood_rate, AVG(p.camps_stacked)::FLOAT AS avg_camps_stacked,
                           AVG(p.obs_placed + p.sen_placed)::FLOAT AS avg_vision_placed,
                           COALESCE(AVG(gold10.avg_gold_10)::FLOAT, 0) AS avg_gold_10,
                           COALESCE(AVG(xp10.avg_xp_10)::FLOAT, 0) AS avg_xp_10,
                           MAX(m.start_time) AS last_played
                    FROM matches m JOIN players p ON p.match_id = m.match_id
                    LEFT JOIN LATERAL (SELECT AVG((pta.gold_t ->> 10)::numeric) AS avg_gold_10 FROM player_time_series_arrays pta WHERE pta.match_id = m.match_id AND pta.player_slot = p.player_slot) gold10 ON TRUE
                    LEFT JOIN LATERAL (SELECT AVG((pta.xp_t ->> 10)::numeric) AS avg_xp_10 FROM player_time_series_arrays pta WHERE pta.match_id = m.match_id AND pta.player_slot = p.player_slot) xp10 ON TRUE
                    WHERE m.radiant_win IS NOT NULL{extra} AND m.patch = %s AND to_timestamp(m.start_time) < %s
                      AND CASE WHEN p.is_radiant THEN m.radiant_team_id ELSE m.dire_team_id END IS NOT NULL
                    GROUP BY team_id, p.hero_id
                ),
                team_hero_bans AS (
                    SELECT CASE WHEN pb.team = 0 THEN m.radiant_team_id ELSE m.dire_team_id END AS team_id, pb.hero_id, COUNT(*) AS bans
                    FROM matches m JOIN picks_bans pb ON pb.match_id = m.match_id
                    WHERE m.radiant_win IS NOT NULL{extra} AND m.patch = %s AND to_timestamp(m.start_time) < %s
                      AND pb.is_pick = FALSE AND pb.team IN (0, 1) AND pb.hero_id IS NOT NULL
                      AND CASE WHEN pb.team = 0 THEN m.radiant_team_id ELSE m.dire_team_id END IS NOT NULL
                    GROUP BY team_id, pb.hero_id
                )
                SELECT COALESCE(p.team_id, b.team_id), COALESCE(p.hero_id, b.hero_id),
                       COALESCE(p.games, 0.0), COALESCE(p.wins, 0.0), COALESCE(b.bans, 0),
                       p.avg_gpm, p.avg_xpm, p.avg_kills, p.avg_deaths, p.avg_assists,
                       p.firstblood_rate, p.avg_camps_stacked, p.avg_vision_placed,
                       p.avg_gold_10, p.avg_xp_10, p.last_played
                FROM team_hero_picks p FULL OUTER JOIN team_hero_bans b ON b.team_id = p.team_id AND b.hero_id = p.hero_id
            """, (patch_id, as_of, patch_id, as_of))

            current = {}
            for r in cur.fetchall():
                tid, hid, c_games, c_wins, c_bans, agpm, axpm, ak_, ad_, aa_, fbr, acs, avp, ag10, ax10, lp = r
                current[(tid, hid)] = {"games":c_games,"wins":c_wins,"bans":c_bans,"avg_gpm":agpm,"avg_xpm":axpm,"avg_kills":ak_,"avg_deaths":ad_,"avg_assists":aa_,"firstblood_rate":fbr,"avg_camps_stacked":acs,"avg_vision_placed":avp,"avg_gold_10":ag10,"avg_xp_10":ax10,"last_played":lp}

            for combo in set(prior.keys()) | set(current.keys()):
                p = prior.get(combo)
                c = current.get(combo)
                tid, hid = combo
                if p and c:
                    games = c["games"]+p["games"]; wins = c["wins"]+p["wins"]; bans = c["bans"]+all_bans.get(combo, 0)
                    tg = games or 1.0
                    ag=_wavg(c["avg_gpm"],c["games"],p["avg_gpm"],p["games"],tg); ax=_wavg(c["avg_xpm"],c["games"],p["avg_xpm"],p["games"],tg)
                    ak_=_wavg(c["avg_kills"],c["games"],p["avg_kills"],p["games"],tg); ad_=_wavg(c["avg_deaths"],c["games"],p["avg_deaths"],p["games"],tg)
                    aa_=_wavg(c["avg_assists"],c["games"],p["avg_assists"],p["games"],tg); fbr=_wavg(c["firstblood_rate"],c["games"],p["firstblood_rate"],p["games"],tg)
                    acs=_wavg(c["avg_camps_stacked"],c["games"],p["avg_camps_stacked"],p["games"],tg); avp=_wavg(c["avg_vision_placed"],c["games"],p["avg_vision_placed"],p["games"],tg)
                    ag10 = p["avg_gold_10"] if c["avg_gold_10"] == 0.0 else _wavg(c["avg_gold_10"],c["games"],p["avg_gold_10"],p["games"],tg)
                    ax10 = p["avg_xp_10"] if c["avg_xp_10"] == 0.0 else _wavg(c["avg_xp_10"],c["games"],p["avg_xp_10"],p["games"],tg)
                    lp=max(c["last_played"] or 0, p["last_played"] or 0)
                elif c:
                    games,wins,bans=c["games"],c["wins"],c["bans"]; ag,ax=c["avg_gpm"],c["avg_xpm"]
                    ak_,ad_,aa_=c["avg_kills"],c["avg_deaths"],c["avg_assists"]
                    fbr,acs,avp=c["firstblood_rate"],c["avg_camps_stacked"],c["avg_vision_placed"]
                    ag10,ax10=c["avg_gold_10"],c["avg_xp_10"]; lp=c["last_played"] or 0
                else:
                    games,wins=p["games"],p["wins"]; bans=all_bans.get(combo, 0)
                    ag,ax=p["avg_gpm"],p["avg_xpm"]; ak_,ad_,aa_=p["avg_kills"],p["avg_deaths"],p["avg_assists"]
                    fbr,acs,avp=p["firstblood_rate"],p["avg_camps_stacked"],p["avg_vision_placed"]
                    ag10,ax10=p["avg_gold_10"],p["avg_xp_10"]; lp=p["last_played"] or 0

                state=(games,wins,bans,ag,ax,ak_,ad_,aa_,fbr,acs,avp,ag10,ax10,lp)
                if last_emitted.get(combo) != state:
                    rows.append((patch_id,as_of,tid,hid,games,wins,bans,_shrunk_wr(wins,games,pg,pw),ag,ax,ak_,ad_,aa_,fbr,acs,avp,ag10,ax10,lp))
                    last_emitted[combo] = state

    total = 0
    with conn.cursor() as cur:
        for batch in _batched(rows, cfg.agg_batch_size):
            psycopg2.extras.execute_values(cur, POPULATE_TEAM_HERO_SNAPSHOT, batch, template=None)
            total += len(batch)
    logger.info("populate_team_hero_snapshot: %s rows for patch %s (lookback=%d, prior_w=%.2f)", total, patch_id, lookback, prior_weight)
    return total


# ---------------------------------------------------------------------------
# 2b. ml.player_hero_snapshot
# ---------------------------------------------------------------------------

POPULATE_PLAYER_HERO_SNAPSHOT = """
    INSERT INTO ml.player_hero_snapshot
        (patch_id, as_of_date, account_id, hero_id, games, wins, win_rate, avg_gpm, avg_xpm, avg_kills, avg_deaths, avg_assists, avg_kda, lane_role, firstblood_rate, avg_camps_stacked, avg_vision_placed, avg_gold_10, avg_xp_10, last_played)
    VALUES %s ON CONFLICT (patch_id, as_of_date, account_id, hero_id) DO UPDATE SET
        games=EXCLUDED.games, wins=EXCLUDED.wins, win_rate=EXCLUDED.win_rate,
        avg_gpm=EXCLUDED.avg_gpm, avg_xpm=EXCLUDED.avg_xpm, avg_kills=EXCLUDED.avg_kills,
        avg_deaths=EXCLUDED.avg_deaths, avg_assists=EXCLUDED.avg_assists, avg_kda=EXCLUDED.avg_kda,
        lane_role=EXCLUDED.lane_role, firstblood_rate=EXCLUDED.firstblood_rate,
        avg_camps_stacked=EXCLUDED.avg_camps_stacked, avg_vision_placed=EXCLUDED.avg_vision_placed,
        avg_gold_10=EXCLUDED.avg_gold_10, avg_xp_10=EXCLUDED.avg_xp_10, last_played=EXCLUDED.last_played;
"""

def populate_player_hero_snapshot(cfg, conn) -> int:
    patch_id = cfg.patch_id; pg = cfg.prior_games; pw = cfg.prior_win_rate
    lookback = cfg.lookback_patches; prior_weight = cfg.prior_patch_weight; min_patch = patch_id - lookback
    _clean_patch_rows(conn, "ml.player_hero_snapshot", patch_id)
    extra = _match_extra_where(cfg, "m")
    with conn.cursor() as cur:
        cur.execute(f"""
            WITH patch_days AS (SELECT generate_series(date_trunc('day', to_timestamp(MIN(start_time)))::date, date_trunc('day', to_timestamp(MAX(start_time)))::date, '7 days'::interval)::date AS as_of_date FROM matches WHERE patch = %s AND radiant_win IS NOT NULL{extra}),
            eligible_matches AS (SELECT d.as_of_date, m.match_id, CASE WHEN m.patch = %s THEN 1.0 ELSE %s END AS patch_weight FROM patch_days d JOIN matches m ON m.radiant_win IS NOT NULL{extra} AND m.patch BETWEEN %s AND %s AND ((m.patch = %s AND to_timestamp(m.start_time) < d.as_of_date) OR m.patch < %s))
            SELECT em.as_of_date, p.account_id, p.hero_id,
                   SUM(em.patch_weight)::FLOAT AS games, SUM(CASE WHEN p.win = 1 THEN em.patch_weight ELSE 0 END)::FLOAT AS wins,
                   AVG(p.gold_per_min)::FLOAT, AVG(p.xp_per_min)::FLOAT, AVG(p.kills)::FLOAT, AVG(p.deaths)::FLOAT, AVG(p.assists)::FLOAT,
                   AVG(p.kda)::FLOAT, MODE() WITHIN GROUP (ORDER BY p.lane_role), AVG(p.firstblood_claimed)::FLOAT,
                   AVG(p.camps_stacked)::FLOAT, AVG(p.obs_placed + p.sen_placed)::FLOAT,
                   COALESCE(AVG(gold10.avg_gold_10)::FLOAT, 0), COALESCE(AVG(xp10.avg_xp_10)::FLOAT, 0), MAX(m.start_time)
            FROM eligible_matches em JOIN matches m ON m.match_id = em.match_id JOIN players p ON p.match_id = m.match_id
            LEFT JOIN LATERAL (SELECT AVG((pta.gold_t ->> 10)::numeric) AS avg_gold_10 FROM player_time_series_arrays pta WHERE pta.match_id = m.match_id AND pta.player_slot = p.player_slot) gold10 ON TRUE
            LEFT JOIN LATERAL (SELECT AVG((pta.xp_t ->> 10)::numeric) AS avg_xp_10 FROM player_time_series_arrays pta WHERE pta.match_id = m.match_id AND pta.player_slot = p.player_slot) xp10 ON TRUE
            WHERE p.account_id IS NOT NULL GROUP BY em.as_of_date, p.account_id, p.hero_id ORDER BY em.as_of_date, p.account_id, p.hero_id
        """, (patch_id, patch_id, prior_weight, min_patch, patch_id, patch_id, patch_id))
        rows = []; last_emitted = {}
        for r in cur.fetchall():
            as_of, aid, hid, games, wins, ag, ax, ak, ad, aa, akda, lr, fbr, acs, avp, ag10, ax10, lp = r
            state = (games, wins, ag, ax, ak, ad, aa, akda, lr, fbr, acs, avp, ag10, ax10, lp)
            if last_emitted.get((aid, hid)) != state:
                rows.append((patch_id, as_of, aid, hid, games, wins, _shrunk_wr(wins, games, pg, pw), ag, ax, ak, ad, aa, akda, lr, fbr, acs, avp, ag10, ax10, lp))
                last_emitted[(aid, hid)] = state
    total = 0
    with conn.cursor() as cur:
        for batch in _batched(rows, cfg.agg_batch_size):
            psycopg2.extras.execute_values(cur, POPULATE_PLAYER_HERO_SNAPSHOT, batch, template=None)
            total += len(batch)
    logger.info("populate_player_hero_snapshot: %s rows for patch %s (lookback=%d, prior_w=%.2f)", total, patch_id, lookback, prior_weight)
    return total


# ---------------------------------------------------------------------------
# 3b. ml.hero_synergy_snapshot
# ---------------------------------------------------------------------------

POPULATE_SYNERGY_SNAPSHOT = """
    INSERT INTO ml.hero_synergy_snapshot (patch_id, as_of_date, hero_a, hero_b, games, wins, win_rate)
    VALUES %s ON CONFLICT (patch_id, as_of_date, hero_a, hero_b) DO UPDATE SET games=EXCLUDED.games, wins=EXCLUDED.wins, win_rate=EXCLUDED.win_rate;
"""

def populate_synergy_snapshot(cfg, conn) -> int:
    patch_id = cfg.patch_id; pg = cfg.prior_games; pw = cfg.prior_win_rate
    lookback = cfg.lookback_patches; prior_weight = cfg.prior_patch_weight; min_patch = patch_id - lookback
    _clean_patch_rows(conn, "ml.hero_synergy_snapshot", patch_id)
    extra = _match_extra_where(cfg, "m")
    with conn.cursor() as cur:
        cur.execute(f"""
            WITH patch_days AS (SELECT generate_series(date_trunc('day', to_timestamp(MIN(start_time)))::date, date_trunc('day', to_timestamp(MAX(start_time)))::date, '7 days'::interval)::date AS as_of_date FROM matches WHERE patch = %s AND radiant_win IS NOT NULL{extra}),
            eligible_matches AS (SELECT d.as_of_date, m.match_id, CASE WHEN m.patch = %s THEN 1.0 ELSE %s END AS patch_weight FROM patch_days d JOIN matches m ON m.radiant_win IS NOT NULL{extra} AND m.patch BETWEEN %s AND %s AND ((m.patch = %s AND to_timestamp(m.start_time) < d.as_of_date) OR m.patch < %s))
            SELECT em.as_of_date, p1.hero_id AS hero_a, p2.hero_id AS hero_b,
                   SUM(em.patch_weight)::FLOAT AS games, SUM(CASE WHEN CASE WHEN p1.is_radiant THEN m.radiant_win ELSE NOT m.radiant_win END THEN em.patch_weight ELSE 0 END)::FLOAT AS wins
            FROM eligible_matches em JOIN matches m ON m.match_id = em.match_id
            JOIN players p1 ON p1.match_id = m.match_id JOIN players p2 ON p2.match_id = m.match_id AND p2.is_radiant = p1.is_radiant AND p2.hero_id > p1.hero_id
            GROUP BY em.as_of_date, p1.hero_id, p2.hero_id HAVING SUM(em.patch_weight) >= 3 ORDER BY em.as_of_date, p1.hero_id, p2.hero_id
        """, (patch_id, patch_id, prior_weight, min_patch, patch_id, patch_id, patch_id))
        rows = []; last_emitted = {}
        for r in cur.fetchall():
            as_of, ha, hb, games, wins = r
            state = (games, wins)
            if last_emitted.get((ha, hb)) != state:
                rows.append((patch_id, as_of, ha, hb, games, wins, _shrunk_wr(wins, games, pg, pw)))
                last_emitted[(ha, hb)] = state
    total = 0
    with conn.cursor() as cur:
        for batch in _batched(rows, cfg.agg_batch_size):
            psycopg2.extras.execute_values(cur, POPULATE_SYNERGY_SNAPSHOT, batch, template=None)
            total += len(batch)
    logger.info("populate_synergy_snapshot: %s rows for patch %s (lookback=%d, prior_w=%.2f)", total, patch_id, lookback, prior_weight)
    return total


# ---------------------------------------------------------------------------
# 4b. ml.hero_counter_snapshot
# ---------------------------------------------------------------------------

POPULATE_COUNTER_SNAPSHOT = """
    INSERT INTO ml.hero_counter_snapshot (patch_id, as_of_date, hero_id, enemy_hero_id, games, wins, win_rate, avg_kd_diff)
    VALUES %s ON CONFLICT (patch_id, as_of_date, hero_id, enemy_hero_id) DO UPDATE SET games=EXCLUDED.games, wins=EXCLUDED.wins, win_rate=EXCLUDED.win_rate, avg_kd_diff=EXCLUDED.avg_kd_diff;
"""

def populate_counter_snapshot(cfg, conn) -> int:
    patch_id = cfg.patch_id; pg = cfg.prior_games; pw = cfg.prior_win_rate
    lookback = cfg.lookback_patches; prior_weight = cfg.prior_patch_weight; min_patch = patch_id - lookback
    _clean_patch_rows(conn, "ml.hero_counter_snapshot", patch_id)
    extra = _match_extra_where(cfg, "m")
    with conn.cursor() as cur:
        cur.execute(f"SELECT generate_series(date_trunc('day', to_timestamp(MIN(start_time)))::date, date_trunc('day', to_timestamp(MAX(start_time)))::date, '7 days'::interval)::date FROM matches WHERE patch = %s AND radiant_win IS NOT NULL{extra}", (patch_id,))
        dates = [r[0] for r in cur.fetchall()]
    total = 0; last_emitted = {}
    for dt in sorted(dates):
        with conn.cursor() as cur:
            cur.execute(f"""
                WITH match_weights AS (SELECT match_id, radiant_win, CASE WHEN patch = %s THEN 1.0 ELSE %s END AS patch_weight FROM matches WHERE patch BETWEEN %s AND %s {extra} AND radiant_win IS NOT NULL AND to_timestamp(start_time) < %s)
                SELECT p1.hero_id, p2.hero_id, SUM(mw.patch_weight)::float,
                       SUM(CASE WHEN (p1.is_radiant = mw.radiant_win) THEN mw.patch_weight ELSE 0 END)::float,
                       SUM((p1.kills - p1.deaths) * mw.patch_weight) / NULLIF(SUM(mw.patch_weight), 0)::float
                FROM match_weights mw JOIN players p1 ON p1.match_id = mw.match_id JOIN players p2 ON p2.match_id = mw.match_id AND p2.is_radiant != p1.is_radiant
                GROUP BY p1.hero_id, p2.hero_id HAVING SUM(mw.patch_weight) >= 3
            """, (patch_id, prior_weight, min_patch, patch_id, dt))
            rows = []
            for r in cur.fetchall():
                hid, ehid, g, w, akd = r
                state = (g, w, akd or 0.0)
                if last_emitted.get((hid, ehid)) != state:
                    rows.append((patch_id, dt, hid, ehid, g, w, _shrunk_wr(w, g, pg, pw), akd or 0.0))
                    last_emitted[(hid, ehid)] = state
            if rows:
                psycopg2.extras.execute_values(cur, POPULATE_COUNTER_SNAPSHOT, rows, template=None)
        total += len(rows)
    logger.info("populate_counter_snapshot: %s rows for patch %s", total, patch_id)
    return total


# ---------------------------------------------------------------------------
# 5b. ml.team_h2h_snapshot
# ---------------------------------------------------------------------------

POPULATE_H2H_SNAPSHOT = """
    INSERT INTO ml.team_h2h_snapshot (as_of_date, snapshot_tier, patch_id, team_id, enemy_team_id, games, wins, win_rate)
    VALUES %s ON CONFLICT (patch_id, as_of_date, team_id, enemy_team_id) DO UPDATE SET games=EXCLUDED.games, wins=EXCLUDED.wins, win_rate=EXCLUDED.win_rate;
"""

def populate_h2h_snapshot(cfg, conn) -> int:
    patch_id = cfg.patch_id; pg = cfg.prior_games; pw = cfg.prior_win_rate
    _clean_patch_rows(conn, "ml.team_h2h_snapshot", patch_id)
    extra = _match_extra_where(cfg, "m")
    with conn.cursor() as cur:
        cur.execute(f"""
            WITH dates AS (SELECT DISTINCT d::DATE AS as_of_date FROM generate_series((SELECT MIN(to_timestamp(m.start_time))::DATE FROM matches m WHERE m.patch = %s{extra}), (SELECT MAX(to_timestamp(m.start_time))::DATE FROM matches m WHERE m.patch = %s{extra}), '1 day'::INTERVAL) d)
            SELECT d.as_of_date, sub.team_id, sub.enemy_team_id, sub.games, sub.wins
            FROM dates d CROSS JOIN LATERAL (
                WITH valid_matches AS (SELECT match_id, radiant_team_id, dire_team_id, radiant_win FROM matches WHERE patch = %s AND to_timestamp(start_time) < d.as_of_date AND radiant_win IS NOT NULL AND radiant_team_id IS NOT NULL AND dire_team_id IS NOT NULL{extra}),
                h2h AS (SELECT radiant_team_id AS team_id, dire_team_id AS enemy_team_id, radiant_win AS won FROM valid_matches UNION ALL SELECT dire_team_id AS team_id, radiant_team_id AS enemy_team_id, NOT radiant_win AS won FROM valid_matches)
                SELECT team_id, enemy_team_id, COUNT(*) AS games, SUM(CASE WHEN won THEN 1 ELSE 0 END) AS wins FROM h2h GROUP BY team_id, enemy_team_id HAVING COUNT(*) >= 2
            ) sub ORDER BY d.as_of_date, sub.team_id, sub.enemy_team_id
        """, (patch_id, patch_id, patch_id))
        rows = []; last_emitted = {}
        for r in cur.fetchall():
            as_of, tid, etid, games, wins = r
            state = (games, wins)
            if last_emitted.get((tid, etid)) != state:
                rows.append((as_of, "daily", patch_id, tid, etid, games, wins, _shrunk_wr(wins, games, pg, pw)))
                last_emitted[(tid, etid)] = state
    total = 0
    with conn.cursor() as cur:
        for batch in _batched(rows, cfg.agg_batch_size):
            psycopg2.extras.execute_values(cur, POPULATE_H2H_SNAPSHOT, batch, template=None)
            total += len(batch)
    logger.info("populate_h2h_snapshot: %s rows for patch %s", total, patch_id)
    return total


# ---------------------------------------------------------------------------
# 6b. ml.hero_baseline_snapshot
# ---------------------------------------------------------------------------

POPULATE_BASELINE_SNAPSHOT = """
    INSERT INTO ml.hero_baseline_snapshot (as_of_date, snapshot_tier, patch_id, hero_id, total_picks, total_wins, total_bans, win_rate, pick_rate, ban_rate, avg_gpm, avg_xpm, avg_kills, avg_deaths, avg_assists, avg_gold_10, avg_xp_10)
    VALUES %s ON CONFLICT (patch_id, as_of_date, hero_id) DO UPDATE SET total_picks=EXCLUDED.total_picks, total_wins=EXCLUDED.total_wins, total_bans=EXCLUDED.total_bans, win_rate=EXCLUDED.win_rate, pick_rate=EXCLUDED.pick_rate, ban_rate=EXCLUDED.ban_rate, avg_gpm=EXCLUDED.avg_gpm, avg_xpm=EXCLUDED.avg_xpm, avg_kills=EXCLUDED.avg_kills, avg_deaths=EXCLUDED.avg_deaths, avg_assists=EXCLUDED.avg_assists, avg_gold_10=EXCLUDED.avg_gold_10, avg_xp_10=EXCLUDED.avg_xp_10;
"""

def populate_baseline_snapshot(cfg, conn) -> int:
    patch_id = cfg.patch_id; pg = cfg.prior_games; pw = cfg.prior_win_rate
    _clean_patch_rows(conn, "ml.hero_baseline_snapshot", patch_id)
    extra = _match_extra_where(cfg, "m")
    with conn.cursor() as cur:
        cur.execute(f"""
            WITH dates AS (SELECT DISTINCT d::DATE AS as_of_date FROM generate_series((SELECT MIN(to_timestamp(m.start_time))::DATE FROM matches m WHERE m.patch = %s{extra}), (SELECT MAX(to_timestamp(m.start_time))::DATE FROM matches m WHERE m.patch = %s{extra}), '1 day'::INTERVAL) d),
            picks_pit AS (SELECT d.as_of_date, sub.hero_id, sub.total_picks, sub.total_wins, sub.avg_gpm, sub.avg_xpm, sub.avg_kills, sub.avg_deaths, sub.avg_assists, sub.avg_gold_10, sub.avg_xp_10
                FROM dates d CROSS JOIN LATERAL (SELECT p.hero_id, COUNT(*) AS total_picks, SUM(CASE WHEN p.win = 1 THEN 1 ELSE 0 END) AS total_wins, AVG(p.gold_per_min)::FLOAT AS avg_gpm, AVG(p.xp_per_min)::FLOAT AS avg_xpm, AVG(p.kills)::FLOAT AS avg_kills, AVG(p.deaths)::FLOAT AS avg_deaths, AVG(p.assists)::FLOAT AS avg_assists, COALESCE(AVG(gold10.avg_gold_10)::FLOAT, 0) AS avg_gold_10, COALESCE(AVG(xp10.avg_xp_10)::FLOAT, 0) AS avg_xp_10 FROM matches m INNER JOIN players p ON p.match_id = m.match_id LEFT JOIN LATERAL (SELECT AVG((pta.gold_t ->> 10)::numeric) AS avg_gold_10 FROM player_time_series_arrays pta WHERE pta.match_id = m.match_id AND pta.player_slot = p.player_slot) gold10 ON TRUE LEFT JOIN LATERAL (SELECT AVG((pta.xp_t ->> 10)::numeric) AS avg_xp_10 FROM player_time_series_arrays pta WHERE pta.match_id = m.match_id AND pta.player_slot = p.player_slot) xp10 ON TRUE WHERE m.patch = %s AND to_timestamp(m.start_time) < d.as_of_date AND m.radiant_win IS NOT NULL{extra} GROUP BY p.hero_id) sub),
            bans_pit AS (SELECT d.as_of_date, sub.hero_id, sub.total_bans FROM dates d CROSS JOIN LATERAL (SELECT pb.hero_id, COUNT(*) AS total_bans FROM matches m INNER JOIN picks_bans pb ON pb.match_id = m.match_id AND pb.is_pick = FALSE WHERE m.patch = %s AND to_timestamp(m.start_time) < d.as_of_date AND m.radiant_win IS NOT NULL{extra} GROUP BY pb.hero_id) sub),
            total_matches_pit AS (SELECT d.as_of_date, sub.total FROM dates d CROSS JOIN LATERAL (SELECT COUNT(DISTINCT match_id) AS total FROM matches WHERE patch = %s AND to_timestamp(start_time) < d.as_of_date AND radiant_win IS NOT NULL{extra}) sub)
            SELECT COALESCE(p.as_of_date, b.as_of_date), COALESCE(p.hero_id, b.hero_id), COALESCE(p.total_picks, 0), COALESCE(p.total_wins, 0), COALESCE(b.total_bans, 0), p.avg_gpm, p.avg_xpm, p.avg_kills, p.avg_deaths, p.avg_assists, p.avg_gold_10, p.avg_xp_10, tm.total
            FROM picks_pit p FULL OUTER JOIN bans_pit b ON b.as_of_date = p.as_of_date AND b.hero_id = p.hero_id
            INNER JOIN total_matches_pit tm ON tm.as_of_date = COALESCE(p.as_of_date, b.as_of_date) ORDER BY COALESCE(p.as_of_date, b.as_of_date), COALESCE(p.hero_id, b.hero_id)
        """, (patch_id, patch_id, patch_id, patch_id, patch_id))
        rows = []; last_emitted = {}
        for r in cur.fetchall():
            as_of, hid, picks, wins, bans, ag, ax, ak, ad, aa, ag10, ax10, tot = r
            pr = picks/tot if tot > 0 else 0; br = bans/tot if tot > 0 else 0
            state = (picks, wins, bans, ag, ax, ak, ad, aa, ag10, ax10, tot)
            if last_emitted.get(hid) != state:
                rows.append((as_of, "daily", patch_id, hid, picks, wins, bans, _shrunk_wr(wins, picks, pg, pw), pr, br, ag, ax, ak, ad, aa, ag10, ax10))
                last_emitted[hid] = state
    total = 0
    with conn.cursor() as cur:
        for batch in _batched(rows, cfg.agg_batch_size):
            psycopg2.extras.execute_values(cur, POPULATE_BASELINE_SNAPSHOT, batch, template=None)
            total += len(batch)
    logger.info("populate_baseline_snapshot: %s rows for patch %s", total, patch_id)
    return total


# ---------------------------------------------------------------------------
# 7b. ml.hero_draft_slot_snapshot
# ---------------------------------------------------------------------------

POPULATE_HERO_DRAFT_SLOT_SNAPSHOT = """
    INSERT INTO ml.hero_draft_slot_snapshot (as_of_date, snapshot_tier, patch_id, hero_id, team_pick_ordinal, games, wins, win_rate)
    VALUES %s ON CONFLICT (patch_id, as_of_date, hero_id, team_pick_ordinal) DO UPDATE SET games=EXCLUDED.games, wins=EXCLUDED.wins, win_rate=EXCLUDED.win_rate;
"""

def populate_hero_draft_slot_snapshot(cfg, conn) -> int:
    patch_id = cfg.patch_id; pg = cfg.prior_games; pw = cfg.prior_win_rate
    _clean_patch_rows(conn, "ml.hero_draft_slot_snapshot", patch_id)
    extra = _match_extra_where(cfg, "m")
    with conn.cursor() as cur:
        cur.execute(f"""
            WITH dates AS (SELECT DISTINCT d::DATE AS as_of_date FROM generate_series((SELECT MIN(to_timestamp(m.start_time))::DATE FROM matches m WHERE m.patch = %s{extra}), (SELECT MAX(to_timestamp(m.start_time))::DATE FROM matches m WHERE m.patch = %s{extra}), '1 day'::INTERVAL) d)
            SELECT d.as_of_date, sub.hero_id, sub.team_pick_ordinal, sub.games, sub.wins
            FROM dates d CROSS JOIN LATERAL (
                SELECT ds.hero_id, ds.team_pick_ordinal, COUNT(*) AS games, SUM(CASE WHEN ds.won THEN 1 ELSE 0 END) AS wins
                FROM (SELECT pb.match_id, pb.hero_id, pb.team, pb."order", pb.is_pick,
                      ROW_NUMBER() OVER (PARTITION BY pb.match_id, pb.team, pb.is_pick ORDER BY pb."order") AS team_pick_ordinal,
                      CASE WHEN (pb.team = 0 AND m.radiant_win) OR (pb.team = 1 AND NOT m.radiant_win) THEN TRUE ELSE FALSE END AS won
                FROM picks_bans pb INNER JOIN matches m ON m.match_id = pb.match_id
                WHERE m.patch = %s AND to_timestamp(m.start_time) < d.as_of_date AND m.radiant_win IS NOT NULL{extra} AND pb.is_pick = TRUE) ds
                WHERE ds.team_pick_ordinal <= 5 GROUP BY ds.hero_id, ds.team_pick_ordinal HAVING COUNT(*) >= 3
            ) sub ORDER BY d.as_of_date, sub.hero_id, sub.team_pick_ordinal
        """, (patch_id, patch_id, patch_id))
        rows = []; last_emitted = {}
        for r in cur.fetchall():
            as_of, hid, tpo, games, wins = r
            state = (games, wins)
            if last_emitted.get((hid, tpo)) != state:
                rows.append((as_of, "daily", patch_id, hid, tpo, games, wins, _shrunk_wr(wins, games, pg, pw)))
                last_emitted[(hid, tpo)] = state
    total = 0
    with conn.cursor() as cur:
        for batch in _batched(rows, cfg.agg_batch_size):
            psycopg2.extras.execute_values(cur, POPULATE_HERO_DRAFT_SLOT_SNAPSHOT, batch, template=None)
            total += len(batch)
    logger.info("populate_hero_draft_slot_snapshot: %s rows for patch %s", total, patch_id)
    return total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

ALL_POPULATORS = [
    ("ml.team_elo", populate_team_elo),
    ("ml.team_hero_agg", populate_team_hero), ("ml.player_hero_agg", populate_player_hero),
    ("ml.hero_synergy_agg", populate_synergy), ("ml.hero_counter_agg", populate_counter),
    ("ml.team_h2h_agg", populate_h2h), ("ml.hero_baseline_agg", populate_baseline),
    ("ml.hero_draft_slot_agg", populate_hero_draft_slot),
    ("ml.team_hero_snapshot", populate_team_hero_snapshot), ("ml.player_hero_snapshot", populate_player_hero_snapshot),
    ("ml.hero_synergy_snapshot", populate_synergy_snapshot), ("ml.hero_counter_snapshot", populate_counter_snapshot),
    ("ml.team_h2h_snapshot", populate_h2h_snapshot), ("ml.hero_baseline_snapshot", populate_baseline_snapshot),
    ("ml.hero_draft_slot_snapshot", populate_hero_draft_slot_snapshot),
]


def _analyze_ml_tables(conn) -> None:
    tables = ["ml.team_hero_agg","ml.player_hero_agg","ml.hero_synergy_agg","ml.hero_counter_agg","ml.team_h2h_agg","ml.hero_baseline_agg","ml.hero_draft_slot_agg","ml.team_hero_snapshot","ml.player_hero_snapshot","ml.hero_synergy_snapshot","ml.hero_counter_snapshot","ml.team_h2h_snapshot","ml.hero_baseline_snapshot","ml.hero_draft_slot_snapshot"]
    old_autocommit = conn.autocommit; conn.autocommit = True
    try:
        with conn.cursor() as cur:
            for tbl in tables: cur.execute(f"VACUUM ANALYZE {tbl}")
    finally:
        conn.autocommit = old_autocommit
    logger.info("Analyzed %d ML tables", len(tables))


def populate_all(cfg: TrainerConfig, conn) -> dict[str, int]:
    """Run all populators in a single transaction. Commit once on success, rollback on failure."""
    counts = {}
    try:
        for name, fn in ALL_POPULATORS:
            logger.info("Populating %s ...", name)
            counts[name] = fn(cfg, conn)
        conn.commit()
        logger.info("All %d populators committed successfully.", len(counts))
    except Exception:
        conn.rollback()
        logger.exception("Populator failed — rolled back all changes.")
        raise
    _analyze_ml_tables(conn)
    return counts
