"""SVD Semantic Embeddings for heroes, teams, and players.

Computes low-dimensional dense representations from sparse co-occurrence
matrices using TruncatedSVD. These embeddings capture drafting style
similarity — heroes played with similar allies cluster together, teams
with similar hero pools get similar playstyle vectors.

Tables created:
  ml.hero_embeddings   (patch_id, hero_id, emb_0..emb_31)   — 32D
  ml.team_embeddings   (patch_id, team_id, emb_0..emb_15)   — 16D
  ml.player_embeddings (patch_id, account_id, emb_0..emb_15) — 16D
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import psycopg2.extras
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)

HERO_EMB_DIM = 32
TEAM_EMB_DIM = 16
PLAYER_EMB_DIM = 16
SPATIAL_EMB_DIM = 16
MAX_HERO_ID = 160


def populate_embeddings(cfg, engine) -> None:
    """Compute SVD embeddings and write to DB tables."""
    patch_id = cfg.patch_id
    logger.info("Computing SVD Semantic Embeddings for patch %s...", patch_id)

    conn = engine.raw_connection()
    try:
        _ensure_tables(conn, patch_id)
        _compute_hero_embeddings(conn, engine, patch_id)
        _compute_team_embeddings(conn, engine, patch_id)
        _compute_player_embeddings(conn, engine, patch_id)
        _compute_spatial_embeddings(conn, engine, patch_id)
    finally:
        conn.close()

    logger.info("SVD Embeddings generated and saved to DB.")


def _ensure_tables(conn, patch_id: int) -> None:
    """Create embedding tables if they don't exist."""
    with conn.cursor() as cur:
        emb_cols_h = ", ".join(f"emb_{i} FLOAT" for i in range(HERO_EMB_DIM))
        emb_cols_t = ", ".join(f"emb_{i} FLOAT" for i in range(TEAM_EMB_DIM))
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS ml.hero_embeddings (
                patch_id INT, hero_id INT, {emb_cols_h},
                PRIMARY KEY (patch_id, hero_id)
            )""")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS ml.team_embeddings (
                patch_id INT, team_id BIGINT, {emb_cols_t},
                PRIMARY KEY (patch_id, team_id)
            )""")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS ml.player_embeddings (
                patch_id INT, account_id BIGINT, {emb_cols_t},
                PRIMARY KEY (patch_id, account_id)
            )""")
        emb_cols_s = ", ".join(f"spatial_emb_{i} FLOAT" for i in range(SPATIAL_EMB_DIM))
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS ml.hero_spatial_embeddings (
                patch_id INT, hero_id INT, {emb_cols_s},
                PRIMARY KEY (patch_id, hero_id)
            )""")
    conn.commit()


def _compute_hero_embeddings(conn, engine, patch_id: int) -> None:
    """SVD on hero co-occurrence matrix (synergy counts)."""
    logger.info("  Hero Embeddings (32-D)...")
    df = pd.read_sql(
        "SELECT hero_a, hero_b, games FROM ml.hero_synergy_agg WHERE patch_id = %s",
        engine, params=(patch_id,),
    )
    if df.empty:
        logger.warning("  No synergy data, skipping hero embeddings")
        return

    mat = np.zeros((MAX_HERO_ID + 1, MAX_HERO_ID + 1))
    for _, row in df.iterrows():
        a, b = int(row["hero_a"]), int(row["hero_b"])
        if a <= MAX_HERO_ID and b <= MAX_HERO_ID:
            mat[a, b] = row["games"]
            mat[b, a] = row["games"]
    mat = np.log1p(mat)

    svd = TruncatedSVD(n_components=HERO_EMB_DIM, random_state=42)
    emb = svd.fit_transform(mat)
    logger.info("  Hero SVD explained variance: %.1f%%", svd.explained_variance_ratio_.sum() * 100)

    rows = [(patch_id, hid, *emb[hid].tolist()) for hid in range(1, MAX_HERO_ID + 1)]
    with conn.cursor() as cur:
        cur.execute("DELETE FROM ml.hero_embeddings WHERE patch_id = %s", (patch_id,))
        psycopg2.extras.execute_values(cur, "INSERT INTO ml.hero_embeddings VALUES %s", rows, page_size=5000)
    conn.commit()
    logger.info("  Saved %d hero embeddings", len(rows))


def _compute_team_embeddings(conn, engine, patch_id: int) -> None:
    """SVD on team×hero frequency matrix."""
    logger.info("  Team Embeddings (16-D)...")
    df = pd.read_sql(
        "SELECT team_id, hero_id, games FROM ml.team_hero_agg WHERE patch_id = %s",
        engine, params=(patch_id,),
    )
    if df.empty:
        logger.warning("  No team-hero data, skipping team embeddings")
        return

    pivot = df.pivot_table(index="team_id", columns="hero_id", values="games", fill_value=0)
    n_teams, n_heroes = len(pivot), MAX_HERO_ID + 1
    T = np.zeros((n_teams, n_heroes))
    for h in pivot.columns:
        if h <= MAX_HERO_ID:
            T[:, h] = pivot[h].values
    T = np.log1p(T)

    n_comp = min(TEAM_EMB_DIM, n_teams - 1, n_heroes - 1)
    if n_comp <= 0:
        logger.warning("  Not enough teams for SVD, skipping")
        return

    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    emb = svd.fit_transform(T)
    if n_comp < TEAM_EMB_DIM:
        emb = np.pad(emb, ((0, 0), (0, TEAM_EMB_DIM - n_comp)))
    logger.info("  Team SVD explained variance: %.1f%%", svd.explained_variance_ratio_.sum() * 100)

    rows = [(patch_id, int(tid), *emb[i].tolist()) for i, tid in enumerate(pivot.index)]
    with conn.cursor() as cur:
        cur.execute("DELETE FROM ml.team_embeddings WHERE patch_id = %s", (patch_id,))
        psycopg2.extras.execute_values(cur, "INSERT INTO ml.team_embeddings VALUES %s", rows, page_size=5000)
    conn.commit()
    logger.info("  Saved %d team embeddings", len(rows))


def _compute_player_embeddings(conn, engine, patch_id: int) -> None:
    """SVD on player×hero frequency matrix (sparse)."""
    logger.info("  Player Embeddings (16-D)...")
    df = pd.read_sql(
        "SELECT account_id, hero_id, games FROM ml.player_hero_agg WHERE patch_id = %s",
        engine, params=(patch_id,),
    )
    if df.empty:
        logger.warning("  No player-hero data, skipping player embeddings")
        return

    players = df["account_id"].unique()
    p2i = {pid: i for i, pid in enumerate(players)}
    row_idx = df["account_id"].map(p2i).values
    col_idx = df["hero_id"].values
    data = np.log1p(df["games"].values.astype(float))

    P = coo_matrix((data, (row_idx, col_idx)), shape=(len(players), MAX_HERO_ID + 1))
    n_comp = min(PLAYER_EMB_DIM, P.shape[0] - 1, P.shape[1] - 1)
    if n_comp <= 0:
        logger.warning("  Not enough players for SVD, skipping")
        return

    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    emb = svd.fit_transform(P)
    if n_comp < PLAYER_EMB_DIM:
        emb = np.pad(emb, ((0, 0), (0, PLAYER_EMB_DIM - n_comp)))
    logger.info("  Player SVD explained variance: %.1f%%", svd.explained_variance_ratio_.sum() * 100)

    rows = [(patch_id, int(pid), *emb[i].tolist()) for pid, i in p2i.items()]
    with conn.cursor() as cur:
        cur.execute("DELETE FROM ml.player_embeddings WHERE patch_id = %s", (patch_id,))
        for start in range(0, len(rows), 5000):
            psycopg2.extras.execute_values(cur, "INSERT INTO ml.player_embeddings VALUES %s", rows[start:start + 5000])
    conn.commit()
    logger.info("  Saved %d player embeddings", len(rows))


def _compute_spatial_embeddings(conn, engine, patch_id: int) -> None:
    """SVD on hero×map-coordinate spatial frequency matrix from lane_pos."""
    logger.info("  Hero Spatial Embeddings (16-D)...")
    import json as _json
    from collections import defaultdict

    hero_spatial = defaultdict(lambda: defaultdict(float))
    sql = """
        SELECT p.hero_id, p.lane_pos
        FROM players p
        JOIN matches m ON m.match_id = p.match_id
        WHERE m.patch = %(patch_id)s AND p.lane_pos IS NOT NULL
    """

    chunk_size = 5000
    conn2 = engine.raw_connection()
    try:
        with conn2.cursor(name="spatial_chunk") as cur:
            cur.itersize = chunk_size
            cur.execute(sql, {"patch_id": patch_id})
            while True:
                rows = cur.fetchmany(chunk_size)
                if not rows:
                    break
                for hero_id, lane_pos in rows:
                    if hero_id is None or hero_id > MAX_HERO_ID:
                        continue
                    lp = lane_pos
                    if isinstance(lp, str):
                        try:
                            lp = _json.loads(lp)
                        except Exception:
                            continue
                    if not isinstance(lp, dict):
                        continue
                    for x, y_dict in lp.items():
                        if not isinstance(y_dict, dict):
                            continue
                        for y, count in y_dict.items():
                            hero_spatial[int(hero_id)][f"{x}_{y}"] += float(count)
    finally:
        conn2.close()

    if not hero_spatial:
        logger.warning("  No lane_pos data, skipping spatial embeddings")
        return

    all_coords = set()
    for coords in hero_spatial.values():
        all_coords.update(coords.keys())
    coord_list = sorted(all_coords)
    coord_idx = {c: i for i, c in enumerate(coord_list)}
    V = len(coord_list)

    S = np.zeros((MAX_HERO_ID + 1, V))
    for hid, coords in hero_spatial.items():
        for c, count in coords.items():
            S[hid, coord_idx[c]] = count
    S = np.log1p(S)

    n_comp = min(SPATIAL_EMB_DIM, S.shape[0] - 1, S.shape[1] - 1)
    if n_comp <= 0:
        logger.warning("  Not enough data for spatial SVD, skipping")
        return

    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    emb = svd.fit_transform(S)
    if n_comp < SPATIAL_EMB_DIM:
        emb = np.pad(emb, ((0, 0), (0, SPATIAL_EMB_DIM - n_comp)))
    logger.info("  Spatial SVD explained variance: %.1f%%", svd.explained_variance_ratio_.sum() * 100)

    rows = [(patch_id, hid, *emb[hid].tolist()) for hid in range(1, MAX_HERO_ID + 1)]
    with conn.cursor() as cur:
        cur.execute("DELETE FROM ml.hero_spatial_embeddings WHERE patch_id = %s", (patch_id,))
        psycopg2.extras.execute_values(cur, "INSERT INTO ml.hero_spatial_embeddings VALUES %s", rows, page_size=5000)
    conn.commit()
    logger.info("  Saved %d hero spatial embeddings", len(rows))
