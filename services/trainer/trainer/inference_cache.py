"""In-memory cache of ML aggregates for fast bot simulations.

Loads all ml.*_agg tables into Python dictionaries at startup so that
the Greedy/MCTS bots can evaluate hypothetical picks in microseconds
without hitting PostgreSQL.

Usage:
    cache = InferenceCache(engine, patch_id=60)
    synergy = cache.get_synergy(14, 53)  # Axe + Crystal Maiden
    baseline = cache.get_baseline(2)     # Axe baseline stats
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# Default values for missing heroes/combos (Bayesian prior)
_DEFAULT_BASELINE: dict[str, Any] = {
    "win_rate": 0.5, "pick_rate": 0.0, "ban_rate": 0.0,
    "avg_gpm": 0.0, "avg_xpm": 0.0, "avg_kills": 0.0,
    "avg_deaths": 0.0, "avg_assists": 0.0,
    "avg_gold_10": 0.0, "avg_xp_10": 0.0,
    "total_picks": 0, "total_wins": 0, "total_bans": 0,
}

_DEFAULT_SYNERGY: dict[str, Any] = {"win_rate": 0.5, "games": 0}
_DEFAULT_COUNTER: dict[str, Any] = {"win_rate": 0.5, "games": 0, "avg_kd_diff": 0.0}
_DEFAULT_TEAM_HERO: dict[str, Any] = {
    "win_rate": 0.5, "games": 0, "wins": 0, "bans": 0,
    "avg_gpm": 0.0, "avg_xpm": 0.0, "avg_kills": 0.0,
    "avg_deaths": 0.0, "avg_assists": 0.0,
    "firstblood_rate": 0.0, "avg_camps_stacked": 0.0, "avg_vision_placed": 0.0,
    "avg_gold_10": 0.0, "avg_xp_10": 0.0,
}
_DEFAULT_PLAYER_HERO: dict[str, Any] = {
    "win_rate": 0.5, "games": 0, "wins": 0,
    "avg_gpm": 0.0, "avg_xpm": 0.0, "avg_kills": 0.0,
    "avg_deaths": 0.0, "avg_assists": 0.0, "avg_kda": 0.0,
    "lane_role": 0, "firstblood_rate": 0.0,
    "avg_camps_stacked": 0.0, "avg_vision_placed": 0.0,
    "avg_gold_10": 0.0, "avg_xp_10": 0.0,
}
_DEFAULT_H2H: dict[str, Any] = {"win_rate": 0.5, "games": 0}
_DEFAULT_DRAFT_SLOT: dict[str, Any] = {"win_rate": 0.5, "games": 0}


class InferenceCache:
    """Loads and holds the latest ML aggregates in RAM for O(1) lookups.

    Essential for fast MCTS / Greedy Bot simulations. Uses the flat
    ml.*_agg tables (not snapshot tables) since the bot operates in the
    present and doesn't need PIT safety.
    """

    def __init__(self, engine: Engine, patch_id: int, core_gpm_threshold: float = 420.0):
        self.engine = engine
        self.patch_id = patch_id
        self.core_gpm_threshold = core_gpm_threshold

        # Caches: key -> dict of stats
        self.baseline: dict[int, dict] = {}
        self.synergy: dict[tuple[int, int], dict] = {}
        self.counter: dict[tuple[int, int], dict] = {}
        self.team_hero: dict[tuple[int, int], dict] = {}
        self.player_hero: dict[tuple[int, int], dict] = {}
        self.team_h2h: dict[tuple[int, int], dict] = {}
        self.draft_slot: dict[tuple[int, int], dict] = {}

        # Valid hero IDs (heroes with >0 games in baseline)
        self.valid_hero_ids: list[int] = []

        self._load_all()

    def _load_all(self) -> None:
        """Load all 7 aggregate tables into memory."""
        logger.info("Loading Inference Cache for patch %s into RAM...", self.patch_id)

        # 1. Hero Baseline — vectorized load (no iterrows)
        df_bl = pd.read_sql(
            "SELECT * FROM ml.hero_baseline_agg WHERE patch_id = %(pid)s",
            self.engine, params={"pid": self.patch_id},
        )
        if not df_bl.empty:
            df_bl["hero_id"] = df_bl["hero_id"].astype(int)
            self.baseline = df_bl.set_index("hero_id").to_dict("index")
        logger.info("  baseline: %d heroes", len(self.baseline))

        # 2. Hero Synergy (hero_a < hero_b in DB)
        df_sy = pd.read_sql(
            "SELECT hero_a, hero_b, win_rate, games FROM ml.hero_synergy_agg WHERE patch_id = %(pid)s",
            self.engine, params={"pid": self.patch_id},
        )
        if not df_sy.empty:
            df_sy["hero_a"] = df_sy["hero_a"].astype(int)
            df_sy["hero_b"] = df_sy["hero_b"].astype(int)
            self.synergy = {(r["hero_a"], r["hero_b"]): r for r in df_sy.to_dict("records")}
        logger.info("  synergy: %d pairs", len(self.synergy))

        # 3. Hero Counter
        df_co = pd.read_sql(
            "SELECT hero_id, enemy_hero_id, win_rate, games, avg_kd_diff "
            "FROM ml.hero_counter_agg WHERE patch_id = %(pid)s",
            self.engine, params={"pid": self.patch_id},
        )
        if not df_co.empty:
            df_co["hero_id"] = df_co["hero_id"].astype(int)
            df_co["enemy_hero_id"] = df_co["enemy_hero_id"].astype(int)
            self.counter = {(r["hero_id"], r["enemy_hero_id"]): r for r in df_co.to_dict("records")}
        logger.info("  counter: %d matchups", len(self.counter))

        # 4. Team-Hero
        df_th = pd.read_sql(
            "SELECT * FROM ml.team_hero_agg WHERE patch_id = %(pid)s",
            self.engine, params={"pid": self.patch_id},
        )
        if not df_th.empty:
            df_th["team_id"] = df_th["team_id"].astype(int)
            df_th["hero_id"] = df_th["hero_id"].astype(int)
            self.team_hero = {(r["team_id"], r["hero_id"]): r for r in df_th.to_dict("records")}
        logger.info("  team_hero: %d combos", len(self.team_hero))

        # 5. Player-Hero
        df_ph = pd.read_sql(
            "SELECT * FROM ml.player_hero_agg WHERE patch_id = %(pid)s",
            self.engine, params={"pid": self.patch_id},
        )
        if not df_ph.empty:
            df_ph["account_id"] = df_ph["account_id"].astype(int)
            df_ph["hero_id"] = df_ph["hero_id"].astype(int)
            self.player_hero = {(r["account_id"], r["hero_id"]): r for r in df_ph.to_dict("records")}
        logger.info("  player_hero: %d combos", len(self.player_hero))

        # 6. Team H2H
        df_h2h = pd.read_sql(
            "SELECT * FROM ml.team_h2h_agg WHERE patch_id = %(pid)s",
            self.engine, params={"pid": self.patch_id},
        )
        if not df_h2h.empty:
            df_h2h["team_id"] = df_h2h["team_id"].astype(int)
            df_h2h["enemy_team_id"] = df_h2h["enemy_team_id"].astype(int)
            self.team_h2h = {(r["team_id"], r["enemy_team_id"]): r for r in df_h2h.to_dict("records")}
        logger.info("  team_h2h: %d matchups", len(self.team_h2h))

        # 7. Hero Draft Slot
        df_ds = pd.read_sql(
            "SELECT * FROM ml.hero_draft_slot_agg WHERE patch_id = %(pid)s",
            self.engine, params={"pid": self.patch_id},
        )
        if not df_ds.empty:
            df_ds["hero_id"] = df_ds["hero_id"].astype(int)
            df_ds["team_pick_ordinal"] = df_ds["team_pick_ordinal"].astype(int)
            self.draft_slot = {(r["hero_id"], r["team_pick_ordinal"]): r for r in df_ds.to_dict("records")}
        logger.info("  draft_slot: %d combos", len(self.draft_slot))

        # 8. SVD Embeddings — vectorized load
        try:
            h_cols = [f"emb_{i}" for i in range(32)]
            t_cols = [f"emb_{i}" for i in range(16)]
            df_he = pd.read_sql(
                "SELECT hero_id, {} FROM ml.hero_embeddings WHERE patch_id = %(pid)s".format(", ".join(h_cols)),
                self.engine, params={"pid": self.patch_id},
            )
            if not df_he.empty:
                df_he["hero_id"] = df_he["hero_id"].astype(int)
                self.hero_embeddings = {r["hero_id"]: [r[c] for c in h_cols] for r in df_he.to_dict("records")}
            df_te = pd.read_sql(
                "SELECT team_id, {} FROM ml.team_embeddings WHERE patch_id = %(pid)s".format(", ".join(t_cols)),
                self.engine, params={"pid": self.patch_id},
            )
            if not df_te.empty:
                df_te["team_id"] = df_te["team_id"].astype(int)
                self.team_embeddings = {r["team_id"]: [r[c] for c in t_cols] for r in df_te.to_dict("records")}
            df_pe = pd.read_sql(
                "SELECT account_id, {} FROM ml.player_embeddings WHERE patch_id = %(pid)s".format(", ".join(t_cols)),
                self.engine, params={"pid": self.patch_id},
            )
            if not df_pe.empty:
                df_pe["account_id"] = df_pe["account_id"].astype(int)
                self.player_embeddings = {r["account_id"]: [r[c] for c in t_cols] for r in df_pe.to_dict("records")}

            s_cols = [f"spatial_emb_{i}" for i in range(16)]
            df_hse = pd.read_sql(
                "SELECT hero_id, {} FROM ml.hero_spatial_embeddings WHERE patch_id = %(pid)s".format(", ".join(s_cols)),
                self.engine, params={"pid": self.patch_id},
            )
            if not df_hse.empty:
                df_hse["hero_id"] = df_hse["hero_id"].astype(int)
                self.hero_spatial_embeddings = {r["hero_id"]: [r[c] for c in s_cols] for r in df_hse.to_dict("records")}

            logger.info("  embeddings: %d heroes, %d spatial, %d teams, %d players",
                        len(self.hero_embeddings), len(self.hero_spatial_embeddings),
                        len(self.team_embeddings), len(self.player_embeddings))
        except Exception as e:
            logger.warning("  Could not load SVD embeddings: %s", e)
            self.hero_embeddings = {}
            self.hero_spatial_embeddings = {}
            self.team_embeddings = {}
            self.player_embeddings = {}

        # Build valid hero list
        self.valid_hero_ids = sorted([
            h for h, bl in self.baseline.items()
            if bl.get("total_picks", 0) > 0
        ])
        logger.info("  valid_heroes: %d heroes with >0 games", len(self.valid_hero_ids))
        logger.info("Inference Cache loaded successfully.")

    # --- Fast Lookup Helpers ---

    def get_baseline(self, hero_id: int) -> dict:
        """Get global baseline stats for a hero."""
        return self.baseline.get(hero_id, _DEFAULT_BASELINE)

    def get_synergy(self, hero1: int, hero2: int) -> dict:
        """Get synergy stats for a hero pair. Handles hero_a < hero_b ordering."""
        ha, hb = min(hero1, hero2), max(hero1, hero2)
        return self.synergy.get((ha, hb), _DEFAULT_SYNERGY)

    def get_counter(self, hero: int, enemy: int) -> dict:
        """Get counter stats for hero vs enemy matchup."""
        return self.counter.get((hero, enemy), _DEFAULT_COUNTER)

    def get_team_hero(self, team_id: int, hero_id: int) -> dict:
        """Get team-specific hero stats."""
        return self.team_hero.get((team_id, hero_id), _DEFAULT_TEAM_HERO)

    def get_player_hero(self, account_id: int, hero_id: int) -> dict:
        """Get player-specific hero stats."""
        return self.player_hero.get((account_id, hero_id), _DEFAULT_PLAYER_HERO)

    def get_h2h(self, team_id: int, enemy_team_id: int) -> dict:
        """Get head-to-head stats between two teams."""
        return self.team_h2h.get((team_id, enemy_team_id), _DEFAULT_H2H)

    def get_draft_slot(self, hero_id: int, pick_ordinal: int) -> dict:
        """Get hero stats for a specific pick position (1-5)."""
        return self.draft_slot.get((hero_id, pick_ordinal), _DEFAULT_DRAFT_SLOT)
