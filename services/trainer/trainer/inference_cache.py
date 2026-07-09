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

    def __init__(self, engine: Engine, patch_id: int):
        self.engine = engine
        self.patch_id = patch_id

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

        # 1. Hero Baseline
        df_bl = pd.read_sql(
            "SELECT * FROM ml.hero_baseline_agg WHERE patch_id = %(pid)s",
            self.engine, params={"pid": self.patch_id},
        )
        self.baseline = {
            int(row["hero_id"]): row.to_dict()
            for _, row in df_bl.iterrows()
        }
        logger.info("  baseline: %d heroes", len(self.baseline))

        # 2. Hero Synergy (hero_a < hero_b in DB)
        df_sy = pd.read_sql(
            "SELECT hero_a, hero_b, win_rate, games FROM ml.hero_synergy_agg WHERE patch_id = %(pid)s",
            self.engine, params={"pid": self.patch_id},
        )
        self.synergy = {
            (int(row["hero_a"]), int(row["hero_b"])): row.to_dict()
            for _, row in df_sy.iterrows()
        }
        logger.info("  synergy: %d pairs", len(self.synergy))

        # 3. Hero Counter
        df_co = pd.read_sql(
            "SELECT hero_id, enemy_hero_id, win_rate, games, avg_kd_diff "
            "FROM ml.hero_counter_agg WHERE patch_id = %(pid)s",
            self.engine, params={"pid": self.patch_id},
        )
        self.counter = {
            (int(row["hero_id"]), int(row["enemy_hero_id"])): row.to_dict()
            for _, row in df_co.iterrows()
        }
        logger.info("  counter: %d matchups", len(self.counter))

        # 4. Team-Hero
        df_th = pd.read_sql(
            "SELECT * FROM ml.team_hero_agg WHERE patch_id = %(pid)s",
            self.engine, params={"pid": self.patch_id},
        )
        self.team_hero = {
            (int(row["team_id"]), int(row["hero_id"])): row.to_dict()
            for _, row in df_th.iterrows()
        }
        logger.info("  team_hero: %d combos", len(self.team_hero))

        # 5. Player-Hero
        df_ph = pd.read_sql(
            "SELECT * FROM ml.player_hero_agg WHERE patch_id = %(pid)s",
            self.engine, params={"pid": self.patch_id},
        )
        self.player_hero = {
            (int(row["account_id"]), int(row["hero_id"])): row.to_dict()
            for _, row in df_ph.iterrows()
        }
        logger.info("  player_hero: %d combos", len(self.player_hero))

        # 6. Team H2H
        df_h2h = pd.read_sql(
            "SELECT * FROM ml.team_h2h_agg WHERE patch_id = %(pid)s",
            self.engine, params={"pid": self.patch_id},
        )
        self.team_h2h = {
            (int(row["team_id"]), int(row["enemy_team_id"])): row.to_dict()
            for _, row in df_h2h.iterrows()
        }
        logger.info("  team_h2h: %d matchups", len(self.team_h2h))

        # 7. Hero Draft Slot
        df_ds = pd.read_sql(
            "SELECT * FROM ml.hero_draft_slot_agg WHERE patch_id = %(pid)s",
            self.engine, params={"pid": self.patch_id},
        )
        self.draft_slot = {
            (int(row["hero_id"]), int(row["team_pick_ordinal"])): row.to_dict()
            for _, row in df_ds.iterrows()
        }
        logger.info("  draft_slot: %d combos", len(self.draft_slot))

        # 8. SVD Embeddings
        try:
            h_cols = [f"emb_{i}" for i in range(32)]
            t_cols = [f"emb_{i}" for i in range(16)]
            df_he = pd.read_sql(
                "SELECT hero_id, {} FROM ml.hero_embeddings WHERE patch_id = %(pid)s".format(", ".join(h_cols)),
                self.engine, params={"pid": self.patch_id},
            )
            self.hero_embeddings = {int(row["hero_id"]): row[h_cols].tolist() for _, row in df_he.iterrows()}
            df_te = pd.read_sql(
                "SELECT team_id, {} FROM ml.team_embeddings WHERE patch_id = %(pid)s".format(", ".join(t_cols)),
                self.engine, params={"pid": self.patch_id},
            )
            self.team_embeddings = {int(row["team_id"]): row[t_cols].tolist() for _, row in df_te.iterrows()}
            df_pe = pd.read_sql(
                "SELECT account_id, {} FROM ml.player_embeddings WHERE patch_id = %(pid)s".format(", ".join(t_cols)),
                self.engine, params={"pid": self.patch_id},
            )
            self.player_embeddings = {int(row["account_id"]): row[t_cols].tolist() for _, row in df_pe.iterrows()}

            s_cols = [f"spatial_emb_{i}" for i in range(16)]
            df_hse = pd.read_sql(
                "SELECT hero_id, {} FROM ml.hero_spatial_embeddings WHERE patch_id = %(pid)s".format(", ".join(s_cols)),
                self.engine, params={"pid": self.patch_id},
            )
            self.hero_spatial_embeddings = {int(row["hero_id"]): row[s_cols].tolist() for _, row in df_hse.iterrows()}

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
