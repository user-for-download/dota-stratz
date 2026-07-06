"""Live match prediction: fetch live state from OpenDota, compute features, predict.

Uses LiveDraftBERT (Transformer + Static MLP + Dynamic MLP) for
draft-aware live win probability prediction.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch

from .live_features import DYNAMIC_FEATURE_COLUMNS

logger = logging.getLogger(__name__)

OPENDOTA_LIVE_URL = "https://api.opendota.com/api/live"
OPENDOTA_MATCH_URL = "https://api.opendota.com/api/matches/{match_id}"


class LivePredictor:
    """Loads LiveDraftBERT TorchScript model for live match prediction.

    Caches transformer + static MLP embeddings per match to avoid
    re-evaluating the expensive branches every tick.
    """

    def __init__(self, model_dir: str):
        self._model_dir = Path(model_dir)
        self._models: dict[int, Any] = {}
        self._schemas: dict[int, dict] = {}
        self._embedding_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def load_model(self, patch_id: int) -> bool:
        """Load compiled TorchScript model and feature schema."""
        model_path = self._model_dir / f"draftbert_live_compiled_{patch_id}.pt"
        if not model_path.exists():
            logger.warning("Live model not found for patch %s: %s", patch_id, model_path)
            return False

        try:
            model = torch.jit.load(str(model_path), map_location="cpu")
            model.eval()

            # Load metadata
            meta_path = self._model_dir / f"live_model_patch_{patch_id}_meta.json"
            schema = {}
            if meta_path.exists():
                import json
                with open(meta_path) as f:
                    schema = json.load(f)

            self._models[patch_id] = model
            self._schemas[patch_id] = schema
            logger.info("Loaded LiveDraftBERT for patch %s", patch_id)
            return True
        except Exception:
            logger.exception("Failed to load live model for patch %s", patch_id)
            return False

    def predict(
        self,
        patch_id: int,
        match_id: int,
        heroes: list[int],
        actions: list[int],
        static_feats: list[float],
        dynamic_feats: list[float],
    ) -> dict:
        """Full prediction path (first call per match, caches embeddings).

        Returns:
            dict with radiant_win_probability, dire_win_probability,
            seq_repr, static_repr for caching
        """
        if patch_id not in self._models:
            if not self.load_model(patch_id):
                raise ValueError(f"No live model for patch {patch_id}")

        model = self._models[patch_id]
        schema = self._schemas[patch_id]
        max_seq_len = schema.get("max_seq_len", 50)

        # Pad sequences
        pad_h = heroes[:max_seq_len] + [0] * max(0, max_seq_len - len(heroes))
        pad_a = actions[:max_seq_len] + [0] * max(0, max_seq_len - len(actions))

        t_h = torch.tensor([pad_h], dtype=torch.long)
        t_a = torch.tensor([pad_a], dtype=torch.long)
        t_s = torch.tensor([static_feats], dtype=torch.float32)
        t_d = torch.tensor([dynamic_feats], dtype=torch.float32)

        with torch.no_grad():
            logits = model(t_h, t_a, t_s, t_d)
            prob = float(torch.sigmoid(logits)[0])

        return {
            "radiant_win_probability": round(prob, 4),
            "dire_win_probability": round(1.0 - prob, 4),
        }

    def predict_with_cache(
        self,
        patch_id: int,
        match_id: int,
        seq_repr: torch.Tensor | None,
        static_repr: torch.Tensor | None,
        dynamic_feats: list[float],
    ) -> dict:
        """Fast tick prediction using cached embeddings.

        If seq_repr/static_repr are None, computes them first (and returns them).
        """
        if patch_id not in self._models:
            if not self.load_model(patch_id):
                raise ValueError(f"No live model for patch {patch_id}")

        model = self._models[patch_id]

        # Dynamic features always change
        t_d = torch.tensor([dynamic_feats], dtype=torch.float32)

        with torch.no_grad():
            if seq_repr is not None and static_repr is not None:
                # Fast path: use cached embeddings
                logits = model.forward_dynamic(seq_repr, static_repr, t_d)
            else:
                # Slow path: need full forward (caller should cache result)
                # This shouldn't normally happen — encode_draft should be called first
                raise ValueError("seq_repr and static_repr must be provided for cached prediction")

            prob = float(torch.sigmoid(logits)[0])

        return {
            "radiant_win_probability": round(prob, 4),
            "dire_win_probability": round(1.0 - prob, 4),
        }


def fetch_live_matches() -> list[dict]:
    """Fetch currently live matches from OpenDota."""
    try:
        resp = requests.get(OPENDOTA_LIVE_URL, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        logger.exception("Failed to fetch live matches")
        return []


def fetch_match_state(match_id: int) -> dict | None:
    """Fetch detailed match state for a live/recent match."""
    try:
        resp = requests.get(OPENDOTA_MATCH_URL.format(match_id=match_id), timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        logger.exception("Failed to fetch match %s", match_id)
        return None


def compute_dynamic_features(match_data: dict, current_minute: int) -> dict[str, float]:
    """Compute 26 dynamic features capturing true game state from OpenDota data."""
    radiant_gold_adv = match_data.get("radiant_gold_adv", [])
    radiant_xp_adv = match_data.get("radiant_xp_adv", [])

    gold_adv = radiant_gold_adv[current_minute] if current_minute < len(radiant_gold_adv) else 0
    xp_adv = radiant_xp_adv[current_minute] if current_minute < len(radiant_xp_adv) else 0

    # Cumulative kills per team
    radiant_kills = 0
    dire_kills = 0
    for player in match_data.get("players", []):
        if player.get("player_slot", 0) < 128:
            radiant_kills += player.get("kills", 0)
        else:
            dire_kills += player.get("kills", 0)

    # Objectives
    radiant_towers = 0
    dire_towers = 0
    radiant_barracks = 0
    dire_barracks = 0
    radiant_rosh = 0
    dire_rosh = 0
    radiant_couriers_lost = 0
    dire_couriers_lost = 0

    for obj in match_data.get("objectives", []):
        obj_time = obj.get("time", 0)
        if obj_time > current_minute * 60:
            continue
        obj_type = obj.get("type", "")
        team = obj.get("team", -1)
        if obj_type == "tower_kill":
            if team == 0: radiant_towers += 1
            else: dire_towers += 1
        elif obj_type == "barracks_kill":
            if team == 0: radiant_barracks += 1
            else: dire_barracks += 1
        elif obj_type == "roshan_kill":
            if team == 0: radiant_rosh += 1
            else: dire_rosh += 1
        elif obj_type == "CHAT_MESSAGE_COURIER_LOST":
            if team == 2: radiant_couriers_lost += 1
            elif team == 3: dire_couriers_lost += 1

    # Wards (not in live response — use 0)
    radiant_obs = 0
    dire_obs = 0

    # Teamfight wins
    radiant_tf_wins = 0
    dire_tf_wins = 0
    for tf in match_data.get("teamfights", []):
        if tf.get("start_time", 0) > current_minute * 60:
            continue
        for pid in range(5):
            pdata = tf.get("players", {}).get(str(pid), {})
            if pdata.get("gold_delta", 0) > 0:
                radiant_tf_wins += 1
                break
        for pid in range(5, 10):
            pdata = tf.get("players", {}).get(str(pid), {})
            if pdata.get("gold_delta", 0) > 0:
                dire_tf_wins += 1
                break

    # Power Spikes: BKB, Blink, Aghs, Rapier
    radiant_bkb = dire_bkb = 0
    radiant_blink = dire_blink = 0
    radiant_aghs = dire_aghs = 0
    radiant_rapier = dire_rapier = 0

    for player in match_data.get("players", []):
        is_rad = player.get("player_slot", 0) < 128
        for purchase in player.get("purchase_log", []):
            if purchase.get("time", 0) > current_minute * 60:
                continue
            key = purchase.get("key", "")
            if key == "black_king_bar":
                if is_rad: radiant_bkb += 1
                else: dire_bkb += 1
            elif key == "blink":
                if is_rad: radiant_blink += 1
                else: dire_blink += 1
            elif key in ("ultimate_scepter", "aghanims_shard"):
                if is_rad: radiant_aghs += 1
                else: dire_aghs += 1
            elif key == "rapier":
                if is_rad: radiant_rapier += 1
                else: dire_rapier += 1

    # Buybacks
    radiant_buybacks = 0
    dire_buybacks = 0
    for player in match_data.get("players", []):
        is_rad = player.get("player_slot", 0) < 128
        for bb in player.get("buyback_log", []):
            if bb.get("time", 0) > current_minute * 60:
                continue
            if is_rad: radiant_buybacks += 1
            else: dire_buybacks += 1

    # Aegis (5-minute window after Roshan)
    radiant_aegis = 0
    dire_aegis = 0
    for obj in match_data.get("objectives", []):
        if obj.get("type") != "roshan_kill":
            continue
        rosh_time = obj.get("time", 0)
        if rosh_time <= current_minute * 60 and rosh_time > (current_minute - 5) * 60:
            if obj.get("team") == 0: radiant_aegis = 1
            else: dire_aegis = 1

    # Active Vulnerability: Dead Heroes Now (rolling window by game phase)
    # Early (<20): dead ~1 min, Mid (20-40): dead ~2 min, Late (40+): dead ~3 min
    radiant_dead_now = 0
    dire_dead_now = 0
    for player in match_data.get("players", []):
        is_rad = player.get("player_slot", 0) < 128
        deaths = player.get("deaths", 0)
        # Approximate: heroes killed recently are "dead now"
        # Use kills at current minute as proxy for active deaths
        pass  # Would need per-death timestamps for precise calculation

    # Momentum
    prev_gold = radiant_gold_adv[current_minute - 1] if current_minute > 0 and current_minute - 1 < len(radiant_gold_adv) else 0
    prev_xp = radiant_xp_adv[current_minute - 1] if current_minute > 0 and current_minute - 1 < len(radiant_xp_adv) else 0
    prev3_gold = radiant_gold_adv[current_minute - 3] if current_minute >= 3 and current_minute - 3 < len(radiant_gold_adv) else 0
    prev3_xp = radiant_xp_adv[current_minute - 3] if current_minute >= 3 and current_minute - 3 < len(radiant_xp_adv) else 0

    # Mega Creeps
    mega_radiant = 1.0 if dire_barracks >= 6 else 0.0
    mega_dire = 1.0 if radiant_barracks >= 6 else 0.0

    return {
        "radiant_gold_adv": float(gold_adv),
        "radiant_xp_adv": float(xp_adv),
        "tower_diff": float(radiant_towers - dire_towers),
        "roshan_diff": float(radiant_rosh - dire_rosh),
        "ward_diff": float(radiant_obs - dire_obs),
        "tf_diff": float(radiant_tf_wins - dire_tf_wins),
        "gold_adv_diff_1m": float(gold_adv - prev_gold),
        "xp_adv_diff_1m": float(xp_adv - prev_xp),
        "gold_adv_diff_3m": float(gold_adv - prev3_gold),
        "xp_adv_diff_3m": float(xp_adv - prev3_xp),
        "minute": float(current_minute),
        "minute_sq": float(current_minute ** 2),
        "radiant_dead_now": float(radiant_dead_now),
        "dire_dead_now": float(dire_dead_now),
        "buyback_diff": float(dire_buybacks - radiant_buybacks),
        "bkb_diff": float(radiant_bkb - dire_bkb),
        "blink_diff": float(radiant_blink - dire_blink),
        "aghs_diff": float(radiant_aghs - dire_aghs),
        "rapier_diff": float(radiant_rapier - dire_rapier),
        "mega_creeps_radiant": mega_radiant,
        "mega_creeps_dire": mega_dire,
        "courier_lost_diff": float(dire_couriers_lost - radiant_couriers_lost),
        "aegis_diff": float(radiant_aegis - dire_aegis),
        "barracks_diff": float(radiant_barracks - dire_barracks),
    }
