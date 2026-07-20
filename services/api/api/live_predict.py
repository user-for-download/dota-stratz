"""Live match prediction: fetch live state from OpenDota, compute features, predict.

Uses LiveDraftBERT (Transformer + Static MLP + Dynamic MLP) for
draft-aware live win probability prediction.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch

from .live_features import DYNAMIC_FEATURE_COLUMNS
from .model_live import LiveDraftBERT

logger = logging.getLogger(__name__)

OPENDOTA_LIVE_URL = "https://api.opendota.com/api/live"
OPENDOTA_MATCH_URL = "https://api.opendota.com/api/matches/{match_id}"


class LivePredictor:
    """Loads LiveDraftBERT via state_dict for live match prediction.

    Caches transformer + static MLP embeddings per match to avoid
    re-evaluating the expensive branches every tick.
    """

    def __init__(self, model_dir: str):
        self._model_dir = Path(model_dir)
        self._models: dict[int, Any] = {}
        self._schemas: dict[int, dict] = {}
        self._static_drift: dict[int, dict] = {}
        self._lock = threading.RLock()

    def _normalize_static(self, static_feats: list[float], patch_id: int) -> list[float]:
        """Apply training-time mean/std scaling to static features."""
        drift = self._static_drift.get(patch_id)
        if drift is None:
            return static_feats
        mean = drift.get("mean", [])
        std = drift.get("std", [])
        if not mean or not std:
            return static_feats
        return [(v - m) / s for v, m, s in zip(static_feats, mean, std)]

    def load_model(self, patch_id: int) -> bool:
        import json
        with self._lock:
            if patch_id in self._models:
                return True
            weights_path = self._model_dir / f"draftbert_live_weights_{patch_id}.pt"
            meta_path = self._model_dir / f"live_model_patch_{patch_id}_meta.json"
            if not weights_path.exists():
                return False
            try:
                schema = {}
                if meta_path.exists():
                    with open(meta_path) as f:
                        schema = json.load(f)
                model = LiveDraftBERT(
                    vocab_size=schema.get("max_hero_id", 160) + 5,
                    d_model=schema.get("d_model", 128),
                    nhead=schema.get("nhead", 4),
                    num_layers=schema.get("num_layers", 3),
                    num_static_features=schema.get("n_static_features", 61),
                    num_dynamic_features=schema.get("n_dynamic_features", 32),
                    max_seq_len=schema.get("max_seq_len", 25),
                    dropout=schema.get("dropout", 0.3),
                    transformer_dropout=schema.get("transformer_dropout", 0.1),
                    static_hidden=schema.get("static_hidden", 64),
                    dynamic_hidden=schema.get("dynamic_hidden", 24),
                    fusion_hidden=schema.get("fusion_hidden", 64),
                    max_patch_id=200,
                )
                model.load_state_dict(torch.load(str(weights_path), map_location="cpu", weights_only=True))
                model.eval()
                self._models[patch_id] = model
                self._schemas[patch_id] = schema
                self._static_drift[patch_id] = schema.get("drift_stats")
                return True
            except Exception:
                logger.exception("Failed to load live model for patch %s", patch_id)
                return False

    def predict(self, patch_id, match_id, heroes, actions, static_feats, dynamic_feats):
        if patch_id not in self._models:
            if not self.load_model(patch_id):
                raise ValueError(f"No live model for patch {patch_id}")
        model = self._models[patch_id]
        schema = self._schemas[patch_id]
        max_seq_len = schema.get("max_seq_len", 25)
        pad_h = heroes[:max_seq_len] + [0] * max(0, max_seq_len - len(heroes))
        pad_a = actions[:max_seq_len] + [0] * max(0, max_seq_len - len(actions))
        static_feats = self._normalize_static(static_feats, patch_id)
        t_h = torch.tensor([pad_h], dtype=torch.long)
        t_a = torch.tensor([pad_a], dtype=torch.long)
        t_s = torch.tensor([static_feats], dtype=torch.float32)
        t_d = torch.tensor([dynamic_feats], dtype=torch.float32)
        t_p = torch.tensor([patch_id], dtype=torch.long)
        with torch.no_grad():
            prob = torch.sigmoid(model(t_h, t_a, t_s, t_d, t_p)).item()
        return {"radiant_win_probability": prob, "dire_win_probability": 1 - prob}

    def predict_with_cache(self, patch_id, match_id, seq_repr, static_repr, dynamic_feats, patch_repr):
        """Fast inference using pre-computed transformer + static embeddings.

        Only runs the Dynamic MLP and Fusion head — skips the expensive
        Transformer and Static MLP branches that don't change between ticks.
        """
        if patch_id not in self._models:
            if not self.load_model(patch_id):
                raise ValueError(f"No live model for patch {patch_id}")
        model = self._models[patch_id]
        t_d = torch.tensor([dynamic_feats], dtype=torch.float32)
        with torch.no_grad():
            logits = model.forward_dynamic(seq_repr, static_repr, t_d, patch_repr)
            prob = torch.sigmoid(logits).item()
        return {"radiant_win_probability": prob, "dire_win_probability": 1 - prob}

    def encode_draft(self, patch_id, match_id, heroes, actions, static_feats):
        if patch_id not in self._models:
            if not self.load_model(patch_id):
                raise ValueError(f"No live model for patch {patch_id}")
        with self._lock:
            model = self._models[patch_id]
            schema = self._schemas[patch_id]
        max_seq_len = schema.get("max_seq_len", 25)
        pad_h = heroes[:max_seq_len] + [0] * max(0, max_seq_len - len(heroes))
        pad_a = actions[:max_seq_len] + [0] * max(0, max_seq_len - len(actions))
        static_feats = self._normalize_static(static_feats, patch_id)
        t_h = torch.tensor([pad_h], dtype=torch.long)
        t_a = torch.tensor([pad_a], dtype=torch.long)
        t_s = torch.tensor([static_feats], dtype=torch.float32)
        t_p = torch.tensor([patch_id], dtype=torch.long)
        with torch.no_grad():
            seq_repr, static_repr, patch_repr = model.encode_draft(t_h, t_a, t_s, t_p)
        return seq_repr, static_repr, patch_repr


def fetch_live_matches() -> list[dict]:
    try:
        resp = requests.get(OPENDOTA_LIVE_URL, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def fetch_match_state(match_id: int) -> dict | None:
    try:
        resp = requests.get(OPENDOTA_MATCH_URL.format(match_id=match_id), timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def compute_dynamic_features(match_data: dict, current_minute: int) -> dict[str, float]:
    """Compute 30 dynamic features from OpenDota match JSON."""
    radiant_gold_adv = match_data.get("radiant_gold_adv", [])
    radiant_xp_adv = match_data.get("radiant_xp_adv", [])
    gold_adv = radiant_gold_adv[current_minute] if current_minute < len(radiant_gold_adv) else 0
    xp_adv = radiant_xp_adv[current_minute] if current_minute < len(radiant_xp_adv) else 0

    # --- Single-pass player aggregation ---
    radiant_kills = dire_kills = 0
    radiant_obs = dire_obs = 0
    r_deep = d_deep = 0
    r_bkb = d_bkb = r_blink = d_blink = r_aghs = d_aghs = r_rapier = d_rapier = 0
    r_buybacks = d_buybacks = 0
    r_neutrals = d_neutrals = 0

    for player in match_data.get("players", []):
        is_rad = player.get("player_slot", 0) < 128

        # Kills
        pk = player.get("kills", 0)
        if is_rad: radiant_kills += pk
        else: dire_kills += pk

        # Wards (obs_log with coordinates)
        for obs in player.get("obs_log", []):
            if obs.get("time", 0) > current_minute * 60:
                continue
            if is_rad:
                radiant_obs += 1
                if obs.get("y", 128) > 128:
                    r_deep += 1
            else:
                dire_obs += 1
                if obs.get("y", 128) < 128:
                    d_deep += 1

        # Purchases (BKB, Blink, Aghs, Rapier)
        for purchase in player.get("purchase_log", []):
            if purchase.get("time", 0) > current_minute * 60:
                continue
            key = purchase.get("key", "")
            if key == "black_king_bar":
                if is_rad: r_bkb += 1
                else: d_bkb += 1
            elif key == "blink":
                if is_rad: r_blink += 1
                else: d_blink += 1
            elif key in ("ultimate_scepter", "aghanims_shard"):
                if is_rad: r_aghs += 1
                else: d_aghs += 1
            elif key == "rapier":
                if is_rad: r_rapier += 1
                else: d_rapier += 1

        # Buybacks
        for bb in player.get("buyback_log", []):
            if bb.get("time", 0) > current_minute * 60:
                continue
            if is_rad: r_buybacks += 1
            else: d_buybacks += 1

        # Neutral items
        for ni in player.get("neutral_item_history", []):
            if ni.get("time", 0) > current_minute * 60:
                continue
            if is_rad: r_neutrals += 1
            else: d_neutrals += 1

    # --- Objectives ---
    r_t1 = d_t1 = r_t2 = d_t2 = r_t3 = d_t3 = r_t4 = d_t4 = 0
    r_melee = d_melee = r_range = d_range = 0
    r_rosh = d_rosh = r_couriers = d_couriers = 0
    for obj in match_data.get("objectives", []):
        if obj.get("time", 0) > current_minute * 60:
            continue
        obj_type = obj.get("type", "")
        team = obj.get("team", -1)
        key = str(obj.get("key", "") or "").lower()
        is_rad = (team == 0)
        if obj_type == "tower_kill":
            if "tower1" in key:
                if is_rad: r_t1 += 1
                else: d_t1 += 1
            elif "tower2" in key:
                if is_rad: r_t2 += 1
                else: d_t2 += 1
            elif "tower3" in key:
                if is_rad: r_t3 += 1
                else: d_t3 += 1
            elif "tower4" in key:
                if is_rad: r_t4 += 1
                else: d_t4 += 1
        elif obj_type == "barracks_kill":
            if "melee" in key:
                if is_rad: r_melee += 1
                else: d_melee += 1
            elif "range" in key:
                if is_rad: r_range += 1
                else: d_range += 1
        elif obj_type == "roshan_kill":
            if is_rad: r_rosh += 1
            else: d_rosh += 1
        elif obj_type == "CHAT_MESSAGE_COURIER_LOST":
            if team == 2: r_couriers += 1
            elif team == 3: d_couriers += 1

    # --- Teamfights ---
    r_tf_wins = d_tf_wins = 0
    for tf in match_data.get("teamfights", []):
        tf_time = tf.get("start", 0)
        if tf_time > current_minute * 60:
            continue
        players = tf.get("players", [])
        if isinstance(players, list):
            for pid in range(min(5, len(players))):
                pdata = players[pid] if isinstance(players[pid], dict) else {}
                if pdata.get("gold_delta", 0) > 0:
                    r_tf_wins += 1
                    break
            for pid in range(5, min(10, len(players))):
                pdata = players[pid] if isinstance(players[pid], dict) else {}
                if pdata.get("gold_delta", 0) > 0:
                    d_tf_wins += 1
                    break

    # --- Aegis ---
    r_aegis = d_aegis = 0
    for obj in match_data.get("objectives", []):
        if obj.get("type") != "roshan_kill":
            continue
        rt = obj.get("time", 0)
        if rt <= current_minute * 60 and rt >= (current_minute - 4) * 60:
            if obj.get("team") == 0: r_aegis = 1
            else: d_aegis = 1

    # --- Dead heroes (rolling window) ---
    kills_by_minute = {}
    for player in match_data.get("players", []):
        is_rad = player.get("player_slot", 0) < 128
        for ke in player.get("kills_log", []):
            km = ke.get("time", 0) // 60
            if km not in kills_by_minute:
                kills_by_minute[km] = [0, 0]
            if is_rad: kills_by_minute[km][1] += 1
            else: kills_by_minute[km][0] += 1
    r_dead = d_dead = 0
    window = 1 if current_minute < 20 else (2 if current_minute < 40 else 3)
    for m in range(max(0, current_minute - window + 1), current_minute + 1):
        if m in kills_by_minute:
            r_dead += kills_by_minute[m][0]
            d_dead += kills_by_minute[m][1]
    r_dead = min(r_dead, 5)
    d_dead = min(d_dead, 5)

    # --- Momentum ---
    prev_g = radiant_gold_adv[current_minute - 1] if current_minute > 0 and current_minute - 1 < len(radiant_gold_adv) else 0
    prev_x = radiant_xp_adv[current_minute - 1] if current_minute > 0 and current_minute - 1 < len(radiant_xp_adv) else 0
    prev3_g = radiant_gold_adv[current_minute - 3] if current_minute >= 3 and current_minute - 3 < len(radiant_gold_adv) else 0
    prev3_x = radiant_xp_adv[current_minute - 3] if current_minute >= 3 and current_minute - 3 < len(radiant_xp_adv) else 0

    # --- Mega Creeps ---
    mega_r = 1.0 if (d_melee + d_range) >= 6 else 0.0
    mega_d = 1.0 if (r_melee + r_range) >= 6 else 0.0

    return {
        "radiant_gold_adv": float(gold_adv),
        "radiant_xp_adv": float(xp_adv),
        "t1_tower_diff": float(r_t1 - d_t1),
        "t2_tower_diff": float(r_t2 - d_t2),
        "t3_tower_diff": float(r_t3 - d_t3),
        "t4_tower_diff": float(r_t4 - d_t4),
        "melee_rax_diff": float(r_melee - d_melee),
        "range_rax_diff": float(r_range - d_range),
        "roshan_diff": float(r_rosh - d_rosh),
        "ward_diff": float(radiant_obs - dire_obs),
        "tf_diff": float(r_tf_wins - d_tf_wins),
        "gold_adv_diff_1m": float(gold_adv - prev_g),
        "xp_adv_diff_1m": float(xp_adv - prev_x),
        "gold_adv_diff_3m": float(gold_adv - prev3_g),
        "xp_adv_diff_3m": float(xp_adv - prev3_x),
        "minute": float(current_minute),
        "minute_sin": float(np.sin(2 * np.pi * current_minute / 5.0)),
        "minute_cos": float(np.cos(2 * np.pi * current_minute / 5.0)),
        "day_night_sin": float(np.sin(2 * np.pi * current_minute / 10.0)),
        "radiant_dead_now": float(r_dead),
        "dire_dead_now": float(d_dead),
        "buyback_diff": float(r_buybacks - d_buybacks),
        "bkb_diff": float(r_bkb - d_bkb),
        "blink_diff": float(r_blink - d_blink),
        "aghs_diff": float(r_aghs - d_aghs),
        "rapier_diff": float(r_rapier - d_rapier),
        "mega_creeps_radiant": mega_r,
        "mega_creeps_dire": mega_d,
        "courier_lost_diff": float(r_couriers - d_couriers),
        "aegis_diff": float(r_aegis - d_aegis),
        "deep_ward_diff": float(r_deep - d_deep),
        "neutral_tier_diff": float(r_neutrals - d_neutrals),
    }
