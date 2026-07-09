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
        self._lock = threading.RLock()

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
                    num_dynamic_features=schema.get("n_dynamic_features", 35),
                    max_seq_len=schema.get("max_seq_len", 25),
                    dropout=schema.get("dropout", 0.3),
                    transformer_dropout=schema.get("transformer_dropout", 0.1),
                    static_hidden=schema.get("static_hidden", 64),
                    dynamic_hidden=schema.get("dynamic_hidden", 32),
                    fusion_hidden=schema.get("fusion_hidden", 64),
                )
                model.load_state_dict(torch.load(str(weights_path), map_location="cpu", weights_only=True))
                model.eval()
                self._models[patch_id] = model
                self._schemas[patch_id] = schema
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
        t_h = torch.tensor([pad_h], dtype=torch.long)
        t_a = torch.tensor([pad_a], dtype=torch.long)
        t_s = torch.tensor([static_feats], dtype=torch.float32)
        t_d = torch.tensor([dynamic_feats], dtype=torch.float32)
        with torch.no_grad():
            prob = torch.sigmoid(model(t_h, t_a, t_s, t_d)).item()
        return {"radiant_win_probability": prob, "dire_win_probability": 1 - prob}

    def encode_draft(self, patch_id, heroes, actions, static_feats):
        if patch_id not in self._models:
            if not self.load_model(patch_id):
                raise ValueError(f"No live model for patch {patch_id}")
        with self._lock:
            model = self._models[patch_id]
            schema = self._schemas[patch_id]
        max_seq_len = schema.get("max_seq_len", 25)
        pad_h = heroes[:max_seq_len] + [0] * max(0, max_seq_len - len(heroes))
        pad_a = actions[:max_seq_len] + [0] * max(0, max_seq_len - len(actions))
        t_h = torch.tensor([pad_h], dtype=torch.long)
        t_a = torch.tensor([pad_a], dtype=torch.long)
        t_s = torch.tensor([static_feats], dtype=torch.float32)
        with torch.no_grad():
            seq_repr, static_repr = model.encode_draft(t_h, t_a, t_s)
        return seq_repr, static_repr


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
    """Compute 45 dynamic features from OpenDota match JSON."""
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
    r_buffs = d_buffs = 0
    r_neutrals = d_neutrals = 0
    r_stuns = d_stuns = 0.0
    r_tfp_list = []
    d_tfp_list = []
    r_td = d_td = 0
    r_gold_t = []
    d_gold_t = []
    r_save = d_save = r_aura = d_aura = 0
    r_runes = d_runes = 0
    r_dewards = d_dewards = 0

    _SAVE_ITEMS = {"glimmer_cape", "force_staff", "eul_scepter", "ghost_scepter", "aeon_disk"}
    _AURA_ITEMS = {"pipe_of_insight", "crimson_guard", "guardian_greaves", "vladmir", "mekansm", "assault"}

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

        # Purchases (BKB, Blink, Aghs, Rapier, Save, Aura)
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
            elif key in _SAVE_ITEMS:
                if is_rad: r_save += 1
                else: d_save += 1
            elif key in _AURA_ITEMS:
                if is_rad: r_aura += 1
                else: d_aura += 1

        # Buybacks
        for bb in player.get("buyback_log", []):
            if bb.get("time", 0) > current_minute * 60:
                continue
            if is_rad: r_buybacks += 1
            else: d_buybacks += 1

        # Permanent buffs (scaling threats)
        for pb in player.get("permanent_buffs", []):
            if is_rad: r_buffs += pb.get("stack_count", 0)
            else: d_buffs += pb.get("stack_count", 0)

        # Neutral items
        for ni in player.get("neutral_item_history", []):
            if ni.get("time", 0) > current_minute * 60:
                continue
            if is_rad: r_neutrals += 1
            else: d_neutrals += 1

        # Stuns, TF participation, tower damage
        stuns = player.get("stuns", 0) or 0
        tfp = player.get("teamfight_participation", 0) or 0
        td = player.get("tower_damage", 0) or 0
        if is_rad:
            r_stuns += stuns
            r_tfp_list.append(tfp)
            r_td += td
        else:
            d_stuns += stuns
            d_tfp_list.append(tfp)
            d_td += td

        # Gold timeline (for carry % and support NW)
        gt = player.get("gold_t", [])
        hero_gold = gt[current_minute] if current_minute < len(gt) else 0
        if is_rad: r_gold_t.append(hero_gold)
        else: d_gold_t.append(hero_gold)

        # Dewards
        r_dewards += (player.get("observer_kills", 0) or 0) + (player.get("sentry_kills", 0) or 0) if is_rad else 0
        d_dewards += (player.get("observer_kills", 0) or 0) + (player.get("sentry_kills", 0) or 0) if not is_rad else 0

        # Runes
        for rune in player.get("runes_log", []):
            if rune.get("time", 0) > current_minute * 60:
                continue
            if is_rad: r_runes += 1
            else: d_runes += 1

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
    tf_gold_swing = tf_xp_swing = 0.0
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
        # Teamfight swing (last minute only)
        if tf_time >= (current_minute - 1) * 60:
            if isinstance(players, list):
                for pid in range(5):
                    pdata = players[pid] if pid < len(players) and isinstance(players[pid], dict) else {}
                    tf_gold_swing += pdata.get("gold_delta", 0)
                    tf_xp_swing += pdata.get("xp_delta", 0)

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

    # --- Carry / Support NW ---
    r_gold_t.sort()
    d_gold_t.sort()
    r_total = sum(r_gold_t) if r_gold_t else 1
    d_total = sum(d_gold_t) if d_gold_t else 1
    r_max = r_gold_t[-1] if r_gold_t else 0
    d_max = d_gold_t[-1] if d_gold_t else 0
    r_carry_pct = r_max / r_total if r_total > 0 else 0.2
    d_carry_pct = d_max / d_total if d_total > 0 else 0.2
    r_support_nw = sum(r_gold_t[:2]) if len(r_gold_t) >= 2 else 0
    d_support_nw = sum(d_gold_t[:2]) if len(d_gold_t) >= 2 else 0

    # --- CC Effectiveness ---
    r_avg_tfp = sum(r_tfp_list) / len(r_tfp_list) if r_tfp_list else 0.5
    d_avg_tfp = sum(d_tfp_list) / len(d_tfp_list) if d_tfp_list else 0.5
    r_cc = r_stuns * max(r_avg_tfp, 0.01)
    d_cc = d_stuns * max(d_avg_tfp, 0.01)

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
        "minute_sq": float(current_minute ** 2),
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
        "rad_carry_nw_pct": r_carry_pct,
        "dire_carry_nw_pct": d_carry_pct,
        "carry_farm_diff": r_carry_pct - d_carry_pct,
        "support_nw_diff": float(r_support_nw - d_support_nw),
        "radiant_cs_adv": 0.0,
        "save_item_diff": float(r_save - d_save),
        "aura_item_diff": float(r_aura - d_aura),
        "dewards_diff": float(r_dewards - d_dewards),
        "deep_ward_diff": float(r_deep - d_deep),
        "rune_control_diff": float(r_runes - d_runes),
        "tf_gold_swing_1m": tf_gold_swing,
        "tf_xp_swing_1m": tf_xp_swing,
        "map_confinement_diff": 0.0,
        "scaling_threat_diff": float(r_buffs - d_buffs),
        "cc_effectiveness_diff": float(r_cc - d_cc),
        "neutral_tier_diff": float(r_neutrals - d_neutrals),
        "tower_damage_diff": float(r_td - d_td),
    }
