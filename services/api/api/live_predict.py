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
        """Load LiveDraftBERT from state_dict + meta JSON."""
        import json

        with self._lock:
            if patch_id in self._models:
                return True

            weights_path = self._model_dir / f"draftbert_live_weights_{patch_id}.pt"
            meta_path = self._model_dir / f"live_model_patch_{patch_id}_meta.json"

            if not weights_path.exists():
                logger.warning("Live weights not found for patch %s: %s", patch_id, weights_path)
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
                logger.info("Loaded LiveDraftBERT for patch %s (state_dict)", patch_id)
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
        max_seq_len = schema.get("max_seq_len", 25)

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

    def encode_draft(
        self,
        patch_id: int,
        match_id: int,
        heroes: list[int],
        actions: list[int],
        static_feats: list[float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode draft + static features, return cached embeddings."""
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
    """Compute 46 dynamic features capturing true game state from OpenDota data."""
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
    r_t1 = d_t1 = r_t2 = d_t2 = r_t3 = d_t3 = r_t4 = d_t4 = 0
    r_melee = d_melee = r_range = d_range = 0
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
        key = str(obj.get("key", "") or "").lower()

        if obj_type == "tower_kill":
            is_rad = (team == 0)
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
            is_rad = (team == 0)
            if "melee" in key:
                if is_rad: r_melee += 1
                else: d_melee += 1
            elif "range" in key:
                if is_rad: r_range += 1
                else: d_range += 1
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
        if tf.get("start", 0) > current_minute * 60:
            continue
        players = tf.get("players", [])
        if isinstance(players, list):
            for pid in range(min(5, len(players))):
                pdata = players[pid] if isinstance(players[pid], dict) else {}
                if pdata.get("gold_delta", 0) > 0:
                    radiant_tf_wins += 1
                    break
            for pid in range(5, min(10, len(players))):
                pdata = players[pid] if isinstance(players[pid], dict) else {}
                if pdata.get("gold_delta", 0) > 0:
                    dire_tf_wins += 1
                    break
        elif isinstance(players, dict):
            for pid in range(5):
                pdata = players.get(str(pid), {})
                if pdata.get("gold_delta", 0) > 0:
                    radiant_tf_wins += 1
                    break
            for pid in range(5, 10):
                pdata = players.get(str(pid), {})
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
        if rosh_time <= current_minute * 60 and rosh_time >= (current_minute - 4) * 60:
            if obj.get("team") == 0: radiant_aegis = 1
            else: dire_aegis = 1

    # Active Vulnerability: Dead Heroes Now (rolling window by game phase)
    # Early (<20): dead ~1 min, Mid (20-40): dead ~2 min, Late (40+): dead ~3 min
    # Use kills_log to track recent kills (victims are "dead now")
    kills_by_minute = {}  # {minute: (r_deaths, d_deaths)}
    for player in match_data.get("players", []):
        is_rad = player.get("player_slot", 0) < 128
        for kill_entry in player.get("kills_log", []):
            k_time = kill_entry.get("time", 0)
            k_min = k_time // 60
            if k_min not in kills_by_minute:
                kills_by_minute[k_min] = [0, 0]
            # This player got a kill → the VICTIM is on the opposite team
            if is_rad:
                kills_by_minute[k_min][1] += 1  # Dire hero died
            else:
                kills_by_minute[k_min][0] += 1  # Radiant hero died

    radiant_dead_now = 0
    dire_dead_now = 0
    window = 1 if current_minute < 20 else (2 if current_minute < 40 else 3)
    for m in range(max(0, current_minute - window + 1), current_minute + 1):
        if m in kills_by_minute:
            radiant_dead_now += kills_by_minute[m][0]
            dire_dead_now += kills_by_minute[m][1]
    # Cap at 5 (team size)
    radiant_dead_now = min(radiant_dead_now, 5)
    dire_dead_now = min(dire_dead_now, 5)

    # Momentum
    prev_gold = radiant_gold_adv[current_minute - 1] if current_minute > 0 and current_minute - 1 < len(radiant_gold_adv) else 0
    prev_xp = radiant_xp_adv[current_minute - 1] if current_minute > 0 and current_minute - 1 < len(radiant_xp_adv) else 0
    prev3_gold = radiant_gold_adv[current_minute - 3] if current_minute >= 3 and current_minute - 3 < len(radiant_gold_adv) else 0
    prev3_xp = radiant_xp_adv[current_minute - 3] if current_minute >= 3 and current_minute - 3 < len(radiant_xp_adv) else 0

    # Mega Creeps
    mega_radiant = 1.0 if (d_melee + d_range) >= 6 else 0.0
    mega_dire = 1.0 if (r_melee + r_range) >= 6 else 0.0

    return {
        "radiant_gold_adv": float(gold_adv),
        "radiant_xp_adv": float(xp_adv),
        "t1_tower_diff": float(r_t1 - d_t1),
        "t2_tower_diff": float(r_t2 - d_t2),
        "t3_tower_diff": float(r_t3 - d_t3),
        "t4_tower_diff": float(r_t4 - d_t4),
        "melee_rax_diff": float(r_melee - d_melee),
        "range_rax_diff": float(r_range - d_range),
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
        "buyback_diff": float(radiant_buybacks - dire_buybacks),
        "bkb_diff": float(radiant_bkb - dire_bkb),
        "blink_diff": float(radiant_blink - dire_blink),
        "aghs_diff": float(radiant_aghs - dire_aghs),
        "rapier_diff": float(radiant_rapier - dire_rapier),
        "mega_creeps_radiant": mega_radiant,
        "mega_creeps_dire": mega_dire,
        "courier_lost_diff": float(radiant_couriers_lost - dire_couriers_lost),
        "aegis_diff": float(radiant_aegis - dire_aegis),
        **_extract_economy_distribution(match_data, current_minute),
        **_extract_cs_advantage(match_data, current_minute),
        **_extract_defensive_items(match_data, current_minute),
        **_extract_dewards(match_data),
        **_extract_deep_vision(match_data),
        **_extract_rune_control(match_data, current_minute),
        **_extract_teamfight_swing(match_data, current_minute),
        **_extract_support_nw(match_data, current_minute),
        **_extract_map_confinement(match_data),
        **_extract_scaling_threats(match_data),
        **_extract_cc_effectiveness(match_data),
        **_extract_neutral_tier(match_data, current_minute),
        **_extract_tower_damage(match_data),
    }


def _extract_economy_distribution(match_data: dict, current_minute: int) -> dict:
    """Feature 1: Economy Distribution — who holds the gold."""
    radiant_gold_total = 0
    dire_gold_total = 0
    radiant_max_gold = 0
    dire_max_gold = 0

    for player in match_data.get("players", []):
        is_rad = player.get("player_slot", 0) < 128
        gold_timeline = player.get("gold_t", [])
        hero_gold = gold_timeline[current_minute] if current_minute < len(gold_timeline) else 0

        if is_rad:
            radiant_gold_total += hero_gold
            radiant_max_gold = max(radiant_max_gold, hero_gold)
        else:
            dire_gold_total += hero_gold
            dire_max_gold = max(dire_max_gold, hero_gold)

    rad_pos1_pct = (radiant_max_gold / radiant_gold_total) if radiant_gold_total > 0 else 0.2
    dire_pos1_pct = (dire_max_gold / dire_gold_total) if dire_gold_total > 0 else 0.2

    return {
        "rad_carry_nw_pct": rad_pos1_pct,
        "dire_carry_nw_pct": dire_pos1_pct,
        "carry_farm_diff": rad_pos1_pct - dire_pos1_pct,
    }


def _extract_cs_advantage(match_data: dict, current_minute: int) -> dict:
    """Feature 2: Laning Phase — Last Hit + Deny advantage."""
    radiant_cs = 0
    dire_cs = 0

    for player in match_data.get("players", []):
        is_rad = player.get("player_slot", 0) < 128
        lh_timeline = player.get("lh_t", [])
        dn_timeline = player.get("dn_t", [])
        minute_idx = min(current_minute, len(lh_timeline) - 1, len(dn_timeline) - 1)
        if minute_idx >= 0:
            lh = lh_timeline[minute_idx] if minute_idx < len(lh_timeline) else 0
            dn = dn_timeline[minute_idx] if minute_idx < len(dn_timeline) else 0
            cs = lh + dn
            if is_rad:
                radiant_cs += cs
            else:
                dire_cs += cs

    return {"radiant_cs_adv": float(radiant_cs - dire_cs)}


# Defensive item keys (save items)
_SAVE_ITEMS = {"glimmer_cape", "force_staff", "eul_scepter", "ghost_scepter", "aeon_disk"}
# Aura item keys (teamfight utility)
_AURA_ITEMS = {"pipe_of_insight", "crimson_guard", "guardian_greaves", "vladmir", "mekansm", "assault"}


def _extract_defensive_items(match_data: dict, current_minute: int) -> dict:
    """Feature 3: Defensive & Utility Power Spikes."""
    radiant_save = 0
    dire_save = 0
    radiant_aura = 0
    dire_aura = 0

    for player in match_data.get("players", []):
        is_rad = player.get("player_slot", 0) < 128
        for purchase in player.get("purchase_log", []):
            if purchase.get("time", 0) > current_minute * 60:
                continue
            key = purchase.get("key", "")
            if key in _SAVE_ITEMS:
                if is_rad: radiant_save += 1
                else: dire_save += 1
            elif key in _AURA_ITEMS:
                if is_rad: radiant_aura += 1
                else: dire_aura += 1

    return {
        "save_item_diff": float(radiant_save - dire_save),
        "aura_item_diff": float(radiant_aura - dire_aura),
    }


def _extract_dewards(match_data: dict) -> dict:
    """Feature 4: Vision Denial — Observer/Sentry kills."""
    radiant_dewards = 0
    dire_dewards = 0

    for player in match_data.get("players", []):
        is_rad = player.get("player_slot", 0) < 128
        obs_kills = player.get("observer_kills", 0) or 0
        sen_kills = player.get("sentry_kills", 0) or 0
        total = obs_kills + sen_kills
        if is_rad:
            radiant_dewards += total
        else:
            dire_dewards += total

    return {"dewards_diff": float(radiant_dewards - dire_dewards)}


def _extract_rune_control(match_data: dict, current_minute: int) -> dict:
    """Feature 5: Rune Control Efficiency."""
    radiant_runes = 0
    dire_runes = 0

    for player in match_data.get("players", []):
        is_rad = player.get("player_slot", 0) < 128
        for rune in player.get("runes_log", []):
            if rune.get("time", 0) > current_minute * 60:
                continue
            if is_rad: radiant_runes += 1
            else: dire_runes += 1

    return {"rune_control_diff": float(radiant_runes - dire_runes)}


def _extract_teamfight_swing(match_data: dict, current_minute: int) -> dict:
    """Feature 6: True Teamfight Efficiency — magnitude of gold/XP swings in last minute."""
    tf_gold_swing = 0.0
    tf_xp_swing = 0.0

    for tf in match_data.get("teamfights", []):
        tf_time = tf.get("start", 0)
        if tf_time < (current_minute - 1) * 60 or tf_time > current_minute * 60:
            continue

        players = tf.get("players", [])
        if isinstance(players, list):
            for pid in range(5):
                pdata = players[pid] if pid < len(players) and isinstance(players[pid], dict) else {}
                tf_gold_swing += pdata.get("gold_delta", 0)
                tf_xp_swing += pdata.get("xp_delta", 0)

    return {
        "tf_gold_swing_1m": tf_gold_swing,
        "tf_xp_swing_1m": tf_xp_swing,
    }


def _extract_deep_vision(match_data: dict) -> dict:
    """Deep Vision Advantage — wards placed past the river on enemy side."""
    # Radiant deep ward = y > 96 (upper half, Dire territory)
    # Dire deep ward = y < 96 (lower half, Radiant territory)
    r_deep = 0
    d_deep = 0
    for player in match_data.get("players", []):
        is_rad = player.get("player_slot", 0) < 128
        for ward in player.get("obs_log", []):
            y = ward.get("y", 0)
            if is_rad and y > 96:
                r_deep += 1
            elif not is_rad and y < 96:
                d_deep += 1
    return {"deep_ward_diff": float(r_deep - d_deep)}


def _extract_support_nw(match_data: dict, current_minute: int) -> dict:
    """Support Net Worth Diff — bottom 2 net worths per team from gold_t."""
    rad_golds = []
    dire_golds = []
    for player in match_data.get("players", []):
        is_rad = player.get("player_slot", 0) < 128
        gt = player.get("gold_t", [])
        hero_gold = gt[current_minute] if current_minute < len(gt) else 0
        if is_rad:
            rad_golds.append(hero_gold)
        else:
            dire_golds.append(hero_gold)
    rad_golds.sort()
    dire_golds.sort()
    # Bottom 2 = supports
    rad_support = sum(rad_golds[:2]) if len(rad_golds) >= 2 else 0
    dire_support = sum(dire_golds[:2]) if len(dire_golds) >= 2 else 0
    return {"support_nw_diff": float(rad_support - dire_support)}


def _extract_map_confinement(match_data: dict) -> dict:
    """Map Confinement — where do teams die? Deaths in enemy territory = aggression."""
    r_home = r_away = d_home = d_away = 0
    for tf in match_data.get("teamfights", []):
        players = tf.get("players", [])
        if not isinstance(players, list):
            continue
        for pid, pdata in enumerate(players):
            if not isinstance(pdata, dict):
                continue
            deaths_pos = pdata.get("deaths_pos", {})
            is_rad = pid < 5
            # deaths_pos is dict with string keys "0"-"9" mapping to {x,y}
            for victim_id_str, pos in deaths_pos.items():
                if not isinstance(pos, dict):
                    continue
                y = pos.get("y", 96)
                if is_rad:
                    if y < 96:
                        r_home += 1  # Radiant died on their own side
                    else:
                        r_away += 1  # Radiant died in enemy territory (aggressive)
                else:
                    if y >= 96:
                        d_home += 1  # Dire died on their own side
                    else:
                        d_away += 1  # Dire died in enemy territory
    total_r = r_home + r_away
    total_d = d_home + d_away
    r_away_pct = r_away / total_r if total_r > 0 else 0.5
    d_away_pct = d_away / total_d if total_d > 0 else 0.5
    return {"map_confinement_diff": float(r_away_pct - d_away_pct)}


def _extract_scaling_threats(match_data: dict) -> dict:
    """Scaling Threats — permanent buff stacks (LC duel damage, Silencer int, etc.)."""
    r_stacks = 0
    d_stacks = 0
    for player in match_data.get("players", []):
        is_rad = player.get("player_slot", 0) < 128
        for buff in player.get("permanent_buffs", []):
            stack = buff.get("stack_count", 0)
            if is_rad:
                r_stacks += stack
            else:
                d_stacks += stack
    return {"scaling_threat_diff": float(r_stacks - d_stacks)}


def _extract_cc_effectiveness(match_data: dict) -> dict:
    """CC Effectiveness — stun duration multiplied by teamfight participation."""
    r_stuns = 0.0
    d_stuns = 0.0
    r_tf = 0.0
    d_tf = 0.0
    r_count = 0
    d_count = 0
    for player in match_data.get("players", []):
        is_rad = player.get("player_slot", 0) < 128
        stuns = player.get("stuns", 0) or 0
        tfp = player.get("teamfight_participation", 0) or 0
        if is_rad:
            r_stuns += stuns
            r_tf += tfp
            r_count += 1
        else:
            d_stuns += stuns
            d_tf += tfp
            d_count += 1
    r_avg_tfp = r_tf / r_count if r_count > 0 else 0.5
    d_avg_tfp = d_tf / d_count if d_count > 0 else 0.5
    r_cc = r_stuns * r_avg_tfp
    d_cc = d_stuns * d_avg_tfp
    return {"cc_effectiveness_diff": float(r_cc - d_cc)}


def _extract_neutral_tier(match_data: dict, current_minute: int) -> dict:
    """Neutral Item Tier Timing — who got their neutrals faster."""
    r_neutrals = 0
    d_neutrals = 0
    for player in match_data.get("players", []):
        is_rad = player.get("player_slot", 0) < 128
        for nih in player.get("neutral_item_history", []):
            if nih.get("time", 0) <= current_minute * 60:
                if is_rad:
                    r_neutrals += 1
                else:
                    d_neutrals += 1
    return {"neutral_tier_diff": float(r_neutrals - d_neutrals)}


def _extract_tower_damage(match_data: dict) -> dict:
    """Tower Damage — cumulative building pressure."""
    r_td = 0
    d_td = 0
    for player in match_data.get("players", []):
        is_rad = player.get("player_slot", 0) < 128
        td = player.get("tower_damage", 0) or 0
        if is_rad:
            r_td += td
        else:
            d_td += td
    return {"tower_damage_diff": float(r_td - d_td)}
