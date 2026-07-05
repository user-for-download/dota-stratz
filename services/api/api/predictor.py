"""Model loading and prediction orchestration for PyTorch DraftBERT.

Loads compiled TorchScript models for fast CPU inference. Uses batched
matrix multiplication to evaluate all candidate heroes simultaneously,
bypassing the Python GIL via C++ execution.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import torch
import numpy as np

from . import db as db_
from .config import APIConfig
from .draft_state import DraftContext
from .features import BatchContext, build_feature_vector, load_schema, pre_fetch_batch
from .reasoning import generate_reasoning
from .models import DraftSlot

logger = logging.getLogger(__name__)

# Valid hero IDs from const_hero table (excludes removed/replaced heroes)
VALID_HERO_IDS = frozenset({
    1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,
    31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,
    58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,
    85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,
    109,110,111,112,113,114,119,120,121,123,126,128,129,131,135,136,137,138,145,155,
})


class Predictor:
    """Manages per-patch PyTorch model loading and batched inference."""

    def __init__(self, cfg: APIConfig):
        self._cfg = cfg
        self._models: dict[int, Any] = {}
        self._schemas: dict[int, dict[str, Any]] = {}
        self._model_dir = Path(cfg.model_dir)
        self._max_hero_id = cfg.max_hero_id
        self._lock = threading.RLock()

    def _model_path(self, patch_id: int) -> Path:
        return self._model_dir / f"draftbert_compiled_{patch_id}.pt"

    def is_loaded(self, patch_id: int) -> bool:
        with self._lock:
            return patch_id in self._models

    def loaded_patches(self) -> list[int]:
        with self._lock:
            return sorted(self._models.keys())

    def load_model(self, patch_id: int) -> bool:
        """Load compiled TorchScript model and schema."""
        model_path = self._model_path(patch_id)
        if not model_path.exists():
            logger.warning("Model not found for patch %s: %s", patch_id, model_path)
            return False

        try:
            model = torch.jit.load(str(model_path), map_location="cpu")
            model.eval()
            schema = load_schema(self._model_dir, patch_id)

            with self._lock:
                self._models[patch_id] = model
                self._schemas[patch_id] = schema
            logger.info("Loaded PyTorch DraftBERT for patch %s", patch_id)
            return True
        except Exception:
            logger.exception("Failed to load model for patch %s", patch_id)
            return False

    def unload_model(self, patch_id: int):
        with self._lock:
            self._models.pop(patch_id, None)
            self._schemas.pop(patch_id, None)

    def reload_all(self):
        new_models = {}
        new_schemas = {}
        count = 0
        for fpath in sorted(self._model_dir.glob("draftbert_compiled_*.pt")):
            try:
                pid = int(fpath.stem.replace("draftbert_compiled_", ""))
                model = torch.jit.load(str(fpath), map_location="cpu")
                model.eval()
                schema = load_schema(self._model_dir, pid)
                new_models[pid] = model
                new_schemas[pid] = schema
                count += 1
            except Exception:
                continue
        with self._lock:
            self._models = new_models
            self._schemas = new_schemas
        return count

    def predict(
        self,
        patch_id: int,
        ctx: DraftContext,
        draft_slots: list[DraftSlot],
        radiant_team_id: int | None,
        dire_team_id: int | None,
        num_recommendations: int = 5,
        account_id: int | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Score all eligible heroes using batched TorchScript inference."""
        with self._lock:
            if patch_id not in self._models:
                if not self.load_model(patch_id):
                    raise ValueError(f"No model for patch {patch_id}.")
            model = self._models[patch_id]
            schema = self._schemas[patch_id]

        taken_heroes = ctx.all_taken
        eligible = [h for h in range(1, 156) if h not in taken_heroes and h in VALID_HERO_IDS]
        if not eligible:
            return [], None

        team_id = radiant_team_id if ctx.recommending_team == 0 else dire_team_id
        enemy_team_id = dire_team_id if ctx.recommending_team == 0 else radiant_team_id

        batch = pre_fetch_batch(patch_id, eligible, team_id, enemy_team_id, ctx, account_id)

        # Build base sequence from draft history
        sorted_draft = sorted(draft_slots, key=lambda x: x.order)
        base_heroes = [d.hero_id for d in sorted_draft]
        base_actions = [d.team * 1 + int(d.is_pick) * 2 + 1 for d in sorted_draft]

        # Vectorize: batch all candidates at once
        num_continuous = len(schema["aggregate_columns"])
        max_seq_len = schema.get("max_seq_len", 50)
        batch_h, batch_a, batch_f = [], [], []

        for hid in eligible:
            h_seq = base_heroes + [hid]
            a_seq = base_actions + [ctx.recommending_team * 1 + 2 + 1]
            pad = max_seq_len - len(h_seq)
            batch_h.append(h_seq + [0] * pad)
            batch_a.append(a_seq + [0] * pad)

            fv = build_feature_vector(hid, ctx, patch_id, batch, schema, {}, schema["max_hero_id"])
            batch_f.append(fv[:num_continuous])

        # Single batched forward pass
        t_h = torch.tensor(batch_h, dtype=torch.long)
        t_a = torch.tensor(batch_a, dtype=torch.long)
        t_f = torch.tensor(np.array(batch_f), dtype=torch.float32)

        with torch.no_grad():
            logits = model(t_h, t_a, t_f)
            probs = torch.sigmoid(logits).numpy()

        if ctx.recommending_team == 1:
            probs = 1.0 - probs

        # Build recommendations with team-hero boost
        recs = []
        for i, hid in enumerate(eligible):
            sc = float(probs[i])
            th = batch.team_hero_agg.get(hid)
            boosted = False
            if th and th.get("games", 0) >= 3:
                twr, tg = th["win_rate"], th["games"]
                if twr >= 0.80 and tg >= 5: sc = min(1.0, sc + 0.25); boosted = True
                elif twr >= 0.75 and tg >= 4: sc = min(1.0, sc + 0.20); boosted = True
                elif twr >= 0.70 and tg >= 3: sc = min(1.0, sc + 0.15); boosted = True
                elif twr >= 0.65 and tg >= 3: sc = min(1.0, sc + 0.10); boosted = True

            recs.append({
                "hero_id": hid, "score": float(probs[i]),
                "pick_probability": round(sc, 4), "win_probability": round(sc, 4),
                "team_games": int(th.get("games") or 0) if th else 0,
                "team_win_rate": round(th.get("win_rate") or 0, 4) if th else None,
                "boosted": boosted,
            })

        recs.sort(key=lambda r: r["win_probability"], reverse=True)

        # Monte Carlo Rollouts: simulate draft completions to re-rank
        try:
            from .lookahead import run_monte_carlo_rollouts
            top_candidates = recs[:15]
            recs = run_monte_carlo_rollouts(
                self, patch_id, ctx, top_candidates, eligible,
                radiant_team_id, dire_team_id, num_simulations=40,
            )
        except Exception as e:
            logger.warning("MCTS rollouts failed, using base policy: %s", e)

        recs = recs[:num_recommendations]

        reasoning = None
        if recs:
            top = recs[0]
            base_reasoning = self._build_reasoning(
                top["hero_id"], top["score"], ctx, patch_id,
                radiant_team_id, dire_team_id, batch=batch,
            )
            mc_prob = top.get("mc_win_probability")
            mc_str = f" | MCTS Rollout WR: {mc_prob*100:.1f}%" if mc_prob else ""
            reasoning = (base_reasoning or "") + mc_str

        return recs, reasoning

    def predict_match_outcome(
        self, patch_id: int, radiant_heroes: list[int], dire_heroes: list[int],
        radiant_team_id: int | None = None, dire_team_id: int | None = None,
    ) -> float:
        """Evaluate a complete 5v5 composition."""
        with self._lock:
            if patch_id not in self._models:
                if not self.load_model(patch_id):
                    raise ValueError(f"No model for patch {patch_id}.")
            model = self._models[patch_id]
            schema = self._schemas[patch_id]

        # Build sequence: alternating Rad/Dire picks
        max_seq_len = schema.get("max_seq_len", 50)
        seq_h, seq_a = [], []
        for i in range(5):
            seq_h.extend([radiant_heroes[i], dire_heroes[i]])
            seq_a.extend([3, 4])
        pad = max_seq_len - len(seq_h)
        seq_h += [0] * pad
        seq_a += [0] * pad

        t_h = torch.tensor([seq_h], dtype=torch.long)
        t_a = torch.tensor([seq_a], dtype=torch.long)

        # Fetch tabular features for team comparison
        ctx = DraftContext(turn=11, recommending_team=-1, is_pick_turn=False,
                           radiant_picks=radiant_heroes, dire_picks=dire_heroes)
        batch = pre_fetch_batch(patch_id, [1], radiant_team_id, dire_team_id, ctx)
        fv = build_feature_vector(1, ctx, patch_id, batch, schema, {}, schema["max_hero_id"])
        num_continuous = len(schema["aggregate_columns"])
        t_f = torch.tensor([fv[:num_continuous]], dtype=torch.float32)

        with torch.no_grad():
            logits = model(t_h, t_a, t_f)
            return float(torch.sigmoid(logits)[0])

    def _build_reasoning(self, hero_id, score, ctx, patch_id,
                         radiant_team_id=None, dire_team_id=None, batch=None):
        if batch:
            bl = batch.baselines.get(hero_id)
            th = batch.team_hero_agg.get(hero_id)
            sy = batch.synergy.get(hero_id)
            co = batch.counter.get(hero_id)
            h2h = batch.h2h_row
            bl_wr = float(bl["win_rate"]) if bl else None
            th_wr = float(th["win_rate"]) if th else None
            sy_wr = float(sy[0]) if sy else None
            co_wr = float(co[0]) if co else None
            h2h_wr = float(h2h["win_rate"]) if h2h else None
        else:
            team_id = radiant_team_id if ctx.recommending_team == 0 else dire_team_id
            bl = db_.fetch_baseline(patch_id, hero_id)
            th = db_.fetch_team_hero_agg(patch_id, team_id, hero_id) if team_id else None
            bl_wr = bl["win_rate"] if bl else None
            th_wr = th["win_rate"] if th else None
            sy_wr, _ = db_.fetch_synergy_avg(patch_id, hero_id, ctx.ally_picks)
            co_wr, _, _ = db_.fetch_counter_avg(patch_id, hero_id, ctx.enemy_picks)
            h2h_wr = None

        return generate_reasoning(
            hero_id=hero_id, score=score, ctx=ctx,
            baseline_win_rate=bl_wr, team_hero_win_rate=th_wr,
            synergy_win_rate=sy_wr if sy_wr and sy_wr != 0.5 else None,
            counter_win_rate=co_wr if co_wr and co_wr != 0.5 else None,
            h2h_win_rate=h2h_wr,
        )
