"""Dataset for LiveDraftBERT training.

Each sample contains:
- heroes: draft hero IDs (same for all minutes of a match)
- actions: action tokens (same for all minutes)
- static_features: 59 pre-game aggregates (same for all minutes)
- dynamic_features: 15 live game state features (CHANGES per minute)
- label: who won (same for all minutes)

Multiple samples per match (one per minute), sharing the same draft
and static features but with different dynamic features.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import TrainerConfig
from .features import feature_column_names, training_features_sql
from .aggregates import _match_extra_where
from .live_features import (
    DYNAMIC_FEATURE_COLUMNS,
    TARGET_COLUMN,
    extract_dynamic_features,
)

logger = logging.getLogger(__name__)


class LiveDraftDataset(Dataset):
    """Pre-tensorized dataset for LiveDraftBERT with 4-tuple inputs."""

    def __init__(
        self,
        heroes_seqs,
        actions_seqs,
        static_feats,
        dynamic_feats,
        labels,
        max_len=50,
    ):
        n = len(labels)

        # Pad sequences using NumPy (much faster than torch.tensor in loop)
        def pad_sequence(seq, max_len):
            return seq[:max_len] + [0] * max(0, max_len - len(seq))

        h_padded = [pad_sequence(h, max_len) for h in heroes_seqs]
        a_padded = [pad_sequence(a, max_len) for a in actions_seqs]

        # Instantiate tensor memory exactly once
        self.heroes = torch.from_numpy(np.array(h_padded, dtype=np.int64))
        self.actions = torch.from_numpy(np.array(a_padded, dtype=np.int64))
        self.static = torch.from_numpy(np.array(static_feats, dtype=np.float32))
        self.dynamic = torch.from_numpy(np.array(dynamic_feats, dtype=np.float32))
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.heroes[idx],
            self.actions[idx],
            self.static[idx],
            self.dynamic[idx],
            self.labels[idx],
        )


def load_live_dataset(cfg: TrainerConfig, engine, max_len: int = 50):
    """Load and prepare LiveDraftBERT training data.

    1. Extract draft sequences + static features (existing pipeline)
    2. Extract per-minute dynamic features (new pipeline)
    3. Merge: each (match_id, minute) gets draft + static + dynamic
    4. Chronological train/val split
    5. Return LiveDraftDataset objects
    """
    logger.info("Loading live training data for patch %s ...", cfg.patch_id)

    # 1. Draft sequences + static features
    draft_df = pd.read_sql(
        training_features_sql(_match_extra_where(cfg), lookback=cfg.lookback_patches),
        engine,
        params={"patch_id": cfg.patch_id},
    )

    if draft_df.empty:
        raise ValueError(f"No draft data found for patch {cfg.patch_id}")

    # Use only the final draft step per match (full draft completed)
    draft_df = draft_df.sort_values(["match_id", "order"])
    final_steps = draft_df.groupby("match_id").last().reset_index()

    agg_cols = feature_column_names(include_onehot=False)
    fill_cols = [c for c in agg_cols if c in final_steps.columns]
    final_steps[fill_cols] = final_steps[fill_cols].fillna(0)

    logger.info("Draft data: %d matches", len(final_steps))

    # 2. Per-minute dynamic features
    dynamic_df = extract_dynamic_features(engine, cfg.patch_id, cfg.lookback_patches)

    # 3. Build draft lookup: match_id → (hero_ids, actions, static_row)
    # Use final_steps for static features (fully completed draft synergies/counters)
    final_lookup = final_steps.set_index("match_id")
    draft_lookup = {}
    for mid, group in draft_df.groupby("match_id", sort=False):
        mh = group["hero_id"].astype(int).tolist()
        ma = (group["team"].astype(int) * 1 + group["is_pick"].astype(int) * 2 + 1).tolist()
        mt = final_lookup.loc[mid, fill_cols].values.astype(np.float32)
        ml = final_lookup.loc[mid, "radiant_win"]
        draft_lookup[mid] = (mh, ma, mt, ml)

    # 4. Chronological split by match start time (use draft_df start_time, not game minute)
    match_start = draft_df.groupby("match_id")["start_time"].first().sort_values()
    match_ids_sorted = match_start.index
    n_train = int(len(match_ids_sorted) * (1 - cfg.val_ratio))
    train_matches = set(match_ids_sorted[:n_train])
    val_matches = set(match_ids_sorted[n_train:])

    logger.info("Split: %d train, %d val matches", len(train_matches), len(val_matches))

    # 5. Build train/val lists
    t_h, t_a, t_s, t_d, t_l = [], [], [], [], []
    v_h, v_a, v_s, v_d, v_l = [], [], [], [], []

    # --- FAST NUMPY EXTRACTION (bypasses Pandas iterrows overhead) ---
    mids = dynamic_df["match_id"].values
    minutes_arr = dynamic_df["minute"].values
    dyn_matrix = dynamic_df[DYNAMIC_FEATURE_COLUMNS].values.astype(np.float32)

    # Vectorized filtering: skip minute 0 and missing draft_lookup entries
    # Skip minute 0 — no dynamic game state yet (gold/xp advantages are 0)
    valid_mask = (minutes_arr != 0) & np.array([mid in draft_lookup for mid in mids])
    valid_indices = np.where(valid_mask)[0]

    for i in valid_indices:
        mid = mids[i]
        mh, ma, mt, ml = draft_lookup[mid]
        dyn = dyn_matrix[i]

        if mid in train_matches:
            t_h.append(mh)
            t_a.append(ma)
            t_s.append(mt)
            t_d.append(dyn)
            t_l.append(float(ml))
        elif mid in val_matches:
            v_h.append(mh)
            v_a.append(ma)
            v_s.append(mt)
            v_d.append(dyn)
            v_l.append(float(ml))

    if not t_l:
        raise ValueError("No training samples after merge")

    train_ds = LiveDraftDataset(t_h, t_a, t_s, t_d, t_l, max_len)
    val_ds = LiveDraftDataset(v_h, v_a, v_s, v_d, v_l, max_len)

    metadata = {
        "n_train_matches": len(train_matches),
        "n_val_matches": len(val_matches),
        "n_train_samples": len(t_l),
        "n_val_samples": len(v_l),
        "n_static_features": len(agg_cols),
        "n_dynamic_features": len(DYNAMIC_FEATURE_COLUMNS),
    }

    logger.info("Final: %d train / %d val samples", len(t_l), len(v_l))

    return train_ds, val_ds, metadata
