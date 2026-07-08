"""Dataset for LiveDraftBERT training.

Each sample contains:
- heroes: draft hero IDs (same for all minutes of a match)
- actions: action tokens (same for all minutes)
- static_features: 61 pre-game aggregates (same for all minutes)
- dynamic_features: 35 live game state features (CHANGES per minute)
- label: who won (same for all minutes)

Multiple samples per match (one per minute), sharing the same draft
and static features but with different dynamic features.

Supports both map-style (Dataset) and streaming (IterableDataset) loading.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .config import TrainerConfig
from .features import feature_column_names, training_features_sql, training_features_sql_fast
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

    # 1. Draft sequences + static features (use fast aggregate SQL)
    from sqlalchemy import text
    import time
    t0 = time.time()
    sql = training_features_sql_fast(_match_extra_where(cfg))
    with engine.connect() as conn:
        result = conn.execute(text(sql), {"patch_id": cfg.patch_id})
        columns = result.keys()
        rows = result.fetchall()
    logger.info("Read %d rows in %.1fs", len(rows), time.time() - t0)
    draft_df = pd.DataFrame(rows, columns=columns)

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


class StreamingLiveDataset(IterableDataset):
    """Memory-efficient streaming dataset for large live training sets.

    Reads data in chunks from the database using server-side cursors,
    yielding one sample at a time without loading the full dataset into memory.
    """

    def __init__(self, engine, match_ids: set[int], patch_id: int,
                 lookback: int = 2, max_len: int = 50, chunk_size: int = 1000,
                 cfg: TrainerConfig = None):
        self.engine = engine
        self.match_ids = match_ids
        self.patch_id = patch_id
        self.lookback = lookback
        self.max_len = max_len
        self.chunk_size = chunk_size
        self.cfg = cfg  # Pass actual config to preserve filters

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is not None:
            # Split match_ids across workers, last worker gets remainder
            per_worker = len(self.match_ids) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = len(self.match_ids) if worker_id == worker_info.num_workers - 1 else start + per_worker
            match_list = sorted(self.match_ids)[start:end]
        else:
            match_list = sorted(self.match_ids)

        if not match_list:
            return

        # Stream draft data in chunks using server-side cursor
        conn = self.engine.raw_connection()
        try:
            conn.cursor_factory = None  # Use default cursor for server-side
            with conn.cursor(name='live_draft_cursor') as cur:
                cur.itersize = self.chunk_size
                agg_cols = feature_column_names(include_onehot=False)
                sql = training_features_sql(_match_extra_where(self.cfg), lookback=self.lookback)
                cur.execute(sql, {"patch_id": self.patch_id})
                col_names = [desc[0] for desc in cur.description]

                # Build draft lookup from streaming rows
                draft_lookup = {}
                for row in cur:
                    row_dict = dict(zip(col_names, row))
                    mid = row_dict["match_id"]
                    if mid not in match_list:
                        continue
                    if mid not in draft_lookup:
                        draft_lookup[mid] = {
                            "heroes": [], "actions": [], "static": None, "label": None
                        }
                    dl = draft_lookup[mid]
                    dl["heroes"].append(int(row_dict["hero_id"]))
                    dl["actions"].append(int(row_dict["team"]) * 1 + int(row_dict["is_pick"]) * 2 + 1)
                    dl["static"] = np.array([float(row_dict.get(c, 0) or 0) for c in agg_cols], dtype=np.float32)
                    dl["label"] = float(row_dict["radiant_win"])

            # Extract dynamic features
            dynamic_df = extract_dynamic_features(self.engine, self.patch_id, self.lookback)
            dyn_matrix = dynamic_df[DYNAMIC_FEATURE_COLUMNS].values.astype(np.float32)
            dyn_mids = dynamic_df["match_id"].values
            dyn_minutes = dynamic_df["minute"].values

            # Yield samples one at a time
            for mid in match_list:
                if mid not in draft_lookup:
                    continue
                dl = draft_lookup[mid]
                mh = dl["heroes"]
                ma = dl["actions"]
                mt = dl["static"]
                ml = dl["label"]

                # Find matching dynamic features
                mask = (dyn_mids == mid) & (dyn_minutes > 0)
                indices = np.where(mask)[0]

                for idx in indices:
                    dyn = dyn_matrix[idx]
                    h_padded = mh[:self.max_len] + [0] * max(0, self.max_len - len(mh))
                    a_padded = ma[:self.max_len] + [0] * max(0, self.max_len - len(ma))
                    yield (
                        torch.tensor(h_padded, dtype=torch.long),
                        torch.tensor(a_padded, dtype=torch.long),
                        torch.tensor(mt, dtype=torch.float32),
                        torch.tensor(dyn, dtype=torch.float32),
                        torch.tensor(ml, dtype=torch.float32),
                    )
        finally:
            conn.close()
