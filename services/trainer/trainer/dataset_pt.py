"""Dataset construction for PyTorch DraftBERT with Multi-Modal tabular features.

Uses Prefix Augmentation and pre-tensorizes the entire dataset at init time
to eliminate Python GIL overhead during training.
"""

import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import TrainerConfig
from .features import training_features_sql, training_features_sql_fast, feature_column_names, make_target
from .aggregates import _match_extra_where

logger = logging.getLogger(__name__)


class DraftSequenceDataset(Dataset):
    """Pre-tensorized dataset for zero-overhead __getitem__ lookups."""

    def __init__(self, heroes_seqs, actions_seqs, tabular_feats, labels, max_len=50):
        n = len(labels)

        # Pad sequences using NumPy (much faster than torch.tensor in loop)
        def pad_sequence(seq, max_len):
            return seq[:max_len] + [0] * max(0, max_len - len(seq))

        h_padded = [pad_sequence(h, max_len) for h in heroes_seqs]
        a_padded = [pad_sequence(a, max_len) for a in actions_seqs]

        # Instantiate tensor memory exactly once
        self.heroes = torch.from_numpy(np.array(h_padded, dtype=np.int64))
        self.actions = torch.from_numpy(np.array(a_padded, dtype=np.int64))
        self.tabular = torch.from_numpy(np.array(tabular_feats, dtype=np.float32))
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.heroes[idx], self.actions[idx], self.tabular[idx], self.labels[idx]


def _build_augmented_data(df, agg_cols):
    """Prefix Augmentation: 1 match → N sequences (one per draft step).

    Action tokens: 1=RadBan, 2=DireBan, 3=RadPick, 4=DirePick.
    """
    heroes_seqs, actions_seqs, tabular_feats, labels = [], [], [], []

    for _, group in df.groupby("match_id", sort=False):
        mh = group["hero_id"].astype(int).tolist()
        ma = (group["team"].astype(int) * 1 + group["is_pick"].astype(int) * 2 + 1).tolist()
        mt = group[agg_cols].to_numpy(dtype=np.float32)
        ml = make_target(group).tolist()

        for i in range(1, len(group) + 1):
            heroes_seqs.append(mh[:i])
            actions_seqs.append(ma[:i])
            tabular_feats.append(mt[i - 1])
            labels.append(ml[i - 1])

    return heroes_seqs, actions_seqs, tabular_feats, labels


def load_sequence_dataset(cfg: TrainerConfig, engine, max_len: int = 50):
    """Load, clean, split, and augment sequence data for DraftBERT.

    Uses fast aggregate tables (no LATERAL joins) for speed.
    """
    logger.info("Loading training data from DB for patch %s ...", cfg.patch_id)

    # Use fast SQL with aggregate tables (no LATERAL joins)
    sql = training_features_sql_fast(_match_extra_where(cfg))
    logger.info("Reading training data from aggregate tables...")
    import time
    t0 = time.time()
    from sqlalchemy import text
    with engine.connect() as conn:
        result = conn.execute(text(sql), {"patch_id": cfg.patch_id})
        columns = result.keys()
        rows = result.fetchall()
    logger.info("Read %d rows in %.1fs", len(rows), time.time() - t0)

    import pandas as pd
    df = pd.DataFrame(rows, columns=columns)

    if df.empty:
        raise ValueError(f"No training data found for patch {cfg.patch_id}.")

    match_sizes = df.groupby("match_id").size()
    valid_matches = match_sizes[match_sizes >= cfg.min_matches_per_group].index
    df = df[df["match_id"].isin(valid_matches)]

    if df.empty:
        raise ValueError("No matches meet min_matches_per_group filter.")

    df = df.sort_values(["match_id", "order"]).reset_index(drop=True)

    agg_cols = feature_column_names(include_onehot=False)
    fill_cols = [c for c in agg_cols if c in df.columns]
    df[fill_cols] = df[fill_cols].fillna(0)

    # Chronological split
    match_start_times = df.groupby("match_id")["start_time"].first().sort_values()
    match_ids_sorted = match_start_times.index
    n_train = int(len(match_ids_sorted) * (1 - cfg.val_ratio))

    train_df = df[df["match_id"].isin(match_ids_sorted[:n_train])]
    val_df = df[df["match_id"].isin(match_ids_sorted[n_train:])]

    logger.info("Split: %d train, %d val matches",
                len(match_ids_sorted[:n_train]), len(match_ids_sorted[n_train:]))

    t0 = time.time()
    t_h, t_a, t_t, t_l = _build_augmented_data(train_df, agg_cols)
    v_h, v_a, v_t, v_l = _build_augmented_data(val_df, agg_cols)
    logger.info("Augmentation done in %.0fs — train: %d seqs, val: %d seqs",
                time.time() - t0, len(t_l), len(v_l))

    train_ds = DraftSequenceDataset(t_h, t_a, t_t, t_l, max_len)
    val_ds = DraftSequenceDataset(v_h, v_a, v_t, v_l, max_len)

    metadata = {
        "n_train_matches": len(match_ids_sorted[:n_train]),
        "n_val_matches": len(match_ids_sorted[n_train:]),
        "n_train_sequences": len(t_l),
        "n_val_sequences": len(v_l),
        "n_continuous_features": len(agg_cols),
    }

    return train_ds, val_ds, metadata
