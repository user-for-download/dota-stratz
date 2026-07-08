"""PyTorch training loop for MultiModal DraftBERT.

Trains the transformer model, exports to TorchScript for fast CPU inference,
and writes schema/metadata files for the API to consume.
"""

import copy
import json
import logging
import time
import uuid
from pathlib import Path

import torch
import torch.nn as nn

from .config import TrainerConfig
from .dataset_pt import load_sequence_dataset
from .model_pt import MultiModalDraftBERT
from .features import feature_column_names, write_schema

logger = logging.getLogger(__name__)


def train_pytorch_model(cfg: TrainerConfig, engine) -> float:
    """Train DraftBERT and export TorchScript model.

    Returns best validation loss.
    """
    torch.set_num_threads(cfg.num_threads)
    torch.set_flush_denormal(True)  # Prevents CPU slowdowns from denormal floats
    run_id = str(uuid.uuid4())[:8]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training DraftBERT run=%s device=%s threads=%d", run_id, device, cfg.num_threads)

    # 1. Load Data
    train_ds, val_ds, metadata = load_sequence_dataset(cfg, engine, max_len=cfg.max_seq_len)

    # --- Standardize Continuous Features ---
    logger.info("Normalizing tabular features (Mean/Std Scaling)...")
    t0 = time.time()
    feature_means = train_ds.tabular.mean(dim=0)
    feature_stds = train_ds.tabular.std(dim=0).clamp(min=1e-6)

    train_ds.tabular = (train_ds.tabular - feature_means) / feature_stds
    val_ds.tabular = (val_ds.tabular - feature_means) / feature_stds

    feature_means_list = feature_means.numpy().tolist()
    feature_stds_list = feature_stds.numpy().tolist()
    logger.info("Normalization done in %.1fs", time.time() - t0)

    num_continuous = metadata["n_continuous_features"]
    n_train = len(train_ds)
    n_val = len(val_ds)
    batch_size = cfg.batch_size
    logger.info("Train: %d samples, Val: %d samples, Features: %d", n_train, n_val, num_continuous)

    # 2. Init Model
    model = MultiModalDraftBERT(
        vocab_size=cfg.max_hero_id + 5,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        num_continuous_features=num_continuous,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout,
        transformer_dropout=cfg.transformer_dropout,
        fusion_hidden=cfg.fusion_hidden,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg.lr_scheduler_factor, patience=cfg.lr_scheduler_patience,
    )

    # 3. Training Loop (vectorized — direct tensor slicing, no DataLoader)
    best_val_loss = float("inf")
    patience_counter = 0
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        t_epoch = time.time()
        # Training
        model.train()
        train_loss = 0.0
        train_samples = 0

        # Shuffle once per epoch for cache-friendly sequential access
        indices = torch.randperm(n_train)
        shuffled_heroes = train_ds.heroes[indices]
        shuffled_actions = train_ds.actions[indices]
        shuffled_tabular = train_ds.tabular[indices]
        shuffled_labels = train_ds.labels[indices]

        for start in range(0, n_train, batch_size):
            end = start + batch_size
            heroes = shuffled_heroes[start:end].to(device)
            actions = shuffled_actions[start:end].to(device)
            tabular = shuffled_tabular[start:end].to(device)
            labels = shuffled_labels[start:end].to(device)

            optimizer.zero_grad()
            logits = model(heroes, actions, tabular)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            train_loss += loss.item() * len(labels)
            train_samples += len(labels)

        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for start in range(0, n_val, batch_size):
                end = start + batch_size
                heroes = val_ds.heroes[start:end].to(device)
                actions = val_ds.actions[start:end].to(device)
                tabular = val_ds.tabular[start:end].to(device)
                labels = val_ds.labels[start:end].to(device)
                logits = model(heroes, actions, tabular)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                val_samples += len(labels)

        avg_train = train_loss / max(train_samples, 1)
        avg_val = val_loss / max(val_samples, 1)
        scheduler.step(avg_val)
        current_lr = optimizer.param_groups[0]['lr']

        logger.info(
            "Epoch %02d/%02d | LR: %.2e | Train Loss: %.4f | Val Loss: %.4f | %.0fs",
            epoch + 1, cfg.epochs, current_lr, avg_train, avg_val, time.time() - t_epoch,
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0

            torch.save(model.state_dict(), model_dir / f"draftbert_w_{cfg.patch_id}_{run_id}.pt")

            model_eval = copy.deepcopy(model).cpu().eval()
            with torch.no_grad():
                dummy_h = torch.tensor([[5, 10, 15] + [0] * (cfg.max_seq_len - 3)], dtype=torch.long)
                dummy_a = torch.tensor([[3, 4, 1] + [0] * (cfg.max_seq_len - 3)], dtype=torch.long)
                dummy_f = torch.zeros((1, num_continuous), dtype=torch.float32)
                traced = torch.jit.trace(model_eval, (dummy_h, dummy_a, dummy_f))
                traced.save(model_dir / f"draftbert_compiled_{cfg.patch_id}_{run_id}.pt")

            torch.save(model.state_dict(), model_dir / f"draftbert_weights_{cfg.patch_id}.pt")
            traced.save(model_dir / f"draftbert_compiled_{cfg.patch_id}.pt")
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    logger.info("Best Val Loss: %.4f", best_val_loss)

    # 4. Save Metadata
    meta = {
        "patch_id": int(cfg.patch_id),
        "best_val_loss": best_val_loss,
        "n_train_sequences": metadata["n_train_sequences"],
        "n_val_sequences": metadata["n_val_sequences"],
        "n_continuous_features": num_continuous,
        "model_type": "draftbert_pytorch",
    }
    (model_dir / f"model_patch_{cfg.patch_id}_meta.json").write_text(json.dumps(meta, indent=2))

    write_schema(model_dir, patch_id=cfg.patch_id, max_hero_id=cfg.max_hero_id,
                 n_embeddings=0, max_seq_len=cfg.max_seq_len,
                 drift_stats={"mean": feature_means_list, "std": feature_stds_list})

    _log_experiment(cfg, run_id, best_val_loss, engine)
    logger.info("Training complete run=%s — loss %.4f", run_id, best_val_loss)
    return best_val_loss


def _log_experiment(cfg: TrainerConfig, run_id: str, val_loss: float, engine):
    try:
        conn = engine.raw_connection()
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ml.experiment_logs (
                    run_id VARCHAR PRIMARY KEY, patch_id INT, val_loss FLOAT,
                    hyperparameters JSONB, created_at TIMESTAMPTZ DEFAULT NOW()
                )""")
            cur.execute("""
                INSERT INTO ml.experiment_logs (run_id, patch_id, val_loss, hyperparameters)
                VALUES (%s, %s, %s, %s)""",
                (run_id, int(cfg.patch_id), val_loss, json.dumps({
                    "d_model": cfg.d_model, "nhead": cfg.nhead,
                    "num_layers": cfg.num_layers, "dropout": cfg.dropout,
                    "lr": cfg.lr, "batch_size": cfg.batch_size,
                    "epochs": cfg.epochs, "max_seq_len": cfg.max_seq_len,
                })))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("Could not log experiment: %s", e)
