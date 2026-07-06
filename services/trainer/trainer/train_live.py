"""Training script for LiveDraftBERT (live match prediction model).

Uses direct tensor slicing for maximum CPU utilization.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn

from .config import TrainerConfig
from .dataset_live import load_live_dataset
from .features import write_schema
from .live_features import DYNAMIC_FEATURE_COLUMNS
from .model_live import LiveDraftBERT

logger = logging.getLogger(__name__)


def train_live_model(cfg: TrainerConfig, engine) -> float:
    torch.set_num_threads(cfg.num_threads)
    patch_id = cfg.patch_id
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset
    logger.info("Loading LiveDraftBERT training data ...")
    train_ds, val_ds, metadata = load_live_dataset(cfg, engine, max_len=cfg.max_seq_len)

    logger.info(
        "Dataset: %d static + %d dynamic features | %d train / %d val samples",
        metadata["n_static_features"], metadata["n_dynamic_features"],
        metadata["n_train_samples"], metadata["n_val_samples"],
    )

    # 2. Create model
    device = torch.device("cpu")
    model = LiveDraftBERT(
        vocab_size=cfg.max_hero_id + 5, d_model=cfg.d_model, nhead=cfg.nhead,
        num_layers=cfg.num_layers, num_static_features=metadata["n_static_features"],
        num_dynamic_features=metadata["n_dynamic_features"], max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout, transformer_dropout=cfg.transformer_dropout,
        static_hidden=cfg.static_hidden, dynamic_hidden=cfg.dynamic_hidden,
        fusion_hidden=cfg.fusion_hidden,
    ).to(device)

    logger.info("LiveDraftBERT: %d parameters", sum(p.numel() for p in model.parameters()))

    # 3. Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=cfg.lr_scheduler_patience, factor=cfg.lr_scheduler_factor,
    )

    # 4. Training loop — direct tensor slicing
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    n_train = len(train_ds)
    n_val = len(val_ds)
    batch_size = min(cfg.batch_size, n_train)

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        train_n = 0
        indices = torch.randperm(n_train)
        n_batches = (n_train + batch_size - 1) // batch_size
        logger.info("Epoch %d/%d starting (%d samples, %d batches, batch_size=%d)",
                    epoch + 1, cfg.epochs, n_train, n_batches, batch_size)

        for batch_i, start in enumerate(range(0, n_train, batch_size)):
            idx = indices[start:start + batch_size]
            heroes = train_ds.heroes[idx]
            actions = train_ds.actions[idx]
            static = train_ds.static[idx]
            dynamic = train_ds.dynamic[idx]
            labels = train_ds.labels[idx]

            optimizer.zero_grad()
            logits = model(heroes, actions, static, dynamic)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            train_loss += loss.item() * len(labels)
            train_n += len(labels)

            if (batch_i + 1) % cfg.log_interval == 0:
                logger.info("  batch %d/%d | samples %d/%d | loss=%.4f",
                            batch_i + 1, n_batches, train_n, n_train, train_loss / train_n)

        train_loss /= train_n

        model.eval()
        val_loss = 0.0
        val_n = 0
        val_correct = 0

        with torch.no_grad():
            val_idx = torch.arange(n_val)
            for start in range(0, n_val, batch_size):
                idx = val_idx[start:start + batch_size]
                heroes = val_ds.heroes[idx]
                actions = val_ds.actions[idx]
                static = val_ds.static[idx]
                dynamic = val_ds.dynamic[idx]
                labels = val_ds.labels[idx]

                logits = model(heroes, actions, static, dynamic)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                val_n += len(labels)
                val_correct += (torch.sigmoid(logits) > 0.5).float().eq(labels).sum().item()

        val_loss = val_loss / val_n if val_n > 0 else float("inf")
        val_acc = val_correct / val_n if val_n > 0 else 0.0
        scheduler.step(val_loss)

        logger.info(
            "Epoch %2d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.3f | lr=%.2e",
            epoch + 1, cfg.epochs, train_loss, val_loss, val_acc, optimizer.param_groups[0]["lr"],
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0

            weights_path = model_dir / f"draftbert_live_weights_{patch_id}.pt"
            torch.save(model.state_dict(), weights_path)
            logger.info("Saved: val_loss=%.4f -> %s", best_val_loss, weights_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    meta = {
        "patch_id": patch_id,
        "weights_filename": f"draftbert_live_weights_{patch_id}.pt",
        "n_static_features": metadata["n_static_features"],
        "n_dynamic_features": metadata["n_dynamic_features"],
        "dynamic_feature_columns": DYNAMIC_FEATURE_COLUMNS,
        "val_loss": best_val_loss, "val_acc": best_val_acc,
        "n_train_matches": metadata["n_train_matches"],
        "n_val_matches": metadata["n_val_matches"],
        "n_train_samples": metadata["n_train_samples"],
        "n_val_samples": metadata["n_val_samples"],
        "model_params": sum(p.numel() for p in model.parameters()),
        "max_seq_len": cfg.max_seq_len, "max_hero_id": cfg.max_hero_id,
        "d_model": cfg.d_model, "nhead": cfg.nhead, "num_layers": cfg.num_layers,
        "dropout": cfg.dropout, "transformer_dropout": cfg.transformer_dropout,
        "static_hidden": cfg.static_hidden, "dynamic_hidden": cfg.dynamic_hidden,
        "fusion_hidden": cfg.fusion_hidden,
    }

    with open(model_dir / f"live_model_patch_{patch_id}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    write_schema(model_dir, patch_id, cfg.max_hero_id, n_embeddings=0, max_seq_len=cfg.max_seq_len)
    _upsert_model_meta(cfg, patch_id, meta, engine)

    logger.info("Training complete. Best val_loss: %.4f", best_val_loss)
    return best_val_loss


def _upsert_model_meta(cfg: TrainerConfig, patch_id: int, meta: dict, engine=None):
    if engine is not None:
        conn = engine.raw_connection()
    else:
        import psycopg2
        conn = psycopg2.connect(cfg.pg_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO live_prediction_models
                    (patch_id, model_filename, weights_filename, feature_columns,
                     val_auc, val_logloss, n_matches, n_samples)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (patch_id) DO UPDATE SET
                    model_filename = EXCLUDED.model_filename,
                    weights_filename = EXCLUDED.weights_filename,
                    feature_columns = EXCLUDED.feature_columns,
                    val_auc = EXCLUDED.val_auc, val_logloss = EXCLUDED.val_logloss,
                    n_matches = EXCLUDED.n_matches, n_samples = EXCLUDED.n_samples,
                    trained_at = NOW()""",
                (patch_id, meta["weights_filename"], meta["weights_filename"],
                 json.dumps(meta["dynamic_feature_columns"]),
                 meta.get("val_acc"), meta.get("val_loss"),
                 meta["n_train_matches"] + meta["n_val_matches"],
                 meta["n_train_samples"] + meta["n_val_samples"]),
            )
        conn.commit()
    except Exception as e:
        logger.warning("Could not save model metadata to DB: %s", e)
    finally:
        conn.close()
