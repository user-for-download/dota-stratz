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


def _roc_auc(labels: torch.Tensor, probs: torch.Tensor) -> float:
    """Rank-based ROC-AUC (Mann-Whitney U), no sklearn dependency.

    Returns 0.5 (uninformative) if only one class is present in the batch.
    """
    labels = labels.view(-1)
    probs = probs.view(-1)
    n_pos = int(labels.sum().item())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = torch.argsort(probs)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(probs) + 1, dtype=torch.float64)
    # average ranks for ties
    sorted_probs = probs[order]
    sorted_ranks = ranks[order]
    i = 0
    while i < len(sorted_probs):
        j = i
        while j + 1 < len(sorted_probs) and sorted_probs[j + 1] == sorted_probs[i]:
            j += 1
        if j > i:
            avg_rank = sorted_ranks[i:j + 1].mean()
            sorted_ranks[i:j + 1] = avg_rank
        i = j + 1
    ranks[order] = sorted_ranks
    sum_ranks_pos = ranks[labels.bool()].sum().item()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)

logger = logging.getLogger(__name__)


def find_learning_rate_live(cfg: TrainerConfig, engine, init_value: float = 1e-7, final_value: float = 10.0, beta: float = 0.98):
    """Runs an LR Range Test for LiveDraftBERT to empirically determine the optimal learning rate."""
    import math
    from pathlib import Path

    torch.set_num_threads(cfg.num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting LR Finder (LiveDraftBERT) on device=%s", device)

    train_ds, _, metadata = load_live_dataset(cfg, engine, max_len=cfg.max_seq_len)

    model = LiveDraftBERT(
        vocab_size=cfg.max_hero_id + 5, d_model=cfg.d_model, nhead=cfg.nhead,
        num_layers=cfg.num_layers, num_static_features=metadata["n_static_features"],
        num_dynamic_features=metadata["n_dynamic_features"], max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout, transformer_dropout=cfg.transformer_dropout,
        static_hidden=cfg.static_hidden, dynamic_hidden=cfg.dynamic_hidden,
        fusion_hidden=cfg.fusion_hidden,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=init_value, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    n_train = len(train_ds)
    batch_size = cfg.batch_size
    num_batches = n_train // batch_size
    if num_batches == 0:
        logger.error("Dataset too small for LR finder (need at least one full batch).")
        return

    num_steps = min(num_batches, 300)
    mult = (final_value / init_value) ** (1 / num_steps)

    lr = init_value
    avg_loss = 0.0
    best_loss = 0.0
    batch_num = 0
    losses, lrs = [], []

    model.train()
    indices = torch.randperm(n_train)

    logger.info("Running LR Finder for %d steps (batch_size=%d)...", num_steps, batch_size)
    for i in range(num_steps):
        batch_num += 1
        start = i * batch_size
        end = start + batch_size

        heroes = train_ds.heroes[indices[start:end]].to(device)
        actions = train_ds.actions[indices[start:end]].to(device)
        static = train_ds.static[indices[start:end]].to(device)
        dynamic = train_ds.dynamic[indices[start:end]].to(device)
        patches = train_ds.patches[indices[start:end]].to(device)
        labels = train_ds.labels[indices[start:end]].to(device)

        optimizer.zero_grad()
        logits = model(heroes, actions, static, dynamic, patches)
        loss = criterion(logits, labels)

        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)

        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            logger.info("Loss diverged at LR=%.2e. Stopping early.", lr)
            break

        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss

        losses.append(smoothed_loss)
        lrs.append(lr)

        loss.backward()
        optimizer.step()

        lr *= mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    out_file = model_dir / f"lr_find_livedraftbert_patch_{cfg.patch_id}.csv"

    with open(out_file, "w") as f:
        f.write("learning_rate,loss\n")
        for r, l in zip(lrs, losses):
            f.write(f"{r:.8e},{l:.6f}\n")

    logger.info("LR Finder complete! Results saved to %s", out_file)
    logger.info("--> Look for the steepest downward slope in the CSV to find your optimal LR.")


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("LiveDraftBERT: using device=%s", device)
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
    best_val_auc = 0.5
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
            heroes = train_ds.heroes[idx].to(device)
            actions = train_ds.actions[idx].to(device)
            static = train_ds.static[idx].to(device)
            dynamic = train_ds.dynamic[idx].to(device)
            patches = train_ds.patches[idx].to(device)
            labels = train_ds.labels[idx].to(device)

            optimizer.zero_grad()
            logits = model(heroes, actions, static, dynamic, patches)
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
        val_probs = []
        val_labels = []

        with torch.no_grad():
            val_idx = torch.arange(n_val)
            for start in range(0, n_val, batch_size):
                idx = val_idx[start:start + batch_size]
                heroes = val_ds.heroes[idx].to(device)
                actions = val_ds.actions[idx].to(device)
                static = val_ds.static[idx].to(device)
                dynamic = val_ds.dynamic[idx].to(device)
                patches = val_ds.patches[idx].to(device)
                labels = val_ds.labels[idx].to(device)

                logits = model(heroes, actions, static, dynamic, patches)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                val_n += len(labels)
                probs = torch.sigmoid(logits)
                val_correct += (probs > 0.5).float().eq(labels).sum().item()
                val_probs.append(probs.detach().cpu())
                val_labels.append(labels.detach().cpu())

        val_loss = val_loss / val_n if val_n > 0 else float("inf")
        val_acc = val_correct / val_n if val_n > 0 else 0.0
        val_auc = _roc_auc(torch.cat(val_labels), torch.cat(val_probs)) if val_n > 0 else 0.5
        scheduler.step(val_loss)

        logger.info(
            "Epoch %2d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.3f | val_auc=%.3f | lr=%.2e",
            epoch + 1, cfg.epochs, train_loss, val_loss, val_acc, val_auc, optimizer.param_groups[0]["lr"],
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_auc = val_auc
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
        "val_loss": best_val_loss, "val_auc": best_val_auc, "val_acc": best_val_acc,
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
    conn = None
    try:
        if engine is not None:
            try:
                conn = engine.raw_connection()
            except (AttributeError, TypeError):
                import psycopg2
                conn = psycopg2.connect(cfg.pg_dsn)
        else:
            import psycopg2
            conn = psycopg2.connect(cfg.pg_dsn)

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
                 meta.get("val_auc"), meta.get("val_loss"),
                 meta["n_train_matches"] + meta["n_val_matches"],
                 meta["n_train_samples"] + meta["n_val_samples"]),
            )
        conn.commit()
    except Exception as e:
        logger.warning("Could not save model metadata to DB: %s", e)
    finally:
        if conn is not None:
            conn.close()
