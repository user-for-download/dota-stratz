"""PyTorch training loop for MultiModal DraftBERT.

Trains the transformer model, exports to TorchScript for fast CPU inference,
and writes schema/metadata files for the API to consume.
"""

import copy
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training DraftBERT on device: %s (%d threads)", device, cfg.num_threads)

    # 1. Load Data
    train_ds, val_ds, metadata = load_sequence_dataset(cfg, engine, max_len=cfg.max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    num_continuous = metadata["n_continuous_features"]
    logger.info("Sequence max_len=%d, Continuous features=%d", cfg.max_seq_len, num_continuous)

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

    # 3. Training Loop
    best_val_loss = float("inf")
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        # Training
        model.train()
        train_loss = 0.0
        for heroes, actions, tabular, labels in train_loader:
            heroes, actions = heroes.to(device), actions.to(device)
            tabular, labels = tabular.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(heroes, actions, tabular)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for heroes, actions, tabular, labels in val_loader:
                heroes, actions = heroes.to(device), actions.to(device)
                tabular, labels = tabular.to(device), labels.to(device)
                logits = model(heroes, actions, tabular)
                loss = criterion(logits, labels)
                val_loss += loss.item()

        avg_train = train_loss / max(len(train_loader), 1)
        avg_val = val_loss / max(len(val_loader), 1)

        scheduler.step(avg_val)
        current_lr = optimizer.param_groups[0]['lr']

        logger.info(
            "Epoch %02d/%02d | LR: %.2e | Train Loss: %.4f | Val Loss: %.4f",
            epoch + 1, cfg.epochs, current_lr, avg_train, avg_val,
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val

            # Save weights (for resume)
            torch.save(model.state_dict(), model_dir / f"draftbert_weights_{cfg.patch_id}.pt")

            # Export to TorchScript on CPU for device-agnostic inference.
            # deepcopy so .cpu() on the clone doesn't orphan the optimizer's
            # parameter references (latent GPU bug — harmless on CPU but
            # breaks training if device were CUDA).
            model_eval = copy.deepcopy(model).cpu().eval()
            with torch.no_grad():
                dummy_h = torch.tensor([[5, 10, 15] + [0] * (cfg.max_seq_len - 3)], dtype=torch.long)
                dummy_a = torch.tensor([[3, 4, 1] + [0] * (cfg.max_seq_len - 3)], dtype=torch.long)
                dummy_f = torch.zeros((1, num_continuous), dtype=torch.float32)
                traced = torch.jit.trace(model_eval, (dummy_h, dummy_a, dummy_f))
                traced.save(model_dir / f"draftbert_compiled_{cfg.patch_id}.pt")

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

    # Write schema (n_embeddings=0, no SVD embeddings needed)
    write_schema(model_dir, patch_id=cfg.patch_id, max_hero_id=cfg.max_hero_id, n_embeddings=0, max_seq_len=cfg.max_seq_len)

    logger.info("PyTorch training complete & TorchScript model exported.")
    return best_val_loss
