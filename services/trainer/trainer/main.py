"""CLI entry point for the ML Training Service.

Usage:

    # Train DraftBERT for a specific patch
    python -m trainer.main --patch 134

    # Train LiveDraftBERT for live match prediction
    python -m trainer.main --patch 134 --live

    # Populate aggregates only (no training)
    python -m trainer.main --patch 134 --agg-only

    # Skip aggregates and go straight to training
    python -m trainer.main --patch 134 --skip-agg

    # Override model directory
    python -m trainer.main --patch 134 --model-dir /tmp/models

All values can also be set via environment variables (see config.py).
"""

from __future__ import annotations

import argparse
import logging
import sys

from .aggregates import populate_all
from .config import TrainerConfig
from .db import connect, make_engine as create_db_engine, fetch_patch_id
from .train_pt import train_pytorch_model

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="dota-stratz ML Training Service",
    )
    parser.add_argument(
        "--patch",
        type=int,
        default=0,
        help="Patch ID to train on (default: auto-detect latest with data)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Override model output directory",
    )
    parser.add_argument(
        "--agg-only",
        action="store_true",
        help="Only populate aggregate tables, skip model training",
    )
    parser.add_argument(
        "--skip-agg",
        action="store_true",
        help="Skip populating aggregate tables (use if DB is already up to date)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Train LiveDraftBERT for live match prediction (instead of DraftBERT)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    cfg = TrainerConfig()
    if args.patch:
        cfg.patch_id = args.patch
    if args.model_dir:
        cfg.model_dir = args.model_dir

    logger.info("Starting training pipeline for patch %s", cfg.patch_id)

    # Database connection
    conn = connect(cfg)
    eng = create_db_engine(cfg)

    try:
        # Resolve patch ID
        patch_id = fetch_patch_id(cfg, conn)
        cfg.patch_id = patch_id
        logger.info("Resolved patch ID: %s", patch_id)

        # Step 1: Populate aggregate tables
        if not args.skip_agg:
            logger.info("Step 1: Populating aggregate tables ...")
            counts = populate_all(cfg, conn)
            for name, cnt in counts.items():
                logger.info("  %s: %d rows", name, cnt)

            if args.agg_only:
                logger.info("Aggregate population complete (--agg-only). Skipping training.")
                return 0
        else:
            logger.info("Step 1 Skipped (--skip-agg provided). Using existing database aggregates.")

        # Step 2: Train model
        if args.live:
            logger.info("Step 2: Training LiveDraftBERT model ...")
            from .train_live import train_live_model
            best_loss = train_live_model(cfg, eng)
            logger.info(
                "LiveDraftBERT training complete. Patch %d | val_loss: %.4f",
                patch_id, best_loss,
            )
        else:
            logger.info("Step 2: Training PyTorch DraftBERT model ...")
            best_loss = train_pytorch_model(cfg, eng)
            logger.info(
                "Training complete. Patch %d | val_loss: %.4f",
                patch_id, best_loss,
            )

    except Exception:
        logger.exception("Training pipeline failed")
        return 1
    finally:
        conn.close()
        eng.dispose()

    logger.info("All done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
