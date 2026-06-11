"""CLI entry point for the ML Training Service.

Usage:

    # Train for a specific patch
    python -m trainer.main --patch 134

    # Train for latest patch (auto-detect)
    python -m trainer.main

    # Populate aggregates only (no training)
    python -m trainer.main --patch 134 --agg-only

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
from .train import train_model

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
        logger.info("Step 1: Populating aggregate tables ...")
        counts = populate_all(cfg, conn)
        for name, cnt in counts.items():
            logger.info("  %s: %d rows", name, cnt)

        if args.agg_only:
            logger.info("Aggregate population complete (--agg-only). Skipping training.")
            return 0

        # Step 2: Train model
        logger.info("Step 2: Training LightGBM binary classification model ...")
        model, best_loss = train_model(cfg, eng)
        logger.info(
            "Training complete. Patch %d | binary_logloss: %.4f",
            patch_id, best_loss,
        )

    except Exception:
        logger.exception("Training pipeline failed")
        return 1
    finally:
        conn.close()

    logger.info("All done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
