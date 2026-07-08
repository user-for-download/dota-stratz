"""Test script for the Drafting Bot components.

This script verifies that the InferenceCache, DraftStateBuilder,
GreedyDraftBot, and MCTSDraftBot work together correctly.

Usage:
    python -m trainer.test_bot
"""

import sys
import time
import logging

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def test_inference_cache():
    """Test that InferenceCache loads data correctly."""
    from trainer.db import make_engine
    from trainer.config import TrainerConfig
    from trainer.inference_cache import InferenceCache

    cfg = TrainerConfig()
    engine = make_engine(cfg)

    cache = InferenceCache(engine, patch_id=cfg.patch_id)

    # Test lookups
    assert len(cache.valid_hero_ids) > 0, "No valid heroes found!"
    logger.info("Valid heroes: %d", len(cache.valid_hero_ids))

    # Test baseline lookup
    hero_id = cache.valid_hero_ids[0]
    bl = cache.get_baseline(hero_id)
    assert "win_rate" in bl, "Baseline missing win_rate"
    logger.info("Hero %d baseline: win_rate=%.3f", hero_id, bl["win_rate"])

    # Test synergy lookup
    if len(cache.valid_hero_ids) >= 2:
        h1, h2 = cache.valid_hero_ids[0], cache.valid_hero_ids[1]
        sy = cache.get_synergy(h1, h2)
        assert "win_rate" in sy, "Synergy missing win_rate"
        logger.info("Synergy %d+%d: win_rate=%.3f, games=%d", h1, h2, sy["win_rate"], sy["games"])

    return cache


def test_draft_state_builder(cache):
    """Test that DraftStateBuilder generates correct feature arrays."""
    from trainer.draft_state import DraftStateBuilder

    builder = DraftStateBuilder(cache)
    expected = len(builder.col_idx)
    assert builder.num_features == expected, f"Expected {expected} features, got {builder.num_features}"

    hero_id = cache.valid_hero_ids[0]

    # Test building features for a pick
    feat = builder.build_tabular_features(
        hypothetical_hero_id=hero_id,
        is_radiant_turn=True,
        is_pick=True,
        radiant_picks=[],
        dire_picks=[],
    )
    assert feat.shape == (expected,), f"Expected shape ({expected},), got {feat.shape}"
    assert feat.dtype == np.float32, f"Expected float32, got {feat.dtype}"

    # Check that baseline values are populated
    idx = builder.col_idx["bl_win_rate"]
    assert feat[idx] != 0.0 or bl["win_rate"] == 0.0, "Baseline win_rate not populated"

    logger.info("Feature array shape: %s, dtype: %s", feat.shape, feat.dtype)
    logger.info("Sample features: is_pick=%.1f, team=%.1f, bl_win_rate=%.3f",
                feat[builder.col_idx["is_pick"]],
                feat[builder.col_idx["team"]],
                feat[builder.col_idx["bl_win_rate"]])

    return builder


def test_greedy_bot(cache, builder):
    """Test that GreedyDraftBot evaluates heroes correctly."""
    from trainer.bot_greedy import GreedyDraftBot

    # Create a simple dummy model for testing
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(builder.num_features, 1)

        def forward(self, heroes, actions, tabular):
            # Just use the tabular features
            return self.linear(tabular).squeeze(-1)

    model = DummyModel()
    bot = GreedyDraftBot(model, builder)

    # Test suggestion
    start = time.time()
    suggestions = bot.suggest_next_action(
        current_heroes=[],
        current_actions=[],
        is_radiant_turn=True,
        is_pick=True,
        radiant_picks=[],
        dire_picks=[],
        top_k=5,
    )
    elapsed = time.time() - start

    assert len(suggestions) == 5, f"Expected 5 suggestions, got {len(suggestions)}"
    assert all("hero_id" in s and "win_probability" in s for s in suggestions)

    logger.info("GreedyBot: 5 suggestions in %.3f seconds", elapsed)
    for i, s in enumerate(suggestions):
        logger.info("  #%d: Hero %d (%.2f%%)", i+1, s["hero_id"], s["win_probability"]*100)

    return bot


def test_mcts_bot(cache, builder):
    """Test that MCTSDraftBot runs correctly."""
    from trainer.bot_mcts import MCTSDraftBot, CAPTAINS_MODE_FORMAT

    # Create a simple dummy model for testing
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(builder.num_features, 1)

        def forward(self, heroes, actions, tabular):
            return self.linear(tabular).squeeze(-1)

    model = DummyModel()
    bot = MCTSDraftBot(model, builder, draft_format=CAPTAINS_MODE_FORMAT)

    # Test MCTS search (start of draft)
    start = time.time()
    best_hero, win_prob = bot.search(
        current_heroes=[],
        current_actions=[],
        radiant_picks=[],
        dire_picks=[],
        turn_idx=0,
        iterations=100,  # Low for testing
    )
    elapsed = time.time() - start

    logger.info("MCTSBot: 100 iterations in %.3f seconds", elapsed)
    logger.info("Best move: Hero %d (%.2f%% win prob)", best_hero, win_prob*100)

    # Test top-k suggestions
    start = time.time()
    suggestions = bot.get_top_k(
        current_heroes=[],
        current_actions=[],
        radiant_picks=[],
        dire_picks=[],
        turn_idx=0,
        iterations=100,
        top_k=5,
    )
    elapsed = time.time() - start

    logger.info("MCTSBot: top-k in %.3f seconds", elapsed)
    for i, s in enumerate(suggestions):
        logger.info("  #%d: Hero %d (%.2f%%, %d visits)",
                    i+1, s["hero_id"], s["win_probability"]*100, s["visits"])

    return bot


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Drafting Bot Component Tests")
    logger.info("=" * 60)

    try:
        # Test 1: InferenceCache
        logger.info("\n--- Test 1: InferenceCache ---")
        cache = test_inference_cache()

        # Test 2: DraftStateBuilder
        logger.info("\n--- Test 2: DraftStateBuilder ---")
        builder = test_draft_state_builder(cache)

        # Test 3: GreedyDraftBot
        logger.info("\n--- Test 3: GreedyDraftBot ---")
        greedy_bot = test_greedy_bot(cache, builder)

        # Test 4: MCTSDraftBot
        logger.info("\n--- Test 4: MCTSDraftBot ---")
        mcts_bot = test_mcts_bot(cache, builder)

        logger.info("\n" + "=" * 60)
        logger.info("All tests passed!")
        logger.info("=" * 60)

    except Exception as e:
        logger.exception("Test failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
