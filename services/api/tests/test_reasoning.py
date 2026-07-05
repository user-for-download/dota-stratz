"""Tests for ``api/reasoning.py`` — human-readable recommendation
explanation generation.

Regression bugs covered:
    - BUG-003: H2H WR segment missing from reasoning string
    - BUG-019: "Hero {hero_id} for" prefix shape
"""

from __future__ import annotations

import pytest

from api.reasoning import generate_reasoning


# ===================================================================
# generate_reasoning
# ===================================================================


class TestGenerateReasoning:
    """``generate_reasoning`` builds a ``" | "``-joined string with
    contextual segments about why a hero was recommended.

    Every output must start with ``"Hero {hero_id} for {team_label}"``
    and end with ``"model score {score:.4f}"``.
    """

    # -- helper -------------------------------------------------------

    @staticmethod
    def _draft_ctx(**overrides):
        """Minimal DraftContext for testing reasoning."""
        from api.draft_state import DraftContext

        defaults = {
            "turn": 7,
            "recommending_team": 0,
            "is_pick_turn": True,
            "radiant_picks": [1],
            "dire_picks": [2],
            "radiant_bans": [3],
            "dire_bans": [4],
        }
        defaults.update(overrides)
        return DraftContext(**defaults)

    # ===============================================================
    # Output structure (BUG-019)
    # ===============================================================

    def test_starts_with_hero_prefix(self):
        """✅ Output starts with 'Hero {hero_id} for ...'.

        Regression guard for BUG-019.
        """
        ctx = self._draft_ctx()
        result = generate_reasoning(
            hero_id=42,
            score=0.85,
            ctx=ctx,
            baseline_win_rate=None,
            team_hero_win_rate=None,
            synergy_win_rate=None,
            counter_win_rate=None,
            h2h_win_rate=None,
        )
        assert result.startswith("Hero 42 for "), (
            f"Expected prefix 'Hero 42 for ', got: {result!r}"
        )

    def test_radiant_label(self):
        """✅ recommending_team=0 → 'for Radiant'."""
        ctx = self._draft_ctx(recommending_team=0)
        result = generate_reasoning(42, 0.85, ctx, None, None, None, None, None)
        assert "for Radiant" in result

    def test_dire_label(self):
        """✅ recommending_team=1 → 'for Dire'."""
        ctx = self._draft_ctx(recommending_team=1)
        result = generate_reasoning(42, 0.85, ctx, None, None, None, None, None)
        assert "for Dire" in result

    def test_ends_with_model_score(self):
        """✅ Last segment is 'model score {score:.4f}'."""
        ctx = self._draft_ctx()
        result = generate_reasoning(
            hero_id=1,
            score=0.85123,
            ctx=ctx,
            baseline_win_rate=None,
            team_hero_win_rate=None,
            synergy_win_rate=None,
            counter_win_rate=None,
            h2h_win_rate=None,
        )
        assert result.endswith("model score 0.8512"), (
            f"Expected suffix 'model score 0.8512', got: {result!r}"
        )

    # ===============================================================
    # H2H WR segment (BUG-003)
    # ===============================================================

    def test_h2h_provided_contains_h2h_wr(self):
        """✅ h2h_win_rate=0.6500 → output contains 'H2H WR 65.0%'.

        Regression guard for BUG-003.
        """
        ctx = self._draft_ctx()
        result = generate_reasoning(
            hero_id=1,
            score=0.80,
            ctx=ctx,
            baseline_win_rate=None,
            team_hero_win_rate=None,
            synergy_win_rate=None,
            counter_win_rate=None,
            h2h_win_rate=0.65,
        )
        assert "H2H WR 65.0%" in result, (
            f"Expected 'H2H WR 65.0%' in output, got: {result!r}"
        )

    def test_h2h_none_no_h2h_segment(self):
        """❌ h2h_win_rate=None → no 'H2H WR' in output."""
        ctx = self._draft_ctx()
        result = generate_reasoning(
            hero_id=1,
            score=0.80,
            ctx=ctx,
            baseline_win_rate=0.50,
            team_hero_win_rate=None,
            synergy_win_rate=None,
            counter_win_rate=None,
            h2h_win_rate=None,
        )
        assert "H2H WR" not in result

    # ===============================================================
    # Synergy segment condition
    # ===============================================================

    def test_synergy_with_ally_picks(self):
        """✅ synergy_win_rate provided + non-empty ally_picks → synergy
        segment appears."""
        ctx = self._draft_ctx()  # default: radiant_picks=[1], recommending_team=0 → ally_picks=[1]
        result = generate_reasoning(
            hero_id=1,
            score=0.80,
            ctx=ctx,
            baseline_win_rate=None,
            team_hero_win_rate=None,
            synergy_win_rate=0.7205,
            counter_win_rate=None,
            h2h_win_rate=None,
        )
        assert "synergy" in result
        assert "72.0%" in result or "72.1%" in result

    def test_synergy_but_empty_ally_picks(self):
        """❌ synergy_win_rate set but ctx.ally_picks is empty → no synergy
        segment."""
        ctx = self._draft_ctx(
            radiant_picks=[],
            dire_picks=[],
            recommending_team=0,
        )
        assert ctx.ally_picks == [], "Precondition: ally_picks must be empty"
        result = generate_reasoning(
            hero_id=1,
            score=0.80,
            ctx=ctx,
            baseline_win_rate=None,
            team_hero_win_rate=None,
            synergy_win_rate=0.7205,
            counter_win_rate=None,
            h2h_win_rate=None,
        )
        assert "synergy" not in result

    # ===============================================================
    # Counter segment condition
    # ===============================================================

    def test_counter_with_enemy_picks(self):
        """✅ counter_win_rate provided + non-empty enemy_picks → counter
        segment appears."""
        # enemy_picks is a property: for recommending_team=0 it returns dire_picks
        ctx = self._draft_ctx(dire_picks=[2])  # default recommending_team=0 → enemy_picks=[2]
        result = generate_reasoning(
            hero_id=1,
            score=0.80,
            ctx=ctx,
            baseline_win_rate=None,
            team_hero_win_rate=None,
            synergy_win_rate=None,
            counter_win_rate=0.5500,
            h2h_win_rate=None,
        )
        assert "counter" in result
        assert "55.0%" in result

    def test_counter_but_empty_enemy_picks(self):
        """❌ counter_win_rate set but ctx.enemy_picks is empty → no
        counter segment."""
        ctx = self._draft_ctx(
            radiant_picks=[1],
            dire_picks=[],
            recommending_team=0,
        )
        assert ctx.enemy_picks == [], "Precondition: enemy_picks must be empty"
        result = generate_reasoning(
            hero_id=1,
            score=0.80,
            ctx=ctx,
            baseline_win_rate=None,
            team_hero_win_rate=None,
            synergy_win_rate=None,
            counter_win_rate=0.5500,
            h2h_win_rate=None,
        )
        assert "counter" not in result

    # ===============================================================
    # Global WR / team WR
    # ===============================================================

    def test_baseline_win_rate_segment(self):
        """✅ baseline_win_rate=0.5000 → contains 'global WR 50.0%'."""
        ctx = self._draft_ctx()
        result = generate_reasoning(
            hero_id=1,
            score=0.80,
            ctx=ctx,
            baseline_win_rate=0.50,
            team_hero_win_rate=None,
            synergy_win_rate=None,
            counter_win_rate=None,
            h2h_win_rate=None,
        )
        assert "global WR 50.0%" in result

    def test_team_hero_win_rate_segment(self):
        """✅ team_hero_win_rate=0.6200 → contains 'team WR 62.0%'."""
        ctx = self._draft_ctx()
        result = generate_reasoning(
            hero_id=1,
            score=0.80,
            ctx=ctx,
            baseline_win_rate=None,
            team_hero_win_rate=0.62,
            synergy_win_rate=None,
            counter_win_rate=None,
            h2h_win_rate=None,
        )
        assert "team WR 62.0%" in result

    # ===============================================================
    # Joining behaviour
    # ===============================================================

    def test_segments_joined_with_pipe(self):
        """✅ Multiple segments separated by ' | '."""
        # Default ctx has radiant_picks=[1] → ally_picks=[1]; dire_picks=[2] → enemy_picks=[2]
        ctx = self._draft_ctx()
        result = generate_reasoning(
            hero_id=5,
            score=0.9012,
            ctx=ctx,
            baseline_win_rate=0.50,
            team_hero_win_rate=0.62,
            synergy_win_rate=0.7205,
            counter_win_rate=0.5500,
            h2h_win_rate=0.6500,
        )
        # Should have at least 5 segments: hero + global + team + synergy + counter + h2h + model
        segments = result.split(" | ")
        assert len(segments) >= 6, (
            f"Expected multiple ' | '-separated segments, got {len(segments)}: {result!r}"
        )
        # First segment is "Hero 5 for Radiant (slot 7)"
        assert segments[0] == "Hero 5 for Radiant (slot 7)"
        # Last segment is model score
        assert segments[-1] == "model score 0.9012"

    def test_minimal_output_only_hero_and_score(self):
        """✅ With all feature values None, output has just hero prefix +
        model score (2 segments)."""
        ctx = self._draft_ctx()
        result = generate_reasoning(
            hero_id=1,
            score=0.50,
            ctx=ctx,
            baseline_win_rate=None,
            team_hero_win_rate=None,
            synergy_win_rate=None,
            counter_win_rate=None,
            h2h_win_rate=None,
        )
        segments = result.split(" | ")
        assert len(segments) == 2, (
            f"Expected exactly 2 segments (hero + model score), got {len(segments)}: {result!r}"
        )
        assert segments[0].startswith("Hero 1 for ")
        assert segments[1] == "model score 0.5000"
