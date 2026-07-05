"""Tests for ``api/draft_state.py`` — draft order parsing, validation,
and context construction.

See regression bugs linked in test names:
    - BUG-005: order upper bound (24-token patterns for patches 46-52)
    - BUG-012: hero_id == 0 should be excluded from taken sets
    - BUG-018: slot.order must match list position
"""

from __future__ import annotations

import pytest

from api.draft_state import (
    DRAFT_PATTERNS,
    DraftContext,
    _validate_draft,
    build_draft_context,
    get_turn_order,
    parse_draft_pattern,
)
from api.models import DraftSlot


# ===================================================================
# parse_draft_pattern
# ===================================================================


class TestParseDraftPattern:
    """``parse_draft_pattern`` converts a whitespace-separated pattern
    string into a tuple of (team, is_pick) tuples, normalizing so that
    0 always represents the first-pick team.
    """

    def test_canonical_radiant_first(self):
        """✅ Parse "B0,B1,P0,P1" — first team is 0, no flip needed."""
        result = parse_draft_pattern("B0 B1 P0 P1")
        assert result == (
            (0, False),  # B0
            (1, False),  # B1
            (0, True),   # P0
            (1, True),   # P1
        )

    def test_normalize_dire_first(self):
        """✅ Pattern starting with B1 — all teams flipped so first-pick
        team becomes 0."""
        result = parse_draft_pattern("B1 B0 P1 P0")
        # Original: B1(team=1), B0(team=0), P1(team=1), P0(team=0)
        # After flip: B0(team=0), B1(team=1), P0(team=0), P1(team=1)
        assert result == (
            (0, False),
            (1, False),
            (0, True),
            (1, True),
        )

    def test_full_20_token(self):
        """✅ Parse a full 20-token CM pattern (patch 12)."""
        pattern = DRAFT_PATTERNS[12]  # 20 tokens
        result = parse_draft_pattern(pattern)
        assert len(result) == 20
        # Every element is a (int, bool) pair
        for team, is_pick in result:
            assert team in (0, 1)
            assert isinstance(is_pick, bool)

    def test_excess_whitespace(self):
        """✅ Extra whitespace between tokens is ignored."""
        result = parse_draft_pattern("  B0   B1  P0  P1  ")
        assert result == (
            (0, False),
            (1, False),
            (0, True),
            (1, True),
        )

    def test_lowercase_input(self):
        """✅ Lowercase "b0 p1" is normalised to uppercase before parsing."""
        result = parse_draft_pattern("b0 b1 p0 p1")
        assert result == (
            (0, False),
            (1, False),
            (0, True),
            (1, True),
        )

    def test_24_token_patches_46_to_52(self):
        """✅ Patches 46-52 produce 24 tokens (→ order length 24).

        Regression guard for BUG-005 where the pattern length was
        unknown and the code assumed max 20 slots.
        """
        for pid in range(46, 53):
            pattern = DRAFT_PATTERNS[pid]
            result = parse_draft_pattern(pattern)
            assert len(result) == 24, (
                f"Patch {pid} expected 24 tokens, got {len(result)}"
            )


# ===================================================================
# get_turn_order
# ===================================================================


class TestGetTurnOrder:
    """``get_turn_order`` selects the correct draft pattern for a patch
    and remaps teams according to ``first_pick_team``.
    """

    def test_first_pick_team_0(self):
        """✅ first_pick_team=0 → Radiant first (order starts with team 0)."""
        order = get_turn_order(patch_id=12, first_pick_team=0)
        assert order[0][0] == 0  # first action is team 0 (Radiant)

    def test_first_pick_team_1(self):
        """✅ first_pick_team=1 → Dire first, teams mirrored from
        normalised base."""
        order = get_turn_order(patch_id=12, first_pick_team=1)
        assert order[0][0] == 1  # first action is team 1 (Dire)

    def test_unknown_patch_falls_back(self):
        """❌ Unknown patch_id (e.g. 9999) falls back to
        LATEST_KNOWN_PATTERN."""
        order_unknown = get_turn_order(patch_id=9999, first_pick_team=0)
        order_latest = get_turn_order(
            patch_id=max(DRAFT_PATTERNS.keys()), first_pick_team=0
        )
        assert order_unknown == order_latest

    def test_24_token_for_patches_46_to_52(self):
        """✅ Patches 46-52 produce 24-token order sequences.

        Regression guard for BUG-005 where the order length was
        previously limited to 20.
        """
        for pid in range(46, 53):
            order = get_turn_order(pid, first_pick_team=0)
            assert len(order) == 24, (
                f"Patch {pid} expected 24 tokens, got {len(order)}"
            )

    def test_team_flip_consistency(self):
        """✅ Flipping first_pick_team toggles every team value."""
        order_0 = get_turn_order(patch_id=8, first_pick_team=0)
        order_1 = get_turn_order(patch_id=8, first_pick_team=1)
        for (t0, _), (t1, _) in zip(order_0, order_1):
            assert t0 == 1 - t1, "Teams should be fully flipped"


# ===================================================================
# _validate_draft
# ===================================================================


class TestValidateDraft:
    """Low-level draft validation logic."""

    # -- helpers -------------------------------------------------------

    @staticmethod
    def _order_8() -> tuple:
        """Return the canonical order for patch 8 (10 actions)."""
        return get_turn_order(patch_id=8, first_pick_team=0)

    @staticmethod
    def _order_46() -> tuple:
        """Return the 24-action order for patch 46."""
        return get_turn_order(patch_id=46, first_pick_team=0)

    # -- positive cases -----------------------------------------------

    def test_valid_draft_no_error(self):
        """✅ A draft that matches the order passes validation."""
        order = self._order_8()
        draft = [
            DraftSlot(hero_id=(i % 120) + 1, is_pick=p, team=t, order=i + 1)
            for i, (t, p) in enumerate(order)
        ]
        # Should not raise
        _validate_draft(draft, order, first_pick_team=0)

    def test_valid_partial_draft(self):
        """✅ A partial draft (2 slots) also passes."""
        order = self._order_8()
        draft = [
            DraftSlot(hero_id=1, is_pick=order[0][1], team=order[0][0], order=1),
            DraftSlot(hero_id=2, is_pick=order[1][1], team=order[1][0], order=2),
        ]
        _validate_draft(draft, order, first_pick_team=0)

    def test_valid_draft_24_slots(self):
        """✅ A full 24-slot draft for patch 46 passes.

        Regression guard for BUG-005.
        """
        order = self._order_46()
        draft = [
            DraftSlot(
                hero_id=(i % 120) + 1,
                is_pick=order[i][1],
                team=order[i][0],
                order=i + 1,
            )
            for i in range(24)
        ]
        _validate_draft(draft, order, first_pick_team=0)

    # -- negative cases -----------------------------------------------

    def test_draft_longer_than_order(self):
        """❌ ValueError when len(draft) > len(order)."""
        order = self._order_8()  # 20 slots for patch 8
        draft = [
            DraftSlot(hero_id=1, is_pick=True, team=0, order=i + 1)
            for i in range(21)  # one too many
        ]
        with pytest.raises(ValueError, match="cannot have more than"):
            _validate_draft(draft, order, first_pick_team=0)

    def test_wrong_team(self):
        """❌ ValueError when slot.team mismatches expected team."""
        order = self._order_8()
        # order[0] = (0, False) — Radiant ban
        draft = [
            DraftSlot(hero_id=1, is_pick=False, team=1, order=1),  # expected team 0
        ]
        with pytest.raises(ValueError, match="expected team 0"):
            _validate_draft(draft, order, first_pick_team=0)

    def test_wrong_action_is_pick(self):
        """❌ ValueError when slot.is_pick mismatches expected action."""
        order = self._order_8()
        # order[0] = (0, False) — ban
        draft = [
            DraftSlot(hero_id=1, is_pick=True, team=0, order=1),  # expected ban
        ]
        with pytest.raises(ValueError, match="expected ban"):
            _validate_draft(draft, order, first_pick_team=0)

    def test_duplicate_hero_id_non_zero(self):
        """❌ ValueError on duplicate non-zero hero_id.

        Both slots must match the expected is_pick and team
        so validation reaches the duplicate check.
        """
        order = self._order_8()
        # order[0] = (0, False) — Radiant ban
        # order[1] = (1, False) — Dire ban
        draft = [
            DraftSlot(hero_id=1, is_pick=order[0][1], team=order[0][0], order=1),
            DraftSlot(hero_id=1, is_pick=order[1][1], team=order[1][0], order=2),  # duplicate
        ]
        with pytest.raises(ValueError, match="Duplicate hero_id 1"):
            _validate_draft(draft, order, first_pick_team=0)

    def test_duplicate_hero_id_zero_allowed(self):
        """✅ Duplicate hero_id=0 does NOT raise ValueError.

        Regression guard for BUG-012 — multiple forfeited bans
        are valid in tournament drafts.
        """
        order = self._order_8()
        draft = [
            DraftSlot(hero_id=0, is_pick=False, team=0, order=1),
            DraftSlot(hero_id=0, is_pick=False, team=1, order=2),
        ]
        # Should not raise
        _validate_draft(draft, order, first_pick_team=0)

    def test_order_mismatch_list_position(self):
        """❌ ValueError when slot.order does not match list position.

        Regression guard for BUG-018.
        """
        order = self._order_8()
        draft = [
            DraftSlot(hero_id=1, is_pick=False, team=0, order=5),  # order should be 1
        ]
        with pytest.raises(ValueError, match="expected order 1, got 5"):
            _validate_draft(draft, order, first_pick_team=0)


# ===================================================================
# DraftContext.all_taken
# ===================================================================


class TestDraftContextAllTaken:
    """The ``all_taken`` property collects all non-zero hero_ids from
    picks and bans."""

    def test_excludes_hero_id_zero(self):
        """✅ Does NOT include hero_id=0 in the returned set.

        Regression guard for BUG-012.
        """
        ctx = DraftContext(
            turn=5,
            recommending_team=0,
            is_pick_turn=True,
            radiant_picks=[1, 0],  # 0 should be filtered
            dire_picks=[2],
            radiant_bans=[0, 3],  # 0 should be filtered
            dire_bans=[4],
        )
        assert ctx.all_taken == {1, 2, 3, 4}

    def test_includes_all_non_zero(self):
        """✅ All non-zero picks and bans are included."""
        ctx = DraftContext(
            turn=10,
            recommending_team=1,
            is_pick_turn=False,
            radiant_picks=[1, 5],
            dire_picks=[2, 6],
            radiant_bans=[3, 7],
            dire_bans=[4, 8],
        )
        assert ctx.all_taken == {1, 2, 3, 4, 5, 6, 7, 8}

    def test_empty_draft(self):
        """✅ Empty sections produce an empty set."""
        ctx = DraftContext(
            turn=1,
            recommending_team=0,
            is_pick_turn=True,
        )
        assert ctx.all_taken == set()

    def test_all_zero_values(self):
        """✅ All sections contain only zeros → empty set."""
        ctx = DraftContext(
            turn=3,
            recommending_team=0,
            is_pick_turn=True,
            radiant_picks=[0, 0],
            dire_picks=[0],
            radiant_bans=[0],
            dire_bans=[0, 0],
        )
        assert ctx.all_taken == set()


# ===================================================================
# build_draft_context
# ===================================================================


class TestBuildDraftContext:
    """``build_draft_context`` constructs a ``DraftContext`` from the
    current draft slots and patch metadata."""

    # -- helpers -------------------------------------------------------

    @staticmethod
    def _draft_from_order(
        order: tuple,
        *hero_id_overrides: dict[int, int],
    ) -> list[DraftSlot]:
        """Build a draft matching *order* with sequential hero IDs."""
        draft = []
        for i, (team, is_pick) in enumerate(order):
            hero_id = (i % 120) + 1
            # Apply any overrides (e.g. to inject a zero)
            for override in hero_id_overrides:
                if i in override:
                    hero_id = override[i]
            draft.append(
                DraftSlot(
                    hero_id=hero_id,
                    is_pick=is_pick,
                    team=team,
                    order=i + 1,
                )
            )
        return draft

    # -- positive cases -----------------------------------------------

    def test_recommending_team_and_is_pick_turn(self):
        """✅ Sets ``recommending_team`` and ``is_pick_turn`` from the
        *next* slot in the order."""
        order = get_turn_order(patch_id=8, first_pick_team=0)
        # After 1 slot the next is order[1]
        draft = [
            DraftSlot(hero_id=1, is_pick=order[0][1], team=order[0][0], order=1),
        ]
        ctx = build_draft_context(draft, patch_id=8, first_pick_team=0)
        next_team, next_is_pick = order[1]
        assert ctx.recommending_team == next_team
        assert ctx.is_pick_turn == next_is_pick

    def test_recommending_team_minus_one_when_complete(self):
        """✅ recommending_team=-1 and is_pick_turn=False when draft is
        fully filled."""
        order = get_turn_order(patch_id=8, first_pick_team=0)
        draft = self._draft_from_order(order)
        ctx = build_draft_context(draft, patch_id=8, first_pick_team=0)
        assert ctx.recommending_team == -1
        assert ctx.is_pick_turn is False

    def test_populates_radiant_picks(self):
        """✅ ``radiant_picks`` contains only picks by team 0."""
        order = get_turn_order(patch_id=8, first_pick_team=0)
        draft = self._draft_from_order(order[:6])
        ctx = build_draft_context(draft, patch_id=8, first_pick_team=0)
        for h in ctx.radiant_picks:
            assert h != 0, "hero_id 0 should not appear in radiant_picks"
        # All should be picks by team 0
        for i, slot in enumerate(draft):
            if slot.team == 0 and slot.is_pick:
                assert slot.hero_id in ctx.radiant_picks

    def test_populates_dire_picks(self):
        """✅ ``dire_picks`` contains only picks by team 1."""
        order = get_turn_order(patch_id=8, first_pick_team=0)
        draft = self._draft_from_order(order[:6])
        ctx = build_draft_context(draft, patch_id=8, first_pick_team=0)
        for i, slot in enumerate(draft):
            if slot.team == 1 and slot.is_pick:
                assert slot.hero_id in ctx.dire_picks

    def test_populates_radiant_bans(self):
        """✅ ``radiant_bans`` contains only bans by team 0."""
        order = get_turn_order(patch_id=8, first_pick_team=0)
        draft = self._draft_from_order(order[:6])
        ctx = build_draft_context(draft, patch_id=8, first_pick_team=0)
        for i, slot in enumerate(draft):
            if slot.team == 0 and not slot.is_pick:
                assert slot.hero_id in ctx.radiant_bans

    def test_populates_dire_bans(self):
        """✅ ``dire_bans`` contains only bans by team 1."""
        order = get_turn_order(patch_id=8, first_pick_team=0)
        draft = self._draft_from_order(order[:6])
        ctx = build_draft_context(draft, patch_id=8, first_pick_team=0)
        for i, slot in enumerate(draft):
            if slot.team == 1 and not slot.is_pick:
                assert slot.hero_id in ctx.dire_bans

    def test_turn_is_len_draft_plus_one(self):
        """✅ ``turn`` equals len(draft) + 1."""
        order = get_turn_order(patch_id=8, first_pick_team=0)
        draft = self._draft_from_order(order[:4])
        ctx = build_draft_context(draft, patch_id=8, first_pick_team=0)
        assert ctx.turn == 5

    # -- negative cases -----------------------------------------------

    def test_invalid_draft_raises(self):
        """❌ An invalid draft (wrong team) propagates the ValueError."""
        order = get_turn_order(patch_id=8, first_pick_team=0)
        # order[0] = (0, False) → Radiant ban
        bad_draft = [
            DraftSlot(hero_id=1, is_pick=False, team=1, order=1),
        ]
        with pytest.raises(ValueError, match="expected team"):
            build_draft_context(bad_draft, patch_id=8, first_pick_team=0)
