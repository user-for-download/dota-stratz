"""Tests for ``api/models.py`` — Pydantic model validation schemas.

Focus on DraftSlot ``order`` field bounds and PredictRequest
``num_recommendations`` / ``draft`` length constraints.

Regression bugs covered:
    - BUG-005: order upper bound raised from 30 → 50
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.models import DraftSlot, PredictRequest


# ===================================================================
# DraftSlot — order field bounds
# ===================================================================


class TestDraftSlotOrder:
    """``DraftSlot.order`` must be between 1 and 50 (inclusive).

    The upper bound was raised from 30 to 50 for future patches
    that may have longer draft phases (BUG-005).
    """

    # -- positive cases -----------------------------------------------

    def test_order_1(self):
        """✅ order=1 is accepted (minimum valid value)."""
        slot = DraftSlot(hero_id=1, is_pick=True, team=0, order=1)
        assert slot.order == 1

    def test_order_24(self):
        """✅ order=24 is accepted — needed for patches 46-52.

        Regression guard for BUG-005.
        """
        slot = DraftSlot(hero_id=1, is_pick=True, team=0, order=24)
        assert slot.order == 24

    def test_order_30(self):
        """✅ order=30 is accepted — the original upper bound from BUG-005.

        Should still work after the bound was raised to 50.
        """
        slot = DraftSlot(hero_id=1, is_pick=True, team=0, order=30)
        assert slot.order == 30

    def test_order_50(self):
        """✅ order=50 is accepted (maximum valid value)."""
        slot = DraftSlot(hero_id=1, is_pick=True, team=0, order=50)
        assert slot.order == 50

    # -- negative cases -----------------------------------------------

    def test_order_0_fails(self):
        """❌ order=0 fails validation (ge=1)."""
        with pytest.raises(ValidationError) as exc:
            DraftSlot(hero_id=1, is_pick=True, team=0, order=0)
        assert "order" in str(exc.value)

    def test_order_negative_fails(self):
        """❌ order=-1 fails validation."""
        with pytest.raises(ValidationError) as exc:
            DraftSlot(hero_id=1, is_pick=True, team=0, order=-1)
        assert "order" in str(exc.value)

    def test_order_51_fails(self):
        """❌ order=51 fails validation (exceeds max 50).

        Regression guard for BUG-005.
        """
        with pytest.raises(ValidationError) as exc:
            DraftSlot(hero_id=1, is_pick=True, team=0, order=51)
        assert "order" in str(exc.value)

    def test_order_100_fails(self):
        """❌ order=100 fails validation."""
        with pytest.raises(ValidationError):
            DraftSlot(hero_id=1, is_pick=True, team=0, order=100)

    # -- other DraftSlot bounds (sanity) ------------------------------

    def test_team_negative_fails(self):
        """❌ team=-1 fails (ge=0)."""
        with pytest.raises(ValidationError):
            DraftSlot(hero_id=1, is_pick=True, team=-1, order=1)

    def test_team_2_fails(self):
        """❌ team=2 fails (le=1)."""
        with pytest.raises(ValidationError):
            DraftSlot(hero_id=1, is_pick=True, team=2, order=1)

    def test_hero_id_negative_fails(self):
        """❌ hero_id=-1 fails (ge=0)."""
        with pytest.raises(ValidationError):
            DraftSlot(hero_id=-1, is_pick=False, team=0, order=1)


# ===================================================================
# PredictRequest — num_recommendations bounds
# ===================================================================


class TestPredictRequestNumRecommendations:
    """``num_recommendations`` must be 1-20 (inclusive)."""

    # -- positive cases -----------------------------------------------

    def test_default_is_5(self):
        """✅ Default value is 5."""
        req = PredictRequest(
            patch_id=1,
            draft=[DraftSlot(hero_id=1, is_pick=True, team=0, order=1)],
        )
        assert req.num_recommendations == 5

    def test_minimum_1(self):
        """✅ num_recommendations=1 is valid."""
        req = PredictRequest(
            patch_id=1,
            draft=[DraftSlot(hero_id=1, is_pick=True, team=0, order=1)],
            num_recommendations=1,
        )
        assert req.num_recommendations == 1

    def test_maximum_20(self):
        """✅ num_recommendations=20 is valid."""
        req = PredictRequest(
            patch_id=1,
            draft=[DraftSlot(hero_id=1, is_pick=True, team=0, order=1)],
            num_recommendations=20,
        )
        assert req.num_recommendations == 20

    # -- negative cases -----------------------------------------------

    def test_0_fails(self):
        """❌ num_recommendations=0 fails (ge=1)."""
        with pytest.raises(ValidationError) as exc:
            PredictRequest(
                patch_id=1,
                draft=[DraftSlot(hero_id=1, is_pick=True, team=0, order=1)],
                num_recommendations=0,
            )
        assert "num_recommendations" in str(exc.value)

    def test_51_fails(self):
        """❌ num_recommendations=51 fails (le=50)."""
        with pytest.raises(ValidationError) as exc:
            PredictRequest(
                patch_id=1,
                draft=[DraftSlot(hero_id=1, is_pick=True, team=0, order=1)],
                num_recommendations=51,
            )
        assert "num_recommendations" in str(exc.value)

    def test_negative_fails(self):
        """❌ num_recommendations=-1 fails."""
        with pytest.raises(ValidationError):
            PredictRequest(
                patch_id=1,
                draft=[DraftSlot(hero_id=1, is_pick=True, team=0, order=1)],
                num_recommendations=-1,
            )


# ===================================================================
# PredictRequest — draft length bounds
# ===================================================================


class TestPredictRequestDraftLength:
    """``draft`` must have at least 1 item and at most 50."""

    # -- positive cases -----------------------------------------------

    def test_single_slot(self):
        """✅ Draft with 1 slot is valid (minimum)."""
        req = PredictRequest(
            patch_id=1,
            draft=[DraftSlot(hero_id=1, is_pick=True, team=0, order=1)],
        )
        assert len(req.draft) == 1

    def test_50_slots(self):
        """✅ Draft with 50 slots is valid (maximum)."""
        draft = [
            DraftSlot(
                hero_id=(i % 120) + 1,
                is_pick=(i % 2 == 0),
                team=i % 2,
                order=i + 1,
            )
            for i in range(50)
        ]
        req = PredictRequest(patch_id=1, draft=draft)
        assert len(req.draft) == 50

    # -- negative cases -----------------------------------------------

    def test_empty_draft_succeeds(self):
        """✅ Empty draft is valid (min_length=0 for pre-draft predictions)."""
        req = PredictRequest(patch_id=1, draft=[])
        assert len(req.draft) == 0

    def test_51_slots_fails(self):
        """❌ Draft with 51 slots (max_length=50) fails.

        Use order values clamped to [1, 50] so each individual
        DraftSlot is valid but the list itself exceeds the limit.
        """
        draft = [
            DraftSlot(
                hero_id=(i % 120) + 1,
                is_pick=(i % 2 == 0),
                team=i % 2,
                order=(i % 50) + 1,
            )
            for i in range(51)
        ]
        with pytest.raises(ValidationError) as exc:
            PredictRequest(patch_id=1, draft=draft)
        assert "draft" in str(exc.value)


# ===================================================================
# PredictRequest — other field bounds
# ===================================================================


class TestPredictRequestOther:
    """Sanity checks on remaining PredictRequest fields."""

    def test_first_pick_team_defaults_to_0(self):
        """✅ first_pick_team defaults to 0."""
        req = PredictRequest(
            patch_id=1,
            draft=[DraftSlot(hero_id=1, is_pick=True, team=0, order=1)],
        )
        assert req.first_pick_team == 0

    def test_first_pick_team_1_valid(self):
        """✅ first_pick_team=1 is valid."""
        req = PredictRequest(
            patch_id=1,
            draft=[DraftSlot(hero_id=1, is_pick=True, team=0, order=1)],
            first_pick_team=1,
        )
        assert req.first_pick_team == 1

    def test_first_pick_team_2_fails(self):
        """❌ first_pick_team=2 fails (le=1)."""
        with pytest.raises(ValidationError):
            PredictRequest(
                patch_id=1,
                draft=[DraftSlot(hero_id=1, is_pick=True, team=0, order=1)],
                first_pick_team=2,
            )

    def test_account_id_optional(self):
        """✅ account_id can be None."""
        req = PredictRequest(
            patch_id=1,
            draft=[DraftSlot(hero_id=1, is_pick=True, team=0, order=1)],
        )
        assert req.account_id is None
