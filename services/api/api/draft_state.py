"""Draft state reconstruction and feature context for a single prediction.

Dota 2 Captain's Mode draft order (20 steps, Radiant first pick):
   1-2:  Phase 1 bans (2 per team, Radiant starts)
   3-8:  Phase 1 picks (3 per team, Radiant starts)
   9-14: Phase 2 bans (3 per team, Dire starts)
  15-18: Phase 2 picks (2 per team, Dire starts)

If ``first_pick_team=1`` (Dire first pick), the pick and ban phases are
swapped so that Dire acts in the "Radiant" slots and vice-versa.

Given a partial draft, this module determines:
  - Whose turn it is.
  - Which heroes are already taken (picked or banned by either side).
  - Ally/enemy hero sets for synergy and counter lookups.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import DraftSlot


# Standard Captain's Mode (Radiant first pick): (team, is_pick) for steps 1..20
_STANDARD_ORDER: tuple[tuple[int, bool], ...] = (
    # Phase 1 bans: 2 per team, Radiant starts (4 total)
    (0, False), (1, False),
    (0, False), (1, False),
    # Phase 1 picks: 3 per team, Radiant starts (6 total)
    (0, True), (1, True),
    (0, True), (1, True),
    (0, True), (1, True),
    # Phase 2 bans: 3 per team, Dire starts (6 total)
    (1, False), (0, False),
    (1, False), (0, False),
    (1, False), (0, False),
    # Phase 2 picks: 2 per team, Dire starts (4 total)
    (1, True), (0, True),
    (1, True), (0, True),
)

_N_STEPS = len(_STANDARD_ORDER)  # 20


def _turn_order(first_pick_team: int) -> tuple[tuple[int, bool], ...]:
    """Return the turn order sequence, flipping Radiant↔Dire if Dire has first pick."""
    if first_pick_team == 0:
        return _STANDARD_ORDER
    # Dire first pick: swap team 0↔1 in every slot
    return tuple((1 - team, is_pick) for team, is_pick in _STANDARD_ORDER)


def _validate_draft(draft: list[DraftSlot], first_pick_team: int = 0):
    """Validate that the draft follows the Captain's Mode order and has no
    duplicate hero IDs."""
    if len(draft) > _N_STEPS:
        raise ValueError(f"Draft cannot have more than {_N_STEPS} slots (got {len(draft)})")

    order = _turn_order(first_pick_team)
    seen_heroes: set[int] = set()
    for i, slot in enumerate(draft):
        expected_team, expected_is_pick = order[i]
        if slot.team != expected_team:
            raise ValueError(
                f"Slot {i + 1}: expected team {expected_team} (first_pick_team={first_pick_team}), got {slot.team}"
            )
        if slot.is_pick != expected_is_pick:
            what = "pick" if expected_is_pick else "ban"
            raise ValueError(
                f"Slot {i + 1}: expected {what}, got {'pick' if slot.is_pick else 'ban'}"
            )
        if slot.hero_id in seen_heroes:
            raise ValueError(f"Duplicate hero_id {slot.hero_id} in draft")
        seen_heroes.add(slot.hero_id)


@dataclass
class DraftContext:
    """Reconstructed context from a partial draft."""

    turn: int  # 1-indexed next slot (1..24)
    recommending_team: int  # 0 = radiant, 1 = dire
    is_pick_turn: bool
    radiant_picks: list[int] = field(default_factory=list)
    dire_picks: list[int] = field(default_factory=list)
    radiant_bans: list[int] = field(default_factory=list)
    dire_bans: list[int] = field(default_factory=list)

    @property
    def all_taken(self) -> set[int]:
        """All heroes that are picked or banned by either side."""
        return set(
            self.radiant_picks + self.dire_picks
            + self.radiant_bans + self.dire_bans
        )

    @property
    def ally_picks(self) -> list[int]:
        """Picks by the team whose turn it is."""
        return self.radiant_picks if self.recommending_team == 0 else self.dire_picks

    @property
    def enemy_picks(self) -> list[int]:
        """Picks by the opposing team."""
        return self.dire_picks if self.recommending_team == 0 else self.radiant_picks


def build_draft_context(draft: list[DraftSlot], first_pick_team: int = 0) -> DraftContext:
    """Build a DraftContext from the current draft state.

    Validates the draft order and reconstructs the full context for
    feature computation.

    Parameters
    ----------
    draft : list[DraftSlot]
        Draft actions so far, in order.
    first_pick_team : int
        0 = Radiant has first pick (standard), 1 = Dire has first pick.
    """
    _validate_draft(draft, first_pick_team)

    ctx = DraftContext(
        turn=len(draft) + 1,
        recommending_team=0,
        is_pick_turn=True,
    )

    for slot in draft:
        if slot.team == 0:
            if slot.is_pick:
                ctx.radiant_picks.append(slot.hero_id)
            else:
                ctx.radiant_bans.append(slot.hero_id)
        else:
            if slot.is_pick:
                ctx.dire_picks.append(slot.hero_id)
            else:
                ctx.dire_bans.append(slot.hero_id)

    # Determine whose turn it is
    if ctx.turn <= _N_STEPS:
        order = _turn_order(first_pick_team)
        ctx.recommending_team, ctx.is_pick_turn = order[ctx.turn - 1]
    else:
        # Draft is complete — no recommendations
        ctx.recommending_team = -1
        ctx.is_pick_turn = False

    return ctx
