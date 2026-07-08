"""Draft state reconstruction and feature context for a single prediction."""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from .models import DraftSlot

# ---------------------------------------------------------------------------
# Dynamic Draft Patterns
# B = Ban, P = Pick. 0 = Radiant, 1 = Dire.
# (Spaces are ignored)
# ---------------------------------------------------------------------------
DRAFT_PATTERNS = {
    8: "B0 B1 B0 B1 P0 P1 P1 P0 P0 P1 B0 B1 B0 B1 B0 B1 P0 P1 P0 P1",
    9: "B1 B0 B1 B0 P1 P0 P0 P1 B1 B0 B1 B0 P0 P1 P0 P1 B0 B1 P0 P1",
    10: "B1 B0 B1 B0 P1 P0 P0 P1 B1 B0 B1 B0 P0 P1 P0 P1 B0 B1 P0 P1",
    11: "B1 B0 B1 B0 P1 P0 P0 P1 B1 B0 B1 B0 P0 P1 P0 P1 B0 B1 P0 P1",
    12: "B0 B1 B0 B1 P0 P1 P1 P0 B0 B1 B0 B1 P1 P0 P1 P0 B1 B0 P1 P0",
    13: "B1 B0 B1 B0 P1 P0 P0 P1 B1 B0 B1 B0 P0 P1 P0 P1 B0 B1 P0 P1",
    14: "B1 B0 B1 B0 P1 P0 P0 P1 B1 B0 B1 B0 P0 P1 P0 P1 B0 B1 P0 P1",
    15: "B0 B1 B0 B1 P0 P1 P1 P0 B0 B1 B0 B1 P1 P0 P1 P0 B1 B0 P1 P0",
    16: "B0 B1 B0 B1 P0 P1 P1 P0 B0 B1 B0 B1 P1 P0 P1 P0 B1 B0 P1 P0",
    17: "B0 B1 B0 B1 P0 P1 P1 P0 B1 B0 B1 B0 P1 P0 P1 P0 B1 B0 P0 P1",
    18: "B0 B1 B0 B1 P0 P1 P1 P0 B1 B0 B1 B0 P1 P0 P1 P0 B1 B0 P0 P1",
    19: "B1 B0 B1 B0 P1 P0 P0 P1 B0 B1 B0 B1 P0 P1 P0 P1 B0 B1 P1 P0",
    20: "B0 B1 B0 B1 P0 P1 P1 P0 B1 B0 B1 B0 P1 P0 P1 P0 B1 B0 P0 P1",
    21: "B1 B0 B1 B0 P1 P0 P0 P1 B0 B1 B0 B1 P0 P1 P0 P1 B0 B1 P1 P0",
    22: "B1 B0 B1 B0 P1 P0 P0 P1 B0 B1 B0 B1 P0 P1 P0 P1 B0 B1 P1 P0",
    23: "B1 B0 B1 B0 P1 P0 P0 P1 B0 B1 B0 B1 P0 P1 P0 P1 B0 B1 P1 P0",
    24: "B0 B1 B0 B1 P0 P1 P1 P0 B1 B0 B1 B0 P1 P0 P1 P0 B1 B0 P0 P1",
    25: "B0 B1 B0 B1 P0 P1 P1 P0 B1 B0 B1 B0 P1 P0 P1 P0 B1 B0 P0 P1",
    26: "B1 B0 B1 B0 B1 B0 P1 P0 P0 P1 B0 B1 B0 B1 P0 P1 P0 P1 B0 B1 P1 P0",
    27: "B1 B0 B1 B0 B1 B0 P1 P0 P0 P1 B0 B1 B0 B1 P0 P1 P0 P1 B0 B1 P1 P0",
    28: "B1 B0 B1 B0 B1 B0 P1 P0 P0 P1 B0 B1 B0 B1 P0 P1 P0 P1 B0 B1 P1 P0",
    29: "B1 B0 B1 B0 B1 B0 P1 P0 P0 P1 B0 B1 B0 B1 P0 P1 P0 P1 B0 B1 P1 P0",
    30: "B1 B0 B1 B0 B1 B0 P1 P0 P0 P1 B0 B1 B0 B1 P0 P1 P0 P1 B0 B1 P1 P0",
    31: "B1 B0 B1 B0 B1 B0 P1 P0 P0 P1 B0 B1 B0 B1 P0 P1 P0 P1 B0 B1 P1 P0",
    32: "B1 B0 B1 B0 B1 B0 P1 P0 P0 P1 B0 B1 B0 B1 P0 P1 P0 P1 B0 B1 P1 P0",
    33: "B1 B0 B1 B0 B1 B0 P1 P0 P0 P1 B0 B1 B0 B1 P0 P1 P0 P1 B0 B1 P1 P0",
    34: "B1 B0 B1 B0 B1 B0 P1 P0 P0 P1 B0 B1 B0 B1 P0 P1 P0 P1 B0 B1 P1 P0",
    35: "B1 B0 B1 B0 B1 B0 P1 P0 P0 P1 B1 B0 B1 B0 P0 P1 P0 P1 B0 B1 P1 P0",
    36: "B1 B0 B1 B0 B1 B0 P1 P0 P0 P1 B1 B0 B1 B0 P0 P1 P0 P1 B0 B1 P1 P0",
    37: "B1 B0 B1 B0 B1 B0 P1 P0 P0 P1 B1 B0 B1 B0 P0 P1 P0 P1 B0 B1 P1 P0",
    38: "B1 B0 B1 B0 B1 B0 P1 P0 P0 P1 B1 B0 B1 B0 P0 P1 P0 P1 B0 B1 P1 P0",
    39: "B1 B0 B1 B0 B1 B0 P1 P0 P0 P1 B1 B0 B1 B0 P0 P1 P0 P1 B0 B1 P1 P0",
    40: "B1 B0 B1 B0 B1 B0 P1 P0 P0 P1 B1 B0 B1 B0 P0 P1 P0 P1 B0 B1 P1 P0",
    41: "B1 B0 B1 B0 B1 B0 P1 P0 P0 P1 B1 B0 B1 B0 P0 P1 P0 P1 B0 B1 P1 P0",
    42: "B0 B1 B0 B1 B0 B1 P0 P1 P1 P0 B0 B1 B0 B1 P1 P0 P1 P0 B1 B0 P0 P1",
    43: "B0 B1 B0 B1 B0 B1 P0 P1 P1 P0 B0 B1 B0 B1 P1 P0 P1 P0 B1 B0 P0 P1",
    44: "B1 B0 B1 B0 B1 B0 P1 P0 P0 P1 B1 B0 B1 B0 P0 P1 P0 P1 B0 B1 P1 P0",
    45: "B0 B1 B0 B1 B0 B1 B0 B1 P0 P1 P1 P0 B0 B1 P1 P0 P1 P0 B1 B0 P0 P1",
    46: "B0 B1 B0 B1 P0 P1 P0 P1 B0 B1 B0 B1 B0 B1 P1 P0 P1 P0 B0 B1 B0 B1 P0 P1",
    47: "B0 B1 B0 B1 P0 P1 P0 P1 B0 B1 B0 B1 B0 B1 P1 P0 P1 P0 B0 B1 B0 B1 P0 P1",
    48: "B1 B0 B1 B0 P1 P0 P0 P1 B1 B0 B1 B0 B1 B0 P0 P1 P0 P1 B1 B0 B1 B0 P1 P0",
    49: "B0 B1 B0 B1 P0 P1 P1 P0 B0 B1 B0 B1 B0 B1 P1 P0 P0 P1 B0 B1 B0 B1 P0 P1",
    50: "B0 B1 B0 B1 P0 P1 P1 P0 B0 B1 B0 B1 B0 B1 P1 P0 P0 P1 B0 B1 B0 B1 P0 P1",
    51: "B0 B1 B0 B1 P0 P1 P1 P0 B0 B1 B0 B1 B0 B1 P1 P0 P0 P1 B0 B1 B0 B1 P0 P1",
    52: "B0 B1 B0 B1 P0 P1 P1 P0 B0 B1 B0 B1 B0 B1 P1 P0 P0 P1 B0 B1 B0 B1 P0 P1",
    53: "B0 B1 B1 B0 B1 B1 B0 P0 P1 B0 B0 B1 P1 P0 P0 P1 P1 P0 B0 B1 B1 B0 P0 P1",
    54: "B0 B1 B1 B0 B1 B1 B0 P0 P1 B0 B0 B1 P1 P0 P0 P1 P1 P0 B0 B1 B1 B0 P0 P1",
    55: "B1 B0 B0 B1 B0 B0 B1 P1 P0 B1 B1 B0 P0 P1 P1 P0 P0 P1 B1 B0 B0 B1 P1 P0",
    56: "B0 B1 B1 B0 B1 B1 B0 P0 P1 B0 B0 B1 P1 P0 P0 P1 P1 P0 B0 B1 B1 B0 P0 P1",
    57: "B1 B0 B0 B1 B0 B0 B1 P1 P0 B1 B1 B0 P0 P1 P1 P0 P0 P1 B1 B0 B0 B1 P1 P0",
    58: "B0 B1 B1 B0 B1 B1 B0 P0 P1 B0 B0 B1 P1 P0 P0 P1 P1 P0 B0 B1 B1 B0 P0 P1",
    59: "B1 B1 B0 B0 B1 B0 B0 P1 P0 B1 B1 B0 P0 P1 P1 P0 P0 P1 B1 B0 B1 B0 P1 P0",
    60: "B1 B1 B0 B0 B1 B0 B0 P1 P0 B1 B1 B0 P0 P1 P1 P0 P0 P1 B1 B0 B1 B0 P1 P0",
}

# Fallback for new patches
LATEST_KNOWN_PATTERN = DRAFT_PATTERNS[60]


def parse_draft_pattern(pattern: str) -> tuple[tuple[int, bool], ...]:
    """Parse a pattern string into a tuple of (team_id, is_pick).

    Normalizes so that 0 always represents the team with first pick:
    if the SQL-sampled pattern happened to start with B1 (Dire first
    pick), flip all teams so 0 = first-pick team. The
    ``first_pick_team`` parameter in ``get_turn_order`` then remaps
    0→Radiant or 0→Dire as appropriate.
    """
    tokens = re.findall(r'[PB][01]', pattern.upper())

    # Normalize: first action's team determines which team is 0 in the
    # canonical representation
    first_team = int(tokens[0][1])

    order = []
    for token in tokens:
        team = int(token[1])
        if first_team == 1:
            team = 1 - team  # Flip 1s to 0s and 0s to 1s
        is_pick = (token[0] == 'P')
        order.append((team, is_pick))

    return tuple(order)


def get_turn_order(patch_id: int, first_pick_team: int) -> tuple[tuple[int, bool], ...]:
    """Get the correct turn order sequence for the given patch.

    Selects the appropriate draft pattern by *patch_id*, normalizes so
    that 0 = first-pick team, then flips to match *first_pick_team*.
    """
    pattern_str = DRAFT_PATTERNS.get(patch_id, LATEST_KNOWN_PATTERN)
    base_order = parse_draft_pattern(pattern_str)

    # If Dire has first pick, swap all 0s to 1s and 1s to 0s
    if first_pick_team == 1:
        return tuple((1 - team, is_pick) for team, is_pick in base_order)
    return base_order


def _validate_draft(draft: list[DraftSlot], order: tuple[tuple[int, bool], ...], first_pick_team: int):
    """Validate that the draft follows the expected order."""
    max_slots = len(order)
    if len(draft) > max_slots:
        raise ValueError(f"Draft cannot have more than {max_slots} slots for this patch (got {len(draft)})")

    seen_heroes: set[int] = set()
    for i, slot in enumerate(draft):
        # Validate that submitted order matches list position (BUG-018).
        if slot.order != i + 1:
            raise ValueError(
                f"Slot {i + 1}: expected order {i + 1}, got {slot.order}. "
                "Draft slots must be sent in order."
            )
        expected_team, expected_is_pick = order[i]
        if slot.team != expected_team:
            raise ValueError(f"Slot {i + 1}: expected team {expected_team} (first_pick_team={first_pick_team}), got {slot.team}")

        if slot.is_pick != expected_is_pick:
            what = "pick" if expected_is_pick else "ban"
            raise ValueError(f"Slot {i + 1}: expected {what}, got {'pick' if slot.is_pick else 'ban'}")

        # hero_id == 0 means the slot was skipped (e.g. a ban was forfeited).
        # Multiple skipped bans are valid in some tournament drafts, so they
        # should not trigger a duplicate error (issue #14).
        if slot.hero_id != 0:
            if slot.hero_id in seen_heroes:
                raise ValueError(f"Duplicate hero_id {slot.hero_id} in draft")
            seen_heroes.add(slot.hero_id)


@dataclass
class DraftContext:
    turn: int
    recommending_team: int
    is_pick_turn: bool
    draft_phase_id: int = 0  # 0=Ban1, 1=Pick1, 2=Ban2, 3=Pick2, 4=Ban3, 5=FinalPick
    radiant_picks: list[int] = field(default_factory=list)
    dire_picks: list[int] = field(default_factory=list)
    radiant_bans: list[int] = field(default_factory=list)
    dire_bans: list[int] = field(default_factory=list)

    @property
    def all_taken(self) -> set[int]:
        return {h for h in (self.radiant_picks + self.dire_picks + self.radiant_bans + self.dire_bans) if h != 0}

    @property
    def ally_picks(self) -> list[int]:
        return self.radiant_picks if self.recommending_team == 0 else self.dire_picks

    @property
    def enemy_picks(self) -> list[int]:
        return self.dire_picks if self.recommending_team == 0 else self.radiant_picks


def _compute_draft_phase(turn: int, order: tuple[tuple[int, bool], ...]) -> int:
    """Compute the CM draft phase (0-5) based on turn number.

    Phases: 0=Ban1, 1=Pick1, 2=Ban2, 3=Pick2, 4=Ban3, 5=FinalPick
    Derived from action transitions (ban→pick or pick→ban).
    """
    if turn <= 1 or turn > len(order):
        return 0
    phase = 0
    for i in range(1, turn):
        prev_is_pick = order[i - 1][1]
        curr_is_pick = order[i][1]
        if prev_is_pick != curr_is_pick:
            phase += 1
    return phase


def build_draft_context(draft: list[DraftSlot], patch_id: int, first_pick_team: int = 0) -> DraftContext:
    order = get_turn_order(patch_id, first_pick_team)
    _validate_draft(draft, order, first_pick_team)

    turn = len(draft) + 1
    draft_phase_id = _compute_draft_phase(turn, order)

    ctx = DraftContext(
        turn=turn,
        recommending_team=0,
        is_pick_turn=True,
        draft_phase_id=draft_phase_id,
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

    if ctx.turn <= len(order):
        ctx.recommending_team, ctx.is_pick_turn = order[ctx.turn - 1]
    else:
        ctx.recommending_team = -1
        ctx.is_pick_turn = False

    return ctx
