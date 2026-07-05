"""Generate human-readable reasoning for the top hero recommendation.

Explains why a particular hero was recommended based on which features
contributed most to the score.
"""

from __future__ import annotations

from .draft_state import DraftContext


def generate_reasoning(
    hero_id: int,
    score: float,
    ctx: DraftContext,
    baseline_win_rate: float | None,
    team_hero_win_rate: float | None,
    synergy_win_rate: float | None,
    counter_win_rate: float | None,
    h2h_win_rate: float | None,
) -> str:
    """Build a concise explanation for the top recommendation.

    Each piece is included only if the data is available and meaningful.
    """
    parts: list[str] = []
    team_label = "Radiant" if ctx.recommending_team == 0 else "Dire"

    parts.append(f"Hero {hero_id} for {team_label} (slot {ctx.turn})")

    if baseline_win_rate is not None:
        parts.append(f"global WR {baseline_win_rate:.1%}")

    if team_hero_win_rate is not None:
        parts.append(f"team WR {team_hero_win_rate:.1%}")

    if synergy_win_rate is not None and ctx.ally_picks:
        parts.append(f"synergy {synergy_win_rate:.1%} with allies")

    if counter_win_rate is not None and ctx.enemy_picks:
        parts.append(f"counter {counter_win_rate:.1%} vs enemies")

    if h2h_win_rate is not None:
        parts.append(f"H2H WR {h2h_win_rate:.1%}")

    parts.append(f"model score {score:.4f}")

    return " | ".join(parts)
