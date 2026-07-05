"""Tests for ``aggregates.py`` — match filter helpers and SQL safety guards.

Regression bugs covered:
    - _VALID_TABLES guard prevents SQL injection via table name parameter
    - _match_extra_where produces consistent filters across all populators
    - _clean_patch_rows does NOT commit (caller owns the transaction)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from trainer.aggregates import _clean_patch_rows, _match_extra_where, _VALID_TABLES
from trainer.config import TrainerConfig


# ===========================================================================
# _VALID_TABLES guard
# ===========================================================================

class TestValidTables:
    """``_clean_patch_rows`` must reject unknown table names to prevent
    SQL injection via the f-string table parameter."""

    def test_known_table_accepted(self):
        conn = MagicMock()
        _clean_patch_rows(conn, "ml.team_hero_agg", 60)
        conn.cursor.assert_called_once()

    def test_unknown_table_rejected(self):
        conn = MagicMock()
        with pytest.raises(ValueError, match="Invalid aggregate table"):
            _clean_patch_rows(conn, "ml.drop_everything", 60)

    def test_all_expected_tables_present(self):
        expected = {
            "ml.team_hero_agg", "ml.player_hero_agg",
            "ml.hero_synergy_agg", "ml.hero_counter_agg",
            "ml.team_h2h_agg", "ml.hero_baseline_agg",
            "ml.hero_draft_slot_agg",
            "ml.team_hero_snapshot", "ml.player_hero_snapshot",
            "ml.hero_synergy_snapshot", "ml.hero_counter_snapshot",
            "ml.team_h2h_snapshot", "ml.hero_baseline_snapshot",
            "ml.hero_draft_slot_snapshot",
        }
        assert expected == _VALID_TABLES


# ===========================================================================
# _clean_patch_rows — transaction semantics
# ===========================================================================

class TestCleanPatchRows:
    """``_clean_patch_rows`` must NOT call ``conn.commit()`` — the caller
    owns the transaction for atomic DELETE+INSERT."""

    def test_no_commit_called(self):
        conn = MagicMock()
        _clean_patch_rows(conn, "ml.team_hero_agg", 60)
        conn.commit.assert_not_called()

    def test_delete_executed_with_patch_id(self):
        conn = MagicMock()
        _clean_patch_rows(conn, "ml.team_hero_agg", 60)
        cur = conn.cursor.return_value.__enter__.return_value
        cur.execute.assert_called_once()
        args = cur.execute.call_args
        assert "DELETE FROM ml.team_hero_agg WHERE patch_id = %s" == args[0][0]
        assert (60,) == args[0][1]


# ===========================================================================
# _match_extra_where
# ===========================================================================

class TestMatchExtraWhere:
    """``_match_extra_where`` must produce consistent SQL fragments
    from the config-driven league/lobby filters."""

    def test_no_filters_returns_empty(self):
        cfg = TrainerConfig(league_only=False, lobby_types="")
        assert _match_extra_where(cfg) == ""

    def test_league_only_filter(self):
        cfg = TrainerConfig(league_only=True, lobby_types="")
        result = _match_extra_where(cfg)
        assert "m.leagueid > 0" in result
        assert result.startswith(" AND ")

    def test_lobby_types_filter(self):
        cfg = TrainerConfig(league_only=False, lobby_types="7,8")
        result = _match_extra_where(cfg)
        assert "m.lobby_type IN (7,8)" in result

    def test_combined_filters(self):
        cfg = TrainerConfig(league_only=True, lobby_types="7,8")
        result = _match_extra_where(cfg)
        assert "m.leagueid > 0" in result
        assert "m.lobby_type IN (7,8)" in result

    def test_custom_alias(self):
        cfg = TrainerConfig(league_only=True, lobby_types="")
        result = _match_extra_where(cfg, alias="t")
        assert "t.leagueid > 0" in result

    def test_empty_lobby_types_no_filter(self):
        cfg = TrainerConfig(league_only=False, lobby_types="  ")
        assert _match_extra_where(cfg) == ""
