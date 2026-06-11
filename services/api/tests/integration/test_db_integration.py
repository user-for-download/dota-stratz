"""Integration tests for ``api/db.py`` — query helpers against a real
Postgres database.

All tests in this module require ``@pytest.mark.integration`` and a
running Postgres instance.  Connection details are read from environment
variables:

    POSTGRES_DSN (preferred)
    or POSTGRES_HOST / POSTGRES_PORT / POSTGRES_DB / POSTGRES_USER / POSTGRES_PASSWORD

Temp tables are created per test and cleaned up in teardown.

Regression bugs covered:
    - BUG-007: account_id=0 should not be skipped by falsy check
"""

from __future__ import annotations

import os
from typing import Any

import psycopg2
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DSN = (
    os.getenv("POSTGRES_DSN")
    or (
        f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
        f"port={os.getenv('POSTGRES_PORT', '5432')} "
        f"dbname={os.getenv('POSTGRES_DB', 'dota2')} "
        f"user={os.getenv('POSTGRES_USER', 'dota2')} "
        f"password={os.getenv('POSTGRES_PASSWORD', 'dota2')}"
    )
)

_SCHEMA = "ml"
_TEMP_TABLES: list[str] = []


def _ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {_SCHEMA}")
    conn.commit()


def _create_temp_table(conn, name: str, ddl: str):
    full_name = f"{_SCHEMA}.{name}"
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {full_name} CASCADE")
        cur.execute(f"CREATE TABLE {full_name} ({ddl})")
    conn.commit()
    _TEMP_TABLES.append(full_name)


def _drop_temp_tables(conn):
    with conn.cursor() as cur:
        for tbl in _TEMP_TABLES:
            cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
    conn.commit()
    _TEMP_TABLES.clear()


# ---------------------------------------------------------------------------
# Fixture: real DB connection
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pg_conn():
    """Provide a real psycopg2 connection to the test database.

    Tests that use this fixture must be marked ``@pytest.mark.integration``
    and require a running Postgres instance.
    """
    conn = psycopg2.connect(DSN)
    _ensure_schema(conn)
    yield conn
    _drop_temp_tables(conn)
    conn.close()


# ---------------------------------------------------------------------------
# fetch_player_hero_agg_batch — account_id=0 (BUG-007)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFetchPlayerHeroAggBatchAccountIdZero:
    """``fetch_player_hero_agg_batch`` must NOT skip account_id=0
    because ``0`` is falsy but a valid Steam account identifier.

    Regression guard for BUG-007.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, pg_conn):
        """Create temp table and insert a row with account_id=0."""
        _create_temp_table(
            pg_conn,
            "player_hero_agg",
            """
            patch_id INT,
            account_id INT,
            hero_id INT,
            games INT DEFAULT 0,
            wins INT DEFAULT 0,
            win_rate REAL DEFAULT 0.0,
            avg_gpm REAL DEFAULT 0.0,
            avg_xpm REAL DEFAULT 0.0,
            avg_kills REAL DEFAULT 0.0,
            avg_deaths REAL DEFAULT 0.0,
            avg_assists REAL DEFAULT 0.0,
            avg_kda REAL DEFAULT 0.0,
            lane_role INT DEFAULT 0,
            firstblood_rate REAL DEFAULT 0.0,
            avg_camps_stacked REAL DEFAULT 0.0,
            avg_vision_placed REAL DEFAULT 0.0,
            avg_gold_10 REAL DEFAULT 0.0,
            avg_xp_10 REAL DEFAULT 0.0,
            PRIMARY KEY (patch_id, account_id, hero_id)
            """,
        )
        with pg_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ml.player_hero_agg
                    (patch_id, account_id, hero_id, games, wins, win_rate)
                VALUES (1, 0, 42, 10, 6, 0.6)
                """,
            )
        pg_conn.commit()
        yield
        # Cleanup is handled by the module-scoped fixture

    def test_row_with_account_id_zero_is_returned(self, pg_conn):
        """✅ Row with account_id=0 is returned, not skipped.

        Regression guard for BUG-007.
        """
        from api.db import fetch_player_hero_agg_batch

        # We need to monkey-patch the module's pool to use our real conn.
        # The simplest approach: call the underlying logic directly by
        # creating a minimal mock pool that provides our real connection.
        with _mock_pool(pg_conn):
            result = fetch_player_hero_agg_batch(
                patch_id=1,
                account_id=0,
                hero_ids=[42],
            )

        assert 42 in result, (
            f"Expected hero_id=42 in result for account_id=0, "
            f"got {result} — BUG-007: falsy account_id was skipped"
        )
        assert result[42]["games"] == 10
        assert result[42]["win_rate"] == 0.6

    def test_account_id_zero_with_unknown_hero_returns_empty(
        self, pg_conn,
    ):
        """✅ account_id=0 + unknown hero_id → empty dict (no crash)."""
        from api.db import fetch_player_hero_agg_batch

        with _mock_pool(pg_conn):
            result = fetch_player_hero_agg_batch(
                patch_id=1,
                account_id=0,
                hero_ids=[999],
            )

        assert result == {}


# ---------------------------------------------------------------------------
# fetch_baselines_batch — correct mapping
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFetchBaselinesBatch:
    """``fetch_baselines_batch`` returns correct hero_id → row mapping."""

    @pytest.fixture(autouse=True)
    def _setup(self, pg_conn):
        """Create temp table and insert 2 baseline rows."""
        _create_temp_table(
            pg_conn,
            "hero_baseline_agg",
            """
            patch_id INT,
            hero_id INT,
            total_picks INT DEFAULT 0,
            total_wins INT DEFAULT 0,
            total_bans INT DEFAULT 0,
            win_rate REAL DEFAULT 0.0,
            pick_rate REAL DEFAULT 0.0,
            ban_rate REAL DEFAULT 0.0,
            avg_gpm REAL DEFAULT 0.0,
            avg_xpm REAL DEFAULT 0.0,
            avg_kills REAL DEFAULT 0.0,
            avg_deaths REAL DEFAULT 0.0,
            avg_assists REAL DEFAULT 0.0,
            avg_gold_10 REAL DEFAULT 0.0,
            avg_xp_10 REAL DEFAULT 0.0,
            PRIMARY KEY (patch_id, hero_id)
            """,
        )
        with pg_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ml.hero_baseline_agg
                    (patch_id, hero_id, total_picks, total_wins, total_bans,
                     win_rate, pick_rate, ban_rate)
                VALUES
                    (1, 1, 100, 55, 20, 0.55, 0.10, 0.05),
                    (1, 2, 200, 110, 30, 0.55, 0.20, 0.08)
                """,
            )
        pg_conn.commit()
        yield

    def test_both_heroes_returned(self, pg_conn):
        """✅ Both heroes appear in the result dict."""
        from api.db import fetch_baselines_batch

        with _mock_pool(pg_conn):
            result = fetch_baselines_batch(patch_id=1, hero_ids=[1, 2])

        assert len(result) == 2
        assert 1 in result
        assert 2 in result

    def test_hero_fields_match_inserted_values(self, pg_conn):
        """✅ All queried fields match what was inserted."""
        from api.db import fetch_baselines_batch

        with _mock_pool(pg_conn):
            result = fetch_baselines_batch(patch_id=1, hero_ids=[1])

        row = result[1]
        assert row["total_picks"] == 100
        assert row["total_wins"] == 55
        assert row["total_bans"] == 20
        assert row["win_rate"] == 0.55
        assert row["pick_rate"] == 0.10
        assert row["ban_rate"] == 0.08

    def test_unknown_hero_returns_empty_dict(self, pg_conn):
        """❌ Querying a hero_id that doesn't exist → {} for that hero."""
        from api.db import fetch_baselines_batch

        with _mock_pool(pg_conn):
            result = fetch_baselines_batch(patch_id=1, hero_ids=[999])

        assert result == {}

    def test_two_heroes_one_unknown_partial_result(self, pg_conn):
        """✅ Mix of known + unknown heroes returns only the known ones."""
        from api.db import fetch_baselines_batch

        with _mock_pool(pg_conn):
            result = fetch_baselines_batch(patch_id=1, hero_ids=[1, 999])

        assert 1 in result
        assert 999 not in result


# ---------------------------------------------------------------------------
# Context manager: replace db._pool with a mock using the real connection
# ---------------------------------------------------------------------------


from contextlib import contextmanager
from unittest import mock


@contextmanager
def _mock_pool(real_conn):
    """Replace ``api.db._pool`` with a mock that returns *real_conn*.

    This allows integration tests to exercise the real query functions
    (e.g. ``fetch_baselines_batch``) while using a real database
    connection, without requiring a full ``init_pool()`` call.
    """
    from api import db as db_module

    fake_pool = mock.MagicMock()
    fake_pool.getconn.return_value = real_conn

    def fake_putconn(conn, close=False):
        # Don't actually close the real connection
        conn.rollback()

    fake_pool.putconn.side_effect = fake_putconn

    with mock.patch.object(db_module, "_pool", fake_pool):
        yield
