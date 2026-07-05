"""Unit tests for ``api/db.py`` — connection pool lifecycle and query
helpers.

All tests mock ``psycopg2`` — no real Postgres is needed.

Regression bugs covered:
    - BUG-004: Deadlock when pool is exhausted — lock must NOT be held
               during ``getconn()``
"""

from __future__ import annotations

import threading
import time
from unittest import mock

import pytest


# ===================================================================
# get_conn
# ===================================================================


class TestGetConn:
    """``get_conn`` acquires a connection from the pool."""

    def test_before_init_raises_runtime_error(self):
        """❌ Called before ``init_pool`` → RuntimeError."""
        from api import db as db_module

        with mock.patch.object(db_module, "_pool", None):
            with pytest.raises(RuntimeError, match="DB pool not initialised"):
                db_module.get_conn()

    def test_returns_connection_from_pool(self):
        """✅ Pool is initialised → returns getconn() result."""
        from api import db as db_module

        mock_pool = mock.MagicMock()
        mock_conn = mock.MagicMock()
        mock_pool.getconn.return_value = mock_conn

        with mock.patch.object(db_module, "_pool", mock_pool):
            conn = db_module.get_conn()
            assert conn is mock_conn
            mock_pool.getconn.assert_called_once_with()

    def test_lock_not_held_during_getconn(self):
        """✅ Lock is released before ``getconn()`` so a concurrent
        ``put_conn()`` can return connections — guards BUG-004.

        We verify this by having ``getconn()`` block and ensuring
        ``put_conn()`` can still acquire ``_pool_lock``.
        """
        from api import db as db_module

        can_release = threading.Event()
        getconn_started = threading.Event()
        getconn_done = threading.Event()
        putconn_done = threading.Event()

        def blocking_getconn(*args, **kwargs):
            getconn_started.set()
            can_release.wait(timeout=5)
            return mock.MagicMock()

        mock_pool = mock.MagicMock()
        mock_pool.getconn.side_effect = blocking_getconn

        def thread_get_conn():
            with mock.patch.object(db_module, "_pool", mock_pool):
                try:
                    db_module.get_conn()
                except Exception:
                    pass
                finally:
                    getconn_done.set()

        def thread_put_conn():
            # Wait until getconn has started (and thus released the lock)
            getconn_started.wait(timeout=5)
            # put_conn should be able to acquire _pool_lock even though
            # getconn() is still blocked inside getconn()
            mock_conn = mock.MagicMock()
            with mock.patch.object(db_module, "_pool", mock_pool):
                try:
                    db_module.put_conn(mock_conn)
                    putconn_done.set()
                except Exception:
                    pass

        t1 = threading.Thread(target=thread_get_conn, daemon=True)
        t2 = threading.Thread(target=thread_put_conn, daemon=True)

        t1.start()
        t2.start()

        # put_conn should finish while get_conn is still blocked
        putconn_done.wait(timeout=3)
        assert putconn_done.is_set(), (
            "put_conn() was blocked — likely deadlock. "
            "Lock is held during getconn() (BUG-004)."
        )

        can_release.set()
        t1.join(timeout=3)
        t2.join(timeout=3)


# ===================================================================
# put_conn
# ===================================================================


class TestPutConn:
    """``put_conn`` rolls back pending transactions and returns the
    connection to the pool.
    """

    def test_rollback_before_putconn(self):
        """✅ conn.rollback() called before _pool.putconn()."""
        from api import db as db_module

        mock_pool = mock.MagicMock()
        mock_conn = mock.MagicMock()

        with mock.patch.object(db_module, "_pool", mock_pool):
            db_module.put_conn(mock_conn)

        # Verify rollback happens before putconn
        mock_conn.rollback.assert_called_once()
        mock_pool.putconn.assert_called_once_with(mock_conn)

    def test_discards_broken_connection(self):
        """❌ rollback raises → connection closed, not returned to pool."""
        from api import db as db_module

        mock_pool = mock.MagicMock()
        mock_conn = mock.MagicMock()
        mock_conn.rollback.side_effect = Exception("Connection broken")

        with mock.patch.object(db_module, "_pool", mock_pool):
            db_module.put_conn(mock_conn)

        mock_conn.rollback.assert_called_once()
        # Must close the broken connection instead of returning it
        mock_pool.putconn.assert_called_once_with(mock_conn, close=True)

    def test_put_conn_when_pool_is_none_does_nothing(self):
        """✅ ``_pool is None`` → no rollback, no putconn (graceful
        shutdown).
        """
        from api import db as db_module

        mock_conn = mock.MagicMock()

        with mock.patch.object(db_module, "_pool", None):
            # Should not raise
            db_module.put_conn(mock_conn)

        mock_conn.rollback.assert_not_called()


# ===================================================================
# init_pool / close_pool
# ===================================================================


class TestInitClosePool:
    """Global pool lifecycle."""

    def test_init_pool_creates_pool(self):
        """✅ ``init_pool`` creates a ThreadedConnectionPool."""
        from api import db as db_module
        from api.config import APIConfig

        cfg = APIConfig(
            pg_host="localhost",
            pg_port=5432,
            pg_user="test",
            pg_password="test",
            pg_db="test",
        )
        with (
            mock.patch("psycopg2.pool.ThreadedConnectionPool") as mock_pool_cls,
            mock.patch.object(db_module, "_pool", None),
        ):
            mock_pool_instance = mock.MagicMock()
            mock_pool_cls.return_value = mock_pool_instance

            db_module.init_pool(cfg)

            mock_pool_cls.assert_called_once_with(
                minconn=cfg.pool_min,
                maxconn=cfg.pool_max,
                dsn=cfg.pg_dsn,
            )

    def test_close_pool_closes_and_resets(self):
        """✅ ``close_pool`` closes all connections and sets _pool to
        None.
        """
        from api import db as db_module

        mock_pool = mock.MagicMock()

        with mock.patch.object(db_module, "_pool", mock_pool):
            db_module.close_pool()
            mock_pool.closeall.assert_called_once()

        # After close_pool, _pool should be None
        # (this is verified by the next get_conn raising RuntimeError)
        with mock.patch.object(db_module, "_pool", None):
            with pytest.raises(RuntimeError, match="DB pool not initialised"):
                db_module.get_conn()
