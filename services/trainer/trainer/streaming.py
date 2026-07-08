"""Server-side cursor utilities for memory-efficient SQL reads.

Provides chunked data loading for large datasets that would OOM with
standard pd.read_sql(). Uses PostgreSQL server-side cursors to stream
results in configurable batches.
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def read_sql_chunked(
    sql: str,
    engine,
    params: dict | None = None,
    chunk_size: int = 10000,
    dtypes: dict[str, str] | None = None,
) -> Iterator[pd.DataFrame]:
    """Read SQL results in chunks using a server-side cursor.

    Yields DataFrames of at most *chunk_size* rows. Memory usage stays
    constant regardless of total result size.

    Parameters
    ----------
    sql : str
        SQL query to execute.
    engine : sqlalchemy.engine
        Database engine.
    params : dict, optional
        Query parameters.
    chunk_size : int
        Number of rows per chunk.
    dtypes : dict, optional
        Column dtype overrides for the output DataFrame.
    """
    conn = engine.raw_connection()
    try:
        cursor_name = f"chunked_{id(sql) % 10000}"
        with conn.cursor(name=cursor_name) as cur:
            cur.itersize = chunk_size
            cur.execute(sql, params or {})
            col_names = [desc[0] for desc in cur.description]

            while True:
                rows = cur.fetchmany(chunk_size)
                if not rows:
                    break
                df = pd.DataFrame(rows, columns=col_names)
                if dtypes:
                    for col, dtype in dtypes.items():
                        if col in df.columns:
                            df[col] = df[col].astype(dtype, errors="ignore")
                yield df
    finally:
        conn.close()


def read_sql_chunked_numpy(
    sql: str,
    engine,
    params: dict | None = None,
    chunk_size: int = 10000,
    float_cols: list[str] | None = None,
    int_cols: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], list]:
    """Read SQL results into numpy arrays using server-side cursor.

    More memory-efficient than DataFrame for numeric-heavy workloads.
    Returns (column_dict, row_count) where column_dict maps column names
    to numpy arrays.
    """
    float_cols = float_cols or []
    int_cols = int_cols or []
    columns: dict[str, list] = {}
    total_rows = 0

    conn = engine.raw_connection()
    try:
        cursor_name = f"numpy_{id(sql) % 10000}"
        with conn.cursor(name=cursor_name) as cur:
            cur.itersize = chunk_size
            cur.execute(sql, params or {})
            col_names = [desc[0] for desc in cur.description]

            for name in col_names:
                columns[name] = []

            while True:
                rows = cur.fetchmany(chunk_size)
                if not rows:
                    break
                total_rows += len(rows)
                for row in rows:
                    for i, name in enumerate(col_names):
                        val = row[i]
                        if name in float_cols:
                            columns[name].append(float(val) if val is not None else 0.0)
                        elif name in int_cols:
                            columns[name].append(int(val) if val is not None else 0)
                        else:
                            columns[name].append(val)
    finally:
        conn.close()

    # Convert lists to numpy arrays
    result = {}
    for name in col_names:
        if name in float_cols:
            result[name] = np.array(columns[name], dtype=np.float32)
        elif name in int_cols:
            result[name] = np.array(columns[name], dtype=np.int64)
        else:
            result[name] = columns[name]  # Keep as list for mixed types

    logger.info("Loaded %d rows via server-side cursor", total_rows)
    return result, total_rows
