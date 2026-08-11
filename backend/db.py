"""Postgres connection pool (pgvector-enabled)."""
from __future__ import annotations

import os

from psycopg_pool import ConnectionPool

DSN = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:mandaflix@localhost:5433/mandaflix"
)

pool = ConnectionPool(DSN, min_size=2, max_size=8, open=True)


def query(sql: str, params: tuple = ()) -> list[tuple]:
    with pool.connection() as con:
        return con.execute(sql, params).fetchall()


def execute(sql: str, params: tuple = ()) -> None:
    with pool.connection() as con:
        con.execute(sql, params)
        con.commit()
