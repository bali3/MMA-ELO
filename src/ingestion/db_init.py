"""Database initialization helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ingestion.schema import SCHEMA_DDL_STATEMENTS


def initialize_sqlite_database(path: Path) -> None:
    """Create all database schema objects in SQLite.

    Assumption: schema creation is explicit and never triggered implicitly during imports.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        create_schema(connection)
        connection.commit()


def create_schema(connection: sqlite3.Connection) -> None:
    """Apply table and index DDL statements to an open SQLite connection.

    The function enforces foreign key checks and runs all DDL in a single
    transaction to keep schema creation atomic.
    """

    connection.execute("PRAGMA foreign_keys = ON;")
    for ddl in SCHEMA_DDL_STATEMENTS:
        connection.execute(ddl)
