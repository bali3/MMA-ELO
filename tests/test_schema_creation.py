"""Tests for SQLite schema creation and structural integrity."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ingestion.db_init import initialize_sqlite_database
from ingestion.schema import REQUIRED_TABLE_NAMES


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name});").fetchall()
    return {str(row[1]) for row in rows}


def _unique_indexes(connection: sqlite3.Connection, table_name: str) -> list[tuple[str, ...]]:
    indexes = connection.execute(f"PRAGMA index_list({table_name});").fetchall()
    unique_columns: list[tuple[str, ...]] = []
    for index in indexes:
        index_name = str(index[1])
        is_unique = int(index[2]) == 1
        if not is_unique:
            continue
        index_info = connection.execute(f"PRAGMA index_info({index_name});").fetchall()
        unique_columns.append(tuple(str(row[2]) for row in index_info))
    return unique_columns


def _has_unique_index(
    connection: sqlite3.Connection,
    table_name: str,
    expected_columns: tuple[str, ...],
) -> bool:
    indexes = _unique_indexes(connection, table_name)
    return any(index == expected_columns for index in indexes)


def test_initialize_sqlite_database_creates_required_tables(tmp_path: Path) -> None:
    """Schema creation should produce all required MMA domain tables."""

    db_path = tmp_path / "mma_schema.sqlite3"
    initialize_sqlite_database(db_path)

    with sqlite3.connect(db_path) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name;"
        ).fetchall()
        table_names = {str(row[0]) for row in tables}

    assert set(REQUIRED_TABLE_NAMES).issubset(table_names)


def test_schema_includes_required_provenance_columns(tmp_path: Path) -> None:
    """All domain tables should include source provenance and lifecycle timestamps."""

    db_path = tmp_path / "mma_schema.sqlite3"
    initialize_sqlite_database(db_path)

    required_common_columns = {
        "source_system",
        "source_record_id",
        "source_updated_at_utc",
        "source_payload_sha256",
        "created_at_utc",
        "updated_at_utc",
    }
    tables_with_ingested_at = {"fighters", "events", "bouts", "fighter_bout_stats", "markets"}

    with sqlite3.connect(db_path) as connection:
        for table_name in REQUIRED_TABLE_NAMES:
            columns = _table_columns(connection, table_name)
            assert required_common_columns.issubset(columns)
            if table_name in tables_with_ingested_at:
                assert "ingested_at_utc" in columns


def test_schema_has_expected_unique_constraints(tmp_path: Path) -> None:
    """Schema must enforce key uniqueness at source and entity levels."""

    db_path = tmp_path / "mma_schema.sqlite3"
    initialize_sqlite_database(db_path)

    with sqlite3.connect(db_path) as connection:
        for table_name in REQUIRED_TABLE_NAMES:
            assert _has_unique_index(connection, table_name, ("source_system", "source_record_id"))

        assert _has_unique_index(connection, "bouts", ("event_id", "bout_order"))
        assert _has_unique_index(connection, "fighter_bout_stats", ("bout_id", "fighter_id"))
        assert _has_unique_index(
            connection,
            "model_predictions",
            ("model_run_id", "bout_id", "target_fighter_id", "prediction_type"),
        )


def test_schema_has_core_foreign_keys(tmp_path: Path) -> None:
    """Relational links should exist between events, bouts, fighters, and stats."""

    db_path = tmp_path / "mma_schema.sqlite3"
    initialize_sqlite_database(db_path)

    with sqlite3.connect(db_path) as connection:
        bout_fks = connection.execute("PRAGMA foreign_key_list(bouts);").fetchall()
        bout_fk_targets = {(str(row[3]), str(row[2])) for row in bout_fks}
        assert ("event_id", "events") in bout_fk_targets
        assert ("fighter_red_id", "fighters") in bout_fk_targets
        assert ("fighter_blue_id", "fighters") in bout_fk_targets

        stats_fks = connection.execute("PRAGMA foreign_key_list(fighter_bout_stats);").fetchall()
        stats_fk_targets = {(str(row[3]), str(row[2])) for row in stats_fks}
        assert ("bout_id", "bouts") in stats_fk_targets
        assert ("fighter_id", "fighters") in stats_fk_targets
        assert ("opponent_fighter_id", "fighters") in stats_fk_targets
