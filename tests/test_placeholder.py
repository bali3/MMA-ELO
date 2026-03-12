"""Placeholder tests for initial scaffold."""

from ingestion.schema import TABLES


def test_schema_tables_exist() -> None:
    """Ensure the initial schema exports at least one table definition."""
    assert len(TABLES) > 0
