"""Tests for UFC Stats fighter metadata idempotent persistence behavior."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ingestion.contracts import ParsedIngestionRecord
from ingestion.sources.ufc_stats import UFCStatsFighterMetadataNormalizer, UFCStatsPersister


def _parsed_record(*, stance: str) -> ParsedIngestionRecord:
    parsed_payload = {
        "fighter_id": "abc123",
        "fighter_url": "http://ufcstats.com/fighter-details/abc123",
        "full_name": "Conor McGregor",
        "date_of_birth_utc": "1988-07-14T00:00:00Z",
        "nationality": None,
        "stance": stance,
        "height_cm": 175.26,
        "reach_cm": 187.96,
    }
    return ParsedIngestionRecord(
        source_system="ufc_stats",
        source_record_id="fighter_profile:abc123",
        parsed_payload=parsed_payload,
        payload_sha256=f"payloadsha_{stance}",
        idempotency_key=f"idem_{stance}",
    )


def test_fighter_metadata_persistence_is_idempotent_and_upserts_updates(tmp_path: Path) -> None:
    """Second identical write should skip, changed payload should update."""

    db_path = tmp_path / "mma.sqlite3"
    persister = UFCStatsPersister(db_path=db_path)
    normalizer = UFCStatsFighterMetadataNormalizer()

    rows_v1 = normalizer.normalize((_parsed_record(stance="Southpaw"),))
    first = persister.persist(rows_v1)
    second = persister.persist(rows_v1)

    rows_v2 = normalizer.normalize((_parsed_record(stance="Orthodox"),))
    third = persister.persist(rows_v2)

    assert first.inserted_count == 1
    assert first.updated_count == 0
    assert first.skipped_count == 0

    assert second.inserted_count == 0
    assert second.updated_count == 0
    assert second.skipped_count == 1

    assert third.inserted_count == 0
    assert third.updated_count == 1
    assert third.skipped_count == 0

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT fighter_id, full_name, stance, height_cm, reach_cm, date_of_birth_utc "
            "FROM fighters WHERE fighter_id = ?",
            ("abc123",),
        ).fetchone()

    assert row == (
        "abc123",
        "Conor McGregor",
        "Orthodox",
        175.26,
        187.96,
        "1988-07-14T00:00:00Z",
    )
