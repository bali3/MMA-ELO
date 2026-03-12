"""Tests for UFC Stats idempotent persistence behavior."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ingestion.contracts import ParsedIngestionRecord
from ingestion.sources.ufc_stats import UFCStatsNormalizer, UFCStatsPersister


def _parsed_record() -> ParsedIngestionRecord:
    parsed_payload = {
        "event_id": "event001",
        "event_url": "http://ufcstats.com/event-details/event001",
        "event_name": "UFC Test Night",
        "event_date_utc": "2024-01-20T00:00:00Z",
        "venue": None,
        "city": "Las Vegas",
        "region": "Nevada",
        "country": "USA",
        "fighters": [
            {"fighter_id": "red001", "full_name": "Red Fighter"},
            {"fighter_id": "blue001", "full_name": "Blue Fighter"},
        ],
        "bouts": [
            {
                "bout_id": "bout001",
                "fight_url": "http://ufcstats.com/fight-details/bout001",
                "bout_order": 1,
                "fighter_red_id": "red001",
                "fighter_red_name": "Red Fighter",
                "fighter_blue_id": "blue001",
                "fighter_blue_name": "Blue Fighter",
                "winner_fighter_id": "red001",
                "result_method": "KO/TKO",
                "result_round": 1,
                "result_time": "2:10",
                "scheduled_rounds": None,
                "weight_class": None,
            }
        ],
    }

    return ParsedIngestionRecord(
        source_system="ufc_stats",
        source_record_id="event_page:event001",
        parsed_payload=parsed_payload,
        payload_sha256="payloadsha001",
        idempotency_key="idem001",
    )


def test_persistence_is_idempotent_for_same_rows(tmp_path: Path) -> None:
    """Persisting the same normalized rows twice should skip on second pass."""

    db_path = tmp_path / "mma.sqlite3"
    rows = UFCStatsNormalizer().normalize((_parsed_record(),))
    persister = UFCStatsPersister(db_path=db_path)

    first = persister.persist(rows)
    second = persister.persist(rows)

    assert first.inserted_count == len(rows)
    assert first.updated_count == 0
    assert first.skipped_count == 0

    assert second.inserted_count == 0
    assert second.updated_count == 0
    assert second.skipped_count == len(rows)

    with sqlite3.connect(db_path) as connection:
        fighter_count = connection.execute("SELECT COUNT(*) FROM fighters;").fetchone()[0]
        event_count = connection.execute("SELECT COUNT(*) FROM events;").fetchone()[0]
        bout_count = connection.execute("SELECT COUNT(*) FROM bouts;").fetchone()[0]

    assert fighter_count == 2
    assert event_count == 1
    assert bout_count == 1
