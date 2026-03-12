"""Tests for UFC Stats fighter-bout stats idempotent persistence behavior."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ingestion.contracts import ParsedIngestionRecord
from ingestion.sources.ufc_stats import UFCStatsBoutStatsNormalizer, UFCStatsPersister


def _parsed_record(*, red_kd: int, red_ctrl_seconds: int) -> ParsedIngestionRecord:
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
        "bout": {
            "bout_id": "bout001",
            "fight_url": "http://ufcstats.com/fight-details/bout001",
            "bout_order": 1,
            "fighter_red_id": "red001",
            "fighter_blue_id": "blue001",
            "winner_fighter_id": "red001",
            "result_method": "Decision",
            "result_round": 3,
            "result_time": "5:00",
            "scheduled_rounds": 3,
            "weight_class": "Lightweight",
        },
        "fighter_bout_stats": [
            {
                "fighter_id": "red001",
                "opponent_fighter_id": "blue001",
                "corner": "red",
                "knockdowns": red_kd,
                "sig_strikes_landed": 45,
                "sig_strikes_attempted": 98,
                "total_strikes_landed": 80,
                "total_strikes_attempted": 160,
                "takedowns_landed": 2,
                "takedowns_attempted": 5,
                "submission_attempts": 1,
                "reversals": 0,
                "control_time_seconds": red_ctrl_seconds,
            },
            {
                "fighter_id": "blue001",
                "opponent_fighter_id": "red001",
                "corner": "blue",
                "knockdowns": 0,
                "sig_strikes_landed": 40,
                "sig_strikes_attempted": 90,
                "total_strikes_landed": 70,
                "total_strikes_attempted": 150,
                "takedowns_landed": 1,
                "takedowns_attempted": 4,
                "submission_attempts": 0,
                "reversals": 1,
                "control_time_seconds": 120,
            },
        ],
    }
    return ParsedIngestionRecord(
        source_system="ufc_stats",
        source_record_id="fight_page:bout001",
        parsed_payload=parsed_payload,
        payload_sha256=f"payloadsha_{red_kd}_{red_ctrl_seconds}",
        idempotency_key=f"idem_{red_kd}_{red_ctrl_seconds}",
    )


def test_fighter_bout_stats_persistence_is_idempotent_and_upserts_updates(tmp_path: Path) -> None:
    """Second identical write should skip, changed fighter_bout_stats should update."""

    db_path = tmp_path / "mma.sqlite3"
    persister = UFCStatsPersister(db_path=db_path)
    normalizer = UFCStatsBoutStatsNormalizer()

    rows_v1 = normalizer.normalize((_parsed_record(red_kd=1, red_ctrl_seconds=240),))
    first = persister.persist(rows_v1)
    second = persister.persist(rows_v1)

    rows_v2 = normalizer.normalize((_parsed_record(red_kd=2, red_ctrl_seconds=300),))
    third = persister.persist(rows_v2)

    assert first.inserted_count == len(rows_v1)
    assert first.updated_count == 0
    assert first.skipped_count == 0

    assert second.inserted_count == 0
    assert second.updated_count == 0
    assert second.skipped_count == len(rows_v1)

    assert third.inserted_count == 0
    assert third.updated_count >= 1
    assert third.skipped_count >= 1

    with sqlite3.connect(db_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM fighter_bout_stats;").fetchone()[0]
        red_row = connection.execute(
            "SELECT knockdowns, control_time_seconds FROM fighter_bout_stats "
            "WHERE bout_id = ? AND fighter_id = ?",
            ("bout001", "red001"),
        ).fetchone()

    assert count == 2
    assert red_row == (2, 300)


def test_fighter_bout_stats_persistence_updates_from_event_fallback_shape(tmp_path: Path) -> None:
    db_path = tmp_path / "mma.sqlite3"
    persister = UFCStatsPersister(db_path=db_path)
    normalizer = UFCStatsBoutStatsNormalizer()

    fallback_rows = normalizer.normalize(
        (
            ParsedIngestionRecord(
                source_system="ufc_stats",
                source_record_id="fight_page:bout001",
                parsed_payload={
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
                    "bout": {
                        "bout_id": "bout001",
                        "fight_url": "http://ufcstats.com/fight-details/bout001",
                        "bout_order": 1,
                        "fighter_red_id": "red001",
                        "fighter_blue_id": "blue001",
                        "winner_fighter_id": "red001",
                        "result_method": "Decision",
                        "result_round": 3,
                        "result_time": "5:00",
                        "scheduled_rounds": 3,
                        "weight_class": "Lightweight",
                        "stats_source_type": "event_details_fallback",
                    },
                    "fighter_bout_stats": [
                        {
                            "fighter_id": "red001",
                            "opponent_fighter_id": "blue001",
                            "corner": "red",
                            "knockdowns": 1,
                            "sig_strikes_landed": 30,
                            "sig_strikes_attempted": None,
                            "total_strikes_landed": None,
                            "total_strikes_attempted": None,
                            "takedowns_landed": 2,
                            "takedowns_attempted": None,
                            "submission_attempts": 1,
                            "reversals": None,
                            "control_time_seconds": None,
                        },
                        {
                            "fighter_id": "blue001",
                            "opponent_fighter_id": "red001",
                            "corner": "blue",
                            "knockdowns": 0,
                            "sig_strikes_landed": 20,
                            "sig_strikes_attempted": None,
                            "total_strikes_landed": None,
                            "total_strikes_attempted": None,
                            "takedowns_landed": 1,
                            "takedowns_attempted": None,
                            "submission_attempts": 0,
                            "reversals": None,
                            "control_time_seconds": None,
                        },
                    ],
                },
                payload_sha256="fallback_sha",
                idempotency_key="fallback_idem",
            ),
        )
    )
    full_rows = normalizer.normalize((_parsed_record(red_kd=2, red_ctrl_seconds=300),))

    first = persister.persist(fallback_rows)
    second = persister.persist(full_rows)

    assert first.inserted_count == len(fallback_rows)
    assert second.updated_count >= 1

    with sqlite3.connect(db_path) as connection:
        red_row = connection.execute(
            """
            SELECT sig_strikes_attempted, total_strikes_landed, total_strikes_attempted, takedowns_attempted, control_time_seconds
            FROM fighter_bout_stats
            WHERE bout_id = ? AND fighter_id = ?
            """,
            ("bout001", "red001"),
        ).fetchone()
    assert red_row == (98, 80, 160, 5, 300)
