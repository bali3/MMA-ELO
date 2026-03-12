"""Tests for The Odds API ingestion, bout matching, and market coverage reporting."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ingestion.contracts import RawIngestionRecord
from ingestion.db_init import initialize_sqlite_database
from ingestion.sources.odds import OddsParser, run_market_odds_ingestion
from markets import run_market_coverage_report


def _seed_fighter(connection: sqlite3.Connection, fighter_id: str, name: str, now: str) -> None:
    connection.execute(
        """
        INSERT INTO fighters (
            fighter_id, full_name, date_of_birth_utc, nationality, stance, height_cm, reach_cm,
            source_system, source_record_id, source_updated_at_utc, source_payload_sha256,
            ingested_at_utc, created_at_utc, updated_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (fighter_id, name, None, None, None, None, None, "fixture", f"fighter:{fighter_id}", None, "sha", now, now, now),
    )


def _seed_event(connection: sqlite3.Connection, event_id: str, event_date_utc: str, now: str) -> None:
    connection.execute(
        """
        INSERT INTO events (
            event_id, promotion, event_name, event_date_utc, venue, city, region, country, timezone_name,
            source_system, source_record_id, source_updated_at_utc, source_payload_sha256,
            ingested_at_utc, created_at_utc, updated_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (event_id, "UFC", event_id, event_date_utc, None, None, None, None, None, "fixture", f"event:{event_id}", None, "sha", now, now, now),
    )


def _seed_bout(
    connection: sqlite3.Connection,
    *,
    bout_id: str,
    event_id: str,
    red_id: str,
    blue_id: str,
    bout_start_utc: str,
    now: str,
) -> None:
    connection.execute(
        """
        INSERT INTO bouts (
            bout_id, event_id, bout_order, fighter_red_id, fighter_blue_id, weight_class, gender,
            is_title_fight, scheduled_rounds, bout_start_time_utc, result_method, result_round,
            result_time_seconds, winner_fighter_id, source_system, source_record_id, source_updated_at_utc,
            source_payload_sha256, ingested_at_utc, created_at_utc, updated_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            bout_id,
            event_id,
            1,
            red_id,
            blue_id,
            "Lightweight",
            "M",
            0,
            3,
            bout_start_utc,
            "Decision",
            3,
            300,
            red_id,
            "fixture",
            f"bout:{bout_id}",
            None,
            "sha",
            now,
            now,
            now,
        ),
    )


def _write_odds_snapshot(raw_root: Path, filename: str, payload: object, *, source_system: str = "odds_api") -> None:
    archive_dir = raw_root / source_system / "2026" / "03" / "12"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / filename).write_text(json.dumps(payload), encoding="utf-8")


def test_odds_parser_extracts_fight_winner_markets_only() -> None:
    parser = OddsParser()
    snapshot_payload = [
        {
            "id": "event_1",
            "commence_time": "2024-01-01T20:00:00Z",
            "bookmakers": [
                {
                    "key": "book_a",
                    "title": "Book A",
                    "last_update": "2024-01-01T19:00:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2024-01-01T19:00:00Z",
                            "outcomes": [
                                {"name": "Red Corner", "price": -150},
                                {"name": "Blue Corner", "price": 130},
                            ],
                        },
                        {
                            "key": "totals",
                            "last_update": "2024-01-01T19:00:00Z",
                            "outcomes": [
                                {"name": "Over 2.5", "price": -110},
                                {"name": "Under 2.5", "price": -110},
                            ],
                        },
                    ],
                }
            ],
        }
    ]

    raw_record = RawIngestionRecord(
        source_system="odds_api",
        source_record_id="odds_snapshot:test",
        fetched_at_utc=datetime(2026, 3, 12, tzinfo=timezone.utc),
        payload={"events": snapshot_payload},
        payload_sha256="sha",
        idempotency_key="idk",
    )

    parsed = parser.parse((raw_record,))
    assert len(parsed) == 2

    first = parsed[0].parsed_payload
    second = parsed[1].parsed_payload
    assert str(first["market_type"]) == "moneyline"
    assert str(second["market_type"]) == "moneyline"
    assert {str(first["fighter_name"]), str(second["fighter_name"])} == {"Red Corner", "Blue Corner"}


def test_market_odds_ingestion_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "mma.sqlite3"
    raw_root = tmp_path / "raw"
    report_path = tmp_path / "market_report.txt"
    initialize_sqlite_database(db_path)

    now = "2026-03-12T00:00:00Z"
    with sqlite3.connect(db_path) as connection:
        _seed_fighter(connection, "f_red", "Red Corner", now)
        _seed_fighter(connection, "f_blue", "Blue Corner", now)
        _seed_event(connection, "event_2024", "2024-01-01T00:00:00Z", now)
        _seed_bout(
            connection,
            bout_id="bout_1",
            event_id="event_2024",
            red_id="f_red",
            blue_id="f_blue",
            bout_start_utc="2024-01-01T20:00:00Z",
            now=now,
        )
        connection.commit()

    snapshot = [
        {
            "id": "event_2024",
            "commence_time": "2024-01-01T20:00:00Z",
            "bookmakers": [
                {
                    "key": "book_a",
                    "title": "Book A",
                    "last_update": "2024-01-01T19:00:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2024-01-01T19:00:00Z",
                            "outcomes": [
                                {"name": "Red Corner", "price": -150},
                                {"name": "Blue Corner", "price": 130},
                            ],
                        }
                    ],
                }
            ],
        }
    ]
    _write_odds_snapshot(raw_root, "120000_snapshot_01.json", snapshot)

    first = run_market_odds_ingestion(
        db_path=db_path,
        raw_root=raw_root,
        promotion="UFC",
        source_system="odds_api",
        report_output_path=report_path,
        from_archives=True,
    )
    second = run_market_odds_ingestion(
        db_path=db_path,
        raw_root=raw_root,
        promotion="UFC",
        source_system="odds_api",
        report_output_path=report_path,
        from_archives=True,
    )

    assert first.inserted_count == 2
    assert first.updated_count == 0
    assert first.skipped_count == 0
    assert first.matched_rows == 2
    assert first.unmatched_rows == 0
    assert first.exact_matches == 1
    assert first.reversed_matches == 1

    assert second.inserted_count == 0
    assert second.updated_count == 0
    assert second.skipped_count == 2
    assert second.matched_rows == 2
    assert second.unmatched_rows == 0
    assert second.exact_matches == 1
    assert second.reversed_matches == 1
    assert first.matching_audit_report_path.exists()

    with sqlite3.connect(db_path) as connection:
        total = connection.execute("SELECT COUNT(*) FROM markets").fetchone()
        linked = connection.execute("SELECT DISTINCT bout_id FROM markets").fetchall()
    assert int(total[0] if total is not None else 0) == 2
    assert len(linked) == 1
    assert str(linked[0][0]) == "bout_1"

    report_text = report_path.read_text(encoding="utf-8")
    assert "market_rows=2" in report_text
    assert "matched_fights=1" in report_text
    assert "unmatched_fights=0" in report_text
    assert "match_rate=1.00000000" in report_text
    assert "opening_odds_coverage covered_fights=1" in report_text
    audit_text = first.matching_audit_report_path.read_text(encoding="utf-8")
    assert "total_market_rows=2" in audit_text
    assert "exact_matches=1" in audit_text
    assert "reversed_matches=1" in audit_text
    assert "unmatched_rows=0" in audit_text


def test_market_matching_respects_chronology_and_coverage_by_year(tmp_path: Path) -> None:
    db_path = tmp_path / "mma.sqlite3"
    raw_root = tmp_path / "raw"
    report_path = tmp_path / "market_report.txt"
    initialize_sqlite_database(db_path)

    now = "2026-03-12T00:00:00Z"
    with sqlite3.connect(db_path) as connection:
        _seed_fighter(connection, "f_red", "Red Fighter", now)
        _seed_fighter(connection, "f_blue", "Blue Fighter", now)
        _seed_fighter(connection, "f_c", "Fighter C", now)
        _seed_fighter(connection, "f_d", "Fighter D", now)
        _seed_event(connection, "event_2024", "2024-03-01T00:00:00Z", now)
        _seed_event(connection, "event_2025", "2025-04-01T00:00:00Z", now)
        _seed_bout(
            connection,
            bout_id="bout_2024",
            event_id="event_2024",
            red_id="f_red",
            blue_id="f_blue",
            bout_start_utc="2024-03-01T20:00:00Z",
            now=now,
        )
        _seed_bout(
            connection,
            bout_id="bout_2025",
            event_id="event_2025",
            red_id="f_c",
            blue_id="f_d",
            bout_start_utc="2025-04-01T20:00:00Z",
            now=now,
        )
        connection.commit()

    open_snapshot = [
        {
            "id": "event_2024",
            "commence_time": "2024-03-01T20:00:00Z",
            "bookmakers": [
                {
                    "key": "book_a",
                    "title": "Book A",
                    "last_update": "2024-03-01T18:00:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2024-03-01T18:00:00Z",
                            "outcomes": [
                                {"name": "Red Fighter", "price": -120},
                                {"name": "Blue Fighter", "price": 110},
                            ],
                        }
                    ],
                }
            ],
        }
    ]
    close_and_late_snapshot = [
        {
            "id": "event_2024",
            "commence_time": "2024-03-01T20:00:00Z",
            "bookmakers": [
                {
                    "key": "book_a",
                    "title": "Book A",
                    "last_update": "2024-03-01T19:50:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2024-03-01T19:50:00Z",
                            "outcomes": [
                                {"name": "Red Fighter", "price": -140},
                                {"name": "Blue Fighter", "price": 125},
                            ],
                        }
                    ],
                }
            ],
        },
        {
            "id": "event_2025",
            "commence_time": "2025-04-01T20:00:00Z",
            "bookmakers": [
                {
                    "key": "book_a",
                    "title": "Book A",
                    "last_update": "2025-04-01T20:30:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2025-04-01T20:30:00Z",
                            "outcomes": [
                                {"name": "Fighter C", "price": -110},
                                {"name": "Fighter D", "price": 100},
                            ],
                        }
                    ],
                }
            ],
        },
    ]

    _write_odds_snapshot(raw_root, "120000_snapshot_01.json", open_snapshot)
    _write_odds_snapshot(raw_root, "130000_snapshot_02.json", close_and_late_snapshot)

    result = run_market_odds_ingestion(
        db_path=db_path,
        raw_root=raw_root,
        promotion="UFC",
        source_system="odds_api",
        report_output_path=report_path,
        from_archives=True,
    )

    assert result.inserted_count == 4
    assert result.updated_count == 0
    assert result.skipped_count == 0
    assert result.matched_rows == 4
    assert result.unmatched_rows == 2

    coverage = run_market_coverage_report(
        db_path=db_path,
        promotion="UFC",
        source_system="odds_api",
        report_output_path=report_path,
    )
    assert coverage.market_rows == 4
    assert coverage.completed_fights == 2
    assert coverage.matched_fights == 1
    assert coverage.unmatched_fights == 1
    assert coverage.match_rate == 0.5
    assert coverage.opening_odds_covered_fights == 1
    assert coverage.closing_odds_covered_fights == 1

    report_text = report_path.read_text(encoding="utf-8")
    assert "year=2024 completed_fights=1 matched_fights=1 coverage=1.00000000" in report_text
    assert "year=2025 completed_fights=1 matched_fights=0 coverage=0.00000000" in report_text
    assert "opening_odds_coverage covered_fights=1 total_fights=2 coverage=0.50000000" in report_text
    assert "closing_odds_coverage covered_fights=1 total_fights=2 coverage=0.50000000" in report_text


def test_market_matching_normalizes_fighter_names(tmp_path: Path) -> None:
    db_path = tmp_path / "mma.sqlite3"
    raw_root = tmp_path / "raw"
    report_path = tmp_path / "market_report.txt"
    initialize_sqlite_database(db_path)

    now = "2026-03-12T00:00:00Z"
    with sqlite3.connect(db_path) as connection:
        _seed_fighter(connection, "f_a", "Jose Aldo Jr", now)
        _seed_fighter(connection, "f_b", "Max Holloway", now)
        _seed_event(connection, "event_1", "2024-01-01T00:00:00Z", now)
        _seed_bout(
            connection,
            bout_id="bout_1",
            event_id="event_1",
            red_id="f_a",
            blue_id="f_b",
            bout_start_utc="2024-01-01T20:00:00Z",
            now=now,
        )
        connection.commit()

    snapshot = [
        {
            "id": "event_1",
            "commence_time": "2024-01-01T20:00:00Z",
            "bookmakers": [
                {
                    "key": "book_a",
                    "title": "Book A",
                    "last_update": "2024-01-01T18:00:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2024-01-01T18:00:00Z",
                            "outcomes": [
                                {"name": "José Aldo", "price": -130},
                                {"name": "Max Holloway", "price": 110},
                            ],
                        }
                    ],
                }
            ],
        }
    ]
    _write_odds_snapshot(raw_root, "120000_snapshot_01.json", snapshot)
    result = run_market_odds_ingestion(
        db_path=db_path,
        raw_root=raw_root,
        promotion="UFC",
        source_system="odds_api",
        report_output_path=report_path,
        from_archives=True,
    )

    assert result.matched_rows == 2
    assert result.normalized_matches >= 1
    assert result.no_match_rows == 0


def test_market_matching_uses_event_date_tolerance(tmp_path: Path) -> None:
    db_path = tmp_path / "mma.sqlite3"
    raw_root = tmp_path / "raw"
    report_path = tmp_path / "market_report.txt"
    initialize_sqlite_database(db_path)

    now = "2026-03-12T00:00:00Z"
    with sqlite3.connect(db_path) as connection:
        _seed_fighter(connection, "f_red", "Red Fighter", now)
        _seed_fighter(connection, "f_blue", "Blue Fighter", now)
        _seed_event(connection, "event_1", "2024-01-01T00:00:00Z", now)
        _seed_bout(
            connection,
            bout_id="bout_1",
            event_id="event_1",
            red_id="f_red",
            blue_id="f_blue",
            bout_start_utc="2024-01-01T20:00:00Z",
            now=now,
        )
        connection.commit()

    snapshot = [
        {
            "id": "event_1",
            "commence_time": "2024-01-03T20:00:00Z",
            "bookmakers": [
                {
                    "key": "book_a",
                    "title": "Book A",
                    "last_update": "2024-01-01T19:00:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2024-01-01T19:00:00Z",
                            "outcomes": [
                                {"name": "Red Fighter", "price": -120},
                                {"name": "Blue Fighter", "price": 110},
                            ],
                        }
                    ],
                }
            ],
        }
    ]
    _write_odds_snapshot(raw_root, "120000_snapshot_01.json", snapshot)
    result = run_market_odds_ingestion(
        db_path=db_path,
        raw_root=raw_root,
        promotion="UFC",
        source_system="odds_api",
        report_output_path=report_path,
        from_archives=True,
    )
    assert result.matched_rows == 2
    assert result.no_match_rows == 0


def test_market_matching_marks_ambiguous_candidate_bouts(tmp_path: Path) -> None:
    db_path = tmp_path / "mma.sqlite3"
    raw_root = tmp_path / "raw"
    report_path = tmp_path / "market_report.txt"
    initialize_sqlite_database(db_path)

    now = "2026-03-12T00:00:00Z"
    with sqlite3.connect(db_path) as connection:
        _seed_fighter(connection, "f_red", "Red Fighter", now)
        _seed_fighter(connection, "f_blue", "Blue Fighter", now)
        _seed_event(connection, "event_1", "2024-01-01T00:00:00Z", now)
        _seed_event(connection, "event_2", "2024-01-01T00:00:00Z", now)
        _seed_bout(
            connection,
            bout_id="bout_1",
            event_id="event_1",
            red_id="f_red",
            blue_id="f_blue",
            bout_start_utc="2024-01-01T20:00:00Z",
            now=now,
        )
        _seed_bout(
            connection,
            bout_id="bout_2",
            event_id="event_2",
            red_id="f_red",
            blue_id="f_blue",
            bout_start_utc="2024-01-01T20:00:00Z",
            now=now,
        )
        connection.commit()

    snapshot = [
        {
            "id": "event_shared",
            "commence_time": "2024-01-01T20:00:00Z",
            "bookmakers": [
                {
                    "key": "book_a",
                    "title": "Book A",
                    "last_update": "2024-01-01T18:00:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2024-01-01T18:00:00Z",
                            "outcomes": [
                                {"name": "Red Fighter", "price": -120},
                                {"name": "Blue Fighter", "price": 110},
                            ],
                        }
                    ],
                }
            ],
        }
    ]
    _write_odds_snapshot(raw_root, "120000_snapshot_01.json", snapshot)
    result = run_market_odds_ingestion(
        db_path=db_path,
        raw_root=raw_root,
        promotion="UFC",
        source_system="odds_api",
        report_output_path=report_path,
        from_archives=True,
    )
    assert result.matched_rows == 0
    assert result.ambiguous_rows == 2
    assert result.unmatched_rows == 2


def test_odds_parser_uses_historical_snapshot_timestamp_fallback() -> None:
    parser = OddsParser(source_mode="historical")
    snapshot_payload = [
        {
            "id": "event_1",
            "commence_time": "2024-01-01T20:00:00Z",
            "bookmakers": [
                {
                    "key": "book_a",
                    "title": "Book A",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Red Corner", "price": -150},
                                {"name": "Blue Corner", "price": 130},
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    raw_record = RawIngestionRecord(
        source_system="odds_api_historical",
        source_record_id="odds_snapshot:test",
        fetched_at_utc=datetime(2026, 3, 12, tzinfo=timezone.utc),
        payload={
            "events": snapshot_payload,
            "historical_snapshot_utc": "2024-01-01T18:00:00Z",
        },
        payload_sha256="sha",
        idempotency_key="idk",
    )

    parsed = parser.parse((raw_record,))
    assert len(parsed) == 2
    assert str(parsed[0].parsed_payload["market_timestamp_utc"]) == "2024-01-01T18:00:00Z"
    assert str(parsed[1].parsed_payload["market_timestamp_utc"]) == "2024-01-01T18:00:00Z"


def test_historical_market_ingestion_completed_bouts_only_with_yearly_coverage(tmp_path: Path) -> None:
    db_path = tmp_path / "mma.sqlite3"
    raw_root = tmp_path / "raw"
    report_path = tmp_path / "historical_market_report.txt"
    initialize_sqlite_database(db_path)

    now = "2026-03-12T00:00:00Z"
    with sqlite3.connect(db_path) as connection:
        _seed_fighter(connection, "f_red", "Red Fighter", now)
        _seed_fighter(connection, "f_blue", "Blue Fighter", now)
        _seed_fighter(connection, "f_c", "Fighter C", now)
        _seed_fighter(connection, "f_d", "Fighter D", now)
        _seed_event(connection, "event_2024", "2024-03-01T00:00:00Z", now)
        _seed_event(connection, "event_2025", "2025-04-01T00:00:00Z", now)
        _seed_bout(
            connection,
            bout_id="bout_2024",
            event_id="event_2024",
            red_id="f_red",
            blue_id="f_blue",
            bout_start_utc="2024-03-01T20:00:00Z",
            now=now,
        )
        _seed_bout(
            connection,
            bout_id="bout_2025",
            event_id="event_2025",
            red_id="f_c",
            blue_id="f_d",
            bout_start_utc="2025-04-01T20:00:00Z",
            now=now,
        )
        connection.execute("UPDATE bouts SET winner_fighter_id = NULL WHERE bout_id = 'bout_2025'")
        connection.commit()

    snapshot_open = [
        {
            "id": "event_2024",
            "commence_time": "2024-03-01T20:00:00Z",
            "bookmakers": [
                {
                    "key": "book_a",
                    "title": "Book A",
                    "last_update": "2024-03-01T18:00:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2024-03-01T18:00:00Z",
                            "outcomes": [
                                {"name": "Red Fighter", "price": -120},
                                {"name": "Blue Fighter", "price": 110},
                            ],
                        }
                    ],
                }
            ],
        },
        {
            "id": "event_2025",
            "commence_time": "2025-04-01T20:00:00Z",
            "bookmakers": [
                {
                    "key": "book_a",
                    "title": "Book A",
                    "last_update": "2025-04-01T18:00:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2025-04-01T18:00:00Z",
                            "outcomes": [
                                {"name": "Fighter C", "price": -110},
                                {"name": "Fighter D", "price": 100},
                            ],
                        }
                    ],
                }
            ],
        },
    ]
    snapshot_close = [
        {
            "id": "event_2024",
            "commence_time": "2024-03-01T20:00:00Z",
            "bookmakers": [
                {
                    "key": "book_a",
                    "title": "Book A",
                    "last_update": "2024-03-01T19:50:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2024-03-01T19:50:00Z",
                            "outcomes": [
                                {"name": "Red Fighter", "price": -140},
                                {"name": "Blue Fighter", "price": 125},
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    _write_odds_snapshot(raw_root, "120000_snapshot_open.json", snapshot_open, source_system="odds_api_historical")
    _write_odds_snapshot(raw_root, "130000_snapshot_close.json", snapshot_close, source_system="odds_api_historical")

    first = run_market_odds_ingestion(
        db_path=db_path,
        raw_root=raw_root,
        promotion="UFC",
        source_system="odds_api_historical",
        source_mode="historical",
        report_output_path=report_path,
        from_archives=True,
    )
    second = run_market_odds_ingestion(
        db_path=db_path,
        raw_root=raw_root,
        promotion="UFC",
        source_system="odds_api_historical",
        source_mode="historical",
        report_output_path=report_path,
        from_archives=True,
    )

    assert first.inserted_count == 4
    assert first.matched_rows == 4
    assert first.unmatched_rows == 2
    assert second.inserted_count == 0
    assert second.skipped_count == 4

    assert first.report.market_rows == 4
    assert first.report.matched_fights == 1
    assert first.report.opening_odds_covered_fights == 1
    assert first.report.closing_odds_covered_fights == 1

    report_text = report_path.read_text(encoding="utf-8")
    assert "opening_odds_coverage_by_year" in report_text
    assert "year=2024 covered_fights=1 total_fights=1 coverage=1.00000000" in report_text
    assert "closing_odds_coverage_by_year" in report_text
    assert "year=2024 covered_fights=1 total_fights=1 coverage=1.00000000 closing_available_fights=1" in report_text
    assert "unmatched_rows_by_reason" in report_text
    assert "reason=no_pre_fight_candidate_within_date_tolerance count=2" in report_text
