"""Tests for sequential Elo rating history generation and reporting."""

from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

import cli
from ingestion.db_init import initialize_sqlite_database
from ratings.elo import EloConfig, build_ratings_report, generate_elo_ratings_history


NOW = "2026-03-11T00:00:00Z"


def _connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def _insert_base_data(connection: sqlite3.Connection) -> None:
    connection.executemany(
        """
        INSERT INTO fighters (
            fighter_id, full_name, date_of_birth_utc, nationality, stance, height_cm, reach_cm,
            source_system, source_record_id, source_updated_at_utc, source_payload_sha256,
            ingested_at_utc, created_at_utc, updated_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                "fighter_a",
                "Fighter A",
                "1990-01-01T00:00:00Z",
                "US",
                "Orthodox",
                180.0,
                185.0,
                "ufc_stats",
                "fighter:a",
                None,
                "sha-fighter-a",
                NOW,
                NOW,
                NOW,
            ),
            (
                "fighter_b",
                "Fighter B",
                "1992-01-01T00:00:00Z",
                "US",
                "Southpaw",
                175.0,
                180.0,
                "ufc_stats",
                "fighter:b",
                None,
                "sha-fighter-b",
                NOW,
                NOW,
                NOW,
            ),
            (
                "fighter_c",
                "Fighter C",
                "1994-01-01T00:00:00Z",
                "US",
                "Orthodox",
                178.0,
                182.0,
                "ufc_stats",
                "fighter:c",
                None,
                "sha-fighter-c",
                NOW,
                NOW,
                NOW,
            ),
        ),
    )

    connection.executemany(
        """
        INSERT INTO events (
            event_id, promotion, event_name, event_date_utc, venue, city, region, country, timezone_name,
            source_system, source_record_id, source_updated_at_utc, source_payload_sha256,
            ingested_at_utc, created_at_utc, updated_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                "event_1",
                "UFC",
                "UFC 1",
                "2024-01-01T00:00:00Z",
                None,
                "Las Vegas",
                "Nevada",
                "USA",
                None,
                "ufc_stats",
                "event:1",
                None,
                "sha-event-1",
                NOW,
                NOW,
                NOW,
            ),
            (
                "event_2",
                "UFC",
                "UFC 2",
                "2024-02-01T00:00:00Z",
                None,
                "Las Vegas",
                "Nevada",
                "USA",
                None,
                "ufc_stats",
                "event:2",
                None,
                "sha-event-2",
                NOW,
                NOW,
                NOW,
            ),
            (
                "event_3",
                "UFC",
                "UFC 3",
                "2024-03-01T00:00:00Z",
                None,
                "Las Vegas",
                "Nevada",
                "USA",
                None,
                "ufc_stats",
                "event:3",
                None,
                "sha-event-3",
                NOW,
                NOW,
                NOW,
            ),
        ),
    )

    connection.executemany(
        """
        INSERT INTO bouts (
            bout_id, event_id, bout_order, fighter_red_id, fighter_blue_id, weight_class, gender,
            is_title_fight, scheduled_rounds, bout_start_time_utc, result_method, result_round,
            result_time_seconds, winner_fighter_id, source_system, source_record_id,
            source_updated_at_utc, source_payload_sha256, ingested_at_utc, created_at_utc, updated_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                "bout_1",
                "event_1",
                1,
                "fighter_a",
                "fighter_b",
                "Lightweight",
                "M",
                0,
                3,
                "2024-01-01T01:00:00Z",
                "Decision",
                3,
                300,
                "fighter_a",
                "ufc_stats",
                "bout:1",
                None,
                "sha-bout-1",
                NOW,
                NOW,
                NOW,
            ),
            (
                "bout_2",
                "event_2",
                1,
                "fighter_a",
                "fighter_b",
                "Welterweight",
                "M",
                0,
                3,
                "2024-02-01T01:00:00Z",
                "KO/TKO",
                1,
                60,
                "fighter_b",
                "ufc_stats",
                "bout:2",
                None,
                "sha-bout-2",
                NOW,
                NOW,
                NOW,
            ),
            (
                "bout_3",
                "event_3",
                1,
                "fighter_a",
                "fighter_c",
                "Welterweight",
                "M",
                0,
                3,
                "2024-03-01T01:00:00Z",
                "",
                None,
                None,
                None,
                "ufc_stats",
                "bout:3",
                None,
                "sha-bout-3",
                NOW,
                NOW,
                NOW,
            ),
        ),
    )


def _insert_base_data_with_reverse_bout_inserts(connection: sqlite3.Connection) -> None:
    _insert_base_data(connection)
    # Replace bouts with reverse insertion order to verify sort-by-time sequencing.
    connection.execute("DELETE FROM bouts")
    connection.executemany(
        """
        INSERT INTO bouts (
            bout_id, event_id, bout_order, fighter_red_id, fighter_blue_id, weight_class, gender,
            is_title_fight, scheduled_rounds, bout_start_time_utc, result_method, result_round,
            result_time_seconds, winner_fighter_id, source_system, source_record_id,
            source_updated_at_utc, source_payload_sha256, ingested_at_utc, created_at_utc, updated_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                "bout_3",
                "event_3",
                1,
                "fighter_a",
                "fighter_c",
                "Welterweight",
                "M",
                0,
                3,
                "2024-03-01T01:00:00Z",
                "",
                None,
                None,
                None,
                "ufc_stats",
                "bout:3",
                None,
                "sha-bout-3",
                NOW,
                NOW,
                NOW,
            ),
            (
                "bout_2",
                "event_2",
                1,
                "fighter_a",
                "fighter_b",
                "Welterweight",
                "M",
                0,
                3,
                "2024-02-01T01:00:00Z",
                "KO/TKO",
                1,
                60,
                "fighter_b",
                "ufc_stats",
                "bout:2",
                None,
                "sha-bout-2",
                NOW,
                NOW,
                NOW,
            ),
            (
                "bout_1",
                "event_1",
                1,
                "fighter_a",
                "fighter_b",
                "Lightweight",
                "M",
                0,
                3,
                "2024-01-01T01:00:00Z",
                "Decision",
                3,
                300,
                "fighter_a",
                "ufc_stats",
                "bout:1",
                None,
                "sha-bout-1",
                NOW,
                NOW,
                NOW,
            ),
        ),
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _row_for(rows: list[dict[str, str]], bout_id: str, fighter_id: str) -> dict[str, str]:
    for row in rows:
        if row["bout_id"] == bout_id and row["fighter_id"] == fighter_id:
            return row
    raise AssertionError(f"Missing row for bout_id={bout_id}, fighter_id={fighter_id}")


def test_elo_history_is_sequential_and_updates_only_after_completed_bouts(tmp_path: Path) -> None:
    db_path = tmp_path / "mma.sqlite3"
    history_path = tmp_path / "elo_history.csv"
    initialize_sqlite_database(db_path)

    with _connect(db_path) as connection:
        _insert_base_data(connection)
        connection.commit()

    result = generate_elo_ratings_history(
        db_path=db_path,
        output_path=history_path,
        config=EloConfig(initial_rating=1500.0, k_factor=32.0, promotion="UFC"),
    )
    assert result.bout_count == 3
    assert result.row_count == 6

    rows = _read_rows(history_path)

    bout_1_a = _row_for(rows, "bout_1", "fighter_a")
    assert float(bout_1_a["pre_overall_elo"]) == 1500.0
    assert float(bout_1_a["post_overall_elo"]) == 1516.0
    assert float(bout_1_a["pre_weight_class_elo"]) == 1500.0
    assert float(bout_1_a["post_weight_class_elo"]) == 1516.0

    bout_2_a = _row_for(rows, "bout_2", "fighter_a")
    assert float(bout_2_a["pre_overall_elo"]) == 1516.0
    assert float(bout_2_a["pre_weight_class_elo"]) == 1500.0
    assert float(bout_2_a["post_overall_elo"]) < float(bout_2_a["pre_overall_elo"])
    assert float(bout_2_a["post_weight_class_elo"]) < float(bout_2_a["pre_weight_class_elo"])

    bout_3_a = _row_for(rows, "bout_3", "fighter_a")
    assert bout_3_a["bout_completed"] == "0"
    assert float(bout_3_a["pre_overall_elo"]) == float(bout_2_a["post_overall_elo"])
    assert float(bout_3_a["post_overall_elo"]) == float(bout_3_a["pre_overall_elo"])
    assert float(bout_3_a["pre_weight_class_elo"]) == float(bout_2_a["post_weight_class_elo"])
    assert float(bout_3_a["post_weight_class_elo"]) == float(bout_3_a["pre_weight_class_elo"])


def test_ratings_cli_build_and_report(tmp_path: Path, capsys: object) -> None:
    db_path = tmp_path / "mma.sqlite3"
    history_path = tmp_path / "elo_history.csv"
    initialize_sqlite_database(db_path)

    with _connect(db_path) as connection:
        _insert_base_data(connection)
        connection.commit()

    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "mma-elo",
            "build-ratings-history",
            "--db-path",
            str(db_path),
            "--output-path",
            str(history_path),
        ]
        assert cli.main() == 0

        sys.argv = [
            "mma-elo",
            "ratings-report",
            "--history-path",
            str(history_path),
            "--db-path",
            str(db_path),
            "--top-n",
            "2",
        ]
        assert cli.main() == 0
    finally:
        sys.argv = original_argv

    captured = capsys.readouterr().out
    assert "ratings_history_build_complete bouts=3 rows=6" in captured
    assert (
        "ratings_report_summary "
        "fighters_rated=2 fighters_total=3 coverage=0.666667 "
        "earliest_rating_date=2024-01-01T01:00:00Z "
        "latest_rating_date=2024-02-01T01:00:00Z"
    ) in captured
    assert "ratings_weight_class_coverage weight_class=Lightweight fighters_rated=2 share_of_rated=1.000000" in captured
    assert "ratings_weight_class_coverage weight_class=Welterweight fighters_rated=2 share_of_rated=1.000000" in captured
    assert "ratings_top_fighter fighter_id=fighter_b" in captured

    report = build_ratings_report(ratings_history_path=history_path, db_path=db_path, top_n=2)
    assert report.total_fighters == 3
    assert report.rated_fighters == 2
    assert report.coverage_rate == 2.0 / 3.0
    assert report.earliest_rating_datetime_utc == "2024-01-01T01:00:00Z"
    assert report.latest_rating_datetime_utc == "2024-02-01T01:00:00Z"
    assert [entry.weight_class for entry in report.weight_class_coverage] == ["Lightweight", "Welterweight"]
    assert all(entry.rated_fighters == 2 for entry in report.weight_class_coverage)
    assert report.top_fighters[0].fighter_id == "fighter_b"


def test_elo_history_is_chronological_even_if_db_insert_order_is_not(tmp_path: Path) -> None:
    db_path = tmp_path / "mma.sqlite3"
    history_path = tmp_path / "elo_history.csv"
    initialize_sqlite_database(db_path)

    with _connect(db_path) as connection:
        _insert_base_data_with_reverse_bout_inserts(connection)
        connection.commit()

    generate_elo_ratings_history(
        db_path=db_path,
        output_path=history_path,
        config=EloConfig(initial_rating=1500.0, k_factor=32.0, promotion="UFC"),
    )
    rows = _read_rows(history_path)

    bout_1_a = _row_for(rows, "bout_1", "fighter_a")
    bout_2_a = _row_for(rows, "bout_2", "fighter_a")
    assert float(bout_1_a["pre_overall_elo"]) == 1500.0
    assert float(bout_2_a["pre_overall_elo"]) == float(bout_1_a["post_overall_elo"])
