"""Tests for data-integrity and chronology validation safeguards."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import cli
from ingestion.db_init import initialize_sqlite_database
from validation.data_integrity import run_data_validations


def _connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def _insert_valid_snapshot(connection: sqlite3.Connection) -> None:
    now = "2026-03-10T12:00:00Z"
    event_date = "2026-03-08T00:00:00Z"
    bout_start = "2026-03-08T04:00:00Z"

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
                "fighter_red",
                "Red Fighter",
                "1990-01-01T00:00:00Z",
                "US",
                "Orthodox",
                180.0,
                185.0,
                "ufc_stats",
                "fighter:red",
                None,
                "sha-red",
                now,
                now,
                now,
            ),
            (
                "fighter_blue",
                "Blue Fighter",
                "1992-02-02T00:00:00Z",
                "US",
                "Southpaw",
                178.0,
                182.0,
                "ufc_stats",
                "fighter:blue",
                None,
                "sha-blue",
                now,
                now,
                now,
            ),
        ),
    )

    connection.execute(
        """
        INSERT INTO events (
            event_id, promotion, event_name, event_date_utc, venue, city, region, country, timezone_name,
            source_system, source_record_id, source_updated_at_utc, source_payload_sha256,
            ingested_at_utc, created_at_utc, updated_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "event_001",
            "UFC",
            "UFC 300",
            event_date,
            "Arena",
            "Las Vegas",
            "Nevada",
            "USA",
            "America/Los_Angeles",
            "ufc_stats",
            "event:001",
            None,
            "sha-event-001",
            now,
            now,
            now,
        ),
    )

    connection.execute(
        """
        INSERT INTO bouts (
            bout_id, event_id, bout_order, fighter_red_id, fighter_blue_id, weight_class, gender,
            is_title_fight, scheduled_rounds, bout_start_time_utc, result_method, result_round,
            result_time_seconds, winner_fighter_id, source_system, source_record_id,
            source_updated_at_utc, source_payload_sha256, ingested_at_utc, created_at_utc, updated_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "bout_001",
            "event_001",
            1,
            "fighter_red",
            "fighter_blue",
            "Lightweight",
            "M",
            0,
            3,
            bout_start,
            "Decision",
            3,
            300,
            "fighter_red",
            "ufc_stats",
            "bout:001",
            None,
            "sha-bout-001",
            now,
            now,
            now,
        ),
    )

    connection.executemany(
        """
        INSERT INTO fighter_bout_stats (
            fighter_bout_stats_id, bout_id, fighter_id, opponent_fighter_id, corner, knockdowns,
            sig_strikes_landed, sig_strikes_attempted, total_strikes_landed, total_strikes_attempted,
            takedowns_landed, takedowns_attempted, submission_attempts, reversals, control_time_seconds,
            source_system, source_record_id, source_updated_at_utc, source_payload_sha256,
            ingested_at_utc, created_at_utc, updated_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                "stats_red",
                "bout_001",
                "fighter_red",
                "fighter_blue",
                "red",
                0,
                30,
                80,
                40,
                95,
                1,
                2,
                0,
                0,
                120,
                "ufc_stats",
                "stats:red",
                None,
                "sha-stats-red",
                now,
                now,
                now,
            ),
            (
                "stats_blue",
                "bout_001",
                "fighter_blue",
                "fighter_red",
                "blue",
                0,
                25,
                70,
                32,
                81,
                0,
                1,
                0,
                0,
                85,
                "ufc_stats",
                "stats:blue",
                None,
                "sha-stats-blue",
                now,
                now,
                now,
            ),
        ),
    )

    connection.execute(
        """
        INSERT INTO markets (
            market_id, bout_id, sportsbook, market_type, selection_fighter_id, selection_label,
            odds_american, odds_decimal, implied_probability, line_value, market_timestamp_utc,
            source_system, source_record_id, source_updated_at_utc, source_payload_sha256,
            ingested_at_utc, created_at_utc, updated_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "market_001",
            "bout_001",
            "book",
            "moneyline",
            "fighter_red",
            "Red Fighter",
            -120,
            1.83,
            0.55,
            None,
            "2026-03-08T03:30:00Z",
            "odds_api",
            "market:001",
            None,
            "sha-market-001",
            now,
            now,
            now,
        ),
    )

    connection.execute(
        """
        INSERT INTO model_predictions (
            model_prediction_id, bout_id, target_fighter_id, model_name, model_version, model_run_id,
            prediction_type, predicted_probability, generated_at_utc, feature_cutoff_utc,
            source_system, source_record_id, source_updated_at_utc, source_payload_sha256,
            created_at_utc, updated_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "pred_001",
            "bout_001",
            "fighter_red",
            "baseline",
            "1.0.0",
            "run_001",
            "win_probability",
            0.57,
            "2026-03-08T03:45:00Z",
            "2026-03-08T03:40:00Z",
            "internal",
            "prediction:001",
            None,
            "sha-prediction-001",
            now,
            now,
        ),
    )


def _issue_names(connection: sqlite3.Connection) -> set[str]:
    report = run_data_validations(connection)
    return {issue.check_name for issue in report.issues}


def test_run_data_validations_passes_for_valid_snapshot(tmp_path: Path) -> None:
    db_path = tmp_path / "validation.sqlite3"
    initialize_sqlite_database(db_path)

    with _connect(db_path) as connection:
        _insert_valid_snapshot(connection)
        connection.commit()
        report = run_data_validations(connection)

    assert report.is_valid
    assert report.issue_count == 0


def test_duplicate_checks_detect_duplicate_fighters_events_and_bouts(tmp_path: Path) -> None:
    db_path = tmp_path / "validation.sqlite3"
    initialize_sqlite_database(db_path)

    with _connect(db_path) as connection:
        _insert_valid_snapshot(connection)

        connection.execute(
            """
            INSERT INTO fighters (
                fighter_id, full_name, date_of_birth_utc, nationality, stance, height_cm, reach_cm,
                source_system, source_record_id, source_updated_at_utc, source_payload_sha256,
                ingested_at_utc, created_at_utc, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "fighter_red_clone",
                "  RED FIGHTER ",
                "1990-01-01T00:00:00Z",
                None,
                None,
                None,
                None,
                "ufc_stats",
                "fighter:red:clone",
                None,
                "sha-red-clone",
                "2026-03-10T12:00:00Z",
                "2026-03-10T12:00:00Z",
                "2026-03-10T12:00:00Z",
            ),
        )

        connection.execute(
            """
            INSERT INTO events (
                event_id, promotion, event_name, event_date_utc, venue, city, region, country, timezone_name,
                source_system, source_record_id, source_updated_at_utc, source_payload_sha256,
                ingested_at_utc, created_at_utc, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "event_002",
                "UFC",
                " ufc 300 ",
                "2026-03-08T00:00:00Z",
                None,
                None,
                None,
                None,
                None,
                "ufc_stats",
                "event:002",
                None,
                "sha-event-002",
                "2026-03-10T12:00:00Z",
                "2026-03-10T12:00:00Z",
                "2026-03-10T12:00:00Z",
            ),
        )

        connection.execute(
            """
            INSERT INTO bouts (
                bout_id, event_id, bout_order, fighter_red_id, fighter_blue_id, weight_class, gender,
                is_title_fight, scheduled_rounds, bout_start_time_utc, result_method, result_round,
                result_time_seconds, winner_fighter_id, source_system, source_record_id,
                source_updated_at_utc, source_payload_sha256, ingested_at_utc, created_at_utc, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "bout_002",
                "event_001",
                2,
                "fighter_blue",
                "fighter_red",
                "Lightweight",
                "M",
                0,
                3,
                "2026-03-08T04:25:00Z",
                "Decision",
                3,
                300,
                "fighter_blue",
                "ufc_stats",
                "bout:002",
                None,
                "sha-bout-002",
                "2026-03-10T12:00:00Z",
                "2026-03-10T12:00:00Z",
                "2026-03-10T12:00:00Z",
            ),
        )

        connection.commit()
        issue_names = _issue_names(connection)

    assert "duplicate_fighters" in issue_names
    assert "duplicate_events" in issue_names
    assert "duplicate_bouts" in issue_names


def test_duplicate_fighter_check_ignores_same_name_without_dob(tmp_path: Path) -> None:
    db_path = tmp_path / "validation.sqlite3"
    initialize_sqlite_database(db_path)

    with _connect(db_path) as connection:
        _insert_valid_snapshot(connection)
        connection.execute(
            """
            INSERT INTO fighters (
                fighter_id, full_name, date_of_birth_utc, nationality, stance, height_cm, reach_cm,
                source_system, source_record_id, source_updated_at_utc, source_payload_sha256,
                ingested_at_utc, created_at_utc, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "fighter_same_name_missing_dob",
                "Red Fighter",
                None,
                None,
                None,
                None,
                None,
                "ufc_stats",
                "fighter:red:missing_dob",
                None,
                "sha-red-missing-dob",
                "2026-03-10T12:00:00Z",
                "2026-03-10T12:00:00Z",
                "2026-03-10T12:00:00Z",
            ),
        )
        connection.commit()
        issue_names = _issue_names(connection)

    assert "duplicate_fighters" not in issue_names


def test_chronology_checks_detect_temporal_leakage_and_record_mismatches(tmp_path: Path) -> None:
    db_path = tmp_path / "validation.sqlite3"
    initialize_sqlite_database(db_path)

    with _connect(db_path) as connection:
        _insert_valid_snapshot(connection)
        connection.execute(
            """
            INSERT INTO fighters (
                fighter_id, full_name, date_of_birth_utc, nationality, stance, height_cm, reach_cm,
                source_system, source_record_id, source_updated_at_utc, source_payload_sha256,
                ingested_at_utc, created_at_utc, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "fighter_extra",
                "Extra Fighter",
                "1994-04-04T00:00:00Z",
                None,
                None,
                None,
                None,
                "ufc_stats",
                "fighter:extra:chronology",
                None,
                "sha-extra-chronology",
                "2026-03-10T12:00:00Z",
                "2026-03-10T12:00:00Z",
                "2026-03-10T12:00:00Z",
            ),
        )

        connection.execute(
            "UPDATE bouts SET bout_start_time_utc = ? WHERE bout_id = ?",
            ("2026-03-07T23:00:00Z", "bout_001"),
        )

        connection.execute(
            """
            UPDATE fighter_bout_stats
            SET fighter_id = ?, opponent_fighter_id = ?
            WHERE fighter_bout_stats_id = ?
            """,
            ("fighter_extra", "fighter_red", "stats_blue"),
        )

        connection.execute(
            "UPDATE markets SET market_timestamp_utc = ? WHERE market_id = ?",
            ("2026-03-08T05:00:00Z", "market_001"),
        )

        connection.execute(
            """
            UPDATE model_predictions
            SET generated_at_utc = ?, feature_cutoff_utc = ?
            WHERE model_prediction_id = ?
            """,
            ("2026-03-08T06:00:00Z", "2026-03-08T06:05:00Z", "pred_001"),
        )

        connection.commit()
        issue_names = _issue_names(connection)

    assert "chronology_bout_before_event_date" in issue_names
    assert "chronology_stats_fighter_mismatch" in issue_names
    assert "chronology_market_after_bout_start" in issue_names
    assert "chronology_prediction_after_bout_start" in issue_names
    assert "chronology_feature_cutoff_after_generated_at" in issue_names
    assert "chronology_feature_cutoff_after_bout_start" in issue_names


def test_missing_critical_fields_detect_blank_values(tmp_path: Path) -> None:
    db_path = tmp_path / "validation.sqlite3"
    initialize_sqlite_database(db_path)

    with _connect(db_path) as connection:
        _insert_valid_snapshot(connection)

        connection.execute("UPDATE fighters SET full_name = '   ' WHERE fighter_id = 'fighter_red'")
        connection.execute("UPDATE events SET event_name = '' WHERE event_id = 'event_001'")
        connection.execute("UPDATE bouts SET source_system = ' ' WHERE bout_id = 'bout_001'")

        connection.commit()
        report = run_data_validations(connection)

    issue_names = {issue.check_name for issue in report.issues}
    messages = [issue.message for issue in report.issues if issue.check_name == "missing_critical_fields"]

    assert "missing_critical_fields" in issue_names
    assert any("'full_name'" in message for message in messages)
    assert any("'event_name'" in message for message in messages)
    assert any("'source_system'" in message for message in messages)


def test_cli_validate_data_returns_nonzero_when_issues_exist(tmp_path: Path, capsys: object) -> None:
    valid_db_path = tmp_path / "valid.sqlite3"
    initialize_sqlite_database(valid_db_path)

    with _connect(valid_db_path) as connection:
        _insert_valid_snapshot(connection)
        connection.commit()

    original_argv = sys.argv[:]
    try:
        sys.argv = ["mma-elo", "validate-data", "--db-path", str(valid_db_path)]
        assert cli.main() == 0
    finally:
        sys.argv = original_argv

    invalid_db_path = tmp_path / "invalid.sqlite3"
    initialize_sqlite_database(invalid_db_path)

    with _connect(invalid_db_path) as connection:
        _insert_valid_snapshot(connection)
        connection.execute("UPDATE fighters SET full_name = '' WHERE fighter_id = 'fighter_red'")
        connection.commit()

    try:
        sys.argv = ["mma-elo", "validate-data", "--db-path", str(invalid_db_path)]
        assert cli.main() == 1
    finally:
        sys.argv = original_argv

    captured = capsys.readouterr()
    assert "data_validation_complete status=ok issues=0" in captured.out
    assert "data_validation_complete status=failed" in captured.out
