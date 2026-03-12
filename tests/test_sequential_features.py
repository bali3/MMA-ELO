"""Tests for sequential, leakage-safe pre-fight feature generation."""

from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

import cli
from features.sequential import build_feature_report, generate_pre_fight_features
from ingestion.db_init import initialize_sqlite_database


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
                "2024-01-31T00:00:00Z",
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
                "Lightweight",
                "M",
                0,
                3,
                "2024-01-31T01:00:00Z",
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
                "fighter_b",
                "Lightweight",
                "M",
                0,
                3,
                "2024-03-01T01:00:00Z",
                "Decision",
                3,
                300,
                "fighter_a",
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
                "stats_b1_a",
                "bout_1",
                "fighter_a",
                "fighter_b",
                "red",
                0,
                30,
                60,
                40,
                90,
                2,
                4,
                1,
                0,
                120,
                "ufc_stats",
                "stats:b1:a",
                None,
                "sha-stats-b1-a",
                NOW,
                NOW,
                NOW,
            ),
            (
                "stats_b1_b",
                "bout_1",
                "fighter_b",
                "fighter_a",
                "blue",
                0,
                20,
                50,
                30,
                70,
                1,
                3,
                0,
                0,
                100,
                "ufc_stats",
                "stats:b1:b",
                None,
                "sha-stats-b1-b",
                NOW,
                NOW,
                NOW,
            ),
            (
                "stats_b2_a",
                "bout_2",
                "fighter_a",
                "fighter_b",
                "red",
                1,
                999,
                1000,
                999,
                1000,
                8,
                10,
                5,
                0,
                50,
                "ufc_stats",
                "stats:b2:a",
                None,
                "sha-stats-b2-a",
                NOW,
                NOW,
                NOW,
            ),
            (
                "stats_b2_b",
                "bout_2",
                "fighter_b",
                "fighter_a",
                "blue",
                1,
                15,
                30,
                20,
                50,
                0,
                1,
                0,
                0,
                25,
                "ufc_stats",
                "stats:b2:b",
                None,
                "sha-stats-b2-b",
                NOW,
                NOW,
                NOW,
            ),
            (
                "stats_b3_a",
                "bout_3",
                "fighter_a",
                "fighter_b",
                "red",
                0,
                10,
                20,
                15,
                30,
                0,
                0,
                99,
                0,
                120,
                "ufc_stats",
                "stats:b3:a",
                None,
                "sha-stats-b3-a",
                NOW,
                NOW,
                NOW,
            ),
            (
                "stats_b3_b",
                "bout_3",
                "fighter_b",
                "fighter_a",
                "blue",
                0,
                12,
                40,
                20,
                60,
                1,
                2,
                0,
                0,
                100,
                "ufc_stats",
                "stats:b3:b",
                None,
                "sha-stats-b3-b",
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


def _as_float(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


def _as_int(value: str) -> int | None:
    if value == "":
        return None
    return int(value)


def test_feature_generation_is_strictly_pre_fight_and_sequential(tmp_path: Path) -> None:
    db_path = tmp_path / "mma.sqlite3"
    output_path = tmp_path / "features.csv"
    initialize_sqlite_database(db_path)

    with _connect(db_path) as connection:
        _insert_base_data(connection)
        connection.commit()

    row_count, persisted_path = generate_pre_fight_features(db_path=db_path, output_path=output_path)

    assert row_count == 6
    assert persisted_path == output_path

    rows = _read_rows(output_path)

    bout_1_a = _row_for(rows, "bout_1", "fighter_a")
    assert _as_int(bout_1_a["prior_fight_count"]) == 0
    assert _as_float(bout_1_a["recent_form_win_rate_l3"]) is None
    assert _as_float(bout_1_a["sig_strikes_landed_per_min_all"]) is None

    bout_2_a = _row_for(rows, "bout_2", "fighter_a")
    assert _as_int(bout_2_a["prior_fight_count"]) == 1
    assert _as_float(bout_2_a["recent_form_win_rate_l3"]) == 1.0
    assert _as_float(bout_2_a["finish_rate_all"]) == 0.0
    assert _as_float(bout_2_a["decision_rate_all"]) == 1.0
    assert _as_float(bout_2_a["sig_strikes_landed_per_min_all"]) == 2.0
    assert _as_int(bout_2_a["days_since_last_fight"]) == 30
    assert _as_int(bout_2_a["rematch_flag"]) == 1

    bout_3_a = _row_for(rows, "bout_3", "fighter_a")
    assert _as_int(bout_3_a["prior_fight_count"]) == 2
    assert _as_float(bout_3_a["recent_form_win_rate_l3"]) == 0.5
    assert _as_float(bout_3_a["sig_strikes_landed_per_min_all"]) == 64.3125
    assert _as_float(bout_3_a["submission_attempts_per_15m_all"]) == 5.625
    assert _as_int(bout_3_a["weight_class_experience"]) == 2


def test_feature_report_and_cli_commands(tmp_path: Path, capsys: object) -> None:
    db_path = tmp_path / "mma.sqlite3"
    output_path = tmp_path / "features.csv"
    initialize_sqlite_database(db_path)

    with _connect(db_path) as connection:
        _insert_base_data(connection)
        connection.commit()

    original_argv = sys.argv[:]
    try:
        sys.argv = ["mma-elo", "generate-features", "--db-path", str(db_path), "--output-path", str(output_path)]
        assert cli.main() == 0

        sys.argv = ["mma-elo", "feature-report", "--features-path", str(output_path)]
        assert cli.main() == 0
    finally:
        sys.argv = original_argv

    captured = capsys.readouterr().out
    assert "feature_generation_complete rows=6" in captured
    assert "feature_report_summary bouts=3 rows=6" in captured
    assert "feature_metric name=fighter_age_years" in captured

    report = build_feature_report(features_path=output_path)
    assert report.bout_count == 3
    assert report.row_count == 6
    assert report.earliest_bout_date_utc == "2024-01-01T01:00:00Z"
    assert report.latest_bout_date_utc == "2024-03-01T01:00:00Z"
    assert report.coverage_rates["fighter_age_years"] == 1.0
    assert report.missing_counts["days_since_last_fight"] == 2


def test_zero_attempt_accuracy_and_defense_are_not_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "mma.sqlite3"
    output_path = tmp_path / "features.csv"
    initialize_sqlite_database(db_path)

    with _connect(db_path) as connection:
        _insert_base_data(connection)
        connection.execute(
            "UPDATE fighter_bout_stats SET takedowns_attempted = 0, takedowns_landed = 0 WHERE bout_id = 'bout_1'"
        )
        connection.commit()

    generate_pre_fight_features(db_path=db_path, output_path=output_path)
    rows = _read_rows(output_path)
    bout_2_a = _row_for(rows, "bout_2", "fighter_a")

    assert _as_float(bout_2_a["takedown_accuracy_all"]) == 0.0
    assert _as_float(bout_2_a["takedown_defense_all"]) == 1.0


def test_missing_static_metadata_is_imputed_for_completeness(tmp_path: Path) -> None:
    db_path = tmp_path / "mma.sqlite3"
    output_path = tmp_path / "features.csv"
    initialize_sqlite_database(db_path)

    with _connect(db_path) as connection:
        _insert_base_data(connection)
        connection.execute(
            "UPDATE fighters SET date_of_birth_utc = NULL, height_cm = NULL, reach_cm = NULL WHERE fighter_id = 'fighter_a'"
        )
        connection.execute(
            "UPDATE fighters SET date_of_birth_utc = NULL, height_cm = NULL, reach_cm = NULL WHERE fighter_id = 'fighter_b'"
        )
        connection.commit()

    generate_pre_fight_features(db_path=db_path, output_path=output_path)
    rows = _read_rows(output_path)
    bout_1_a = _row_for(rows, "bout_1", "fighter_a")

    assert _as_float(bout_1_a["fighter_age_years"]) == 30.0
    assert _as_float(bout_1_a["height_diff_cm"]) == 0.0
    assert _as_float(bout_1_a["reach_diff_cm"]) == 0.0
