"""Tests for chronology-safe baseline model training and prediction flow."""

from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

import cli
from ingestion.db_init import initialize_sqlite_database
from models.baselines import run_baseline_models


NOW = "2026-03-11T00:00:00Z"


def _connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def _insert_fixture_data(connection: sqlite3.Connection) -> None:
    connection.executemany(
        """
        INSERT INTO fighters (
            fighter_id, full_name, date_of_birth_utc, nationality, stance, height_cm, reach_cm,
            source_system, source_record_id, source_updated_at_utc, source_payload_sha256,
            ingested_at_utc, created_at_utc, updated_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ("fighter_a", "Fighter A", "1990-01-01T00:00:00Z", "US", "Orthodox", 180.0, 184.0, "ufc_stats", "fighter:a", None, "sha-fighter-a", NOW, NOW, NOW),
            ("fighter_b", "Fighter B", "1991-01-01T00:00:00Z", "US", "Southpaw", 178.0, 182.0, "ufc_stats", "fighter:b", None, "sha-fighter-b", NOW, NOW, NOW),
            ("fighter_c", "Fighter C", "1992-01-01T00:00:00Z", "US", "Orthodox", 176.0, 180.0, "ufc_stats", "fighter:c", None, "sha-fighter-c", NOW, NOW, NOW),
        ),
    )

    events = (
        ("event_1", "UFC", "UFC 1", "2024-01-01T00:00:00Z"),
        ("event_2", "UFC", "UFC 2", "2024-02-01T00:00:00Z"),
        ("event_3", "UFC", "UFC 3", "2024-03-01T00:00:00Z"),
        ("event_4", "UFC", "UFC 4", "2024-04-01T00:00:00Z"),
        ("event_5", "UFC", "UFC 5", "2024-05-01T00:00:00Z"),
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
            (event_id, promotion, name, date_utc, None, "Las Vegas", "Nevada", "USA", None, "ufc_stats", f"event:{event_id}", None, f"sha-{event_id}", NOW, NOW, NOW)
            for event_id, promotion, name, date_utc in events
        ),
    )

    bouts = (
        ("bout_1", "event_1", "fighter_a", "fighter_b", "fighter_a", "2024-01-01T01:00:00Z", "Decision", 3, 300),
        ("bout_2", "event_2", "fighter_b", "fighter_c", "fighter_b", "2024-02-01T01:00:00Z", "Decision", 3, 300),
        ("bout_3", "event_3", "fighter_a", "fighter_c", "fighter_c", "2024-03-01T01:00:00Z", "KO/TKO", 2, 150),
        ("bout_4", "event_4", "fighter_a", "fighter_b", "fighter_b", "2024-04-01T01:00:00Z", "Decision", 3, 300),
        ("bout_5", "event_5", "fighter_c", "fighter_a", "fighter_a", "2024-05-01T01:00:00Z", "Submission", 1, 90),
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
                bout_id,
                event_id,
                1,
                red_id,
                blue_id,
                "Lightweight",
                "M",
                0,
                3,
                start_utc,
                method,
                result_round,
                result_time_seconds,
                winner_id,
                "ufc_stats",
                f"bout:{bout_id}",
                None,
                f"sha-{bout_id}",
                NOW,
                NOW,
                NOW,
            )
            for bout_id, event_id, red_id, blue_id, winner_id, start_utc, method, result_round, result_time_seconds in bouts
        ),
    )

    stats_rows: list[tuple[object, ...]] = []
    for bout_id, _, red_id, blue_id, _, _, _, _, _ in bouts:
        stats_rows.append(
            (
                f"stats_{bout_id}_red",
                bout_id,
                red_id,
                blue_id,
                "red",
                0,
                20,
                40,
                30,
                60,
                1,
                2,
                0,
                0,
                120,
                "ufc_stats",
                f"stats:{bout_id}:red",
                None,
                f"sha-stats-{bout_id}-red",
                NOW,
                NOW,
                NOW,
            )
        )
        stats_rows.append(
            (
                f"stats_{bout_id}_blue",
                bout_id,
                blue_id,
                red_id,
                "blue",
                0,
                18,
                38,
                26,
                56,
                0,
                1,
                1,
                0,
                95,
                "ufc_stats",
                f"stats:{bout_id}:blue",
                None,
                f"sha-stats-{bout_id}-blue",
                NOW,
                NOW,
                NOW,
            )
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
        stats_rows,
    )

    markets = (
        ("market_b4_a", "bout_4", "fighter_a", 0.45, "2024-04-01T00:30:00Z"),
        ("market_b4_b", "bout_4", "fighter_b", 0.55, "2024-04-01T00:30:00Z"),
        ("market_b5_c", "bout_5", "fighter_c", 0.40, "2024-05-01T00:30:00Z"),
        ("market_b5_a", "bout_5", "fighter_a", 0.60, "2024-05-01T00:30:00Z"),
    )
    connection.executemany(
        """
        INSERT INTO markets (
            market_id, bout_id, sportsbook, market_type, selection_fighter_id, selection_label,
            odds_american, odds_decimal, implied_probability, line_value, market_timestamp_utc,
            source_system, source_record_id, source_updated_at_utc, source_payload_sha256,
            ingested_at_utc, created_at_utc, updated_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                market_id,
                bout_id,
                "book_a",
                "moneyline",
                fighter_id,
                fighter_id,
                None,
                None,
                implied_probability,
                None,
                timestamp_utc,
                "odds_api",
                f"market:{market_id}",
                None,
                f"sha-market-{market_id}",
                NOW,
                NOW,
                NOW,
            )
            for market_id, bout_id, fighter_id, implied_probability, timestamp_utc in markets
        ),
    )


def test_run_baseline_models_persists_predictions_and_report(tmp_path: Path) -> None:
    db_path = tmp_path / "mma.sqlite3"
    initialize_sqlite_database(db_path)
    with _connect(db_path) as connection:
        _insert_fixture_data(connection)
        connection.commit()

    features_path = tmp_path / "pre_fight_features.csv"
    ratings_path = tmp_path / "elo_ratings_history.csv"
    predictions_path = tmp_path / "baseline_predictions.csv"
    report_path = tmp_path / "baseline_report.txt"

    result = run_baseline_models(
        db_path=db_path,
        promotion="UFC",
        train_fraction=0.6,
        model_version="v1",
        model_run_id="test_run_001",
        features_output_path=features_path,
        ratings_output_path=ratings_path,
        predictions_output_path=predictions_path,
        report_output_path=report_path,
    )

    assert result.model_run_id == "test_run_001"
    assert result.prediction_count > 0
    assert result.scored_bout_count == 2
    assert result.train_test_split.train_end_utc == "2024-03-01T01:00:00Z"
    assert result.train_test_split.test_start_utc == "2024-04-01T01:00:00Z"

    with predictions_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    predicted_bouts = {row["bout_id"] for row in rows}
    predicted_models = {row["model_name"] for row in rows}
    assert predicted_bouts == {"bout_4", "bout_5"}
    assert {"market_implied_probability", "elo_logistic", "tabular_logistic", "tabular_tree_stump"} <= predicted_models
    assert all(row["split"] == "test" for row in rows)

    with sqlite3.connect(db_path) as connection:
        persisted_count = connection.execute(
            "SELECT COUNT(*) FROM model_predictions WHERE model_run_id = ?",
            ("test_run_001",),
        ).fetchone()[0]
    assert int(persisted_count) == len(rows)

    report_text = report_path.read_text(encoding="utf-8")
    assert "bouts_scored=2" in report_text
    assert "train_end_utc=2024-03-01T01:00:00Z" in report_text
    assert "test_start_utc=2024-04-01T01:00:00Z" in report_text
    assert "feature_coverage" in report_text


def test_train_baselines_cli_runs_end_to_end(tmp_path: Path) -> None:
    db_path = tmp_path / "mma_cli.sqlite3"
    initialize_sqlite_database(db_path)
    with _connect(db_path) as connection:
        _insert_fixture_data(connection)
        connection.commit()

    predictions_path = tmp_path / "cli_baseline_predictions.csv"
    report_path = tmp_path / "cli_baseline_report.txt"
    features_path = tmp_path / "cli_features.csv"
    ratings_path = tmp_path / "cli_ratings.csv"

    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "mma-elo",
            "train-baselines",
            "--db-path",
            str(db_path),
            "--promotion",
            "UFC",
            "--train-fraction",
            "0.6",
            "--model-run-id",
            "cli_run_001",
            "--features-output-path",
            str(features_path),
            "--ratings-output-path",
            str(ratings_path),
            "--predictions-output-path",
            str(predictions_path),
            "--report-output-path",
            str(report_path),
        ]
        assert cli.main() == 0
    finally:
        sys.argv = original_argv

    assert features_path.exists()
    assert ratings_path.exists()
    assert predictions_path.exists()
    assert report_path.exists()

    with predictions_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) > 0
    assert all(row["model_run_id"] == "cli_run_001" for row in rows)
