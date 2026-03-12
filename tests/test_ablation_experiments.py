"""Tests for chronology-safe ablation experiments and CLI wiring."""

from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

import cli
from evaluation import run_ablation_experiments
from ingestion.db_init import initialize_sqlite_database

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
            ("fighter_d", "Fighter D", "1993-01-01T00:00:00Z", "US", "Orthodox", 182.0, 186.0, "ufc_stats", "fighter:d", None, "sha-fighter-d", NOW, NOW, NOW),
        ),
    )

    events = (
        ("event_1", "UFC", "UFC 1", "2024-01-01T00:00:00Z"),
        ("event_2", "UFC", "UFC 2", "2024-02-01T00:00:00Z"),
        ("event_3", "UFC", "UFC 3", "2024-03-01T00:00:00Z"),
        ("event_4", "UFC", "UFC 4", "2024-04-01T00:00:00Z"),
        ("event_5", "UFC", "UFC 5", "2024-05-01T00:00:00Z"),
        ("event_6", "UFC", "UFC 6", "2024-06-01T00:00:00Z"),
        ("event_7", "UFC", "UFC 7", "2024-07-01T00:00:00Z"),
        ("event_8", "UFC", "UFC 8", "2024-08-01T00:00:00Z"),
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
        ("bout_1", "event_1", "fighter_a", "fighter_b", "fighter_a", "2024-01-01T01:00:00Z"),
        ("bout_2", "event_2", "fighter_c", "fighter_d", "fighter_d", "2024-02-01T01:00:00Z"),
        ("bout_3", "event_3", "fighter_a", "fighter_c", "fighter_c", "2024-03-01T01:00:00Z"),
        ("bout_4", "event_4", "fighter_b", "fighter_d", "fighter_b", "2024-04-01T01:00:00Z"),
        ("bout_5", "event_5", "fighter_a", "fighter_d", "fighter_d", "2024-05-01T01:00:00Z"),
        ("bout_6", "event_6", "fighter_b", "fighter_c", "fighter_b", "2024-06-01T01:00:00Z"),
        ("bout_7", "event_7", "fighter_a", "fighter_b", "fighter_a", "2024-07-01T01:00:00Z"),
        ("bout_8", "event_8", "fighter_c", "fighter_d", "fighter_c", "2024-08-01T01:00:00Z"),
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
                "Decision",
                3,
                300,
                winner_id,
                "ufc_stats",
                f"bout:{bout_id}",
                None,
                f"sha-{bout_id}",
                NOW,
                NOW,
                NOW,
            )
            for bout_id, event_id, red_id, blue_id, winner_id, start_utc in bouts
        ),
    )

    stats_rows: list[tuple[object, ...]] = []
    for bout_id, _, red_id, blue_id, _, _ in bouts:
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

    market_rows: list[tuple[object, ...]] = []
    for bout_id, _, red_id, blue_id, _, start_utc in bouts:
        market_time = start_utc.replace("01:00:00Z", "00:30:00Z")
        market_rows.append(
            (
                f"market_{bout_id}_red",
                bout_id,
                "book_a",
                "moneyline",
                red_id,
                red_id,
                None,
                None,
                0.52,
                None,
                market_time,
                "odds_api",
                f"market:{bout_id}:red",
                None,
                f"sha-market-{bout_id}-red",
                NOW,
                NOW,
                NOW,
            )
        )
        market_rows.append(
            (
                f"market_{bout_id}_blue",
                bout_id,
                "book_a",
                "moneyline",
                blue_id,
                blue_id,
                None,
                None,
                0.48,
                None,
                market_time,
                "odds_api",
                f"market:{bout_id}:blue",
                None,
                f"sha-market-{bout_id}-blue",
                NOW,
                NOW,
                NOW,
            )
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
        market_rows,
    )


def test_run_ablation_experiments_outputs_reports_and_predictions(tmp_path: Path) -> None:
    db_path = tmp_path / "mma_ablation.sqlite3"
    initialize_sqlite_database(db_path)
    with _connect(db_path) as connection:
        _insert_fixture_data(connection)
        connection.commit()

    predictions_path = tmp_path / "ablation_predictions.csv"
    comparison_report_path = tmp_path / "ablation_comparison_report.txt"
    contribution_report_path = tmp_path / "ablation_contribution_report.txt"
    features_path = tmp_path / "features.csv"
    ratings_path = tmp_path / "ratings.csv"

    result = run_ablation_experiments(
        db_path=db_path,
        promotion="UFC",
        window_type="expanding",
        min_train_bouts=3,
        test_bouts=2,
        rolling_train_bouts=None,
        model_version="v1",
        model_run_id="ablation_run_001",
        calibration_bins=5,
        features_output_path=features_path,
        ratings_output_path=ratings_path,
        predictions_output_path=predictions_path,
        comparison_report_output_path=comparison_report_path,
        contribution_report_output_path=contribution_report_path,
    )

    expected_ids = {
        "market_only_no_interactions",
        "market_only_with_interactions",
        "elo_only_no_interactions",
        "elo_only_with_interactions",
        "features_only_no_interactions",
        "features_only_with_interactions",
        "features_elo_no_interactions",
        "features_elo_with_interactions",
        "features_elo_market_no_interactions",
        "features_elo_market_with_interactions",
    }
    assert {row.config.scenario_id for row in result.scenario_metrics} == expected_ids

    assert predictions_path.exists()
    assert comparison_report_path.exists()
    assert contribution_report_path.exists()

    with predictions_path.open("r", encoding="utf-8", newline="") as handle:
        prediction_rows = list(csv.DictReader(handle))
    assert len(prediction_rows) > 0
    assert {row["scenario_id"] for row in prediction_rows} == expected_ids
    assert all(row["model_run_id"] == "ablation_run_001" for row in prediction_rows)
    assert all(row["feature_cutoff_utc"] == row["bout_datetime_utc"] for row in prediction_rows)

    comparison_text = comparison_report_path.read_text(encoding="utf-8")
    assert "ablation_comparison_report" in comparison_text
    assert "features_elo_market_with_interactions" in comparison_text

    contribution_text = contribution_report_path.read_text(encoding="utf-8")
    assert "ablation_contribution_report" in contribution_text
    assert "interaction_terms_given_features_elo_market" in contribution_text


def test_ablation_cli_runs_with_feature_group_filters(tmp_path: Path) -> None:
    db_path = tmp_path / "mma_ablation_cli.sqlite3"
    initialize_sqlite_database(db_path)
    with _connect(db_path) as connection:
        _insert_fixture_data(connection)
        connection.commit()

    predictions_path = tmp_path / "cli_ablation_predictions.csv"
    comparison_report_path = tmp_path / "cli_ablation_comparison_report.txt"
    contribution_report_path = tmp_path / "cli_ablation_contribution_report.txt"
    features_path = tmp_path / "cli_features.csv"
    ratings_path = tmp_path / "cli_ratings.csv"

    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "mma-elo",
            "run-ablation-experiments",
            "--db-path",
            str(db_path),
            "--promotion",
            "UFC",
            "--window-type",
            "rolling",
            "--min-train-bouts",
            "3",
            "--test-bouts",
            "2",
            "--rolling-train-bouts",
            "4",
            "--model-run-id",
            "ablation_cli_001",
            "--include-group",
            "elo",
            "--exclude-group",
            "market",
            "--features-output-path",
            str(features_path),
            "--ratings-output-path",
            str(ratings_path),
            "--predictions-output-path",
            str(predictions_path),
            "--comparison-report-output-path",
            str(comparison_report_path),
            "--contribution-report-output-path",
            str(contribution_report_path),
        ]
        assert cli.main() == 0
    finally:
        sys.argv = original_argv

    with predictions_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    scenario_ids = {row["scenario_id"] for row in rows}
    assert scenario_ids == {
        "elo_only_no_interactions",
        "elo_only_with_interactions",
        "features_elo_no_interactions",
        "features_elo_with_interactions",
    }
    assert all(row["model_run_id"] == "ablation_cli_001" for row in rows)
