"""Tests for model registry, champion/challenger comparison, and CLI selection command."""

from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

import cli
from evaluation import compare_registered_models, discover_model_registry
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

    connection.executemany(
        """
        INSERT INTO events (
            event_id, promotion, event_name, event_date_utc, venue, city, region, country, timezone_name,
            source_system, source_record_id, source_updated_at_utc, source_payload_sha256,
            ingested_at_utc, created_at_utc, updated_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ("event_1", "UFC", "UFC 1", "2024-01-01T00:00:00Z", None, "Las Vegas", "Nevada", "USA", None, "ufc_stats", "event:event_1", None, "sha-event-1", NOW, NOW, NOW),
            ("event_2", "UFC", "UFC 2", "2024-02-01T00:00:00Z", None, "Las Vegas", "Nevada", "USA", None, "ufc_stats", "event:event_2", None, "sha-event-2", NOW, NOW, NOW),
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
                "bout:bout_1",
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
                "fighter_c",
                "fighter_d",
                "Lightweight",
                "M",
                0,
                3,
                "2024-02-01T01:00:00Z",
                "Decision",
                3,
                300,
                "fighter_d",
                "ufc_stats",
                "bout:bout_2",
                None,
                "sha-bout-2",
                NOW,
                NOW,
                NOW,
            ),
        ),
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
            ("market_b1_a", "bout_1", "book_a", "moneyline", "fighter_a", "fighter_a", None, None, 0.45, None, "2024-01-01T00:30:00Z", "odds_api", "market:market_b1_a", None, "sha-b1-a", NOW, NOW, NOW),
            ("market_b1_b", "bout_1", "book_a", "moneyline", "fighter_b", "fighter_b", None, None, 0.55, None, "2024-01-01T00:30:00Z", "odds_api", "market:market_b1_b", None, "sha-b1-b", NOW, NOW, NOW),
            ("market_b2_c", "bout_2", "book_a", "moneyline", "fighter_c", "fighter_c", None, None, 0.55, None, "2024-02-01T00:30:00Z", "odds_api", "market:market_b2_c", None, "sha-b2-c", NOW, NOW, NOW),
            ("market_b2_d", "bout_2", "book_a", "moneyline", "fighter_d", "fighter_d", None, None, 0.45, None, "2024-02-01T00:30:00Z", "odds_api", "market:market_b2_d", None, "sha-b2-d", NOW, NOW, NOW),
        ),
    )


def _write_predictions(path: Path) -> None:
    rows = [
        {
            "model_run_id": "run_alpha",
            "model_name": "elo_logistic",
            "model_version": "v1",
            "window_type": "expanding",
            "fold_index": "1",
            "train_start_utc": "2023-01-01T00:00:00Z",
            "train_end_utc": "2023-12-01T00:00:00Z",
            "test_start_utc": "2024-01-01T01:00:00Z",
            "test_end_utc": "2024-01-01T01:00:00Z",
            "bout_id": "bout_1",
            "event_id": "event_1",
            "bout_datetime_utc": "2024-01-01T01:00:00Z",
            "fighter_id": "fighter_a",
            "opponent_id": "fighter_b",
            "split": "test",
            "label": "1.0",
            "predicted_probability": "0.65",
            "feature_cutoff_utc": "2024-01-01T01:00:00Z",
        },
        {
            "model_run_id": "run_alpha",
            "model_name": "elo_logistic",
            "model_version": "v1",
            "window_type": "expanding",
            "fold_index": "1",
            "train_start_utc": "2023-01-01T00:00:00Z",
            "train_end_utc": "2023-12-01T00:00:00Z",
            "test_start_utc": "2024-01-01T01:00:00Z",
            "test_end_utc": "2024-01-01T01:00:00Z",
            "bout_id": "bout_1",
            "event_id": "event_1",
            "bout_datetime_utc": "2024-01-01T01:00:00Z",
            "fighter_id": "fighter_b",
            "opponent_id": "fighter_a",
            "split": "test",
            "label": "0.0",
            "predicted_probability": "0.35",
            "feature_cutoff_utc": "2024-01-01T01:00:00Z",
        },
        {
            "model_run_id": "run_alpha",
            "model_name": "elo_logistic",
            "model_version": "v1",
            "window_type": "expanding",
            "fold_index": "2",
            "train_start_utc": "2023-01-01T00:00:00Z",
            "train_end_utc": "2024-01-01T01:00:00Z",
            "test_start_utc": "2024-02-01T01:00:00Z",
            "test_end_utc": "2024-02-01T01:00:00Z",
            "bout_id": "bout_2",
            "event_id": "event_2",
            "bout_datetime_utc": "2024-02-01T01:00:00Z",
            "fighter_id": "fighter_c",
            "opponent_id": "fighter_d",
            "split": "test",
            "label": "0.0",
            "predicted_probability": "0.65",
            "feature_cutoff_utc": "2024-02-01T01:00:00Z",
        },
        {
            "model_run_id": "run_alpha",
            "model_name": "elo_logistic",
            "model_version": "v1",
            "window_type": "expanding",
            "fold_index": "2",
            "train_start_utc": "2023-01-01T00:00:00Z",
            "train_end_utc": "2024-01-01T01:00:00Z",
            "test_start_utc": "2024-02-01T01:00:00Z",
            "test_end_utc": "2024-02-01T01:00:00Z",
            "bout_id": "bout_2",
            "event_id": "event_2",
            "bout_datetime_utc": "2024-02-01T01:00:00Z",
            "fighter_id": "fighter_d",
            "opponent_id": "fighter_c",
            "split": "test",
            "label": "1.0",
            "predicted_probability": "0.35",
            "feature_cutoff_utc": "2024-02-01T01:00:00Z",
        },
        {
            "model_run_id": "run_beta",
            "model_name": "tabular_logistic",
            "model_version": "v1",
            "window_type": "expanding",
            "fold_index": "1",
            "train_start_utc": "2023-01-01T00:00:00Z",
            "train_end_utc": "2023-12-01T00:00:00Z",
            "test_start_utc": "2024-01-01T01:00:00Z",
            "test_end_utc": "2024-01-01T01:00:00Z",
            "bout_id": "bout_1",
            "event_id": "event_1",
            "bout_datetime_utc": "2024-01-01T01:00:00Z",
            "fighter_id": "fighter_a",
            "opponent_id": "fighter_b",
            "split": "test",
            "label": "1.0",
            "predicted_probability": "0.52",
            "feature_cutoff_utc": "2024-01-01T01:00:00Z",
        },
        {
            "model_run_id": "run_beta",
            "model_name": "tabular_logistic",
            "model_version": "v1",
            "window_type": "expanding",
            "fold_index": "1",
            "train_start_utc": "2023-01-01T00:00:00Z",
            "train_end_utc": "2023-12-01T00:00:00Z",
            "test_start_utc": "2024-01-01T01:00:00Z",
            "test_end_utc": "2024-01-01T01:00:00Z",
            "bout_id": "bout_1",
            "event_id": "event_1",
            "bout_datetime_utc": "2024-01-01T01:00:00Z",
            "fighter_id": "fighter_b",
            "opponent_id": "fighter_a",
            "split": "test",
            "label": "0.0",
            "predicted_probability": "0.48",
            "feature_cutoff_utc": "2024-01-01T01:00:00Z",
        },
        {
            "model_run_id": "run_beta",
            "model_name": "tabular_logistic",
            "model_version": "v1",
            "window_type": "expanding",
            "fold_index": "2",
            "train_start_utc": "2023-01-01T00:00:00Z",
            "train_end_utc": "2024-01-01T01:00:00Z",
            "test_start_utc": "2024-02-01T01:00:00Z",
            "test_end_utc": "2024-02-01T01:00:00Z",
            "bout_id": "bout_2",
            "event_id": "event_2",
            "bout_datetime_utc": "2024-02-01T01:00:00Z",
            "fighter_id": "fighter_c",
            "opponent_id": "fighter_d",
            "split": "test",
            "label": "0.0",
            "predicted_probability": "0.40",
            "feature_cutoff_utc": "2024-02-01T01:00:00Z",
        },
        {
            "model_run_id": "run_beta",
            "model_name": "tabular_logistic",
            "model_version": "v1",
            "window_type": "expanding",
            "fold_index": "2",
            "train_start_utc": "2023-01-01T00:00:00Z",
            "train_end_utc": "2024-01-01T01:00:00Z",
            "test_start_utc": "2024-02-01T01:00:00Z",
            "test_end_utc": "2024-02-01T01:00:00Z",
            "bout_id": "bout_2",
            "event_id": "event_2",
            "bout_datetime_utc": "2024-02-01T01:00:00Z",
            "fighter_id": "fighter_d",
            "opponent_id": "fighter_c",
            "split": "test",
            "label": "1.0",
            "predicted_probability": "0.60",
            "feature_cutoff_utc": "2024-02-01T01:00:00Z",
        },
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_discover_model_registry_finds_models(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.csv"
    _write_predictions(predictions_path)

    registry = discover_model_registry(predictions_paths=[predictions_path])

    assert len(registry.entries) == 2
    assert {entry.model_identifier for entry in registry.entries} == {
        "elo_logistic:run_alpha",
        "tabular_logistic:run_beta",
    }


def test_compare_registered_models_selects_best_and_periods(tmp_path: Path) -> None:
    db_path = tmp_path / "mma.sqlite3"
    initialize_sqlite_database(db_path)
    with _connect(db_path) as connection:
        _insert_fixture_data(connection)
        connection.commit()

    predictions_path = tmp_path / "predictions.csv"
    _write_predictions(predictions_path)
    report_path = tmp_path / "model_comparison_report.txt"

    registry = discover_model_registry(predictions_paths=[predictions_path])
    result = compare_registered_models(
        db_path=db_path,
        promotion="UFC",
        registry=registry,
        report_output_path=report_path,
        calibration_bins=5,
        min_edge=0.02,
        flat_stake=1.0,
        kelly_fraction=None,
        initial_bankroll=100.0,
        vig_adjustment="proportional",
        one_bet_per_bout=True,
    )

    assert result.report_path.exists()
    assert result.best_by_metric["log_loss"] == "tabular_logistic:run_beta"
    assert result.best_by_metric["roi"] == "tabular_logistic:run_beta"
    assert result.champion_challenger.champion.entry.model_identifier == "tabular_logistic:run_beta"
    assert all(len(summary.periods) == 2 for summary in result.model_summaries)

    report = result.report_path.read_text(encoding="utf-8")
    assert "best_log_loss_model=tabular_logistic:run_beta" in report
    assert "best_brier_score_model=tabular_logistic:run_beta" in report
    assert "best_calibration_model=" in report
    assert "best_roi_model=tabular_logistic:run_beta" in report


def test_compare_models_cli_runs_end_to_end(tmp_path: Path) -> None:
    db_path = tmp_path / "mma_cli.sqlite3"
    initialize_sqlite_database(db_path)
    with _connect(db_path) as connection:
        _insert_fixture_data(connection)
        connection.commit()

    predictions_path = tmp_path / "predictions_cli.csv"
    _write_predictions(predictions_path)
    report_path = tmp_path / "model_comparison_cli_report.txt"

    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "mma-elo",
            "compare-models",
            "--db-path",
            str(db_path),
            "--promotion",
            "UFC",
            "--predictions-path",
            str(predictions_path),
            "--report-output-path",
            str(report_path),
        ]
        assert cli.main() == 0
    finally:
        sys.argv = original_argv

    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "champion_vs_challenger" in report
    assert "best_drawdown_model=" in report
