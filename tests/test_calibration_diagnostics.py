"""Tests for calibration diagnostics metrics and CLI integration."""

from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

import cli
from evaluation import compute_calibration_metrics, run_calibration_diagnostics
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
        ("bout_1", "event_1", "fighter_a", "fighter_b", "fighter_a", "Lightweight", "2024-01-01T01:00:00Z"),
        ("bout_2", "event_2", "fighter_c", "fighter_d", "fighter_d", "Featherweight", "2024-02-01T01:00:00Z"),
        ("bout_3", "event_3", "fighter_a", "fighter_d", "fighter_a", "Lightweight", "2024-03-01T01:00:00Z"),
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
                weight_class,
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
            for bout_id, event_id, red_id, blue_id, winner_id, weight_class, start_utc in bouts
        ),
    )


def _write_predictions(path: Path) -> None:
    rows = [
        {
            "model_run_id": "run_001",
            "model_name": "elo_logistic",
            "model_version": "v1",
            "split": "test",
            "bout_id": "bout_1",
            "event_id": "event_1",
            "bout_datetime_utc": "2024-01-01T01:00:00Z",
            "fighter_id": "fighter_a",
            "opponent_id": "fighter_b",
            "label": "1.0",
            "predicted_probability": "0.70",
            "feature_cutoff_utc": "2024-01-01T01:00:00Z",
        },
        {
            "model_run_id": "run_001",
            "model_name": "elo_logistic",
            "model_version": "v1",
            "split": "test",
            "bout_id": "bout_1",
            "event_id": "event_1",
            "bout_datetime_utc": "2024-01-01T01:00:00Z",
            "fighter_id": "fighter_b",
            "opponent_id": "fighter_a",
            "label": "0.0",
            "predicted_probability": "0.30",
            "feature_cutoff_utc": "2024-01-01T01:00:00Z",
        },
        {
            "model_run_id": "run_001",
            "model_name": "elo_logistic",
            "model_version": "v1",
            "split": "test",
            "bout_id": "bout_2",
            "event_id": "event_2",
            "bout_datetime_utc": "2024-02-01T01:00:00Z",
            "fighter_id": "fighter_c",
            "opponent_id": "fighter_d",
            "label": "0.0",
            "predicted_probability": "0.40",
            "feature_cutoff_utc": "2024-02-01T01:00:00Z",
        },
        {
            "model_run_id": "run_001",
            "model_name": "elo_logistic",
            "model_version": "v1",
            "split": "test",
            "bout_id": "bout_2",
            "event_id": "event_2",
            "bout_datetime_utc": "2024-02-01T01:00:00Z",
            "fighter_id": "fighter_d",
            "opponent_id": "fighter_c",
            "label": "1.0",
            "predicted_probability": "0.60",
            "feature_cutoff_utc": "2024-02-01T01:00:00Z",
        },
        {
            "model_run_id": "run_001",
            "model_name": "elo_logistic",
            "model_version": "v1",
            "split": "test",
            "bout_id": "bout_3",
            "event_id": "event_3",
            "bout_datetime_utc": "2024-03-01T01:00:00Z",
            "fighter_id": "fighter_a",
            "opponent_id": "fighter_d",
            "label": "1.0",
            "predicted_probability": "0.55",
            "feature_cutoff_utc": "2024-03-01T02:00:00Z",
        },
        {
            "model_run_id": "run_001",
            "model_name": "elo_logistic",
            "model_version": "v1",
            "split": "test",
            "bout_id": "bout_3",
            "event_id": "event_3",
            "bout_datetime_utc": "2024-03-01T01:00:00Z",
            "fighter_id": "fighter_d",
            "opponent_id": "fighter_a",
            "label": "0.0",
            "predicted_probability": "",
            "feature_cutoff_utc": "2024-03-01T01:00:00Z",
        },
        {
            "model_run_id": "run_001",
            "model_name": "elo_logistic",
            "model_version": "v1",
            "split": "test",
            "bout_id": "bout_1",
            "event_id": "event_1",
            "bout_datetime_utc": "2024-01-01T01:00:00Z",
            "fighter_id": "fighter_a",
            "opponent_id": "fighter_b",
            "label": "",
            "predicted_probability": "0.70",
            "feature_cutoff_utc": "2024-01-01T01:00:00Z",
        },
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_compute_calibration_metrics_expected_values() -> None:
    metrics = compute_calibration_metrics(
        labels=[1.0, 0.0, 1.0, 0.0],
        probabilities=[0.9, 0.7, 0.2, 0.1],
        num_bins=2,
    )

    assert len(metrics.bins) == 2
    assert abs(metrics.expected_calibration_error - 0.325) < 1e-9
    assert abs(metrics.maximum_calibration_error - 0.35) < 1e-9


def test_run_calibration_diagnostics_generates_artifacts(tmp_path: Path) -> None:
    db_path = tmp_path / "mma.sqlite3"
    initialize_sqlite_database(db_path)
    with _connect(db_path) as connection:
        _insert_fixture_data(connection)
        connection.commit()

    predictions_path = tmp_path / "predictions.csv"
    _write_predictions(predictions_path)

    output_dir = tmp_path / "diagnostics"
    report_path = output_dir / "report.txt"
    result = run_calibration_diagnostics(
        db_path=db_path,
        predictions_path=predictions_path,
        output_dir=output_dir,
        report_output_path=report_path,
        promotion="UFC",
        model_name="elo_logistic",
        model_run_id="run_001",
        calibration_bins=5,
    )

    assert result.predictions_loaded == 5
    assert result.predictions_used == 4
    assert result.dropped_chronology_invalid == 1
    assert result.dropped_missing_probability == 1
    assert result.dropped_invalid_label == 1

    assert result.calibration_table_path.exists()
    assert result.reliability_curve_path.exists()
    assert result.probability_bucket_summary_path.exists()
    assert result.performance_breakdown_path.exists()
    assert result.report_path.exists()

    breakdown_text = result.performance_breakdown_path.read_text(encoding="utf-8")
    assert "favourite_status" in breakdown_text
    assert "weight_class" in breakdown_text

    report_text = result.report_path.read_text(encoding="utf-8")
    assert "calibration_quality=" in report_text
    assert "prediction_filtering" in report_text


def test_calibration_diagnostics_cli_runs_end_to_end(tmp_path: Path) -> None:
    db_path = tmp_path / "mma_cli.sqlite3"
    initialize_sqlite_database(db_path)
    with _connect(db_path) as connection:
        _insert_fixture_data(connection)
        connection.commit()

    predictions_path = tmp_path / "predictions_cli.csv"
    _write_predictions(predictions_path)

    output_dir = tmp_path / "cli_diagnostics"
    report_path = output_dir / "report.txt"

    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "mma-elo",
            "calibration-diagnostics",
            "--db-path",
            str(db_path),
            "--predictions-path",
            str(predictions_path),
            "--promotion",
            "UFC",
            "--model-name",
            "elo_logistic",
            "--model-run-id",
            "run_001",
            "--calibration-bins",
            "5",
            "--output-dir",
            str(output_dir),
            "--report-output-path",
            str(report_path),
        ]
        assert cli.main() == 0
    finally:
        sys.argv = original_argv

    assert (output_dir / "calibration_table.csv").exists()
    assert (output_dir / "reliability_curve.csv").exists()
    assert (output_dir / "probability_bucket_summary.csv").exists()
    assert (output_dir / "performance_breakdowns.csv").exists()
    assert report_path.exists()
