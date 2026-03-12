"""Tests for chronological walk-forward evaluation and leakage guards."""

from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

import cli
from evaluation import BoutTimelineEntry, build_walk_forward_folds, run_walk_forward_evaluation
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


def test_build_walk_forward_folds_never_uses_future_bouts() -> None:
    timeline = [
        BoutTimelineEntry(bout_id="b1", bout_datetime_utc="2024-01-01T00:00:00Z", bout_order=1),
        BoutTimelineEntry(bout_id="b2", bout_datetime_utc="2024-02-01T00:00:00Z", bout_order=1),
        BoutTimelineEntry(bout_id="b3", bout_datetime_utc="2024-03-01T00:00:00Z", bout_order=1),
        BoutTimelineEntry(bout_id="b4", bout_datetime_utc="2024-04-01T00:00:00Z", bout_order=1),
        BoutTimelineEntry(bout_id="b5", bout_datetime_utc="2024-05-01T00:00:00Z", bout_order=1),
        BoutTimelineEntry(bout_id="b6", bout_datetime_utc="2024-06-01T00:00:00Z", bout_order=1),
    ]

    folds = build_walk_forward_folds(
        timeline=timeline,
        window_type="expanding",
        min_train_bouts=3,
        test_bouts=2,
        rolling_train_bouts=None,
    )

    assert len(folds) == 2
    for fold in folds:
        assert set(fold.train_bout_ids).isdisjoint(set(fold.test_bout_ids))
        assert fold.train_end_utc <= fold.test_start_utc


def test_run_walk_forward_evaluation_outputs_period_metrics(tmp_path: Path) -> None:
    db_path = tmp_path / "mma_walk_forward.sqlite3"
    initialize_sqlite_database(db_path)
    with _connect(db_path) as connection:
        _insert_fixture_data(connection)
        connection.commit()

    ratings_path = tmp_path / "ratings.csv"
    predictions_path = tmp_path / "walk_forward_predictions.csv"
    report_path = tmp_path / "walk_forward_report.txt"

    result = run_walk_forward_evaluation(
        db_path=db_path,
        promotion="UFC",
        window_type="expanding",
        min_train_bouts=3,
        test_bouts=2,
        rolling_train_bouts=None,
        model_version="v1",
        model_run_id="wf_run_001",
        calibration_bins=5,
        ratings_output_path=ratings_path,
        predictions_output_path=predictions_path,
        report_output_path=report_path,
    )

    assert result.model_run_id == "wf_run_001"
    assert result.fold_count == 3
    assert result.predictions_count > 0
    assert predictions_path.exists()
    assert report_path.exists()

    with predictions_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == result.predictions_count
    fold_ids = {int(row["fold_index"]) for row in rows}
    assert fold_ids == {1, 2, 3}
    assert all(row["model_name"] == "elo_logistic" for row in rows)
    assert all(row["window_type"] == "expanding" for row in rows)
    assert all(row["feature_cutoff_utc"] == row["bout_datetime_utc"] for row in rows)

    report_text = report_path.read_text(encoding="utf-8")
    assert "aggregate_metrics" in report_text
    assert "log_loss=" in report_text
    assert "brier_score=" in report_text
    assert "hit_rate=" in report_text
    assert "period_performance" in report_text
    assert "fold=1" in report_text
    assert "calibration_bins" in report_text


def test_walk_forward_cli_runs_end_to_end(tmp_path: Path) -> None:
    db_path = tmp_path / "mma_walk_forward_cli.sqlite3"
    initialize_sqlite_database(db_path)
    with _connect(db_path) as connection:
        _insert_fixture_data(connection)
        connection.commit()

    predictions_path = tmp_path / "cli_walk_forward_predictions.csv"
    report_path = tmp_path / "cli_walk_forward_report.txt"
    ratings_path = tmp_path / "cli_walk_forward_ratings.csv"

    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "mma-elo",
            "walk-forward-eval",
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
            "wf_cli_001",
            "--calibration-bins",
            "5",
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

    assert predictions_path.exists()
    assert report_path.exists()
    assert ratings_path.exists()

    with predictions_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) > 0
    assert all(row["model_run_id"] == "wf_cli_001" for row in rows)
    assert all(row["window_type"] == "rolling" for row in rows)
