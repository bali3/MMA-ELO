"""Tests for segmented edge and model-vs-market reporting."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from evaluation import run_segmented_edge_analysis
from ingestion.db_init import initialize_sqlite_database

NOW = "2026-03-11T00:00:00Z"


def _connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def _insert_fixture_data(connection: sqlite3.Connection) -> None:
    fighters = (
        ("fighter_a", "Fighter A"),
        ("fighter_b", "Fighter B"),
        ("fighter_c", "Fighter C"),
        ("fighter_d", "Fighter D"),
        ("fighter_e", "Fighter E"),
        ("fighter_f", "Fighter F"),
        ("fighter_g", "Fighter G"),
        ("fighter_h", "Fighter H"),
    )
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
                fighter_id,
                full_name,
                "1990-01-01T00:00:00Z",
                "US",
                "Orthodox",
                180.0,
                182.0,
                "ufc_stats",
                f"fighter:{fighter_id}",
                None,
                f"sha-{fighter_id}",
                NOW,
                NOW,
                NOW,
            )
            for fighter_id, full_name in fighters
        ),
    )

    events = (
        ("event_1", "UFC 1", "2024-01-01T00:00:00Z"),
        ("event_2", "UFC 2", "2024-02-01T00:00:00Z"),
        ("event_3", "UFC 3", "2025-01-01T00:00:00Z"),
        ("event_4", "UFC 4", "2025-02-01T00:00:00Z"),
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
                event_id,
                "UFC",
                event_name,
                event_date_utc,
                None,
                "Las Vegas",
                "Nevada",
                "USA",
                None,
                "ufc_stats",
                f"event:{event_id}",
                None,
                f"sha-{event_id}",
                NOW,
                NOW,
                NOW,
            )
            for event_id, event_name, event_date_utc in events
        ),
    )

    bouts = (
        ("bout_1", "event_1", "fighter_a", "fighter_b", "fighter_a", "Lightweight", "2024-01-01T01:00:00Z"),
        ("bout_2", "event_2", "fighter_c", "fighter_d", "fighter_d", "Featherweight", "2024-02-01T01:00:00Z"),
        ("bout_3", "event_3", "fighter_e", "fighter_f", "fighter_e", "Lightweight", "2025-01-01T01:00:00Z"),
        ("bout_4", "event_4", "fighter_g", "fighter_h", "fighter_h", "Welterweight", "2025-02-01T01:00:00Z"),
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

    markets = (
        ("m1a", "bout_1", "book_a", "fighter_a", 0.58, 150.0, "2024-01-01T00:30:00Z"),
        ("m1b", "bout_1", "book_a", "fighter_b", 0.47, 120.0, "2024-01-01T00:30:00Z"),
        ("m2a", "bout_2", "book_b", "fighter_c", 0.62, 110.0, "2024-02-01T00:30:00Z"),
        ("m2b", "bout_2", "book_b", "fighter_d", 0.43, 115.0, "2024-02-01T00:30:00Z"),
        ("m3a", "bout_3", "book_a", "fighter_e", 0.52, 140.0, "2025-01-01T00:30:00Z"),
        ("m3b", "bout_3", "book_a", "fighter_f", 0.53, 140.0, "2025-01-01T00:30:00Z"),
        ("m4a", "bout_4", "book_c", "fighter_g", 0.39, 95.0, "2025-02-01T00:30:00Z"),
        ("m4b", "bout_4", "book_c", "fighter_h", 0.66, 90.0, "2025-02-01T00:30:00Z"),
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
                sportsbook,
                "moneyline",
                fighter_id,
                fighter_id,
                None,
                None,
                implied_probability,
                line_value,
                timestamp_utc,
                "odds_api",
                f"market:{market_id}",
                None,
                f"sha-{market_id}",
                NOW,
                NOW,
                NOW,
            )
            for market_id, bout_id, sportsbook, fighter_id, implied_probability, line_value, timestamp_utc in markets
        ),
    )


def _write_predictions(path: Path) -> None:
    rows = [
        ("elo_logistic", "run_001", "bout_1", "event_1", "2024-01-01T01:00:00Z", "fighter_a", "fighter_b", 1.0, 0.66),
        ("elo_logistic", "run_001", "bout_1", "event_1", "2024-01-01T01:00:00Z", "fighter_b", "fighter_a", 0.0, 0.34),
        ("market_implied_probability", "run_001", "bout_1", "event_1", "2024-01-01T01:00:00Z", "fighter_a", "fighter_b", 1.0, 0.58),
        ("market_implied_probability", "run_001", "bout_1", "event_1", "2024-01-01T01:00:00Z", "fighter_b", "fighter_a", 0.0, 0.47),
        ("elo_logistic", "run_001", "bout_2", "event_2", "2024-02-01T01:00:00Z", "fighter_c", "fighter_d", 0.0, 0.58),
        ("elo_logistic", "run_001", "bout_2", "event_2", "2024-02-01T01:00:00Z", "fighter_d", "fighter_c", 1.0, 0.42),
        ("market_implied_probability", "run_001", "bout_2", "event_2", "2024-02-01T01:00:00Z", "fighter_c", "fighter_d", 0.0, 0.62),
        ("market_implied_probability", "run_001", "bout_2", "event_2", "2024-02-01T01:00:00Z", "fighter_d", "fighter_c", 1.0, 0.43),
        ("elo_logistic", "run_001", "bout_3", "event_3", "2025-01-01T01:00:00Z", "fighter_e", "fighter_f", 1.0, 0.62),
        ("elo_logistic", "run_001", "bout_3", "event_3", "2025-01-01T01:00:00Z", "fighter_f", "fighter_e", 0.0, 0.38),
        ("market_implied_probability", "run_001", "bout_3", "event_3", "2025-01-01T01:00:00Z", "fighter_e", "fighter_f", 1.0, 0.52),
        ("market_implied_probability", "run_001", "bout_3", "event_3", "2025-01-01T01:00:00Z", "fighter_f", "fighter_e", 0.0, 0.53),
        ("elo_logistic", "run_001", "bout_4", "event_4", "2025-02-01T01:00:00Z", "fighter_g", "fighter_h", 0.0, 0.46),
        ("elo_logistic", "run_001", "bout_4", "event_4", "2025-02-01T01:00:00Z", "fighter_h", "fighter_g", 1.0, 0.54),
        ("market_implied_probability", "run_001", "bout_4", "event_4", "2025-02-01T01:00:00Z", "fighter_g", "fighter_h", 0.0, 0.39),
        ("market_implied_probability", "run_001", "bout_4", "event_4", "2025-02-01T01:00:00Z", "fighter_h", "fighter_g", 1.0, 0.66),
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_run_id",
                "model_name",
                "model_version",
                "bout_id",
                "event_id",
                "bout_datetime_utc",
                "fighter_id",
                "opponent_id",
                "split",
                "label",
                "predicted_probability",
                "feature_cutoff_utc",
            ],
        )
        writer.writeheader()
        for model_name, run_id, bout_id, event_id, bout_datetime_utc, fighter_id, opponent_id, label, probability in rows:
            writer.writerow(
                {
                    "model_run_id": run_id,
                    "model_name": model_name,
                    "model_version": "v1",
                    "bout_id": bout_id,
                    "event_id": event_id,
                    "bout_datetime_utc": bout_datetime_utc,
                    "fighter_id": fighter_id,
                    "opponent_id": opponent_id,
                    "split": "test",
                    "label": f"{label:.1f}",
                    "predicted_probability": f"{probability:.2f}",
                    "feature_cutoff_utc": bout_datetime_utc,
                }
            )


def _write_features(path: Path) -> None:
    header = [
        "bout_id",
        "event_id",
        "bout_order",
        "bout_datetime_utc",
        "fighter_id",
        "opponent_id",
        "corner",
        "weight_class",
        "fighter_age_years",
        "height_diff_cm",
        "reach_diff_cm",
        "stance_matchup",
        "days_since_last_fight",
        "prior_fight_count",
        "recent_form_win_rate_l3",
        "recent_form_win_rate_l5",
        "opponent_strength_avg_l3",
        "opponent_strength_avg_l5",
        "finish_rate_all",
        "decision_rate_all",
        "sig_strikes_landed_per_min_all",
        "sig_striking_accuracy_all",
        "sig_striking_defense_all",
        "takedown_attempts_per_15m_all",
        "takedown_accuracy_all",
        "takedown_defense_all",
        "submission_attempts_per_15m_all",
        "avg_fight_duration_seconds_all",
        "rematch_flag",
        "weight_class_experience",
    ]
    rows = [
        ["bout_1", "event_1", 1, "2024-01-01T01:00:00Z", "fighter_a", "fighter_b", "red", "Lightweight", 30, 1, 1, "open", 45, 5, 0.7, 0.7, 0.6, 0.6, 0.4, 0.6, 3.0, 0.5, 0.5, 2.0, 0.4, 0.7, 0.3, 600, 0, 3],
        ["bout_1", "event_1", 1, "2024-01-01T01:00:00Z", "fighter_b", "fighter_a", "blue", "Lightweight", 31, -1, -1, "open", 90, 8, 0.6, 0.6, 0.5, 0.5, 0.3, 0.7, 2.8, 0.4, 0.4, 1.5, 0.3, 0.6, 0.2, 620, 0, 4],
        ["bout_2", "event_2", 1, "2024-02-01T01:00:00Z", "fighter_c", "fighter_d", "red", "Featherweight", 29, 0, 0, "open", 210, 1, "", "", "", "", "", "", "", "", "", "", "", "", "", "", 0, 0],
        ["bout_2", "event_2", 1, "2024-02-01T01:00:00Z", "fighter_d", "fighter_c", "blue", "Featherweight", 28, 0, 0, "open", 25, 12, 0.8, 0.8, 0.7, 0.7, 0.5, 0.5, 3.4, 0.5, 0.6, 3.1, 0.5, 0.8, 0.6, 700, 0, 6],
        ["bout_3", "event_3", 1, "2025-01-01T01:00:00Z", "fighter_e", "fighter_f", "red", "Lightweight", 27, 2, 2, "open", 370, 0, "", "", "", "", "", "", "", "", "", "", "", "", "", "", 0, 0],
        ["bout_3", "event_3", 1, "2025-01-01T01:00:00Z", "fighter_f", "fighter_e", "blue", "Lightweight", 30, -2, -2, "open", 120, 3, 0.4, 0.4, 0.4, 0.4, 0.2, 0.8, 2.1, 0.3, 0.5, 1.0, 0.2, 0.5, 0.1, 500, 0, 1],
        ["bout_4", "event_4", 1, "2025-02-01T01:00:00Z", "fighter_g", "fighter_h", "red", "Welterweight", 32, 0, 0, "open", 75, 9, 0.5, 0.5, 0.5, 0.5, 0.4, 0.6, 2.9, 0.4, 0.5, 2.4, 0.4, 0.7, 0.2, 640, 0, 5],
        ["bout_4", "event_4", 1, "2025-02-01T01:00:00Z", "fighter_h", "fighter_g", "blue", "Welterweight", 29, 0, 0, "open", 40, 10, 0.7, 0.7, 0.7, 0.7, 0.6, 0.4, 3.3, 0.6, 0.6, 2.8, 0.5, 0.8, 0.4, 710, 0, 7],
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def test_segmented_edge_analysis_generates_report_and_csv(tmp_path: Path) -> None:
    db_path = tmp_path / "mma.sqlite3"
    initialize_sqlite_database(db_path)
    with _connect(db_path) as connection:
        _insert_fixture_data(connection)
        connection.commit()

    predictions_path = tmp_path / "baseline_predictions.csv"
    features_path = tmp_path / "pre_fight_features.csv"
    _write_predictions(predictions_path)
    _write_features(features_path)

    report_path = tmp_path / "edge_pockets_report.txt"
    metrics_path = tmp_path / "edge_pockets_segments.csv"
    result = run_segmented_edge_analysis(
        db_path=db_path,
        features_path=features_path,
        predictions_path=predictions_path,
        report_output_path=report_path,
        segment_metrics_output_path=metrics_path,
        promotion="UFC",
        target_model_name="elo_logistic",
        target_model_run_id="run_001",
        baseline_model_name="market_implied_probability",
        baseline_model_run_id="run_001",
        min_edge=0.02,
        flat_stake=1.0,
        kelly_fraction=None,
        initial_bankroll=100.0,
        vig_adjustment="proportional",
        one_bet_per_bout=True,
        min_credible_predictions=2,
        min_credible_bets=1,
        min_credible_bouts=1,
    )

    assert result.report_path.exists()
    assert result.segment_metrics_path.exists()
    assert result.best_log_loss_segment != ""
    assert result.best_brier_segment != ""
    assert result.best_roi_segment != ""

    report_text = report_path.read_text(encoding="utf-8")
    assert "segmented_edge_analysis_report" in report_text
    assert "credible_segments" in report_text
    assert "best_roi_segment=" in report_text
    assert "segment_comparisons" in report_text
    assert "type=bookmaker" in report_text
    assert "type=edge_threshold_bucket" in report_text

    metrics_text = metrics_path.read_text(encoding="utf-8")
    assert "segment_type,segment_value" in metrics_text
    assert "year,2024" in metrics_text
    assert "weight_class,Lightweight" in metrics_text
    assert "bookmaker,book_a" in metrics_text
    assert "feature_completeness_bucket,complete_95_plus" in metrics_text
