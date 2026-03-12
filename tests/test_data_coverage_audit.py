"""Tests for data coverage audit report generation."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ingestion.db_init import initialize_sqlite_database
from validation import run_data_coverage_audit


def test_data_coverage_audit_writes_missingness_reasons(tmp_path: Path) -> None:
    db_path = tmp_path / "mma.sqlite3"
    features_path = tmp_path / "features.csv"
    report_path = tmp_path / "coverage_audit.txt"
    initialize_sqlite_database(db_path)

    now = "2026-03-11T00:00:00Z"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO fighters (
                fighter_id, full_name, date_of_birth_utc, nationality, stance, height_cm, reach_cm,
                source_system, source_record_id, source_updated_at_utc, source_payload_sha256,
                ingested_at_utc, created_at_utc, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("f1", "Fighter One", None, None, None, None, None, "ufc_stats", "fighter:f1", None, "sha1", now, now, now),
        )
        connection.execute(
            """
            INSERT INTO fighters (
                fighter_id, full_name, date_of_birth_utc, nationality, stance, height_cm, reach_cm,
                source_system, source_record_id, source_updated_at_utc, source_payload_sha256,
                ingested_at_utc, created_at_utc, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("f2", "Fighter Two", "1990-01-01T00:00:00Z", None, None, 180.0, 182.0, "ufc_stats", "fighter:f2", None, "sha2", now, now, now),
        )
        connection.execute(
            """
            INSERT INTO events (
                event_id, promotion, event_name, event_date_utc, venue, city, region, country, timezone_name,
                source_system, source_record_id, source_updated_at_utc, source_payload_sha256,
                ingested_at_utc, created_at_utc, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("e1", "UFC", "UFC Test", "2024-01-01T00:00:00Z", None, None, None, None, None, "ufc_stats", "event:e1", None, "shae1", now, now, now),
        )
        connection.execute(
            """
            INSERT INTO bouts (
                bout_id, event_id, bout_order, fighter_red_id, fighter_blue_id, weight_class, gender,
                is_title_fight, scheduled_rounds, bout_start_time_utc, result_method, result_round,
                result_time_seconds, winner_fighter_id, source_system, source_record_id, source_updated_at_utc,
                source_payload_sha256, ingested_at_utc, created_at_utc, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("b1", "e1", 1, "f1", "f2", "Lightweight", "M", 0, 3, "2024-01-01T01:00:00Z", "Decision", 3, 300, "f2", "ufc_stats", "bout:b1", None, "shab1", now, now, now),
        )
        connection.execute(
            """
            INSERT INTO fighter_bout_stats (
                fighter_bout_stats_id, bout_id, fighter_id, opponent_fighter_id, corner,
                knockdowns, sig_strikes_landed, sig_strikes_attempted, total_strikes_landed,
                total_strikes_attempted, takedowns_landed, takedowns_attempted, submission_attempts,
                reversals, control_time_seconds, source_system, source_record_id, source_updated_at_utc,
                source_payload_sha256, ingested_at_utc, created_at_utc, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "b1:f1",
                "b1",
                "f1",
                "f2",
                "red",
                1,
                10,
                None,
                None,
                None,
                1,
                None,
                0,
                None,
                None,
                "ufc_stats",
                "fighter_bout_stats:b1:f1",
                None,
                "sha_stat_1",
                now,
                now,
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO fighter_bout_stats (
                fighter_bout_stats_id, bout_id, fighter_id, opponent_fighter_id, corner,
                knockdowns, sig_strikes_landed, sig_strikes_attempted, total_strikes_landed,
                total_strikes_attempted, takedowns_landed, takedowns_attempted, submission_attempts,
                reversals, control_time_seconds, source_system, source_record_id, source_updated_at_utc,
                source_payload_sha256, ingested_at_utc, created_at_utc, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "b1:f2",
                "b1",
                "f2",
                "f1",
                "blue",
                0,
                8,
                None,
                None,
                None,
                0,
                None,
                0,
                None,
                None,
                "ufc_stats",
                "fighter_bout_stats:b1:f2",
                None,
                "sha_stat_2",
                now,
                now,
                now,
            ),
        )
        connection.commit()

    features_path.write_text(
        (
            "bout_id,event_id,bout_order,bout_datetime_utc,fighter_id,opponent_id,corner,weight_class,"
            "fighter_age_years,height_diff_cm,reach_diff_cm,stance_matchup,days_since_last_fight,prior_fight_count,"
            "recent_form_win_rate_l3,recent_form_win_rate_l5,opponent_strength_avg_l3,opponent_strength_avg_l5,"
            "finish_rate_all,decision_rate_all,sig_strikes_landed_per_min_all,sig_striking_accuracy_all,"
            "sig_striking_defense_all,takedown_attempts_per_15m_all,takedown_accuracy_all,takedown_defense_all,"
            "submission_attempts_per_15m_all,avg_fight_duration_seconds_all,rematch_flag,weight_class_experience\n"
            "b1,e1,1,2024-01-01T01:00:00Z,f1,f2,red,Lightweight,,,,,,"  # age/height/reach/days/prior missing
            "0,,,,,,,,,,,,,,,\n"
        ),
        encoding="utf-8",
    )

    result = run_data_coverage_audit(
        db_path=db_path,
        promotion="UFC",
        features_path=features_path,
        report_output_path=report_path,
    )

    assert result.total_events == 1
    assert result.total_completed_bouts == 1
    assert result.total_fighters == 2
    assert result.completed_bouts_with_two_stats_rows == 1
    assert result.completed_bouts_with_one_stats_row == 0
    assert result.completed_bouts_with_zero_stats_rows == 0
    text = report_path.read_text(encoding="utf-8")
    assert "bout_stats_coverage_totals" in text
    assert "bouts_with_two_stats_rows=1" in text
    assert "bout_stats_coverage_by_year" in text
    assert "year=2024 completed_bouts=1 bouts_with_two_stats_rows=1 bouts_with_one_stats_row=0 bouts_with_zero_stats_rows=0" in text
    assert "bout_stats_parser_path_proxy" in text
    assert "rows_with_event_details_fallback_shape=2" in text
    assert "feature=fighter_age_years reason=missing_fighter_dob count=1" in text
    assert "feature=height_diff_cm reason=missing_height_metadata count=1" in text
    assert "feature=reach_diff_cm reason=missing_reach_metadata count=1" in text
    assert "join_matching_failures" in text
