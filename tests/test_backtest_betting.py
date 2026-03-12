"""Tests for chronology-safe betting simulation selection and staking behavior."""

from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

import cli
from backtest.betting import (
    BetCandidate,
    DecisionRules,
    calculate_stake,
    run_betting_backtest,
    run_betting_sensitivity_analysis,
    select_bets,
)
from ingestion.db_init import initialize_sqlite_database

NOW = "2026-03-11T00:00:00Z"


def _connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def _insert_base_fixture_data(connection: sqlite3.Connection) -> None:
    connection.executemany(
        """
        INSERT INTO fighters (
            fighter_id, full_name, date_of_birth_utc, nationality, stance, height_cm, reach_cm,
            source_system, source_record_id, source_updated_at_utc, source_payload_sha256,
            ingested_at_utc, created_at_utc, updated_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ("fighter_a", "Fighter A", "1990-01-01T00:00:00Z", "US", "Orthodox", 180.0, 183.0, "ufc_stats", "fighter:a", None, "sha-a", NOW, NOW, NOW),
            ("fighter_b", "Fighter B", "1991-01-01T00:00:00Z", "US", "Southpaw", 178.0, 181.0, "ufc_stats", "fighter:b", None, "sha-b", NOW, NOW, NOW),
            ("fighter_c", "Fighter C", "1992-01-01T00:00:00Z", "US", "Orthodox", 179.0, 182.0, "ufc_stats", "fighter:c", None, "sha-c", NOW, NOW, NOW),
            ("fighter_d", "Fighter D", "1993-01-01T00:00:00Z", "US", "Orthodox", 181.0, 184.0, "ufc_stats", "fighter:d", None, "sha-d", NOW, NOW, NOW),
        ),
    )

    events = (
        ("event_1", "UFC", "UFC 1", "2024-01-01T00:00:00Z"),
        ("event_2", "UFC", "UFC 2", "2024-02-01T00:00:00Z"),
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


def _insert_market_rows(connection: sqlite3.Connection) -> None:
    markets = (
        ("market_b1_a", "bout_1", "fighter_a", 0.40, "2024-01-01T00:30:00Z", 200.0),
        ("market_b1_b", "bout_1", "fighter_b", 0.60, "2024-01-01T00:30:00Z", 120.0),
        ("market_b2_c", "bout_2", "fighter_c", 0.52, "2024-02-01T00:30:00Z", 80.0),
        ("market_b2_d", "bout_2", "fighter_d", 0.52, "2024-02-01T00:30:00Z", 90.0),
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
                line_value,
                timestamp_utc,
                "odds_api",
                f"market:{market_id}",
                None,
                f"sha-market-{market_id}",
                NOW,
                NOW,
                NOW,
            )
            for market_id, bout_id, fighter_id, implied_probability, timestamp_utc, line_value in markets
        ),
    )


def _write_predictions_csv(path: Path) -> None:
    rows = [
        {
            "model_run_id": "run_001",
            "model_name": "elo_logistic",
            "model_version": "v1",
            "bout_id": "bout_1",
            "event_id": "event_1",
            "bout_datetime_utc": "2024-01-01T01:00:00Z",
            "fighter_id": "fighter_a",
            "opponent_id": "fighter_b",
            "split": "test",
            "label": "1.0",
            "predicted_probability": "0.55",
            "feature_cutoff_utc": "2024-01-01T01:00:00Z",
        },
        {
            "model_run_id": "run_001",
            "model_name": "elo_logistic",
            "model_version": "v1",
            "bout_id": "bout_1",
            "event_id": "event_1",
            "bout_datetime_utc": "2024-01-01T01:00:00Z",
            "fighter_id": "fighter_b",
            "opponent_id": "fighter_a",
            "split": "test",
            "label": "0.0",
            "predicted_probability": "0.45",
            "feature_cutoff_utc": "2024-01-01T01:00:00Z",
        },
        {
            "model_run_id": "run_001",
            "model_name": "elo_logistic",
            "model_version": "v1",
            "bout_id": "bout_2",
            "event_id": "event_2",
            "bout_datetime_utc": "2024-02-01T01:00:00Z",
            "fighter_id": "fighter_c",
            "opponent_id": "fighter_d",
            "split": "test",
            "label": "0.0",
            "predicted_probability": "0.53",
            "feature_cutoff_utc": "2024-02-01T01:00:00Z",
        },
        {
            "model_run_id": "run_001",
            "model_name": "elo_logistic",
            "model_version": "v1",
            "bout_id": "bout_2",
            "event_id": "event_2",
            "bout_datetime_utc": "2024-02-01T01:00:00Z",
            "fighter_id": "fighter_d",
            "opponent_id": "fighter_c",
            "split": "test",
            "label": "1.0",
            "predicted_probability": "0.47",
            "feature_cutoff_utc": "2024-02-01T01:00:00Z",
        },
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_backtest_selection_and_flat_staking(tmp_path: Path) -> None:
    db_path = tmp_path / "mma.sqlite3"
    initialize_sqlite_database(db_path)
    with _connect(db_path) as connection:
        _insert_base_fixture_data(connection)
        _insert_market_rows(connection)
        connection.commit()

    predictions_path = tmp_path / "predictions.csv"
    _write_predictions_csv(predictions_path)

    report_path = tmp_path / "backtest_report.txt"
    result = run_betting_backtest(
        db_path=db_path,
        predictions_path=predictions_path,
        report_output_path=report_path,
        promotion="UFC",
        model_name="elo_logistic",
        model_run_id="run_001",
        min_edge=0.02,
        flat_stake=10.0,
        kelly_fraction=None,
        vig_adjustment="proportional",
        one_bet_per_bout=True,
    )

    assert result.bets_count == 2
    assert len(result.bets) == 2
    assert all(abs(bet.stake - 10.0) < 1e-9 for bet in result.bets)

    assert abs(result.total_staked - 20.0) < 1e-9
    assert abs(result.total_pnl - 5.0) < 1e-9
    assert abs(result.roi - 0.25) < 1e-9
    assert abs(result.max_drawdown - 10.0) < 1e-9
    assert abs(result.average_edge - 0.09) < 1e-9

    report = report_path.read_text(encoding="utf-8")
    assert "market_implied_probability_comparison" in report
    assert "roi=0.250000" in report
    assert "bets_count=2" in report


def test_backtest_fractional_kelly_staking(tmp_path: Path) -> None:
    db_path = tmp_path / "mma_kelly.sqlite3"
    initialize_sqlite_database(db_path)
    with _connect(db_path) as connection:
        _insert_base_fixture_data(connection)
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
                    "market_k1_a",
                    "bout_1",
                    "book_a",
                    "moneyline",
                    "fighter_a",
                    "fighter_a",
                    None,
                    2.0,
                    0.50,
                    None,
                    "2024-01-01T00:30:00Z",
                    "odds_api",
                    "market:market_k1_a",
                    None,
                    "sha-market-k1-a",
                    NOW,
                    NOW,
                    NOW,
                ),
                (
                    "market_k1_b",
                    "bout_1",
                    "book_a",
                    "moneyline",
                    "fighter_b",
                    "fighter_b",
                    None,
                    2.0,
                    0.50,
                    None,
                    "2024-01-01T00:30:00Z",
                    "odds_api",
                    "market:market_k1_b",
                    None,
                    "sha-market-k1-b",
                    NOW,
                    NOW,
                    NOW,
                ),
            ),
        )
        connection.commit()

    predictions_path = tmp_path / "kelly_predictions.csv"
    rows = [
        {
            "model_run_id": "run_kelly",
            "model_name": "elo_logistic",
            "model_version": "v1",
            "bout_id": "bout_1",
            "event_id": "event_1",
            "bout_datetime_utc": "2024-01-01T01:00:00Z",
            "fighter_id": "fighter_a",
            "opponent_id": "fighter_b",
            "split": "test",
            "label": "1.0",
            "predicted_probability": "0.60",
            "feature_cutoff_utc": "2024-01-01T01:00:00Z",
        }
    ]
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report_path = tmp_path / "kelly_report.txt"
    result = run_betting_backtest(
        db_path=db_path,
        predictions_path=predictions_path,
        report_output_path=report_path,
        promotion="UFC",
        model_name="elo_logistic",
        model_run_id="run_kelly",
        min_edge=0.01,
        flat_stake=2.0,
        kelly_fraction=0.5,
        initial_bankroll=100.0,
        vig_adjustment="none",
        one_bet_per_bout=True,
    )

    assert result.bets_count == 1
    bet = result.bets[0]
    assert abs(bet.stake - 10.0) < 1e-9
    assert abs(bet.pnl - 10.0) < 1e-9
    assert abs(result.roi - 1.0) < 1e-9


def test_run_backtest_cli_writes_report(tmp_path: Path) -> None:
    db_path = tmp_path / "mma_cli.sqlite3"
    initialize_sqlite_database(db_path)
    with _connect(db_path) as connection:
        _insert_base_fixture_data(connection)
        _insert_market_rows(connection)
        connection.commit()

    predictions_path = tmp_path / "predictions_cli.csv"
    _write_predictions_csv(predictions_path)
    report_path = tmp_path / "backtest_cli_report.txt"

    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "mma-elo",
            "run-backtest",
            "--db-path",
            str(db_path),
            "--predictions-path",
            str(predictions_path),
            "--report-output-path",
            str(report_path),
            "--promotion",
            "UFC",
            "--model-name",
            "elo_logistic",
            "--model-run-id",
            "run_001",
            "--min-edge",
            "0.02",
            "--flat-stake",
            "10",
        ]
        assert cli.main() == 0
    finally:
        sys.argv = original_argv

    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "betting_backtest_report" in report
    assert "market_implied_probability_comparison" in report


def test_select_bets_applies_threshold_and_one_per_bout() -> None:
    candidates = [
        BetCandidate(
            bout_id="bout_a",
            event_id="event_1",
            bout_datetime_utc="2024-01-01T01:00:00Z",
            fighter_id="fighter_a",
            opponent_id="fighter_b",
            label=1.0,
            model_probability=0.60,
            market_probability_raw=0.50,
            market_probability_fair=0.50,
            decimal_odds=2.0,
            edge=0.10,
        ),
        BetCandidate(
            bout_id="bout_a",
            event_id="event_1",
            bout_datetime_utc="2024-01-01T01:00:00Z",
            fighter_id="fighter_b",
            opponent_id="fighter_a",
            label=0.0,
            model_probability=0.46,
            market_probability_raw=0.50,
            market_probability_fair=0.50,
            decimal_odds=2.0,
            edge=-0.04,
        ),
        BetCandidate(
            bout_id="bout_b",
            event_id="event_2",
            bout_datetime_utc="2024-02-01T01:00:00Z",
            fighter_id="fighter_c",
            opponent_id="fighter_d",
            label=0.0,
            model_probability=0.56,
            market_probability_raw=0.50,
            market_probability_fair=0.50,
            decimal_odds=2.0,
            edge=0.06,
        ),
        BetCandidate(
            bout_id="bout_b",
            event_id="event_2",
            bout_datetime_utc="2024-02-01T01:00:00Z",
            fighter_id="fighter_d",
            opponent_id="fighter_c",
            label=1.0,
            model_probability=0.57,
            market_probability_raw=0.50,
            market_probability_fair=0.50,
            decimal_odds=2.0,
            edge=0.07,
        ),
    ]

    selected = select_bets(candidates=candidates, min_edge=0.05, one_bet_per_bout=True)
    assert [row.fighter_id for row in selected] == ["fighter_a", "fighter_d"]
    assert all(row.edge >= 0.05 for row in selected)

    selected_multi = select_bets(candidates=candidates, min_edge=0.05, one_bet_per_bout=False)
    assert [row.fighter_id for row in selected_multi] == ["fighter_a", "fighter_d", "fighter_c"]


def test_calculate_stake_uses_custom_kelly_hook() -> None:
    candidate = BetCandidate(
        bout_id="bout_hook",
        event_id="event_hook",
        bout_datetime_utc="2024-01-01T01:00:00Z",
        fighter_id="fighter_a",
        opponent_id="fighter_b",
        label=1.0,
        model_probability=0.60,
        market_probability_raw=0.50,
        market_probability_fair=0.50,
        decimal_odds=2.0,
        edge=0.10,
    )

    observed: list[tuple[float, float]] = []

    def custom_sizer(row: BetCandidate, bankroll: float, fraction: float) -> float:
        observed.append((bankroll, fraction))
        assert row.bout_id == "bout_hook"
        return 999.0

    stake = calculate_stake(
        candidate=candidate,
        bankroll=75.0,
        flat_stake=2.0,
        kelly_fraction=0.25,
        kelly_sizer=custom_sizer,
    )

    assert observed == [(75.0, 0.25)]
    assert abs(stake - 75.0) < 1e-9


def test_decision_rules_block_low_confidence_and_low_data(tmp_path: Path) -> None:
    db_path = tmp_path / "mma_rules.sqlite3"
    initialize_sqlite_database(db_path)
    with _connect(db_path) as connection:
        _insert_base_fixture_data(connection)
        _insert_market_rows(connection)
        connection.commit()

    predictions_path = tmp_path / "predictions_rules.csv"
    _write_predictions_csv(predictions_path)

    result = run_betting_backtest(
        db_path=db_path,
        predictions_path=predictions_path,
        report_output_path=tmp_path / "rules_report.txt",
        promotion="UFC",
        model_name="elo_logistic",
        model_run_id="run_001",
        min_edge=0.02,
        min_confidence=0.10,
        min_pre_bout_fights=1,
        require_pre_bout_data=True,
        flat_stake=10.0,
        kelly_fraction=None,
        vig_adjustment="proportional",
        one_bet_per_bout=True,
    )

    # All fighters in fixture are in their first tracked bout, so low-data guard should skip all bets.
    assert result.bets_count == 0
    assert abs(result.total_staked - 0.0) < 1e-9


def test_decision_rules_enforce_liquidity_and_stake_caps(tmp_path: Path) -> None:
    db_path = tmp_path / "mma_liquidity_caps.sqlite3"
    initialize_sqlite_database(db_path)
    with _connect(db_path) as connection:
        _insert_base_fixture_data(connection)
        _insert_market_rows(connection)
        connection.commit()

    predictions_path = tmp_path / "predictions_caps.csv"
    _write_predictions_csv(predictions_path)

    result = run_betting_backtest(
        db_path=db_path,
        predictions_path=predictions_path,
        report_output_path=tmp_path / "caps_report.txt",
        promotion="UFC",
        model_name="elo_logistic",
        model_run_id="run_001",
        min_edge=0.02,
        min_liquidity=100.0,
        require_liquidity=True,
        max_stake_per_bet=6.0,
        max_event_exposure=5.0,
        flat_stake=10.0,
        kelly_fraction=None,
        vig_adjustment="proportional",
        one_bet_per_bout=True,
    )

    # Liquidity filter keeps only bout_1 side, then stake and event caps reduce stake to 5.
    assert result.bets_count == 1
    assert abs(result.total_staked - 5.0) < 1e-9
    assert abs(result.bets[0].stake - 5.0) < 1e-9


def test_run_backtest_sensitivity_analysis_writes_plain_text_report(tmp_path: Path) -> None:
    db_path = tmp_path / "mma_sensitivity.sqlite3"
    initialize_sqlite_database(db_path)
    with _connect(db_path) as connection:
        _insert_base_fixture_data(connection)
        _insert_market_rows(connection)
        connection.commit()

    predictions_path = tmp_path / "predictions_sensitivity.csv"
    _write_predictions_csv(predictions_path)
    report_path = tmp_path / "sensitivity_report.txt"

    result = run_betting_sensitivity_analysis(
        db_path=db_path,
        predictions_path=predictions_path,
        report_output_path=report_path,
        promotion="UFC",
        model_name="elo_logistic",
        model_run_id="run_001",
        min_edge_values=[0.01, 0.02],
        min_confidence_values=[0.0, 0.1],
        min_pre_bout_fights_values=[0],
        max_stake_per_bet_values=[None, 5.0],
        max_event_exposure_values=[None],
        max_day_exposure_values=[None],
        min_liquidity_values=[None],
        flat_stake=10.0,
        kelly_fraction=None,
        vig_adjustment="proportional",
        one_bet_per_bout=True,
    )

    assert report_path.exists()
    assert len(result.scenarios) == 8
    report = report_path.read_text(encoding="utf-8")
    assert "betting_rule_sensitivity_report" in report
    assert "roi=" in report
    assert "drawdown=" in report


def test_run_backtest_cli_accepts_hardened_rule_flags(tmp_path: Path) -> None:
    db_path = tmp_path / "mma_cli_rules.sqlite3"
    initialize_sqlite_database(db_path)
    with _connect(db_path) as connection:
        _insert_base_fixture_data(connection)
        _insert_market_rows(connection)
        connection.commit()

    predictions_path = tmp_path / "predictions_cli_rules.csv"
    _write_predictions_csv(predictions_path)
    report_path = tmp_path / "backtest_cli_rules_report.txt"

    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "mma-elo",
            "run-backtest",
            "--db-path",
            str(db_path),
            "--predictions-path",
            str(predictions_path),
            "--report-output-path",
            str(report_path),
            "--promotion",
            "UFC",
            "--model-name",
            "elo_logistic",
            "--model-run-id",
            "run_001",
            "--min-edge",
            "0.02",
            "--min-confidence",
            "0.01",
            "--min-pre-bout-fights",
            "0",
            "--max-stake-per-bet",
            "8",
            "--max-event-exposure",
            "8",
            "--max-day-exposure",
            "20",
            "--min-liquidity",
            "50",
            "--flat-stake",
            "10",
        ]
        assert cli.main() == 0
    finally:
        sys.argv = original_argv

    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "decision_rules" in report


def test_run_backtest_sensitivity_cli_writes_report(tmp_path: Path) -> None:
    db_path = tmp_path / "mma_cli_sensitivity.sqlite3"
    initialize_sqlite_database(db_path)
    with _connect(db_path) as connection:
        _insert_base_fixture_data(connection)
        _insert_market_rows(connection)
        connection.commit()

    predictions_path = tmp_path / "predictions_cli_sensitivity.csv"
    _write_predictions_csv(predictions_path)
    report_path = tmp_path / "backtest_cli_sensitivity_report.txt"

    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "mma-elo",
            "run-backtest-sensitivity",
            "--db-path",
            str(db_path),
            "--predictions-path",
            str(predictions_path),
            "--report-output-path",
            str(report_path),
            "--promotion",
            "UFC",
            "--model-name",
            "elo_logistic",
            "--model-run-id",
            "run_001",
            "--min-edge-values",
            "0.01,0.02",
            "--min-confidence-values",
            "0.0,0.1",
            "--min-pre-bout-fights-values",
            "0",
            "--max-stake-per-bet-values",
            "none,5",
            "--max-event-exposure-values",
            "none",
            "--max-day-exposure-values",
            "none",
            "--min-liquidity-values",
            "none",
            "--flat-stake",
            "10",
        ]
        assert cli.main() == 0
    finally:
        sys.argv = original_argv

    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "betting_rule_sensitivity_report" in report


def test_select_bets_supports_decision_rules_object() -> None:
    candidates = [
        BetCandidate(
            bout_id="bout_a",
            event_id="event_1",
            bout_datetime_utc="2024-01-01T01:00:00Z",
            fighter_id="fighter_a",
            opponent_id="fighter_b",
            label=1.0,
            model_probability=0.60,
            market_probability_raw=0.50,
            market_probability_fair=0.50,
            decimal_odds=2.0,
            edge=0.10,
            confidence=0.10,
            fighter_pre_bout_count=3,
            opponent_pre_bout_count=3,
            market_liquidity=100.0,
        ),
        BetCandidate(
            bout_id="bout_b",
            event_id="event_2",
            bout_datetime_utc="2024-02-01T01:00:00Z",
            fighter_id="fighter_c",
            opponent_id="fighter_d",
            label=1.0,
            model_probability=0.56,
            market_probability_raw=0.50,
            market_probability_fair=0.50,
            decimal_odds=2.0,
            edge=0.06,
            confidence=0.06,
            fighter_pre_bout_count=0,
            opponent_pre_bout_count=0,
            market_liquidity=20.0,
        ),
    ]
    rules = DecisionRules(
        min_edge=0.05,
        min_confidence=0.08,
        min_pre_bout_fights=2,
        max_stake_per_bet=None,
        max_event_exposure=None,
        max_day_exposure=None,
        min_liquidity=50.0,
        require_liquidity=True,
    )

    selected = select_bets(
        candidates=candidates,
        min_edge=0.0,
        one_bet_per_bout=True,
        decision_rules=rules,
    )
    assert [row.fighter_id for row in selected] == ["fighter_a"]
