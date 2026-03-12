"""CLI entrypoint for MMA model system."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from datetime import datetime

from backtest import run_betting_backtest, run_betting_sensitivity_analysis
from evaluation import (
    compare_registered_models,
    discover_model_registry,
    run_ablation_experiments,
    run_calibration_diagnostics,
    run_segmented_edge_analysis,
    run_walk_forward_evaluation,
)
from features import build_feature_report, generate_pre_fight_features
from ingestion.schema import TABLES
from ingestion.sources.odds import run_market_odds_ingestion
from ingestion.sources.ufc_stats import (
    run_ufc_stats_bout_stats_ingestion_from_archives,
    run_ufc_stats_bout_stats_ingestion,
    run_ufc_stats_events_ingestion_from_archives,
    run_ufc_stats_events_ingestion,
    run_ufc_stats_fighter_metadata_ingestion_from_archives,
    run_ufc_stats_fighter_metadata_ingestion,
)
from markets import run_market_coverage_report
from models import run_baseline_models
from orchestration import run_research_workflow
from ratings import EloConfig, build_ratings_report, generate_elo_ratings_history
from utils.config import load_settings
from utils.logging import configure_logging
from validation import run_data_coverage_audit, run_data_validations


def build_parser() -> argparse.ArgumentParser:
    """Build top-level CLI parser."""

    parser = argparse.ArgumentParser(prog="mma-elo", description="MMA model research CLI")
    parser.add_argument("--env", help="Override MMA_ELO_ENV for this run.")
    parser.add_argument("--log-level", help="Override MMA_ELO_LOG_LEVEL for this run.")

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "print-schema-summary",
        help="Print schema table names and exit.",
    )
    subparsers.add_parser(
        "print-data-dirs",
        help="Print expected local data directories and exit.",
    )
    validation_parser = subparsers.add_parser(
        "validate-data",
        help="Run data integrity and chronology validations.",
    )
    validation_parser.add_argument(
        "--db-path",
        help="SQLite database path. Defaults to MMA_ELO_DATABASE_URL sqlite target.",
    )
    coverage_audit_parser = subparsers.add_parser(
        "data-coverage-audit",
        help="Generate plain-text data/feature coverage audit report.",
    )
    coverage_audit_parser.add_argument(
        "--db-path",
        help="SQLite database path. Defaults to MMA_ELO_DATABASE_URL sqlite target.",
    )
    coverage_audit_parser.add_argument(
        "--promotion",
        default="UFC",
        help="Promotion filter for date spans and join checks (default: UFC).",
    )
    coverage_audit_parser.add_argument(
        "--features-path",
        default="data/processed/features/pre_fight_features.csv",
        help="Path to generated feature CSV used for coverage/missingness checks.",
    )
    coverage_audit_parser.add_argument(
        "--report-output-path",
        default="data/processed/reports/data_coverage_audit.txt",
        help="Path to persist the plain-text data coverage audit report.",
    )
    market_ingest_parser = subparsers.add_parser(
        "ingest-market-odds",
        help="Ingest The Odds API market data (uses MMA_ELO_ODDS_API_KEY), match to bouts, and persist into markets table.",
    )
    market_ingest_parser.add_argument(
        "--db-path",
        help="SQLite database path. Defaults to MMA_ELO_DATABASE_URL sqlite target.",
    )
    market_ingest_parser.add_argument(
        "--raw-root",
        default="data/raw",
        help="Directory root for raw Odds API payload archival and archive replay.",
    )
    market_ingest_parser.add_argument(
        "--promotion",
        default="UFC",
        help="Promotion filter used for fight matching and coverage report.",
    )
    market_ingest_parser.add_argument(
        "--source-system",
        default="odds_api",
        help="Source system label persisted for ingested market rows.",
    )
    market_ingest_parser.add_argument(
        "--source-mode",
        choices=("upcoming", "historical"),
        default="upcoming",
        help="Odds ingestion mode: upcoming snapshots or historical snapshots (default: upcoming).",
    )
    market_ingest_parser.add_argument(
        "--sport-key",
        default="mma_mixed_martial_arts",
        help="The Odds API sport key (default: mma_mixed_martial_arts).",
    )
    market_ingest_parser.add_argument(
        "--regions",
        default="us",
        help="The Odds API regions parameter (default: us).",
    )
    market_ingest_parser.add_argument(
        "--markets",
        default="h2h",
        help="The Odds API markets parameter (default: h2h for fight winner).",
    )
    market_ingest_parser.add_argument(
        "--odds-format",
        choices=("american", "decimal"),
        default="american",
        help="The Odds API odds format (default: american).",
    )
    market_ingest_parser.add_argument(
        "--from-archives",
        action="store_true",
        help="Replay archived Odds API JSON payloads from --raw-root instead of live fetching.",
    )
    market_ingest_parser.add_argument(
        "--historical-start-utc",
        default=None,
        help="Historical mode: inclusive UTC snapshot start timestamp (ISO-8601).",
    )
    market_ingest_parser.add_argument(
        "--historical-end-utc",
        default=None,
        help="Historical mode: inclusive UTC snapshot end timestamp (ISO-8601).",
    )
    market_ingest_parser.add_argument(
        "--historical-interval-hours",
        type=int,
        default=24,
        help="Historical mode: interval hours between snapshot pulls (default: 24).",
    )
    market_ingest_parser.add_argument(
        "--completed-bouts-only",
        action="store_true",
        help="Restrict bout matching to completed bouts only (default enabled for historical mode).",
    )
    market_ingest_parser.add_argument(
        "--report-output-path",
        default="data/processed/reports/market_data_validation_report.txt",
        help="Path to persist market matching/coverage validation report.",
    )

    market_report_parser = subparsers.add_parser(
        "market-coverage-report",
        help="Generate market matching and coverage report from persisted markets data.",
    )
    market_report_parser.add_argument(
        "--db-path",
        help="SQLite database path. Defaults to MMA_ELO_DATABASE_URL sqlite target.",
    )
    market_report_parser.add_argument(
        "--promotion",
        default="UFC",
        help="Promotion filter for market coverage report.",
    )
    market_report_parser.add_argument(
        "--source-system",
        default=None,
        help="Optional source_system filter for market rows.",
    )
    market_report_parser.add_argument(
        "--report-output-path",
        default="data/processed/reports/market_data_validation_report.txt",
        help="Path to persist market matching/coverage report.",
    )

    ingest_parser = subparsers.add_parser(
        "ingest-ufc-stats-events",
        help="Ingest UFC Stats completed events and bout results.",
    )
    ingest_parser.add_argument(
        "--db-path",
        help="SQLite database path. Defaults to MMA_ELO_DATABASE_URL sqlite target.",
    )
    ingest_parser.add_argument(
        "--raw-root",
        default="data/raw",
        help="Directory root for raw response archival.",
    )
    ingest_parser.add_argument(
        "--event-limit",
        type=int,
        default=None,
        help="Optional max number of completed events to ingest.",
    )
    ingest_parser.add_argument(
        "--from-archives",
        action="store_true",
        help="Replay archived UFC Stats HTML from --raw-root instead of live fetching.",
    )

    fighter_ingest_parser = subparsers.add_parser(
        "ingest-ufc-stats-fighters",
        help="Ingest UFC Stats fighter profile metadata only.",
    )
    fighter_ingest_parser.add_argument(
        "--db-path",
        help="SQLite database path. Defaults to MMA_ELO_DATABASE_URL sqlite target.",
    )
    fighter_ingest_parser.add_argument(
        "--raw-root",
        default="data/raw",
        help="Directory root for raw response archival.",
    )
    fighter_ingest_parser.add_argument(
        "--fighter-limit",
        type=int,
        default=None,
        help="Optional max number of fighter profiles to ingest.",
    )
    fighter_ingest_parser.add_argument(
        "--from-archives",
        action="store_true",
        help="Replay archived UFC Stats HTML from --raw-root instead of live fetching.",
    )

    bout_stats_parser = subparsers.add_parser(
        "ingest-ufc-stats-bout-stats",
        help="Ingest UFC Stats per-bout fighter statistics from fight pages.",
    )
    bout_stats_parser.add_argument(
        "--db-path",
        help="SQLite database path. Defaults to MMA_ELO_DATABASE_URL sqlite target.",
    )
    bout_stats_parser.add_argument(
        "--raw-root",
        default="data/raw",
        help="Directory root for raw response archival.",
    )
    bout_stats_parser.add_argument(
        "--event-limit",
        type=int,
        default=None,
        help="Optional max number of completed events to ingest.",
    )
    bout_stats_parser.add_argument(
        "--from-archives",
        action="store_true",
        help="Replay archived UFC Stats HTML from --raw-root instead of live fetching.",
    )

    feature_gen_parser = subparsers.add_parser(
        "generate-features",
        help="Generate sequential pre-fight features in strict chronological order.",
    )
    feature_gen_parser.add_argument(
        "--db-path",
        help="SQLite database path. Defaults to MMA_ELO_DATABASE_URL sqlite target.",
    )
    feature_gen_parser.add_argument(
        "--output-path",
        default="data/processed/features/pre_fight_features.csv",
        help="CSV output path for generated pre-fight features.",
    )
    feature_gen_parser.add_argument(
        "--promotion",
        default="UFC",
        help="Promotion filter for feature generation (default: UFC).",
    )

    feature_report_parser = subparsers.add_parser(
        "feature-report",
        help="Show feature coverage and missingness summary from persisted features.",
    )
    feature_report_parser.add_argument(
        "--features-path",
        default="data/processed/features/pre_fight_features.csv",
        help="Path to generated feature CSV.",
    )

    ratings_history_parser = subparsers.add_parser(
        "build-ratings-history",
        help="Generate chronological Elo rating history (overall + weight class).",
    )
    ratings_history_parser.add_argument(
        "--db-path",
        help="SQLite database path. Defaults to MMA_ELO_DATABASE_URL sqlite target.",
    )
    ratings_history_parser.add_argument(
        "--output-path",
        default="data/processed/ratings/elo_ratings_history.csv",
        help="CSV output path for generated Elo history.",
    )
    ratings_history_parser.add_argument(
        "--promotion",
        default="UFC",
        help="Promotion filter for ratings history generation (default: UFC).",
    )
    ratings_history_parser.add_argument(
        "--initial-rating",
        type=float,
        default=1500.0,
        help="Initial Elo rating for unseen fighters (default: 1500).",
    )
    ratings_history_parser.add_argument(
        "--k-factor",
        type=float,
        default=32.0,
        help="Elo K-factor for completed bout updates (default: 32).",
    )

    ratings_report_parser = subparsers.add_parser(
        "ratings-report",
        help="Show current top-rated fighters and rating coverage.",
    )
    ratings_report_parser.add_argument(
        "--history-path",
        default="data/processed/ratings/elo_ratings_history.csv",
        help="Path to generated Elo history CSV.",
    )
    ratings_report_parser.add_argument(
        "--db-path",
        help="Optional SQLite database path used for fighter coverage denominator and names.",
    )
    ratings_report_parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top fighters to print (default: 10).",
    )

    baselines_parser = subparsers.add_parser(
        "train-baselines",
        help="Train chronology-safe baseline models and persist test-period predictions.",
    )
    baselines_parser.add_argument(
        "--db-path",
        help="SQLite database path. Defaults to MMA_ELO_DATABASE_URL sqlite target.",
    )
    baselines_parser.add_argument(
        "--promotion",
        default="UFC",
        help="Promotion filter for baseline run (default: UFC).",
    )
    baselines_parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Chronological train split fraction by completed bouts (default: 0.8).",
    )
    baselines_parser.add_argument(
        "--model-version",
        default="v1",
        help="Model version label used for persisted predictions.",
    )
    baselines_parser.add_argument(
        "--model-run-id",
        default=None,
        help="Optional deterministic run id. Defaults to generated UTC id.",
    )
    baselines_parser.add_argument(
        "--features-output-path",
        default="data/processed/features/pre_fight_features.csv",
        help="Path to persist generated pre-fight features.",
    )
    baselines_parser.add_argument(
        "--ratings-output-path",
        default="data/processed/ratings/elo_ratings_history.csv",
        help="Path to persist generated Elo rating history.",
    )
    baselines_parser.add_argument(
        "--predictions-output-path",
        default="data/processed/predictions/baseline_predictions.csv",
        help="Path to persist baseline predictions CSV.",
    )
    baselines_parser.add_argument(
        "--report-output-path",
        default="data/processed/reports/baseline_report.txt",
        help="Path to persist baseline summary report.",
    )

    walk_forward_parser = subparsers.add_parser(
        "walk-forward-eval",
        help="Run strict chronological walk-forward evaluation (expanding or rolling windows).",
    )
    walk_forward_parser.add_argument(
        "--db-path",
        help="SQLite database path. Defaults to MMA_ELO_DATABASE_URL sqlite target.",
    )
    walk_forward_parser.add_argument(
        "--promotion",
        default="UFC",
        help="Promotion filter for evaluation (default: UFC).",
    )
    walk_forward_parser.add_argument(
        "--window-type",
        choices=("expanding", "rolling"),
        default="expanding",
        help="Walk-forward window policy (default: expanding).",
    )
    walk_forward_parser.add_argument(
        "--min-train-bouts",
        type=int,
        default=200,
        help="Minimum completed bouts in the initial train window (default: 200).",
    )
    walk_forward_parser.add_argument(
        "--test-bouts",
        type=int,
        default=50,
        help="Completed bouts per forward test period (default: 50).",
    )
    walk_forward_parser.add_argument(
        "--rolling-train-bouts",
        type=int,
        default=None,
        help="Train-window size for rolling evaluation. Required when --window-type rolling.",
    )
    walk_forward_parser.add_argument(
        "--model-version",
        default="v1",
        help="Model version label used for persisted predictions.",
    )
    walk_forward_parser.add_argument(
        "--model-run-id",
        default=None,
        help="Optional deterministic run id. Defaults to generated UTC id.",
    )
    walk_forward_parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Number of calibration bins for summaries (default: 10).",
    )
    walk_forward_parser.add_argument(
        "--ratings-output-path",
        default="data/processed/ratings/elo_ratings_history.csv",
        help="Path to persist generated Elo rating history used for evaluation.",
    )
    walk_forward_parser.add_argument(
        "--predictions-output-path",
        default="data/processed/predictions/walk_forward_predictions.csv",
        help="Path to persist walk-forward prediction rows.",
    )
    walk_forward_parser.add_argument(
        "--report-output-path",
        default="data/processed/reports/walk_forward_report.txt",
        help="Path to persist walk-forward evaluation report.",
    )

    ablation_parser = subparsers.add_parser(
        "run-ablation-experiments",
        help="Run chronology-safe walk-forward ablation experiments for major feature groups.",
    )
    ablation_parser.add_argument(
        "--db-path",
        help="SQLite database path. Defaults to MMA_ELO_DATABASE_URL sqlite target.",
    )
    ablation_parser.add_argument(
        "--promotion",
        default="UFC",
        help="Promotion filter for evaluation (default: UFC).",
    )
    ablation_parser.add_argument(
        "--window-type",
        choices=("expanding", "rolling"),
        default="expanding",
        help="Walk-forward window policy (default: expanding).",
    )
    ablation_parser.add_argument(
        "--min-train-bouts",
        type=int,
        default=200,
        help="Minimum completed bouts in the initial train window (default: 200).",
    )
    ablation_parser.add_argument(
        "--test-bouts",
        type=int,
        default=50,
        help="Completed bouts per forward test period (default: 50).",
    )
    ablation_parser.add_argument(
        "--rolling-train-bouts",
        type=int,
        default=None,
        help="Train-window size for rolling evaluation. Required when --window-type rolling.",
    )
    ablation_parser.add_argument(
        "--model-version",
        default="v1",
        help="Model version label used for persisted predictions.",
    )
    ablation_parser.add_argument(
        "--model-run-id",
        default=None,
        help="Optional deterministic run id. Defaults to generated UTC id.",
    )
    ablation_parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Number of calibration bins for summaries (default: 10).",
    )
    ablation_parser.add_argument(
        "--include-group",
        action="append",
        default=None,
        choices=("features", "elo", "market"),
        help="Optional feature-group inclusion filter. Repeat flag to include multiple groups.",
    )
    ablation_parser.add_argument(
        "--exclude-group",
        action="append",
        default=None,
        choices=("features", "elo", "market"),
        help="Optional feature-group exclusion filter. Repeat flag to exclude multiple groups.",
    )
    ablation_parser.add_argument(
        "--features-output-path",
        default="data/processed/features/pre_fight_features.csv",
        help="Path to persist generated pre-fight features used for ablations.",
    )
    ablation_parser.add_argument(
        "--ratings-output-path",
        default="data/processed/ratings/elo_ratings_history.csv",
        help="Path to persist generated Elo rating history used for ablations.",
    )
    ablation_parser.add_argument(
        "--predictions-output-path",
        default="data/processed/predictions/ablation_predictions.csv",
        help="Path to persist ablation prediction rows.",
    )
    ablation_parser.add_argument(
        "--comparison-report-output-path",
        default="data/processed/reports/ablation_comparison_report.txt",
        help="Path to persist ablation comparison report.",
    )
    ablation_parser.add_argument(
        "--contribution-report-output-path",
        default="data/processed/reports/ablation_contribution_report.txt",
        help="Path to persist ablation contribution summary report.",
    )

    backtest_parser = subparsers.add_parser(
        "run-backtest",
        help="Run chronology-safe betting simulation from precomputed model predictions.",
    )
    backtest_parser.add_argument(
        "--db-path",
        help="SQLite database path. Defaults to MMA_ELO_DATABASE_URL sqlite target.",
    )
    backtest_parser.add_argument(
        "--predictions-path",
        default="data/processed/predictions/baseline_predictions.csv",
        help="Path to prediction CSV used for backtesting.",
    )
    backtest_parser.add_argument(
        "--report-output-path",
        default="data/processed/reports/backtest_report.txt",
        help="Path to persist plain-text backtest report.",
    )
    backtest_parser.add_argument(
        "--promotion",
        default="UFC",
        help="Promotion filter for market snapshots (default: UFC).",
    )
    backtest_parser.add_argument(
        "--model-name",
        default="elo_logistic",
        help="Model name to evaluate from predictions CSV (default: elo_logistic).",
    )
    backtest_parser.add_argument(
        "--model-run-id",
        default=None,
        help="Optional model run id filter. Defaults to latest run id in CSV.",
    )
    backtest_parser.add_argument(
        "--min-edge",
        type=float,
        default=0.02,
        help="Minimum model-vs-market edge required to place a bet (default: 0.02).",
    )
    backtest_parser.add_argument(
        "--flat-stake",
        type=float,
        default=1.0,
        help="Flat stake size per bet in units (default: 1.0).",
    )
    backtest_parser.add_argument(
        "--kelly-fraction",
        type=float,
        default=None,
        help="Optional fractional Kelly multiplier in [0,1]. If set, Kelly sizing is used.",
    )
    backtest_parser.add_argument(
        "--initial-bankroll",
        type=float,
        default=100.0,
        help="Initial bankroll used when Kelly sizing is enabled (default: 100).",
    )
    backtest_parser.add_argument(
        "--vig-adjustment",
        choices=("none", "proportional"),
        default="proportional",
        help="Vig-adjustment mode for market implied probabilities (default: proportional).",
    )
    backtest_parser.add_argument(
        "--allow-multiple-bets-per-bout",
        action="store_true",
        help="Allow betting both sides in the same bout if both pass threshold.",
    )
    backtest_parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum confidence abs(p-0.5) required to place a bet (default: 0.0).",
    )
    backtest_parser.add_argument(
        "--min-pre-bout-fights",
        type=int,
        default=0,
        help="Minimum historical fights required for both fighters (default: 0).",
    )
    backtest_parser.add_argument(
        "--require-pre-bout-data",
        action="store_true",
        help="Skip bets when pre-bout fight counts are unavailable.",
    )
    backtest_parser.add_argument(
        "--max-stake-per-bet",
        type=float,
        default=None,
        help="Optional hard cap for stake size per bet.",
    )
    backtest_parser.add_argument(
        "--max-event-exposure",
        type=float,
        default=None,
        help="Optional max total stake for any single event/card.",
    )
    backtest_parser.add_argument(
        "--max-day-exposure",
        type=float,
        default=None,
        help="Optional max total stake for any single UTC day.",
    )
    backtest_parser.add_argument(
        "--min-liquidity",
        type=float,
        default=None,
        help="Optional minimum market liquidity proxy from line_value.",
    )
    backtest_parser.add_argument(
        "--require-liquidity",
        action="store_true",
        help="Skip bets when liquidity data is unavailable.",
    )

    backtest_sensitivity_parser = subparsers.add_parser(
        "run-backtest-sensitivity",
        help="Run backtests across decision-rule grids and write ROI/drawdown sensitivity report.",
    )
    backtest_sensitivity_parser.add_argument(
        "--db-path",
        help="SQLite database path. Defaults to MMA_ELO_DATABASE_URL sqlite target.",
    )
    backtest_sensitivity_parser.add_argument(
        "--predictions-path",
        default="data/processed/predictions/baseline_predictions.csv",
        help="Path to prediction CSV used for backtesting.",
    )
    backtest_sensitivity_parser.add_argument(
        "--report-output-path",
        default="data/processed/reports/backtest_sensitivity_report.txt",
        help="Path to persist plain-text sensitivity report.",
    )
    backtest_sensitivity_parser.add_argument(
        "--promotion",
        default="UFC",
        help="Promotion filter for market snapshots (default: UFC).",
    )
    backtest_sensitivity_parser.add_argument(
        "--model-name",
        default="elo_logistic",
        help="Model name to evaluate from predictions CSV (default: elo_logistic).",
    )
    backtest_sensitivity_parser.add_argument(
        "--model-run-id",
        default=None,
        help="Optional model run id filter. Defaults to latest run id in CSV.",
    )
    backtest_sensitivity_parser.add_argument(
        "--min-edge-values",
        default="0.02",
        help="Comma-separated min-edge values.",
    )
    backtest_sensitivity_parser.add_argument(
        "--min-confidence-values",
        default="0.0",
        help="Comma-separated confidence threshold values.",
    )
    backtest_sensitivity_parser.add_argument(
        "--min-pre-bout-fights-values",
        default="0",
        help="Comma-separated min pre-bout fight-count values.",
    )
    backtest_sensitivity_parser.add_argument(
        "--max-stake-per-bet-values",
        default="none",
        help="Comma-separated per-bet cap values; use 'none' for no cap.",
    )
    backtest_sensitivity_parser.add_argument(
        "--max-event-exposure-values",
        default="none",
        help="Comma-separated per-event exposure values; use 'none' for no cap.",
    )
    backtest_sensitivity_parser.add_argument(
        "--max-day-exposure-values",
        default="none",
        help="Comma-separated per-day exposure values; use 'none' for no cap.",
    )
    backtest_sensitivity_parser.add_argument(
        "--min-liquidity-values",
        default="none",
        help="Comma-separated liquidity thresholds; use 'none' for no filter.",
    )
    backtest_sensitivity_parser.add_argument(
        "--require-pre-bout-data",
        action="store_true",
        help="Skip bets when pre-bout fight counts are unavailable.",
    )
    backtest_sensitivity_parser.add_argument(
        "--require-liquidity",
        action="store_true",
        help="Skip bets when liquidity data is unavailable.",
    )
    backtest_sensitivity_parser.add_argument(
        "--flat-stake",
        type=float,
        default=1.0,
        help="Flat stake size per bet in units (default: 1.0).",
    )
    backtest_sensitivity_parser.add_argument(
        "--kelly-fraction",
        type=float,
        default=None,
        help="Optional fractional Kelly multiplier in [0,1]. If set, Kelly sizing is used.",
    )
    backtest_sensitivity_parser.add_argument(
        "--initial-bankroll",
        type=float,
        default=100.0,
        help="Initial bankroll used when Kelly sizing is enabled (default: 100).",
    )
    backtest_sensitivity_parser.add_argument(
        "--vig-adjustment",
        choices=("none", "proportional"),
        default="proportional",
        help="Vig-adjustment mode for market implied probabilities (default: proportional).",
    )
    backtest_sensitivity_parser.add_argument(
        "--allow-multiple-bets-per-bout",
        action="store_true",
        help="Allow betting both sides in the same bout if both pass threshold.",
    )

    calibration_parser = subparsers.add_parser(
        "calibration-diagnostics",
        help="Generate calibration diagnostics from chronologically valid predictions.",
    )
    calibration_parser.add_argument(
        "--db-path",
        help="SQLite database path. Defaults to MMA_ELO_DATABASE_URL sqlite target.",
    )
    calibration_parser.add_argument(
        "--predictions-path",
        default="data/processed/predictions/walk_forward_predictions.csv",
        help="Path to prediction CSV used for calibration diagnostics.",
    )
    calibration_parser.add_argument(
        "--promotion",
        default="UFC",
        help="Promotion filter for bout metadata joins (default: UFC).",
    )
    calibration_parser.add_argument(
        "--model-name",
        default="elo_logistic",
        help="Model name to evaluate from predictions CSV (default: elo_logistic).",
    )
    calibration_parser.add_argument(
        "--model-run-id",
        default=None,
        help="Optional model run id filter. Defaults to latest run id in CSV.",
    )
    calibration_parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Number of probability bins for diagnostics (default: 10).",
    )
    calibration_parser.add_argument(
        "--output-dir",
        default="data/processed/reports/calibration_diagnostics",
        help="Directory for CSV diagnostics artifacts.",
    )
    calibration_parser.add_argument(
        "--report-output-path",
        default="data/processed/reports/calibration_diagnostics/report.txt",
        help="Path to persist plain-text calibration report.",
    )

    compare_models_parser = subparsers.add_parser(
        "compare-models",
        help="Compare all available models from prediction artifacts and select champion/challenger.",
    )
    compare_models_parser.add_argument(
        "--db-path",
        help="SQLite database path. Defaults to MMA_ELO_DATABASE_URL sqlite target.",
    )
    compare_models_parser.add_argument(
        "--promotion",
        default="UFC",
        help="Promotion filter for market snapshots (default: UFC).",
    )
    compare_models_parser.add_argument(
        "--predictions-path",
        action="append",
        default=None,
        help=(
            "Prediction CSV path to scan for model runs. "
            "Repeat this flag to include multiple files."
        ),
    )
    compare_models_parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Number of calibration bins used in aggregate model comparisons (default: 10).",
    )
    compare_models_parser.add_argument(
        "--min-edge",
        type=float,
        default=0.02,
        help="Minimum model-vs-market edge required to place a bet (default: 0.02).",
    )
    compare_models_parser.add_argument(
        "--flat-stake",
        type=float,
        default=1.0,
        help="Flat stake size per bet in units (default: 1.0).",
    )
    compare_models_parser.add_argument(
        "--kelly-fraction",
        type=float,
        default=None,
        help="Optional fractional Kelly multiplier in [0,1]. If set, Kelly sizing is used.",
    )
    compare_models_parser.add_argument(
        "--initial-bankroll",
        type=float,
        default=100.0,
        help="Initial bankroll used when Kelly sizing is enabled (default: 100).",
    )
    compare_models_parser.add_argument(
        "--vig-adjustment",
        choices=("none", "proportional"),
        default="proportional",
        help="Vig-adjustment mode for market implied probabilities (default: proportional).",
    )
    compare_models_parser.add_argument(
        "--allow-multiple-bets-per-bout",
        action="store_true",
        help="Allow betting both sides in the same bout if both pass threshold.",
    )
    compare_models_parser.add_argument(
        "--report-output-path",
        default="data/processed/reports/model_comparison_report.txt",
        help="Path to persist plain-text model comparison report.",
    )

    workflow_parser = subparsers.add_parser(
        "run-research-workflow",
        help="Run full reproducible research workflow in strict stage order with fail-fast logging.",
    )
    workflow_parser.add_argument(
        "--db-path",
        help="SQLite database path. Defaults to MMA_ELO_DATABASE_URL sqlite target.",
    )
    workflow_parser.add_argument(
        "--raw-root",
        default="data/raw",
        help="Directory root for raw response archival.",
    )
    workflow_parser.add_argument(
        "--run-ingestion",
        action="store_true",
        help="Enable live ingestion stages. By default workflow runs in offline snapshot mode.",
    )
    workflow_parser.add_argument(
        "--event-limit",
        type=int,
        default=None,
        help="Optional max number of completed events to ingest when --run-ingestion is enabled.",
    )
    workflow_parser.add_argument(
        "--fighter-limit",
        type=int,
        default=None,
        help="Optional max number of fighters to ingest when --run-ingestion is enabled.",
    )
    workflow_parser.add_argument(
        "--promotion",
        default="UFC",
        help="Promotion filter for feature/model/evaluation stages (default: UFC).",
    )
    workflow_parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Chronological train split fraction for baseline stage (default: 0.8).",
    )
    workflow_parser.add_argument(
        "--window-type",
        choices=("expanding", "rolling"),
        default="expanding",
        help="Walk-forward window policy (default: expanding).",
    )
    workflow_parser.add_argument(
        "--min-train-bouts",
        type=int,
        default=200,
        help="Minimum completed bouts in initial walk-forward train window (default: 200).",
    )
    workflow_parser.add_argument(
        "--test-bouts",
        type=int,
        default=50,
        help="Completed bouts per forward test period (default: 50).",
    )
    workflow_parser.add_argument(
        "--rolling-train-bouts",
        type=int,
        default=None,
        help="Train-window size for rolling walk-forward; required for --window-type rolling.",
    )
    workflow_parser.add_argument(
        "--model-version",
        default="v1",
        help="Model version label used for persisted predictions.",
    )
    workflow_parser.add_argument(
        "--model-run-id",
        default=None,
        help="Optional deterministic model run id reused across stages.",
    )
    workflow_parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Calibration bins for walk-forward reporting (default: 10).",
    )
    workflow_parser.add_argument(
        "--min-edge",
        type=float,
        default=0.02,
        help="Minimum model-vs-market edge for backtest betting decisions.",
    )
    workflow_parser.add_argument(
        "--flat-stake",
        type=float,
        default=1.0,
        help="Flat stake size per bet in units (default: 1.0).",
    )
    workflow_parser.add_argument(
        "--kelly-fraction",
        type=float,
        default=None,
        help="Optional fractional Kelly multiplier in [0,1]. If set, Kelly sizing is used.",
    )
    workflow_parser.add_argument(
        "--initial-bankroll",
        type=float,
        default=100.0,
        help="Initial bankroll used when Kelly sizing is enabled (default: 100).",
    )
    workflow_parser.add_argument(
        "--vig-adjustment",
        choices=("none", "proportional"),
        default="proportional",
        help="Vig-adjustment mode for market implied probabilities (default: proportional).",
    )
    workflow_parser.add_argument(
        "--allow-multiple-bets-per-bout",
        action="store_true",
        help="Allow betting both sides in the same bout if both pass threshold.",
    )
    workflow_parser.add_argument(
        "--features-output-path",
        default="data/processed/features/pre_fight_features.csv",
        help="Path to persist generated pre-fight features.",
    )
    workflow_parser.add_argument(
        "--ratings-output-path",
        default="data/processed/ratings/elo_ratings_history.csv",
        help="Path to persist generated Elo rating history.",
    )
    workflow_parser.add_argument(
        "--baseline-predictions-output-path",
        default="data/processed/predictions/baseline_predictions.csv",
        help="Path to persist baseline predictions CSV.",
    )
    workflow_parser.add_argument(
        "--baseline-report-output-path",
        default="data/processed/reports/baseline_report.txt",
        help="Path to persist baseline summary report.",
    )
    workflow_parser.add_argument(
        "--walk-forward-predictions-output-path",
        default="data/processed/predictions/walk_forward_predictions.csv",
        help="Path to persist walk-forward prediction rows.",
    )
    workflow_parser.add_argument(
        "--walk-forward-report-output-path",
        default="data/processed/reports/walk_forward_report.txt",
        help="Path to persist walk-forward report.",
    )
    workflow_parser.add_argument(
        "--ablation-predictions-output-path",
        default="data/processed/predictions/ablation_predictions.csv",
        help="Path to persist ablation prediction rows.",
    )
    workflow_parser.add_argument(
        "--ablation-comparison-report-output-path",
        default="data/processed/reports/ablation_comparison_report.txt",
        help="Path to persist ablation comparison report.",
    )
    workflow_parser.add_argument(
        "--ablation-contribution-report-output-path",
        default="data/processed/reports/ablation_contribution_report.txt",
        help="Path to persist ablation contribution summary report.",
    )
    workflow_parser.add_argument(
        "--calibration-output-dir",
        default="data/processed/reports/calibration_diagnostics",
        help="Directory for calibration diagnostics artifacts.",
    )
    workflow_parser.add_argument(
        "--calibration-report-output-path",
        default="data/processed/reports/calibration_diagnostics/report.txt",
        help="Path to persist calibration diagnostics report.",
    )
    workflow_parser.add_argument(
        "--model-comparison-report-output-path",
        default="data/processed/reports/model_comparison_report.txt",
        help="Path to persist model comparison report.",
    )
    workflow_parser.add_argument(
        "--data-coverage-audit-report-output-path",
        default="data/processed/reports/data_coverage_audit.txt",
        help="Path to persist the data coverage audit report.",
    )
    workflow_parser.add_argument(
        "--market-coverage-report-output-path",
        default="data/processed/reports/market_data_validation_report.txt",
        help="Path to persist the market matching and coverage validation report.",
    )
    workflow_parser.add_argument(
        "--backtest-report-output-path",
        default="data/processed/reports/backtest_report.txt",
        help="Path to persist backtest report.",
    )
    workflow_parser.add_argument(
        "--edge-pockets-report-output-path",
        default="data/processed/reports/edge_pockets_report.txt",
        help="Path to persist segmented edge-analysis report.",
    )
    workflow_parser.add_argument(
        "--edge-pockets-segment-metrics-output-path",
        default="data/processed/reports/edge_pockets_segments.csv",
        help="Path to persist segmented edge-analysis comparison CSV.",
    )
    workflow_parser.add_argument(
        "--summary-report-path",
        default="data/processed/reports/final_research_summary.txt",
        help="Path to persist final end-to-end workflow summary report.",
    )
    workflow_parser.add_argument(
        "--log-dir",
        default="data/processed/logs/research_workflow",
        help="Directory for per-stage workflow logs.",
    )
    workflow_parser.add_argument(
        "--workflow-log-path",
        default="data/processed/logs/research_workflow/workflow.log",
        help="Path to persist high-level workflow execution log.",
    )

    return parser


def main() -> int:
    """Run CLI command."""

    parser = build_parser()
    args = parser.parse_args()

    settings = load_settings()
    log_level = args.log_level or settings.log_level
    configure_logging(log_level)

    logger = logging.getLogger(__name__)
    effective_env = args.env or settings.environment
    logger.info("mma-elo CLI started", extra={"environment": effective_env})

    if args.command == "print-schema-summary":
        for table in TABLES:
            print(table.name)
        return 0

    if args.command == "print-data-dirs":
        base = Path("data")
        for dirname in ("raw", "interim", "processed"):
            print((base / dirname).as_posix())
        return 0

    if args.command == "ingest-ufc-stats-events":
        db_path = Path(args.db_path) if args.db_path else _sqlite_path_from_database_url(settings.database_url)
        raw_root = Path(args.raw_root)
        result = (
            run_ufc_stats_events_ingestion_from_archives(
                db_path=db_path,
                raw_root=raw_root,
                event_limit=args.event_limit,
            )
            if args.from_archives
            else run_ufc_stats_events_ingestion(
                db_path=db_path,
                raw_root=raw_root,
                event_limit=args.event_limit,
            )
        )
        print(
            "ufc_stats_events_ingestion_complete "
            f"inserted={result.inserted_count} updated={result.updated_count} skipped={result.skipped_count}"
        )
        return 0

    if args.command == "ingest-ufc-stats-fighters":
        db_path = Path(args.db_path) if args.db_path else _sqlite_path_from_database_url(settings.database_url)
        raw_root = Path(args.raw_root)
        result = (
            run_ufc_stats_fighter_metadata_ingestion_from_archives(
                db_path=db_path,
                raw_root=raw_root,
                fighter_limit=args.fighter_limit,
            )
            if args.from_archives
            else run_ufc_stats_fighter_metadata_ingestion(
                db_path=db_path,
                raw_root=raw_root,
                fighter_limit=args.fighter_limit,
            )
        )
        print(
            "ufc_stats_fighter_metadata_ingestion_complete "
            f"inserted={result.inserted_count} updated={result.updated_count} skipped={result.skipped_count}"
        )
        return 0

    if args.command == "ingest-ufc-stats-bout-stats":
        db_path = Path(args.db_path) if args.db_path else _sqlite_path_from_database_url(settings.database_url)
        raw_root = Path(args.raw_root)
        result = (
            run_ufc_stats_bout_stats_ingestion_from_archives(
                db_path=db_path,
                raw_root=raw_root,
                event_limit=args.event_limit,
            )
            if args.from_archives
            else run_ufc_stats_bout_stats_ingestion(
                db_path=db_path,
                raw_root=raw_root,
                event_limit=args.event_limit,
            )
        )
        print(
            "ufc_stats_bout_stats_ingestion_complete "
            f"inserted={result.inserted_count} updated={result.updated_count} skipped={result.skipped_count}"
        )
        return 0

    if args.command == "validate-data":
        db_path = Path(args.db_path) if args.db_path else _sqlite_path_from_database_url(settings.database_url)
        issue_count = _run_data_validation_command(db_path=db_path)
        return 0 if issue_count == 0 else 1

    if args.command == "ingest-market-odds":
        db_path = Path(args.db_path) if args.db_path else _sqlite_path_from_database_url(settings.database_url)
        historical_start_utc = _parse_optional_utc_datetime(args.historical_start_utc)
        historical_end_utc = _parse_optional_utc_datetime(args.historical_end_utc)
        result = run_market_odds_ingestion(
            db_path=db_path,
            raw_root=Path(args.raw_root),
            promotion=args.promotion,
            source_system=args.source_system,
            source_mode=args.source_mode,
            odds_api_key=settings.odds_api_key,
            odds_api_base_url=settings.odds_api_base_url,
            sport_key=args.sport_key,
            regions=args.regions,
            markets=args.markets,
            odds_format=args.odds_format,
            from_archives=args.from_archives,
            historical_start_utc=historical_start_utc,
            historical_end_utc=historical_end_utc,
            historical_interval_hours=args.historical_interval_hours,
            completed_bouts_only=True if args.completed_bouts_only else None,
            report_output_path=Path(args.report_output_path),
        )
        print(
            "market_odds_ingestion_complete "
            f"inserted={result.inserted_count} "
            f"updated={result.updated_count} "
            f"skipped={result.skipped_count} "
            f"matched_rows={result.matched_rows} "
            f"unmatched_rows={result.unmatched_rows} "
            f"exact_matches={result.exact_matches} "
            f"normalized_matches={result.normalized_matches} "
            f"reversed_matches={result.reversed_matches} "
            f"ambiguous_rows={result.ambiguous_rows} "
            f"no_match_rows={result.no_match_rows} "
            f"matching_audit_report_path={result.matching_audit_report_path.as_posix()} "
            f"report_path={result.report.report_path.as_posix()}"
        )
        return 0

    if args.command == "market-coverage-report":
        db_path = Path(args.db_path) if args.db_path else _sqlite_path_from_database_url(settings.database_url)
        result = run_market_coverage_report(
            db_path=db_path,
            promotion=args.promotion,
            source_system=args.source_system,
            report_output_path=Path(args.report_output_path),
        )
        print(
            "market_coverage_report_complete "
            f"market_rows={result.market_rows} "
            f"matched_fights={result.matched_fights} "
            f"unmatched_fights={result.unmatched_fights} "
            f"match_rate={result.match_rate:.8f} "
            f"opening_odds_covered_fights={result.opening_odds_covered_fights} "
            f"closing_odds_covered_fights={result.closing_odds_covered_fights if result.closing_odds_covered_fights is not None else 'n/a'} "
            f"report_path={result.report_path.as_posix()}"
        )
        return 0

    if args.command == "data-coverage-audit":
        db_path = Path(args.db_path) if args.db_path else _sqlite_path_from_database_url(settings.database_url)
        result = run_data_coverage_audit(
            db_path=db_path,
            promotion=args.promotion,
            features_path=Path(args.features_path),
            report_output_path=Path(args.report_output_path),
        )
        print(
            "data_coverage_audit_complete "
            f"total_events={result.total_events} "
            f"total_completed_bouts={result.total_completed_bouts} "
            f"total_fighters={result.total_fighters} "
            f"total_fighter_bout_stats_rows={result.total_fighter_bout_stats_rows} "
            f"report_path={result.report_path.as_posix()}"
        )
        return 0

    if args.command == "generate-features":
        db_path = Path(args.db_path) if args.db_path else _sqlite_path_from_database_url(settings.database_url)
        output_path = Path(args.output_path)
        row_count, persisted_path = generate_pre_fight_features(
            db_path=db_path,
            output_path=output_path,
            promotion=args.promotion,
        )
        print(f"feature_generation_complete rows={row_count} path={persisted_path.as_posix()}")
        return 0

    if args.command == "feature-report":
        report = build_feature_report(features_path=Path(args.features_path))
        print(
            "feature_report_summary "
            f"bouts={report.bout_count} rows={report.row_count} "
            f"earliest={report.earliest_bout_date_utc} latest={report.latest_bout_date_utc}"
        )
        for name, coverage in sorted(report.coverage_rates.items()):
            missing = report.missing_counts.get(name, 0)
            print(f"feature_metric name={name} coverage={coverage:.6f} missing={missing}")
        return 0

    if args.command == "build-ratings-history":
        db_path = Path(args.db_path) if args.db_path else _sqlite_path_from_database_url(settings.database_url)
        output_path = Path(args.output_path)
        config = EloConfig(
            initial_rating=args.initial_rating,
            k_factor=args.k_factor,
            promotion=args.promotion,
        )
        result = generate_elo_ratings_history(
            db_path=db_path,
            output_path=output_path,
            config=config,
        )
        print(
            "ratings_history_build_complete "
            f"bouts={result.bout_count} rows={result.row_count} path={result.output_path.as_posix()}"
        )
        return 0

    if args.command == "ratings-report":
        report = build_ratings_report(
            ratings_history_path=Path(args.history_path),
            db_path=Path(args.db_path) if args.db_path else None,
            top_n=args.top_n,
        )
        print(
            "ratings_report_summary "
            f"fighters_rated={report.rated_fighters} "
            f"fighters_total={report.total_fighters} "
            f"coverage={report.coverage_rate:.6f} "
            f"earliest_rating_date={report.earliest_rating_datetime_utc or 'n/a'} "
            f"latest_rating_date={report.latest_rating_datetime_utc or 'n/a'}"
        )
        for coverage in report.weight_class_coverage:
            print(
                "ratings_weight_class_coverage "
                f"weight_class={coverage.weight_class} "
                f"fighters_rated={coverage.rated_fighters} "
                f"share_of_rated={coverage.share_of_rated_fighters:.6f}"
            )
        for fighter in report.top_fighters:
            print(
                "ratings_top_fighter "
                f"fighter_id={fighter.fighter_id} "
                f"name={fighter.full_name} "
                f"overall_elo={fighter.overall_elo:.6f} "
                f"completed_bouts={fighter.completed_bouts}"
            )
        return 0

    if args.command == "train-baselines":
        db_path = Path(args.db_path) if args.db_path else _sqlite_path_from_database_url(settings.database_url)
        result = run_baseline_models(
            db_path=db_path,
            promotion=args.promotion,
            train_fraction=args.train_fraction,
            model_version=args.model_version,
            model_run_id=args.model_run_id,
            features_output_path=Path(args.features_output_path),
            ratings_output_path=Path(args.ratings_output_path),
            predictions_output_path=Path(args.predictions_output_path),
            report_output_path=Path(args.report_output_path),
        )
        print(
            "baseline_training_complete "
            f"model_run_id={result.model_run_id} "
            f"predictions={result.prediction_count} "
            f"bouts_scored={result.scored_bout_count} "
            f"predictions_path={result.predictions_path.as_posix()} "
            f"report_path={result.report_path.as_posix()}"
        )
        print(
            "baseline_train_test_boundaries "
            f"train_start_utc={result.train_test_split.train_start_utc} "
            f"train_end_utc={result.train_test_split.train_end_utc} "
            f"test_start_utc={result.train_test_split.test_start_utc} "
            f"test_end_utc={result.train_test_split.test_end_utc}"
        )
        for metric in result.coverage_metrics:
            print(
                "baseline_model_coverage "
                f"model={metric.model_name} "
                f"scored_rows={metric.scored_rows} "
                f"eligible_rows={metric.eligible_rows} "
                f"coverage={metric.coverage_rate:.6f}"
            )
        return 0

    if args.command == "walk-forward-eval":
        db_path = Path(args.db_path) if args.db_path else _sqlite_path_from_database_url(settings.database_url)
        result = run_walk_forward_evaluation(
            db_path=db_path,
            promotion=args.promotion,
            window_type=args.window_type,
            min_train_bouts=args.min_train_bouts,
            test_bouts=args.test_bouts,
            rolling_train_bouts=args.rolling_train_bouts,
            model_version=args.model_version,
            model_run_id=args.model_run_id,
            calibration_bins=args.calibration_bins,
            ratings_output_path=Path(args.ratings_output_path),
            predictions_output_path=Path(args.predictions_output_path),
            report_output_path=Path(args.report_output_path),
        )
        print(
            "walk_forward_eval_complete "
            f"model_run_id={result.model_run_id} "
            f"model={result.model_name} "
            f"window_type={result.window_type} "
            f"folds={result.fold_count} "
            f"predictions={result.predictions_count} "
            f"predictions_path={result.predictions_path.as_posix()} "
            f"report_path={result.report_path.as_posix()}"
        )
        print(
            "walk_forward_aggregate_metrics "
            f"log_loss={result.aggregate_log_loss:.8f} "
            f"brier_score={result.aggregate_brier_score:.8f} "
            f"hit_rate={result.aggregate_hit_rate:.8f} "
            f"calibration_ece={result.aggregate_calibration.expected_calibration_error:.8f} "
            f"calibration_mce={result.aggregate_calibration.maximum_calibration_error:.8f}"
        )
        for fold in result.fold_metrics:
            print(
                "walk_forward_period "
                f"fold={fold.fold_index} "
                f"train_start_utc={fold.train_start_utc} "
                f"train_end_utc={fold.train_end_utc} "
                f"test_start_utc={fold.test_start_utc} "
                f"test_end_utc={fold.test_end_utc} "
                f"log_loss={fold.log_loss:.8f} "
                f"brier_score={fold.brier_score:.8f} "
                f"hit_rate={fold.hit_rate:.8f} "
                f"calibration_ece={fold.calibration_ece:.8f} "
                f"calibration_mce={fold.calibration_mce:.8f}"
            )
        return 0

    if args.command == "run-ablation-experiments":
        db_path = Path(args.db_path) if args.db_path else _sqlite_path_from_database_url(settings.database_url)
        result = run_ablation_experiments(
            db_path=db_path,
            promotion=args.promotion,
            window_type=args.window_type,
            min_train_bouts=args.min_train_bouts,
            test_bouts=args.test_bouts,
            rolling_train_bouts=args.rolling_train_bouts,
            model_version=args.model_version,
            model_run_id=args.model_run_id,
            calibration_bins=args.calibration_bins,
            include_groups=args.include_group,
            exclude_groups=args.exclude_group,
            features_output_path=Path(args.features_output_path),
            ratings_output_path=Path(args.ratings_output_path),
            predictions_output_path=Path(args.predictions_output_path),
            comparison_report_output_path=Path(args.comparison_report_output_path),
            contribution_report_output_path=Path(args.contribution_report_output_path),
        )
        print(
            "ablation_experiments_complete "
            f"model_run_id={result.model_run_id} "
            f"window_type={result.window_type} "
            f"folds={result.fold_count} "
            f"scenarios={len(result.scenario_metrics)} "
            f"predictions_path={result.predictions_path.as_posix()} "
            f"comparison_report_path={result.comparison_report_path.as_posix()} "
            f"contribution_report_path={result.contribution_report_path.as_posix()}"
        )
        for metrics in sorted(result.scenario_metrics, key=lambda row: row.log_loss):
            print(
                "ablation_scenario "
                f"scenario_id={metrics.config.scenario_id} "
                f"groups={','.join(metrics.config.active_groups())} "
                f"interactions={int(metrics.config.include_matchup_interactions)} "
                f"predictions={metrics.predictions_count} "
                f"log_loss={metrics.log_loss:.8f} "
                f"brier_score={metrics.brier_score:.8f} "
                f"hit_rate={metrics.hit_rate:.8f} "
                f"calibration_ece={metrics.calibration_ece:.8f}"
            )
        return 0

    if args.command == "run-backtest":
        db_path = Path(args.db_path) if args.db_path else _sqlite_path_from_database_url(settings.database_url)
        result = run_betting_backtest(
            db_path=db_path,
            predictions_path=Path(args.predictions_path),
            report_output_path=Path(args.report_output_path),
            promotion=args.promotion,
            model_name=args.model_name,
            model_run_id=args.model_run_id,
            min_edge=args.min_edge,
            min_confidence=args.min_confidence,
            min_pre_bout_fights=args.min_pre_bout_fights,
            require_pre_bout_data=args.require_pre_bout_data,
            max_stake_per_bet=args.max_stake_per_bet,
            max_event_exposure=args.max_event_exposure,
            max_day_exposure=args.max_day_exposure,
            min_liquidity=args.min_liquidity,
            require_liquidity=args.require_liquidity,
            flat_stake=args.flat_stake,
            kelly_fraction=args.kelly_fraction,
            initial_bankroll=args.initial_bankroll,
            vig_adjustment=args.vig_adjustment,
            one_bet_per_bout=not args.allow_multiple_bets_per_bout,
        )
        print(
            "betting_backtest_complete "
            f"model_name={result.model_name} "
            f"model_run_id={result.model_run_id} "
            f"bets={result.bets_count} "
            f"roi={result.roi:.6f} "
            f"drawdown={result.max_drawdown:.6f} "
            f"avg_edge={result.average_edge:.6f} "
            f"report_path={result.report_path.as_posix()}"
        )
        print(
            "betting_market_comparison "
            f"avg_model_probability={result.average_model_probability:.6f} "
            f"avg_market_probability_raw={result.average_market_probability_raw:.6f} "
            f"avg_market_probability_fair={result.average_market_probability_fair:.6f}"
        )
        return 0

    if args.command == "run-backtest-sensitivity":
        db_path = Path(args.db_path) if args.db_path else _sqlite_path_from_database_url(settings.database_url)
        result = run_betting_sensitivity_analysis(
            db_path=db_path,
            predictions_path=Path(args.predictions_path),
            report_output_path=Path(args.report_output_path),
            promotion=args.promotion,
            model_name=args.model_name,
            model_run_id=args.model_run_id,
            min_edge_values=_parse_csv_float_list(args.min_edge_values),
            min_confidence_values=_parse_csv_float_list(args.min_confidence_values),
            min_pre_bout_fights_values=_parse_csv_int_list(args.min_pre_bout_fights_values),
            max_stake_per_bet_values=_parse_csv_optional_float_list(args.max_stake_per_bet_values),
            max_event_exposure_values=_parse_csv_optional_float_list(args.max_event_exposure_values),
            max_day_exposure_values=_parse_csv_optional_float_list(args.max_day_exposure_values),
            min_liquidity_values=_parse_csv_optional_float_list(args.min_liquidity_values),
            require_pre_bout_data=args.require_pre_bout_data,
            require_liquidity=args.require_liquidity,
            flat_stake=args.flat_stake,
            kelly_fraction=args.kelly_fraction,
            initial_bankroll=args.initial_bankroll,
            vig_adjustment=args.vig_adjustment,
            one_bet_per_bout=not args.allow_multiple_bets_per_bout,
        )
        print(
            "betting_sensitivity_complete "
            f"model_name={result.model_name} "
            f"model_run_id={result.model_run_id} "
            f"scenarios={len(result.scenarios)} "
            f"report_path={result.report_path.as_posix()}"
        )
        return 0

    if args.command == "calibration-diagnostics":
        db_path = Path(args.db_path) if args.db_path else _sqlite_path_from_database_url(settings.database_url)
        result = run_calibration_diagnostics(
            db_path=db_path,
            predictions_path=Path(args.predictions_path),
            output_dir=Path(args.output_dir),
            report_output_path=Path(args.report_output_path),
            promotion=args.promotion,
            model_name=args.model_name,
            model_run_id=args.model_run_id,
            calibration_bins=args.calibration_bins,
        )
        print(
            "calibration_diagnostics_complete "
            f"model_name={result.model_name} "
            f"model_run_id={result.model_run_id} "
            f"predictions_used={result.predictions_used} "
            f"ece={result.aggregate_calibration.expected_calibration_error:.8f} "
            f"mce={result.aggregate_calibration.maximum_calibration_error:.8f} "
            f"log_loss={result.aggregate_log_loss:.8f} "
            f"brier_score={result.aggregate_brier_score:.8f} "
            f"report_path={result.report_path.as_posix()}"
        )
        print(
            "calibration_diagnostics_artifacts "
            f"calibration_table={result.calibration_table_path.as_posix()} "
            f"reliability_curve={result.reliability_curve_path.as_posix()} "
            f"probability_bucket_summary={result.probability_bucket_summary_path.as_posix()} "
            f"performance_breakdowns={result.performance_breakdown_path.as_posix()}"
        )
        return 0

    if args.command == "compare-models":
        db_path = Path(args.db_path) if args.db_path else _sqlite_path_from_database_url(settings.database_url)
        predictions_paths = (
            [Path(path) for path in args.predictions_path]
            if args.predictions_path
            else [
                Path("data/processed/predictions/baseline_predictions.csv"),
                Path("data/processed/predictions/walk_forward_predictions.csv"),
            ]
        )
        registry = discover_model_registry(predictions_paths=predictions_paths)
        result = compare_registered_models(
            db_path=db_path,
            promotion=args.promotion,
            registry=registry,
            report_output_path=Path(args.report_output_path),
            calibration_bins=args.calibration_bins,
            min_edge=args.min_edge,
            flat_stake=args.flat_stake,
            kelly_fraction=args.kelly_fraction,
            initial_bankroll=args.initial_bankroll,
            vig_adjustment=args.vig_adjustment,
            one_bet_per_bout=not args.allow_multiple_bets_per_bout,
        )
        print(
            "model_comparison_complete "
            f"models={len(result.model_summaries)} "
            f"champion={result.champion_challenger.champion.entry.model_identifier} "
            f"challenger={result.champion_challenger.challenger.entry.model_identifier} "
            f"report_path={result.report_path.as_posix()}"
        )
        print(
            "model_comparison_best_by_metric "
            f"log_loss={result.best_by_metric['log_loss']} "
            f"brier_score={result.best_by_metric['brier_score']} "
            f"calibration={result.best_by_metric['calibration']} "
            f"roi={result.best_by_metric['roi']} "
            f"drawdown={result.best_by_metric['drawdown']}"
        )
        return 0

    if args.command == "run-research-workflow":
        db_path = Path(args.db_path) if args.db_path else _sqlite_path_from_database_url(settings.database_url)
        result = run_research_workflow(
            db_path=db_path,
            raw_root=Path(args.raw_root),
            run_ingestion=args.run_ingestion,
            promotion=args.promotion,
            event_limit=args.event_limit,
            fighter_limit=args.fighter_limit,
            train_fraction=args.train_fraction,
            window_type=args.window_type,
            min_train_bouts=args.min_train_bouts,
            test_bouts=args.test_bouts,
            rolling_train_bouts=args.rolling_train_bouts,
            model_version=args.model_version,
            model_run_id=args.model_run_id,
            calibration_bins=args.calibration_bins,
            min_edge=args.min_edge,
            flat_stake=args.flat_stake,
            kelly_fraction=args.kelly_fraction,
            initial_bankroll=args.initial_bankroll,
            vig_adjustment=args.vig_adjustment,
            allow_multiple_bets_per_bout=args.allow_multiple_bets_per_bout,
            features_output_path=Path(args.features_output_path),
            ratings_output_path=Path(args.ratings_output_path),
            baseline_predictions_output_path=Path(args.baseline_predictions_output_path),
            baseline_report_output_path=Path(args.baseline_report_output_path),
            walk_forward_predictions_output_path=Path(args.walk_forward_predictions_output_path),
            walk_forward_report_output_path=Path(args.walk_forward_report_output_path),
            ablation_predictions_output_path=Path(args.ablation_predictions_output_path),
            ablation_comparison_report_output_path=Path(args.ablation_comparison_report_output_path),
            ablation_contribution_report_output_path=Path(args.ablation_contribution_report_output_path),
            calibration_output_dir=Path(args.calibration_output_dir),
            calibration_report_output_path=Path(args.calibration_report_output_path),
            model_comparison_report_output_path=Path(args.model_comparison_report_output_path),
            data_coverage_audit_report_output_path=Path(args.data_coverage_audit_report_output_path),
            market_coverage_report_output_path=Path(args.market_coverage_report_output_path),
            backtest_report_output_path=Path(args.backtest_report_output_path),
            edge_pockets_report_output_path=Path(args.edge_pockets_report_output_path),
            edge_pockets_segment_metrics_output_path=Path(args.edge_pockets_segment_metrics_output_path),
            summary_report_path=Path(args.summary_report_path),
            log_dir=Path(args.log_dir),
            workflow_log_path=Path(args.workflow_log_path),
        )
        print(
            "research_workflow_complete "
            f"status={'ok' if result.success else 'failed'} "
            f"stages={len(result.sequence.records)} "
            f"summary_report_path={result.summary_report_path.as_posix()}"
        )
        for record in result.sequence.records:
            print(
                "research_workflow_stage "
                f"name={record.name} "
                f"status={record.status} "
                f"log_path={record.log_path.as_posix()}"
            )
        if not result.success and result.sequence.failed_stage_name is not None:
            print(f"research_workflow_failed_stage name={result.sequence.failed_stage_name}")
            return 1
        return 0

    # Assumption: no side-effectful orchestration runs by default without command selection.
    parser.print_help()
    return 0


def _sqlite_path_from_database_url(database_url: str) -> Path:
    """Resolve local SQLite file path from ``sqlite:///`` URL syntax."""

    prefix = "sqlite:///"
    if database_url.startswith(prefix):
        return Path(database_url[len(prefix) :])
    return Path(database_url)


def _parse_optional_utc_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    return datetime.fromisoformat(text)


def _run_data_validation_command(*, db_path: Path) -> int:
    """Run all data validations and print a machine-readable summary."""

    import sqlite3

    with sqlite3.connect(db_path) as connection:
        report = run_data_validations(connection)

    if report.is_valid:
        print("data_validation_complete status=ok issues=0")
        return 0

    for issue in report.issues:
        print(
            "validation_issue "
            f"check={issue.check_name} table={issue.table_name} row={issue.row_ref} message={issue.message}"
        )
    print(f"data_validation_complete status=failed issues={report.issue_count}")
    return report.issue_count


def _parse_csv_float_list(value: str) -> list[float]:
    parsed: list[float] = []
    for token in value.split(","):
        text = token.strip()
        if text == "":
            continue
        parsed.append(float(text))
    if len(parsed) == 0:
        raise ValueError("Expected at least one numeric value.")
    return parsed


def _parse_csv_int_list(value: str) -> list[int]:
    parsed: list[int] = []
    for token in value.split(","):
        text = token.strip()
        if text == "":
            continue
        parsed.append(int(text))
    if len(parsed) == 0:
        raise ValueError("Expected at least one integer value.")
    return parsed


def _parse_csv_optional_float_list(value: str) -> list[float | None]:
    parsed: list[float | None] = []
    for token in value.split(","):
        text = token.strip().lower()
        if text == "":
            continue
        if text in {"none", "null"}:
            parsed.append(None)
            continue
        parsed.append(float(text))
    if len(parsed) == 0:
        raise ValueError("Expected at least one value (numeric or none).")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
