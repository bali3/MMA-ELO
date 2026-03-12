"""Reproducible, fail-fast workflow orchestration for MMA research pipeline."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from backtest import run_betting_backtest
from evaluation import (
    compare_registered_models,
    discover_model_registry,
    run_ablation_experiments,
    run_calibration_diagnostics,
    run_segmented_edge_analysis,
    run_walk_forward_evaluation,
)
from features import build_feature_report, generate_pre_fight_features
from ingestion.sources.ufc_stats import (
    run_ufc_stats_bout_stats_ingestion,
    run_ufc_stats_events_ingestion,
    run_ufc_stats_fighter_metadata_ingestion,
)
from models import run_baseline_models
from markets import run_market_coverage_report
from ratings import EloConfig, build_ratings_report, generate_elo_ratings_history
from validation import run_data_coverage_audit, run_data_validations

StageDetails = Mapping[str, Any]
StageRunner = Callable[[], Optional[StageDetails]]


@dataclass(frozen=True)
class WorkflowStage:
    """Single executable stage definition."""

    name: str
    description: str
    runner: StageRunner


@dataclass(frozen=True)
class StageExecutionRecord:
    """Observed stage execution metadata."""

    name: str
    description: str
    status: str
    started_at_utc: str
    finished_at_utc: str
    log_path: Path
    details: dict[str, str]


@dataclass(frozen=True)
class StageSequenceResult:
    """Result for a complete stage sequence run."""

    success: bool
    records: tuple[StageExecutionRecord, ...]
    failed_stage_name: str | None


@dataclass(frozen=True)
class ResearchWorkflowResult:
    """End-to-end workflow execution output."""

    success: bool
    sequence: StageSequenceResult
    summary_report_path: Path


def run_stage_sequence(
    *,
    stages: Sequence[WorkflowStage],
    log_dir: Path,
    workflow_log_path: Path,
) -> StageSequenceResult:
    """Execute stages in order with per-stage logging and fail-fast behavior."""

    log_dir.mkdir(parents=True, exist_ok=True)
    workflow_log_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[StageExecutionRecord] = []

    _append_line(workflow_log_path, f"workflow_started_at_utc={_utc_now_iso()}")

    for index, stage in enumerate(stages, start=1):
        stage_log_path = log_dir / f"{index:02d}_{stage.name}.log"
        started_at = _utc_now_iso()
        _append_line(stage_log_path, f"stage={stage.name}")
        _append_line(stage_log_path, f"description={stage.description}")
        _append_line(stage_log_path, f"started_at_utc={started_at}")

        status = "ok"
        details: dict[str, str]

        try:
            details = _normalize_details(stage.runner())
        except Exception as exc:  # noqa: BLE001 - explicit fail-fast boundary for stage runners
            status = "failed"
            details = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }

        finished_at = _utc_now_iso()
        _append_line(stage_log_path, f"finished_at_utc={finished_at}")
        _append_line(stage_log_path, f"status={status}")
        for key, value in sorted(details.items()):
            _append_line(stage_log_path, f"{key}={value}")

        record = StageExecutionRecord(
            name=stage.name,
            description=stage.description,
            status=status,
            started_at_utc=started_at,
            finished_at_utc=finished_at,
            log_path=stage_log_path,
            details=details,
        )
        records.append(record)

        _append_line(
            workflow_log_path,
            (
                f"stage={record.name} status={record.status} "
                f"started_at_utc={record.started_at_utc} finished_at_utc={record.finished_at_utc} "
                f"log_path={record.log_path.as_posix()}"
            ),
        )

        if status != "ok":
            _append_line(workflow_log_path, f"workflow_status=failed failed_stage={record.name}")
            return StageSequenceResult(
                success=False,
                records=tuple(records),
                failed_stage_name=record.name,
            )

    _append_line(workflow_log_path, "workflow_status=ok")
    return StageSequenceResult(success=True, records=tuple(records), failed_stage_name=None)


def run_research_workflow(
    *,
    db_path: Path,
    raw_root: Path,
    run_ingestion: bool,
    promotion: str,
    event_limit: int | None,
    fighter_limit: int | None,
    train_fraction: float,
    window_type: str,
    min_train_bouts: int,
    test_bouts: int,
    rolling_train_bouts: int | None,
    model_version: str,
    model_run_id: str | None,
    calibration_bins: int,
    min_edge: float,
    flat_stake: float,
    kelly_fraction: float | None,
    initial_bankroll: float,
    vig_adjustment: str,
    allow_multiple_bets_per_bout: bool,
    features_output_path: Path,
    ratings_output_path: Path,
    baseline_predictions_output_path: Path,
    baseline_report_output_path: Path,
    walk_forward_predictions_output_path: Path,
    walk_forward_report_output_path: Path,
    ablation_predictions_output_path: Path,
    ablation_comparison_report_output_path: Path,
    ablation_contribution_report_output_path: Path,
    calibration_output_dir: Path,
    calibration_report_output_path: Path,
    model_comparison_report_output_path: Path,
    data_coverage_audit_report_output_path: Path,
    market_coverage_report_output_path: Path,
    backtest_report_output_path: Path,
    edge_pockets_report_output_path: Path,
    edge_pockets_segment_metrics_output_path: Path,
    summary_report_path: Path,
    log_dir: Path,
    workflow_log_path: Path,
) -> ResearchWorkflowResult:
    """Run the full research workflow and write a final stage-by-stage summary report."""

    state: dict[str, str] = {}

    def _ingestion_snapshot_stage() -> StageDetails:
        return _collect_ingestion_snapshot(db_path)

    def _ingest_events_stage() -> StageDetails:
        result = run_ufc_stats_events_ingestion(
            db_path=db_path,
            raw_root=raw_root,
            event_limit=event_limit,
        )
        return {
            "inserted": result.inserted_count,
            "updated": result.updated_count,
            "skipped": result.skipped_count,
        }

    def _ingest_fighters_stage() -> StageDetails:
        result = run_ufc_stats_fighter_metadata_ingestion(
            db_path=db_path,
            raw_root=raw_root,
            fighter_limit=fighter_limit,
        )
        return {
            "inserted": result.inserted_count,
            "updated": result.updated_count,
            "skipped": result.skipped_count,
        }

    def _ingest_bout_stats_stage() -> StageDetails:
        result = run_ufc_stats_bout_stats_ingestion(
            db_path=db_path,
            raw_root=raw_root,
            event_limit=event_limit,
        )
        return {
            "inserted": result.inserted_count,
            "updated": result.updated_count,
            "skipped": result.skipped_count,
        }

    def _validate_data_stage() -> StageDetails:
        with sqlite3.connect(db_path) as connection:
            report = run_data_validations(connection)
        if not report.is_valid:
            raise ValueError(f"data_validations_failed issue_count={report.issue_count}")
        return {"issues": report.issue_count, "status": "ok"}

    def _data_coverage_stage() -> StageDetails:
        return _collect_data_coverage_snapshot(db_path=db_path, promotion=promotion)

    def _generate_features_stage() -> StageDetails:
        row_count, persisted_path = generate_pre_fight_features(
            db_path=db_path,
            output_path=features_output_path,
            promotion=promotion,
        )
        return {
            "rows": row_count,
            "features_path": persisted_path.as_posix(),
        }

    def _feature_coverage_stage() -> StageDetails:
        report = build_feature_report(features_path=features_output_path)
        coverage_values = tuple(report.coverage_rates.values())
        avg_coverage = 0.0 if len(coverage_values) == 0 else sum(coverage_values) / float(len(coverage_values))
        min_coverage = 0.0 if len(coverage_values) == 0 else min(coverage_values)
        return {
            "rows": report.row_count,
            "bouts": report.bout_count,
            "earliest_bout_utc": report.earliest_bout_date_utc or "n/a",
            "latest_bout_utc": report.latest_bout_date_utc or "n/a",
            "avg_key_feature_coverage": f"{avg_coverage:.8f}",
            "min_key_feature_coverage": f"{min_coverage:.8f}",
        }

    def _data_coverage_audit_stage() -> StageDetails:
        audit = run_data_coverage_audit(
            db_path=db_path,
            promotion=promotion,
            features_path=features_output_path,
            report_output_path=data_coverage_audit_report_output_path,
        )
        return {
            "total_events": audit.total_events,
            "total_completed_bouts": audit.total_completed_bouts,
            "total_fighters": audit.total_fighters,
            "total_fighter_bout_stats_rows": audit.total_fighter_bout_stats_rows,
            "completed_bouts_with_two_stats_rows": audit.completed_bouts_with_two_stats_rows,
            "completed_bouts_with_one_stats_row": audit.completed_bouts_with_one_stats_row,
            "completed_bouts_with_zero_stats_rows": audit.completed_bouts_with_zero_stats_rows,
            "report_path": audit.report_path.as_posix(),
        }

    def _market_coverage_stage() -> StageDetails:
        report = run_market_coverage_report(
            db_path=db_path,
            promotion=promotion,
            report_output_path=market_coverage_report_output_path,
        )
        return {
            "market_rows": report.market_rows,
            "completed_fights": report.completed_fights,
            "matched_fights": report.matched_fights,
            "unmatched_fights": report.unmatched_fights,
            "match_rate": f"{report.match_rate:.8f}",
            "opening_odds_covered_fights": report.opening_odds_covered_fights,
            "closing_odds_covered_fights": (
                "n/a" if report.closing_odds_covered_fights is None else report.closing_odds_covered_fights
            ),
            "report_path": report.report_path.as_posix(),
        }

    def _build_ratings_stage() -> StageDetails:
        result = generate_elo_ratings_history(
            db_path=db_path,
            output_path=ratings_output_path,
            config=EloConfig(promotion=promotion),
        )
        return {
            "bouts": result.bout_count,
            "rows": result.row_count,
            "ratings_path": result.output_path.as_posix(),
        }

    def _ratings_coverage_stage() -> StageDetails:
        report = build_ratings_report(
            ratings_history_path=ratings_output_path,
            db_path=db_path,
            top_n=10,
        )
        return {
            "rated_fighters": report.rated_fighters,
            "total_fighters": report.total_fighters,
            "coverage_rate": f"{report.coverage_rate:.8f}",
            "earliest_rating_utc": report.earliest_rating_datetime_utc or "n/a",
            "latest_rating_utc": report.latest_rating_datetime_utc or "n/a",
            "top_weight_class_count": len(report.weight_class_coverage),
        }

    def _train_baselines_stage() -> StageDetails:
        result = run_baseline_models(
            db_path=db_path,
            promotion=promotion,
            train_fraction=train_fraction,
            model_version=model_version,
            model_run_id=model_run_id,
            features_output_path=features_output_path,
            ratings_output_path=ratings_output_path,
            predictions_output_path=baseline_predictions_output_path,
            report_output_path=baseline_report_output_path,
        )
        state["baseline_model_run_id"] = result.model_run_id
        return {
            "model_run_id": result.model_run_id,
            "predictions": result.prediction_count,
            "scored_bouts": result.scored_bout_count,
            "predictions_path": result.predictions_path.as_posix(),
            "report_path": result.report_path.as_posix(),
        }

    def _walk_forward_stage() -> StageDetails:
        window_settings = _resolve_walk_forward_window_settings(
            db_path=db_path,
            promotion=promotion,
            window_type=window_type,
            min_train_bouts=min_train_bouts,
            test_bouts=test_bouts,
            rolling_train_bouts=rolling_train_bouts,
        )
        result = run_walk_forward_evaluation(
            db_path=db_path,
            promotion=promotion,
            window_type=window_type,
            min_train_bouts=int(window_settings["effective_min_train_bouts"]),
            test_bouts=int(window_settings["effective_test_bouts"]),
            rolling_train_bouts=(
                None
                if window_settings["effective_rolling_train_bouts"] == "n/a"
                else int(window_settings["effective_rolling_train_bouts"])
            ),
            model_version=model_version,
            model_run_id=model_run_id,
            calibration_bins=calibration_bins,
            ratings_output_path=ratings_output_path,
            predictions_output_path=walk_forward_predictions_output_path,
            report_output_path=walk_forward_report_output_path,
        )
        state["walk_forward_model_run_id"] = result.model_run_id
        return {
            "model_run_id": result.model_run_id,
            "folds": result.fold_count,
            "predictions": result.predictions_count,
            "log_loss": f"{result.aggregate_log_loss:.8f}",
            "brier_score": f"{result.aggregate_brier_score:.8f}",
            "hit_rate": f"{result.aggregate_hit_rate:.8f}",
            "completed_bouts_available": window_settings["completed_bouts_available"],
            "configured_min_train_bouts": window_settings["configured_min_train_bouts"],
            "configured_test_bouts": window_settings["configured_test_bouts"],
            "effective_min_train_bouts": window_settings["effective_min_train_bouts"],
            "effective_test_bouts": window_settings["effective_test_bouts"],
            "effective_rolling_train_bouts": window_settings["effective_rolling_train_bouts"],
            "window_auto_adjusted": window_settings["window_auto_adjusted"],
            "predictions_path": result.predictions_path.as_posix(),
            "report_path": result.report_path.as_posix(),
        }

    def _calibration_stage() -> StageDetails:
        result = run_calibration_diagnostics(
            db_path=db_path,
            predictions_path=walk_forward_predictions_output_path,
            output_dir=calibration_output_dir,
            report_output_path=calibration_report_output_path,
            promotion=promotion,
            model_name="elo_logistic",
            model_run_id=state.get("walk_forward_model_run_id"),
            calibration_bins=calibration_bins,
        )
        return {
            "model_run_id": result.model_run_id,
            "predictions_loaded": result.predictions_loaded,
            "predictions_used": result.predictions_used,
            "dropped_chronology_invalid": result.dropped_chronology_invalid,
            "dropped_missing_probability": result.dropped_missing_probability,
            "dropped_invalid_label": result.dropped_invalid_label,
            "ece": f"{result.aggregate_calibration.expected_calibration_error:.8f}",
            "mce": f"{result.aggregate_calibration.maximum_calibration_error:.8f}",
            "log_loss": f"{result.aggregate_log_loss:.8f}",
            "brier_score": f"{result.aggregate_brier_score:.8f}",
            "report_path": result.report_path.as_posix(),
        }

    def _ablation_stage() -> StageDetails:
        window_settings = _resolve_walk_forward_window_settings(
            db_path=db_path,
            promotion=promotion,
            window_type=window_type,
            min_train_bouts=min_train_bouts,
            test_bouts=test_bouts,
            rolling_train_bouts=rolling_train_bouts,
        )
        result = run_ablation_experiments(
            db_path=db_path,
            promotion=promotion,
            window_type=window_type,
            min_train_bouts=int(window_settings["effective_min_train_bouts"]),
            test_bouts=int(window_settings["effective_test_bouts"]),
            rolling_train_bouts=(
                None
                if window_settings["effective_rolling_train_bouts"] == "n/a"
                else int(window_settings["effective_rolling_train_bouts"])
            ),
            model_version=model_version,
            model_run_id=model_run_id,
            calibration_bins=calibration_bins,
            features_output_path=features_output_path,
            ratings_output_path=ratings_output_path,
            predictions_output_path=ablation_predictions_output_path,
            comparison_report_output_path=ablation_comparison_report_output_path,
            contribution_report_output_path=ablation_contribution_report_output_path,
        )
        best = min(result.scenario_metrics, key=lambda row: row.log_loss)
        return {
            "model_run_id": result.model_run_id,
            "folds": result.fold_count,
            "scenarios": len(result.scenario_metrics),
            "best_scenario": best.config.scenario_id,
            "best_log_loss": f"{best.log_loss:.8f}",
            "best_calibration_ece": f"{best.calibration_ece:.8f}",
            "completed_bouts_available": window_settings["completed_bouts_available"],
            "effective_min_train_bouts": window_settings["effective_min_train_bouts"],
            "effective_test_bouts": window_settings["effective_test_bouts"],
            "window_auto_adjusted": window_settings["window_auto_adjusted"],
            "predictions_path": result.predictions_path.as_posix(),
            "comparison_report_path": result.comparison_report_path.as_posix(),
            "contribution_report_path": result.contribution_report_path.as_posix(),
        }

    def _model_comparison_stage() -> StageDetails:
        registry = discover_model_registry(
            predictions_paths=(
                baseline_predictions_output_path,
                walk_forward_predictions_output_path,
                ablation_predictions_output_path,
            )
        )
        result = compare_registered_models(
            db_path=db_path,
            promotion=promotion,
            registry=registry,
            report_output_path=model_comparison_report_output_path,
            calibration_bins=calibration_bins,
            min_edge=min_edge,
            flat_stake=flat_stake,
            kelly_fraction=kelly_fraction,
            initial_bankroll=initial_bankroll,
            vig_adjustment=vig_adjustment,
            one_bet_per_bout=not allow_multiple_bets_per_bout,
        )
        return {
            "models_compared": len(result.model_summaries),
            "champion": result.champion_challenger.champion.entry.model_identifier,
            "challenger": result.champion_challenger.challenger.entry.model_identifier,
            "best_log_loss_model": result.best_by_metric["log_loss"],
            "best_roi_model": result.best_by_metric["roi"],
            "report_path": result.report_path.as_posix(),
        }

    def _backtest_stage() -> StageDetails:
        resolved_run_id = state.get("baseline_model_run_id")
        result = run_betting_backtest(
            db_path=db_path,
            predictions_path=baseline_predictions_output_path,
            report_output_path=backtest_report_output_path,
            promotion=promotion,
            model_name="elo_logistic",
            model_run_id=resolved_run_id,
            min_edge=min_edge,
            flat_stake=flat_stake,
            kelly_fraction=kelly_fraction,
            initial_bankroll=initial_bankroll,
            vig_adjustment=vig_adjustment,
            one_bet_per_bout=not allow_multiple_bets_per_bout,
        )
        return {
            "model_run_id": result.model_run_id,
            "bets": result.bets_count,
            "roi": f"{result.roi:.8f}",
            "max_drawdown": f"{result.max_drawdown:.8f}",
            "report_path": result.report_path.as_posix(),
        }

    def _edge_pockets_stage() -> StageDetails:
        result = run_segmented_edge_analysis(
            db_path=db_path,
            features_path=features_output_path,
            predictions_path=baseline_predictions_output_path,
            report_output_path=edge_pockets_report_output_path,
            segment_metrics_output_path=edge_pockets_segment_metrics_output_path,
            promotion=promotion,
            target_model_name="elo_logistic",
            target_model_run_id=state.get("baseline_model_run_id"),
            baseline_model_name="market_implied_probability",
            baseline_model_run_id=state.get("baseline_model_run_id"),
            min_edge=min_edge,
            flat_stake=flat_stake,
            kelly_fraction=kelly_fraction,
            initial_bankroll=initial_bankroll,
            vig_adjustment=vig_adjustment,
            one_bet_per_bout=not allow_multiple_bets_per_bout,
        )
        return {
            "target_model_name": result.target_model_name,
            "target_model_run_id": result.target_model_run_id,
            "baseline_model_name": result.baseline_model_name,
            "baseline_model_run_id": result.baseline_model_run_id,
            "segments_evaluated": result.segments_evaluated,
            "credible_segments": result.credible_segments,
            "best_log_loss_segment": result.best_log_loss_segment,
            "best_brier_segment": result.best_brier_segment,
            "best_roi_segment": result.best_roi_segment,
            "worst_roi_segment": result.worst_roi_segment,
            "worst_drawdown_segment": result.worst_drawdown_segment,
            "report_path": result.report_path.as_posix(),
            "segment_metrics_path": result.segment_metrics_path.as_posix(),
        }

    stages: list[WorkflowStage] = []
    if run_ingestion:
        stages.extend(
            (
                WorkflowStage(
                    name="ingestion_events",
                    description="Ingest UFC completed events and bout outcomes.",
                    runner=_ingest_events_stage,
                ),
                WorkflowStage(
                    name="ingestion_fighters",
                    description="Ingest UFC fighter metadata.",
                    runner=_ingest_fighters_stage,
                ),
                WorkflowStage(
                    name="ingestion_bout_stats",
                    description="Ingest UFC per-bout fighter stats.",
                    runner=_ingest_bout_stats_stage,
                ),
            )
        )
    else:
        stages.append(
            WorkflowStage(
                name="ingestion_snapshot",
                description="Summarize existing ingested data snapshot (offline mode).",
                runner=_ingestion_snapshot_stage,
            )
        )

    stages.extend(
        (
            WorkflowStage(
                name="validate_data",
                description="Run chronology and data-integrity validations.",
                runner=_validate_data_stage,
            ),
            WorkflowStage(
                name="data_coverage",
                description="Build UFC data coverage snapshot.",
                runner=_data_coverage_stage,
            ),
            WorkflowStage(
                name="generate_features",
                description="Generate pre-fight sequential features.",
                runner=_generate_features_stage,
            ),
            WorkflowStage(
                name="feature_coverage",
                description="Summarize key feature coverage on generated dataset.",
                runner=_feature_coverage_stage,
            ),
            WorkflowStage(
                name="data_coverage_audit",
                description="Generate data coverage audit report and join-failure diagnostics.",
                runner=_data_coverage_audit_stage,
            ),
            WorkflowStage(
                name="market_coverage",
                description="Generate market matching and odds coverage report.",
                runner=_market_coverage_stage,
            ),
            WorkflowStage(
                name="build_ratings_history",
                description="Generate Elo rating history.",
                runner=_build_ratings_stage,
            ),
            WorkflowStage(
                name="rating_coverage",
                description="Summarize rating coverage and temporal span.",
                runner=_ratings_coverage_stage,
            ),
            WorkflowStage(
                name="train_baselines",
                description="Train chronology-safe baseline models.",
                runner=_train_baselines_stage,
            ),
            WorkflowStage(
                name="walk_forward_eval",
                description="Run chronological walk-forward evaluation.",
                runner=_walk_forward_stage,
            ),
            WorkflowStage(
                name="calibration_diagnostics",
                description="Generate calibration diagnostics from walk-forward predictions.",
                runner=_calibration_stage,
            ),
            WorkflowStage(
                name="run_ablation_experiments",
                description="Run walk-forward ablation experiments.",
                runner=_ablation_stage,
            ),
            WorkflowStage(
                name="compare_models",
                description="Run multi-model comparison and champion/challenger selection.",
                runner=_model_comparison_stage,
            ),
            WorkflowStage(
                name="run_backtest",
                description="Run betting simulation on baseline predictions.",
                runner=_backtest_stage,
            ),
            WorkflowStage(
                name="edge_pockets_analysis",
                description="Analyze segmented model-vs-market performance and betting edge.",
                runner=_edge_pockets_stage,
            ),
        )
    )

    sequence = run_stage_sequence(stages=stages, log_dir=log_dir, workflow_log_path=workflow_log_path)
    _write_final_summary(
        summary_report_path=summary_report_path,
        sequence=sequence,
        run_ingestion=run_ingestion,
    )
    return ResearchWorkflowResult(success=sequence.success, sequence=sequence, summary_report_path=summary_report_path)


def _write_final_summary(
    *,
    summary_report_path: Path,
    sequence: StageSequenceResult,
    run_ingestion: bool,
) -> None:
    summary_report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_text = build_consolidated_research_summary_text(
        sequence=sequence,
        run_ingestion=run_ingestion,
    )
    summary_report_path.write_text(summary_text, encoding="utf-8")


def build_consolidated_research_summary_text(
    *,
    sequence: StageSequenceResult,
    run_ingestion: bool,
) -> str:
    lines = [
        "MMA Research Workflow Final Summary",
        f"generated_at_utc: {_utc_now_iso()}",
        f"status: {'ok' if sequence.success else 'failed'}",
        f"ingestion_mode: {'live' if run_ingestion else 'offline_snapshot'}",
        "",
        "Consolidated research summary:",
    ]

    lines.extend(_summary_section("data_coverage", _summary_pairs(sequence, "data_coverage")))
    lines.extend(_summary_section("data_coverage_audit", _summary_pairs(sequence, "data_coverage_audit")))
    lines.extend(_summary_section("validation_results", _summary_pairs(sequence, "validate_data")))
    lines.extend(_summary_section("feature_coverage", _summary_pairs(sequence, "feature_coverage")))
    lines.extend(_summary_section("market_coverage", _summary_pairs(sequence, "market_coverage")))
    lines.extend(_summary_section("rating_coverage", _summary_pairs(sequence, "rating_coverage")))
    lines.extend(_summary_section("model_comparison", _summary_pairs(sequence, "compare_models")))
    lines.extend(_summary_section("walk_forward_evaluation_metrics", _summary_pairs(sequence, "walk_forward_eval")))
    lines.extend(_summary_section("calibration_diagnostics", _summary_pairs(sequence, "calibration_diagnostics")))
    lines.extend(_summary_section("betting_simulation_results", _summary_pairs(sequence, "run_backtest")))
    lines.extend(_summary_section("edge_pockets", _summary_pairs(sequence, "edge_pockets_analysis")))
    lines.extend(_summary_section("ablation_results", _summary_pairs(sequence, "run_ablation_experiments")))
    lines.extend(_build_evaluation_readiness_flags(sequence))

    lines.extend(
        (
            "",
            "Stage outcomes:",
        )
    )
    for record in sequence.records:
        lines.append(f"- stage: {record.name}")
        lines.append(f"  description: {record.description}")
        lines.append(f"  status: {record.status}")
        lines.append(f"  started_at_utc: {record.started_at_utc}")
        lines.append(f"  finished_at_utc: {record.finished_at_utc}")
        lines.append(f"  log_path: {record.log_path.as_posix()}")
        for key, value in sorted(record.details.items()):
            lines.append(f"  {key}: {value}")

    lines.extend(
        (
            "",
            "Required summary coverage:",
            f"- ingestion: {_section_status(sequence.records, 'ingestion')}",
            f"- data_coverage: {_section_status(sequence.records, 'data_coverage')}",
            f"- validation: {_section_status(sequence.records, 'validate_data')}",
            f"- feature_coverage: {_section_status(sequence.records, 'feature_coverage')}",
            f"- market_coverage: {_section_status(sequence.records, 'market_coverage')}",
            f"- data_coverage_audit: {_section_status(sequence.records, 'data_coverage_audit')}",
            f"- rating_coverage: {_section_status(sequence.records, 'rating_coverage')}",
            f"- model_comparison: {_section_status(sequence.records, 'compare_models')}",
            f"- walk_forward_evaluation: {_section_status(sequence.records, 'walk_forward_eval')}",
            f"- calibration_diagnostics: {_section_status(sequence.records, 'calibration_diagnostics')}",
            f"- betting_simulation: {_section_status(sequence.records, 'run_backtest')}",
            f"- edge_pockets: {_section_status(sequence.records, 'edge_pockets_analysis')}",
            f"- ablation_results: {_section_status(sequence.records, 'run_ablation_experiments')}",
        )
    )

    if not sequence.success and sequence.failed_stage_name is not None:
        lines.append(f"failed_stage: {sequence.failed_stage_name}")

    return "\n".join(lines) + "\n"


def _summary_pairs(sequence: StageSequenceResult, stage_name: str) -> dict[str, str]:
    record = next((item for item in sequence.records if item.name == stage_name), None)
    if record is None:
        return {"status": "not_run"}
    rows = dict(record.details)
    rows["status"] = record.status
    return rows


def _summary_section(name: str, rows: Mapping[str, str]) -> list[str]:
    lines = ["", f"{name}:"]
    for key, value in sorted(rows.items()):
        lines.append(f"- {key}: {value}")
    return lines


def _build_evaluation_readiness_flags(sequence: StageSequenceResult) -> list[str]:
    data_rows = _summary_pairs(sequence, "data_coverage")
    audit_rows = _summary_pairs(sequence, "data_coverage_audit")
    market_rows_data = _summary_pairs(sequence, "market_coverage")
    walk_rows = _summary_pairs(sequence, "walk_forward_eval")
    backtest_rows = _summary_pairs(sequence, "run_backtest")

    completed_bouts = _as_int(data_rows.get("ufc_completed_bouts_count"))
    bouts_with_two_stats_rows = _as_int(audit_rows.get("completed_bouts_with_two_stats_rows"))
    matched_completed_fights = _as_int(market_rows_data.get("matched_fights"))
    opening_covered = _as_int(market_rows_data.get("opening_odds_covered_fights"))
    closing_covered = _as_int(market_rows_data.get("closing_odds_covered_fights"))
    completed_fights_market = _as_int(market_rows_data.get("completed_fights"))
    fold_count = _as_int(walk_rows.get("folds"))
    bets_count = _as_int(backtest_rows.get("bets"))

    lines = ["", "evaluation_readiness_flags:"]
    min_completed_bouts = 200
    min_matched_completed_fights = 200
    min_folds = 3

    if completed_bouts is None:
        lines.append("- sample_size_flag: unknown_completed_bout_count")
    elif completed_bouts < min_completed_bouts:
        lines.append(
            f"- sample_size_flag: insufficient (completed_bouts={completed_bouts} recommended_min={min_completed_bouts})"
        )
    else:
        lines.append(f"- sample_size_flag: ok (completed_bouts={completed_bouts})")

    if completed_bouts is None or bouts_with_two_stats_rows is None:
        lines.append("- bout_stats_coverage_flag: unknown")
    elif completed_bouts == 0:
        lines.append("- bout_stats_coverage_flag: no_completed_bouts")
    else:
        coverage = bouts_with_two_stats_rows / completed_bouts
        if coverage < 0.90:
            lines.append(
                (
                    "- bout_stats_coverage_flag: insufficient "
                    f"(two_stats_rows={bouts_with_two_stats_rows} completed_bouts={completed_bouts} "
                    f"coverage={coverage:.6f} recommended_min=0.900000)"
                )
            )
        else:
            lines.append(
                (
                    "- bout_stats_coverage_flag: ok "
                    f"(two_stats_rows={bouts_with_two_stats_rows} completed_bouts={completed_bouts} "
                    f"coverage={coverage:.6f})"
                )
            )

    if fold_count is None:
        lines.append("- fold_stability_flag: unknown_fold_count")
    elif fold_count < min_folds:
        lines.append(f"- fold_stability_flag: insufficient (folds={fold_count} recommended_min={min_folds})")
    else:
        lines.append(f"- fold_stability_flag: ok (folds={fold_count})")

    if matched_completed_fights is None:
        lines.append("- market_coverage_flag: unknown_matched_completed_fights")
    elif matched_completed_fights < min_matched_completed_fights:
        lines.append(
            (
                "- market_coverage_flag: insufficient "
                f"(matched_completed_fights={matched_completed_fights} "
                f"recommended_min={min_matched_completed_fights})"
            )
        )
    else:
        lines.append(f"- market_coverage_flag: ok (matched_completed_fights={matched_completed_fights})")

    if bets_count is None:
        lines.append("- betting_activity_flag: unknown_bets_count")
    elif bets_count == 0:
        lines.append("- betting_activity_flag: no_bets_placed")
    else:
        lines.append(f"- betting_activity_flag: ok (bets={bets_count})")

    if opening_covered is None or completed_fights_market is None:
        lines.append("- opening_odds_coverage_flag: unknown")
    elif completed_fights_market == 0:
        lines.append("- opening_odds_coverage_flag: no_completed_fights")
    else:
        opening_rate = opening_covered / completed_fights_market
        if opening_rate < 0.70:
            lines.append(
                (
                    "- opening_odds_coverage_flag: insufficient "
                    f"(covered_fights={opening_covered} total_fights={completed_fights_market} "
                    f"coverage={opening_rate:.6f} recommended_min=0.700000)"
                )
            )
        else:
            lines.append(
                (
                    "- opening_odds_coverage_flag: ok "
                    f"(covered_fights={opening_covered} total_fights={completed_fights_market} "
                    f"coverage={opening_rate:.6f})"
                )
            )

    if market_rows_data.get("closing_odds_covered_fights") == "n/a":
        lines.append("- closing_odds_coverage_flag: not_available")
    elif closing_covered is None or completed_fights_market is None:
        lines.append("- closing_odds_coverage_flag: unknown")
    elif completed_fights_market == 0:
        lines.append("- closing_odds_coverage_flag: no_completed_fights")
    else:
        closing_rate = closing_covered / completed_fights_market
        if closing_rate < 0.50:
            lines.append(
                (
                    "- closing_odds_coverage_flag: insufficient "
                    f"(covered_fights={closing_covered} total_fights={completed_fights_market} "
                    f"coverage={closing_rate:.6f} recommended_min=0.500000)"
                )
            )
        else:
            lines.append(
                (
                    "- closing_odds_coverage_flag: ok "
                    f"(covered_fights={closing_covered} total_fights={completed_fights_market} "
                    f"coverage={closing_rate:.6f})"
                )
            )

    return lines


def _section_status(records: Sequence[StageExecutionRecord], section_hint: str) -> str:
    matching = [record.status for record in records if section_hint in record.name]
    if len(matching) == 0:
        return "not_run"
    if any(status == "failed" for status in matching):
        return "failed"
    if all(status == "ok" for status in matching):
        return "ok"
    return "partial"


def _collect_ingestion_snapshot(db_path: Path) -> dict[str, str]:
    counts: dict[str, str] = {}
    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()
        for table in ("events", "fighters", "bouts", "fighter_bout_stats", "markets"):
            row = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            counts[f"{table}_count"] = str(int(row[0] if row is not None else 0))
    return counts


def _collect_data_coverage_snapshot(*, db_path: Path, promotion: str) -> dict[str, str]:
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        cursor = connection.cursor()
        counts = {
            "events_count": _single_count(cursor, "SELECT COUNT(*) FROM events"),
            "fighters_count": _single_count(cursor, "SELECT COUNT(*) FROM fighters"),
            "bouts_count": _single_count(cursor, "SELECT COUNT(*) FROM bouts"),
            "fighter_bout_stats_count": _single_count(cursor, "SELECT COUNT(*) FROM fighter_bout_stats"),
            "markets_count": _single_count(cursor, "SELECT COUNT(*) FROM markets"),
            "model_predictions_count": _single_count(cursor, "SELECT COUNT(*) FROM model_predictions"),
            "ufc_events_count": _single_count(
                cursor,
                "SELECT COUNT(*) FROM events WHERE UPPER(promotion) = UPPER(?)",
                promotion,
            ),
            "ufc_bouts_count": _single_count(
                cursor,
                """
                SELECT COUNT(*)
                FROM bouts AS b
                JOIN events AS e ON e.event_id = b.event_id
                WHERE UPPER(e.promotion) = UPPER(?)
                """,
                promotion,
            ),
            "ufc_completed_bouts_count": _single_count(
                cursor,
                """
                SELECT COUNT(*)
                FROM bouts AS b
                JOIN events AS e ON e.event_id = b.event_id
                WHERE UPPER(e.promotion) = UPPER(?)
                  AND b.winner_fighter_id IS NOT NULL
                """,
                promotion,
            ),
            "ufc_market_rows_count": _single_count(
                cursor,
                """
                SELECT COUNT(*)
                FROM markets AS m
                JOIN bouts AS b ON b.bout_id = m.bout_id
                JOIN events AS e ON e.event_id = b.event_id
                WHERE UPPER(e.promotion) = UPPER(?)
                """,
                promotion,
            ),
        }

        date_row = cursor.execute(
            """
            SELECT
                MIN(COALESCE(b.bout_start_time_utc, e.event_date_utc)) AS first_bout_utc,
                MAX(COALESCE(b.bout_start_time_utc, e.event_date_utc)) AS last_bout_utc
            FROM bouts AS b
            JOIN events AS e ON e.event_id = b.event_id
            WHERE UPPER(e.promotion) = UPPER(?)
            """,
            (promotion,),
        ).fetchone()

    return {
        **counts,
        "promotion": promotion,
        "first_ufc_bout_utc": str(date_row["first_bout_utc"]) if date_row and date_row["first_bout_utc"] else "n/a",
        "last_ufc_bout_utc": str(date_row["last_bout_utc"]) if date_row and date_row["last_bout_utc"] else "n/a",
    }


def _resolve_walk_forward_window_settings(
    *,
    db_path: Path,
    promotion: str,
    window_type: str,
    min_train_bouts: int,
    test_bouts: int,
    rolling_train_bouts: int | None,
) -> dict[str, str]:
    completed_bouts_available = int(_count_completed_bouts(db_path=db_path, promotion=promotion))
    if completed_bouts_available < 4:
        raise ValueError("Need at least 4 completed bouts for walk-forward evaluation.")

    effective_min_train = min_train_bouts
    if effective_min_train >= completed_bouts_available:
        effective_min_train = max(1, completed_bouts_available // 2)

    if effective_min_train >= completed_bouts_available:
        effective_min_train = completed_bouts_available - 1
    if effective_min_train < 1:
        raise ValueError("Unable to derive a valid min_train_bouts setting.")

    max_test_bouts = completed_bouts_available - effective_min_train
    effective_test = min(test_bouts, max_test_bouts)
    if effective_test < 1:
        raise ValueError("Unable to derive a valid test_bouts setting.")

    effective_rolling: int | None = rolling_train_bouts
    if window_type == "rolling":
        if effective_rolling is None or effective_rolling < 1:
            effective_rolling = effective_min_train
        effective_rolling = min(effective_rolling, effective_min_train)

    auto_adjusted = (
        (effective_min_train != min_train_bouts)
        or (effective_test != test_bouts)
        or (window_type == "rolling" and effective_rolling != rolling_train_bouts)
    )

    return {
        "completed_bouts_available": str(completed_bouts_available),
        "configured_min_train_bouts": str(min_train_bouts),
        "configured_test_bouts": str(test_bouts),
        "effective_min_train_bouts": str(effective_min_train),
        "effective_test_bouts": str(effective_test),
        "effective_rolling_train_bouts": "n/a" if effective_rolling is None else str(effective_rolling),
        "window_auto_adjusted": "true" if auto_adjusted else "false",
    }


def _count_completed_bouts(*, db_path: Path, promotion: str) -> int:
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT COUNT(*)
            FROM bouts AS b
            JOIN events AS e ON e.event_id = b.event_id
            WHERE UPPER(e.promotion) = UPPER(?)
              AND b.winner_fighter_id IS NOT NULL
            """,
            (promotion,),
        ).fetchone()
    return int(row[0] if row is not None else 0)


def _single_count(cursor: sqlite3.Cursor, query: str, *params: object) -> str:
    row = cursor.execute(query, params).fetchone()
    return str(int(row[0] if row is not None else 0))


def _as_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _normalize_details(details: StageDetails | None) -> dict[str, str]:
    if details is None:
        return {}

    normalized: dict[str, str] = {}
    for key, value in details.items():
        normalized[str(key)] = _to_text(value)
    return normalized


def _to_text(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.8f}"
    return str(value)


def _append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.write("\n")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
