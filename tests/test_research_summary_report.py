"""Tests for consolidated research summary/report generation."""

from __future__ import annotations

from pathlib import Path

from orchestration.workflow import (
    StageExecutionRecord,
    StageSequenceResult,
    build_consolidated_research_summary_text,
)


def _record(
    *,
    name: str,
    status: str = "ok",
    details: dict[str, str] | None = None,
) -> StageExecutionRecord:
    return StageExecutionRecord(
        name=name,
        description=f"stage {name}",
        status=status,
        started_at_utc="2024-01-01T00:00:00Z",
        finished_at_utc="2024-01-01T00:01:00Z",
        log_path=Path(f"/tmp/{name}.log"),
        details=details or {},
    )


def test_consolidated_summary_contains_all_required_sections() -> None:
    sequence = StageSequenceResult(
        success=True,
        failed_stage_name=None,
        records=(
            _record(
                name="data_coverage",
                details={
                    "ufc_bouts_count": "100",
                    "ufc_completed_bouts_count": "100",
                    "first_ufc_bout_utc": "2020-01-01T00:00:00Z",
                },
            ),
            _record(
                name="data_coverage_audit",
                details={
                    "report_path": "/tmp/audit.txt",
                    "total_completed_bouts": "100",
                    "completed_bouts_with_two_stats_rows": "95",
                },
            ),
            _record(name="validate_data", details={"issues": "0"}),
            _record(name="feature_coverage", details={"avg_key_feature_coverage": "0.90000000"}),
            _record(
                name="market_coverage",
                details={
                    "completed_fights": "100",
                    "opening_odds_covered_fights": "85",
                    "closing_odds_covered_fights": "55",
                    "matched_fights": "90",
                    "match_rate": "0.90000000",
                },
            ),
            _record(name="rating_coverage", details={"coverage_rate": "0.85000000"}),
            _record(name="walk_forward_eval", details={"log_loss": "0.64000000", "folds": "3"}),
            _record(name="calibration_diagnostics", details={"ece": "0.03000000"}),
            _record(name="run_backtest", details={"roi": "0.12000000", "bets": "42"}),
            _record(name="edge_pockets_analysis", details={"best_roi_segment": "weight_class=Lightweight"}),
            _record(name="run_ablation_experiments", details={"best_scenario": "features_elo_market_with_interactions"}),
            _record(name="compare_models", details={"champion": "elo_logistic:run_001"}),
        ),
    )

    text = build_consolidated_research_summary_text(sequence=sequence, run_ingestion=False)

    assert "Consolidated research summary:" in text
    assert "data_coverage:" in text
    assert "data_coverage_audit:" in text
    assert "validation_results:" in text
    assert "feature_coverage:" in text
    assert "market_coverage:" in text
    assert "- match_rate: 0.90000000" in text
    assert "rating_coverage:" in text
    assert "model_comparison:" in text
    assert "walk_forward_evaluation_metrics:" in text
    assert "calibration_diagnostics:" in text
    assert "betting_simulation_results:" in text
    assert "edge_pockets:" in text
    assert "ablation_results:" in text
    assert "- champion: elo_logistic:run_001" in text
    assert "- best_roi_segment: weight_class=Lightweight" in text
    assert "- best_scenario: features_elo_market_with_interactions" in text
    assert "- roi: 0.12000000" in text
    assert "evaluation_readiness_flags:" in text
    assert "bout_stats_coverage_flag: ok" in text
    assert "market_coverage_flag: insufficient (matched_completed_fights=90 recommended_min=200)" in text
    assert "opening_odds_coverage_flag: ok" in text
    assert "closing_odds_coverage_flag: ok" in text


def test_consolidated_summary_marks_missing_sections_not_run_and_failed_stage() -> None:
    sequence = StageSequenceResult(
        success=False,
        failed_stage_name="walk_forward_eval",
        records=(
            _record(name="validate_data", details={"issues": "0"}),
            _record(name="walk_forward_eval", status="failed", details={"error_message": "no folds"}),
        ),
    )

    text = build_consolidated_research_summary_text(sequence=sequence, run_ingestion=True)

    assert "status: failed" in text
    assert "ingestion_mode: live" in text
    assert "data_coverage:" in text
    assert "- status: not_run" in text
    assert "sample_size_flag: unknown_completed_bout_count" in text
    assert "market_coverage_flag: unknown_matched_completed_fights" in text
    assert "failed_stage: walk_forward_eval" in text
