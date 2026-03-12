# MMA ELO

Production-oriented MMA fight prediction and betting-edge research system with strict chronological integrity.

## Principles

- Strict chronological integrity (no future data in features).
- Chronological validation only (no random splits).
- UFC-first scope.
- Profitability and calibration prioritized over headline accuracy.
- Modular, testable Python code in `src/`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Reproducible End-to-End Workflow

All stages remain independently runnable.

### Documented Full Pipeline Command Sequence (modular, no dashboard)

This sequence runs ingestion, validation, feature generation, modeling, evaluation, and betting simulation.

```bash
# 0) Optional: initialize and inspect
mma-elo print-data-dirs
mma-elo print-schema-summary

# 1) Ingestion
mma-elo ingest-ufc-stats-events --db-path data/processed/mma_elo.db --raw-root data/raw --from-archives
mma-elo ingest-ufc-stats-fighters --db-path data/processed/mma_elo.db --raw-root data/raw --from-archives
mma-elo ingest-ufc-stats-bout-stats --db-path data/processed/mma_elo.db --raw-root data/raw --from-archives
export MMA_ELO_ODDS_API_KEY=\"your_the_odds_api_key\"
mma-elo ingest-market-odds \
  --db-path data/processed/mma_elo.db \
  --raw-root data/raw \
  --promotion UFC \
  --source-mode upcoming \
  --report-output-path data/processed/reports/market_data_validation_report.txt
# Also writes:
# data/processed/reports/market_matching_audit_report.txt

# Historical odds replay for completed-bout backtesting coverage
mma-elo ingest-market-odds \
  --db-path data/processed/mma_elo.db \
  --raw-root data/raw \
  --promotion UFC \
  --source-system odds_api_historical \
  --source-mode historical \
  --from-archives \
  --report-output-path data/processed/reports/market_data_validation_report.txt

# 2) Chronology and integrity validation
mma-elo validate-data --db-path data/processed/mma_elo.db

# 3) Sequential feature generation
mma-elo generate-features --db-path data/processed/mma_elo.db --promotion UFC --output-path data/processed/features/pre_fight_features.csv

# 4) Data coverage audit (coverage, missingness reasons, join failures)
mma-elo data-coverage-audit \
  --db-path data/processed/mma_elo.db \
  --promotion UFC \
  --features-path data/processed/features/pre_fight_features.csv \
  --report-output-path data/processed/reports/data_coverage_audit.txt

# 5) Ratings (Elo as prior)
mma-elo build-ratings-history --db-path data/processed/mma_elo.db --promotion UFC --output-path data/processed/ratings/elo_ratings_history.csv

# 6) Baseline modeling
mma-elo train-baselines \
  --db-path data/processed/mma_elo.db \
  --promotion UFC \
  --train-fraction 0.8 \
  --features-output-path data/processed/features/pre_fight_features.csv \
  --ratings-output-path data/processed/ratings/elo_ratings_history.csv \
  --predictions-output-path data/processed/predictions/baseline_predictions.csv \
  --report-output-path data/processed/reports/baseline_report.txt

# 7) Walk-forward evaluation
mma-elo walk-forward-eval \
  --db-path data/processed/mma_elo.db \
  --promotion UFC \
  --window-type expanding \
  --min-train-bouts 200 \
  --test-bouts 50 \
  --ratings-output-path data/processed/ratings/elo_ratings_history.csv \
  --predictions-output-path data/processed/predictions/walk_forward_predictions.csv \
  --report-output-path data/processed/reports/walk_forward_report.txt

# 8) Betting simulation
mma-elo run-backtest \
  --db-path data/processed/mma_elo.db \
  --predictions-path data/processed/predictions/baseline_predictions.csv \
  --promotion UFC \
  --model-name elo_logistic \
  --min-edge 0.02 \
  --flat-stake 1.0 \
  --report-output-path data/processed/reports/backtest_report.txt
```

### Single Command: Full-History Research Run + Consolidated Summary

Default is offline snapshot mode (`--run-ingestion` not set), which uses the full locally available UFC history in your database.

```bash
mma-elo run-research-workflow \
  --db-path data/processed/mma_elo.db \
  --promotion UFC \
  --window-type expanding \
  --min-train-bouts 200 \
  --test-bouts 50 \
  --calibration-bins 10 \
  --ablation-predictions-output-path data/processed/predictions/ablation_predictions.csv \
  --ablation-comparison-report-output-path data/processed/reports/ablation_comparison_report.txt \
  --ablation-contribution-report-output-path data/processed/reports/ablation_contribution_report.txt \
  --calibration-output-dir data/processed/reports/calibration_diagnostics \
  --calibration-report-output-path data/processed/reports/calibration_diagnostics/report.txt \
  --model-comparison-report-output-path data/processed/reports/model_comparison_report.txt \
  --data-coverage-audit-report-output-path data/processed/reports/data_coverage_audit.txt \
  --summary-report-path data/processed/reports/final_research_summary.txt \
  --workflow-log-path data/processed/logs/research_workflow/workflow.log \
  --log-dir data/processed/logs/research_workflow
```

Enable live ingestion only when intentionally required:

```bash
mma-elo run-research-workflow \
  --db-path data/processed/mma_elo.db \
  --run-ingestion \
  --raw-root data/raw
```

## Failure Handling and Logging

- Workflow is strict-order and fail-fast.
- Each stage writes its own log file in `data/processed/logs/research_workflow/`.
- A high-level workflow log is written to `data/processed/logs/research_workflow/workflow.log`.
- On failure, execution stops immediately and reports the failed stage.

## Final Summary Report

`mma-elo run-research-workflow` writes:

- `data/processed/reports/final_research_summary.txt`
- `data/processed/reports/market_data_validation_report.txt`
- `data/processed/reports/market_matching_audit_report.txt`

The consolidated summary includes:

- data coverage
- data coverage audit
- validation results
- feature coverage
- rating coverage
- model comparison
- walk-forward evaluation metrics
- calibration diagnostics
- betting simulation results
- ablation results

It also includes explicit `evaluation_readiness_flags` to warn when completed-bout sample size, fold count, matched completed-fight market coverage, opening/closing odds coverage, or bet activity are too small for meaningful evaluation.

## Tests

Run the full suite:

```bash
pytest
```

Run orchestration-specific tests:

```bash
pytest tests/test_workflow_orchestration.py
```

Run summary/report-specific tests:

```bash
pytest tests/test_research_summary_report.py
```

Run ingestion/audit-specific tests:

```bash
pytest tests/test_ufc_stats_archive_replay_ingestion.py tests/test_data_coverage_audit.py
```
