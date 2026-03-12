"""Model registry, champion-vs-challenger comparison, and selection reporting."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Sequence

from backtest import PlacedBet, run_betting_backtest
from evaluation.calibration import compute_calibration_metrics

MetricName = Literal["log_loss", "brier_score", "calibration", "roi", "drawdown"]


@dataclass(frozen=True)
class ModelRegistryEntry:
    """Single registered model source discovered from predictions artifacts."""

    model_name: str
    model_run_id: str
    predictions_path: Path
    prediction_rows: int

    @property
    def model_identifier(self) -> str:
        return f"{self.model_name}:{self.model_run_id}"


@dataclass(frozen=True)
class ModelRegistry:
    """Collection of discovered model entries."""

    entries: tuple[ModelRegistryEntry, ...]


@dataclass(frozen=True)
class ModelPeriodSummary:
    """Period-level out-of-sample metrics for one model."""

    period_id: str
    period_start_utc: str
    period_end_utc: str
    predictions_count: int
    log_loss: float
    brier_score: float
    calibration_ece: float
    roi: float
    max_drawdown: float
    bets_count: int


@dataclass(frozen=True)
class ModelEvaluationSummary:
    """Aggregate and period metrics for a single model entry."""

    entry: ModelRegistryEntry
    log_loss: float
    brier_score: float
    calibration_ece: float
    roi: float
    max_drawdown: float
    predictions_count: int
    bets_count: int
    periods: tuple[ModelPeriodSummary, ...]


@dataclass(frozen=True)
class ChampionChallengerMetric:
    """Head-to-head result for one metric."""

    metric: MetricName
    winner_model_identifier: str
    champion_value: float
    challenger_value: float


@dataclass(frozen=True)
class ChampionChallengerComparison:
    """Champion-versus-challenger summary."""

    champion: ModelEvaluationSummary
    challenger: ModelEvaluationSummary
    metric_results: tuple[ChampionChallengerMetric, ...]


@dataclass(frozen=True)
class ModelComparisonResult:
    """Full model selection output used by CLI/reporting."""

    registry: ModelRegistry
    model_summaries: tuple[ModelEvaluationSummary, ...]
    best_by_metric: dict[MetricName, str]
    champion_challenger: ChampionChallengerComparison
    report_path: Path


def discover_model_registry(*, predictions_paths: Sequence[Path]) -> ModelRegistry:
    """Build a simple registry from one or more predictions CSV files."""

    counts: dict[tuple[str, str, str], int] = {}

    for predictions_path in predictions_paths:
        if not predictions_path.exists():
            continue

        with predictions_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                split_value = str(raw.get("split", "test")).strip()
                if split_value != "test":
                    continue

                model_name = str(raw.get("model_name", "")).strip()
                model_run_id = str(raw.get("model_run_id", "")).strip() or "unknown"
                if model_name == "":
                    continue

                key = (model_name, model_run_id, predictions_path.as_posix())
                counts[key] = counts.get(key, 0) + 1

    entries = [
        ModelRegistryEntry(
            model_name=model_name,
            model_run_id=model_run_id,
            predictions_path=Path(path_text),
            prediction_rows=prediction_rows,
        )
        for (model_name, model_run_id, path_text), prediction_rows in counts.items()
    ]
    entries.sort(key=lambda row: (row.model_name, row.model_run_id, row.predictions_path.as_posix()))

    return ModelRegistry(entries=tuple(entries))


def compare_registered_models(
    *,
    db_path: Path,
    promotion: str,
    registry: ModelRegistry,
    report_output_path: Path,
    calibration_bins: int,
    min_edge: float,
    flat_stake: float,
    kelly_fraction: float | None,
    initial_bankroll: float,
    vig_adjustment: Literal["none", "proportional"],
    one_bet_per_bout: bool,
) -> ModelComparisonResult:
    """Evaluate registered models with chronology-safe metrics and select champion/challenger."""

    if len(registry.entries) < 2:
        raise ValueError("Model comparison requires at least two registered models.")

    model_summaries = [
        _evaluate_model_entry(
            db_path=db_path,
            promotion=promotion,
            entry=entry,
            calibration_bins=calibration_bins,
            min_edge=min_edge,
            flat_stake=flat_stake,
            kelly_fraction=kelly_fraction,
            initial_bankroll=initial_bankroll,
            vig_adjustment=vig_adjustment,
            one_bet_per_bout=one_bet_per_bout,
            report_output_path=report_output_path,
        )
        for entry in registry.entries
    ]

    best_by_metric: dict[MetricName, str] = {
        "log_loss": _best_model_identifier(model_summaries, metric="log_loss", better="lower"),
        "brier_score": _best_model_identifier(model_summaries, metric="brier_score", better="lower"),
        "calibration": _best_model_identifier(model_summaries, metric="calibration_ece", better="lower"),
        "roi": _best_model_identifier(model_summaries, metric="roi", better="higher"),
        "drawdown": _best_model_identifier(model_summaries, metric="max_drawdown", better="lower"),
    }

    champion, challenger = _select_champion_and_challenger(model_summaries)
    champion_challenger = compare_champion_vs_challenger(champion=champion, challenger=challenger)

    _write_model_comparison_report(
        output_path=report_output_path,
        registry=registry,
        model_summaries=model_summaries,
        best_by_metric=best_by_metric,
        champion_challenger=champion_challenger,
    )

    return ModelComparisonResult(
        registry=registry,
        model_summaries=tuple(model_summaries),
        best_by_metric=best_by_metric,
        champion_challenger=champion_challenger,
        report_path=report_output_path,
    )


def compare_champion_vs_challenger(
    *,
    champion: ModelEvaluationSummary,
    challenger: ModelEvaluationSummary,
) -> ChampionChallengerComparison:
    """Create explicit metric-level head-to-head between two model summaries."""

    metric_rows: list[ChampionChallengerMetric] = []

    for metric in ("log_loss", "brier_score", "calibration", "roi", "drawdown"):
        if metric == "log_loss":
            winner = champion if champion.log_loss <= challenger.log_loss else challenger
            metric_rows.append(
                ChampionChallengerMetric(
                    metric="log_loss",
                    winner_model_identifier=winner.entry.model_identifier,
                    champion_value=champion.log_loss,
                    challenger_value=challenger.log_loss,
                )
            )
            continue

        if metric == "brier_score":
            winner = champion if champion.brier_score <= challenger.brier_score else challenger
            metric_rows.append(
                ChampionChallengerMetric(
                    metric="brier_score",
                    winner_model_identifier=winner.entry.model_identifier,
                    champion_value=champion.brier_score,
                    challenger_value=challenger.brier_score,
                )
            )
            continue

        if metric == "calibration":
            winner = champion if champion.calibration_ece <= challenger.calibration_ece else challenger
            metric_rows.append(
                ChampionChallengerMetric(
                    metric="calibration",
                    winner_model_identifier=winner.entry.model_identifier,
                    champion_value=champion.calibration_ece,
                    challenger_value=challenger.calibration_ece,
                )
            )
            continue

        if metric == "roi":
            winner = champion if champion.roi >= challenger.roi else challenger
            metric_rows.append(
                ChampionChallengerMetric(
                    metric="roi",
                    winner_model_identifier=winner.entry.model_identifier,
                    champion_value=champion.roi,
                    challenger_value=challenger.roi,
                )
            )
            continue

        winner = champion if champion.max_drawdown <= challenger.max_drawdown else challenger
        metric_rows.append(
            ChampionChallengerMetric(
                metric="drawdown",
                winner_model_identifier=winner.entry.model_identifier,
                champion_value=champion.max_drawdown,
                challenger_value=challenger.max_drawdown,
            )
        )

    return ChampionChallengerComparison(
        champion=champion,
        challenger=challenger,
        metric_results=tuple(metric_rows),
    )


def _evaluate_model_entry(
    *,
    db_path: Path,
    promotion: str,
    entry: ModelRegistryEntry,
    calibration_bins: int,
    min_edge: float,
    flat_stake: float,
    kelly_fraction: float | None,
    initial_bankroll: float,
    vig_adjustment: Literal["none", "proportional"],
    one_bet_per_bout: bool,
    report_output_path: Path,
) -> ModelEvaluationSummary:
    rows = _load_model_rows(entry=entry)
    if len(rows) == 0:
        raise ValueError(f"No chronologically valid predictions for model={entry.model_identifier}.")

    labels = [row.label for row in rows]
    probabilities = [row.predicted_probability for row in rows]
    calibration = compute_calibration_metrics(
        labels=labels,
        probabilities=probabilities,
        num_bins=calibration_bins,
    )

    backtest_report_path = report_output_path.parent / "model_backtests" / (
        f"{_slugify(entry.model_name)}_{_slugify(entry.model_run_id)}.txt"
    )
    backtest_result = run_betting_backtest(
        db_path=db_path,
        predictions_path=entry.predictions_path,
        report_output_path=backtest_report_path,
        promotion=promotion,
        model_name=entry.model_name,
        model_run_id=None if entry.model_run_id == "unknown" else entry.model_run_id,
        min_edge=min_edge,
        flat_stake=flat_stake,
        kelly_fraction=kelly_fraction,
        initial_bankroll=initial_bankroll,
        vig_adjustment=vig_adjustment,
        one_bet_per_bout=one_bet_per_bout,
    )

    period_summaries = _build_period_summaries(
        rows=rows,
        bets=backtest_result.bets,
    )

    return ModelEvaluationSummary(
        entry=entry,
        log_loss=_log_loss(labels=labels, probabilities=probabilities),
        brier_score=_brier_score(labels=labels, probabilities=probabilities),
        calibration_ece=calibration.expected_calibration_error,
        roi=backtest_result.roi,
        max_drawdown=backtest_result.max_drawdown,
        predictions_count=len(rows),
        bets_count=backtest_result.bets_count,
        periods=tuple(period_summaries),
    )


@dataclass(frozen=True)
class _PredictionRow:
    bout_id: str
    bout_datetime_utc: str
    fold_index: int | None
    test_start_utc: str | None
    test_end_utc: str | None
    label: float
    predicted_probability: float


def _load_model_rows(*, entry: ModelRegistryEntry) -> list[_PredictionRow]:
    rows: list[_PredictionRow] = []

    with entry.predictions_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            if str(raw.get("model_name", "")).strip() != entry.model_name:
                continue

            raw_run_id = str(raw.get("model_run_id", "")).strip() or "unknown"
            if raw_run_id != entry.model_run_id:
                continue

            if str(raw.get("split", "test")).strip() != "test":
                continue

            label = _as_optional_float(raw.get("label"))
            probability = _as_optional_float(raw.get("predicted_probability"))
            if label is None or probability is None:
                continue

            bout_datetime = str(raw.get("bout_datetime_utc", "")).strip()
            feature_cutoff = str(raw.get("feature_cutoff_utc", "")).strip()
            if bout_datetime == "" or feature_cutoff == "":
                continue

            # Chronology guard: features must be available before the bout starts.
            if feature_cutoff > bout_datetime:
                continue

            fold_index_raw = str(raw.get("fold_index", "")).strip()
            fold_index = int(fold_index_raw) if fold_index_raw != "" else None
            test_start_utc = str(raw.get("test_start_utc", "")).strip() or None
            test_end_utc = str(raw.get("test_end_utc", "")).strip() or None

            rows.append(
                _PredictionRow(
                    bout_id=str(raw.get("bout_id", "")).strip(),
                    bout_datetime_utc=bout_datetime,
                    fold_index=fold_index,
                    test_start_utc=test_start_utc,
                    test_end_utc=test_end_utc,
                    label=label,
                    predicted_probability=_clamp_probability(probability),
                )
            )

    return rows


def _build_period_summaries(
    *,
    rows: Sequence[_PredictionRow],
    bets: Sequence[PlacedBet],
) -> list[ModelPeriodSummary]:
    uses_folds = any(row.fold_index is not None for row in rows)

    period_by_row: dict[str, list[_PredictionRow]] = {}
    bout_period_map: dict[str, str] = {}

    for row in rows:
        period_id = _period_id(row=row, uses_folds=uses_folds)
        period_by_row.setdefault(period_id, []).append(row)
        bout_period_map[row.bout_id] = period_id

    bets_by_period: dict[str, list[PlacedBet]] = {}
    for bet in bets:
        period_id = bout_period_map.get(bet.bout_id)
        if period_id is None:
            continue
        bets_by_period.setdefault(period_id, []).append(bet)

    summaries: list[ModelPeriodSummary] = []
    for period_id in sorted(period_by_row.keys()):
        period_rows = period_by_row[period_id]
        labels = [row.label for row in period_rows]
        probabilities = [row.predicted_probability for row in period_rows]
        calibration_ece = compute_calibration_metrics(
            labels=labels,
            probabilities=probabilities,
            num_bins=5,
        ).expected_calibration_error

        period_bets = sorted(
            bets_by_period.get(period_id, []),
            key=lambda row: (_parse_utc(row.bout_datetime_utc), row.bout_id, row.fighter_id),
        )
        total_staked = sum(row.stake for row in period_bets)
        total_pnl = sum(row.pnl for row in period_bets)
        roi = 0.0 if total_staked == 0.0 else total_pnl / total_staked

        equity = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for bet in period_bets:
            equity += bet.pnl
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, peak - equity)

        start_utc, end_utc = _period_boundaries(period_id=period_id, rows=period_rows, uses_folds=uses_folds)
        summaries.append(
            ModelPeriodSummary(
                period_id=period_id,
                period_start_utc=start_utc,
                period_end_utc=end_utc,
                predictions_count=len(period_rows),
                log_loss=_log_loss(labels=labels, probabilities=probabilities),
                brier_score=_brier_score(labels=labels, probabilities=probabilities),
                calibration_ece=calibration_ece,
                roi=roi,
                max_drawdown=max_drawdown,
                bets_count=len(period_bets),
            )
        )

    return summaries


def _period_id(*, row: _PredictionRow, uses_folds: bool) -> str:
    if uses_folds and row.fold_index is not None:
        return f"fold_{row.fold_index:03d}"
    return row.bout_datetime_utc[:7]


def _period_boundaries(*, period_id: str, rows: Sequence[_PredictionRow], uses_folds: bool) -> tuple[str, str]:
    if uses_folds:
        starts = [row.test_start_utc for row in rows if row.test_start_utc is not None]
        ends = [row.test_end_utc for row in rows if row.test_end_utc is not None]
        start_utc = min(starts) if len(starts) > 0 else min(row.bout_datetime_utc for row in rows)
        end_utc = max(ends) if len(ends) > 0 else max(row.bout_datetime_utc for row in rows)
        return start_utc, end_utc

    datetimes = [row.bout_datetime_utc for row in rows]
    return min(datetimes), max(datetimes)


def _best_model_identifier(
    model_summaries: Sequence[ModelEvaluationSummary],
    *,
    metric: str,
    better: Literal["lower", "higher"],
) -> str:
    if better == "lower":
        winner = min(model_summaries, key=lambda row: getattr(row, metric))
        return winner.entry.model_identifier

    winner = max(model_summaries, key=lambda row: getattr(row, metric))
    return winner.entry.model_identifier


def _select_champion_and_challenger(
    model_summaries: Sequence[ModelEvaluationSummary],
) -> tuple[ModelEvaluationSummary, ModelEvaluationSummary]:
    ranked = sorted(
        model_summaries,
        key=lambda row: (
            _rank_value(metric="log_loss", value=row.log_loss, better="lower"),
            _rank_value(metric="brier_score", value=row.brier_score, better="lower"),
            _rank_value(metric="calibration", value=row.calibration_ece, better="lower"),
            _rank_value(metric="roi", value=row.roi, better="higher"),
            _rank_value(metric="drawdown", value=row.max_drawdown, better="lower"),
            row.entry.model_identifier,
        ),
    )
    return ranked[0], ranked[1]


def _rank_value(*, metric: str, value: float, better: Literal["lower", "higher"]) -> float:
    if better == "lower":
        return value

    # Higher is better for ROI; invert so smaller sort key remains better.
    return -value


def _write_model_comparison_report(
    *,
    output_path: Path,
    registry: ModelRegistry,
    model_summaries: Sequence[ModelEvaluationSummary],
    best_by_metric: dict[MetricName, str],
    champion_challenger: ChampionChallengerComparison,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "model_comparison_report",
        "registry",
    ]

    for entry in registry.entries:
        lines.append(
            "registered_model "
            f"model={entry.model_identifier} rows={entry.prediction_rows} "
            f"predictions_path={entry.predictions_path.as_posix()}"
        )

    lines.extend(
        [
            "best_model_by_metric",
            f"best_log_loss_model={best_by_metric['log_loss']}",
            f"best_brier_score_model={best_by_metric['brier_score']}",
            f"best_calibration_model={best_by_metric['calibration']}",
            f"best_roi_model={best_by_metric['roi']}",
            f"best_drawdown_model={best_by_metric['drawdown']}",
            "model_aggregate_metrics",
        ]
    )

    for summary in sorted(model_summaries, key=lambda row: row.entry.model_identifier):
        lines.append(
            "model_metrics "
            f"model={summary.entry.model_identifier} "
            f"predictions={summary.predictions_count} bets={summary.bets_count} "
            f"log_loss={summary.log_loss:.8f} "
            f"brier_score={summary.brier_score:.8f} "
            f"calibration_ece={summary.calibration_ece:.8f} "
            f"roi={summary.roi:.8f} "
            f"drawdown={summary.max_drawdown:.8f}"
        )

    lines.extend(
        [
            "champion_vs_challenger",
            f"champion={champion_challenger.champion.entry.model_identifier}",
            f"challenger={champion_challenger.challenger.entry.model_identifier}",
        ]
    )

    for metric in champion_challenger.metric_results:
        lines.append(
            "head_to_head "
            f"metric={metric.metric} "
            f"winner={metric.winner_model_identifier} "
            f"champion_value={metric.champion_value:.8f} "
            f"challenger_value={metric.challenger_value:.8f}"
        )

    lines.append("period_by_period")
    for summary in sorted(model_summaries, key=lambda row: row.entry.model_identifier):
        for period in summary.periods:
            lines.append(
                "period_metrics "
                f"model={summary.entry.model_identifier} "
                f"period={period.period_id} "
                f"start_utc={period.period_start_utc} end_utc={period.period_end_utc} "
                f"predictions={period.predictions_count} bets={period.bets_count} "
                f"log_loss={period.log_loss:.8f} "
                f"brier_score={period.brier_score:.8f} "
                f"calibration_ece={period.calibration_ece:.8f} "
                f"roi={period.roi:.8f} "
                f"drawdown={period.max_drawdown:.8f}"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _log_loss(*, labels: Sequence[float], probabilities: Sequence[float]) -> float:
    if len(labels) == 0 or len(probabilities) == 0:
        raise ValueError("log_loss requires at least one prediction.")
    if len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities length mismatch.")

    total = 0.0
    for label, probability in zip(labels, probabilities):
        p = _clamp_probability(probability)
        total += -(label * math.log(p) + ((1.0 - label) * math.log(1.0 - p)))
    return total / float(len(labels))


def _brier_score(*, labels: Sequence[float], probabilities: Sequence[float]) -> float:
    if len(labels) == 0 or len(probabilities) == 0:
        raise ValueError("brier_score requires at least one prediction.")
    if len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities length mismatch.")

    total = 0.0
    for label, probability in zip(labels, probabilities):
        p = _clamp_probability(probability)
        total += (p - label) ** 2
    return total / float(len(labels))


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _as_optional_float(value: object | None) -> float | None:
    if value is None:
        return None

    text = str(value).strip()
    if text == "":
        return None

    try:
        return float(text)
    except ValueError:
        return None


def _clamp_probability(value: float) -> float:
    if value < 1e-6:
        return 1e-6
    if value > 1.0 - 1e-6:
        return 1.0 - 1e-6
    return value


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "unknown"
