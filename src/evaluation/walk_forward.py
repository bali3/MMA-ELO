"""Chronology-safe walk-forward model evaluation."""

from __future__ import annotations

import csv
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Sequence

from ratings import EloConfig, generate_elo_ratings_history

WindowType = Literal["expanding", "rolling"]


@dataclass(frozen=True)
class BoutExample:
    """Single fighter-side bout example for fold training/evaluation."""

    bout_id: str
    event_id: str
    bout_datetime_utc: str
    bout_order: int
    fighter_id: str
    opponent_id: str
    label: float | None
    feature_cutoff_utc: str
    pre_overall_elo: float | None
    opponent_pre_overall_elo: float | None

    @property
    def elo_diff(self) -> float | None:
        if self.pre_overall_elo is None or self.opponent_pre_overall_elo is None:
            return None
        return self.pre_overall_elo - self.opponent_pre_overall_elo


@dataclass(frozen=True)
class BoutTimelineEntry:
    """Unique completed bout timeline entry used for chronological folds."""

    bout_id: str
    bout_datetime_utc: str
    bout_order: int


@dataclass(frozen=True)
class WalkForwardFold:
    """Single time-ordered fold definition."""

    fold_index: int
    train_bout_ids: tuple[str, ...]
    test_bout_ids: tuple[str, ...]
    train_start_utc: str
    train_end_utc: str
    test_start_utc: str
    test_end_utc: str


@dataclass(frozen=True)
class FoldMetrics:
    """Per-fold metrics summary."""

    fold_index: int
    train_bouts: int
    test_bouts: int
    train_start_utc: str
    train_end_utc: str
    test_start_utc: str
    test_end_utc: str
    log_loss: float
    brier_score: float
    hit_rate: float
    calibration_ece: float
    calibration_mce: float


@dataclass(frozen=True)
class CalibrationBin:
    """Probability bin summary for calibration diagnostics."""

    bin_index: int
    lower_bound: float
    upper_bound: float
    count: int
    avg_predicted_probability: float | None
    observed_win_rate: float | None
    absolute_error: float | None


@dataclass(frozen=True)
class CalibrationSummary:
    """Aggregate calibration error summary."""

    bins: tuple[CalibrationBin, ...]
    expected_calibration_error: float
    maximum_calibration_error: float


@dataclass(frozen=True)
class WalkForwardEvaluationResult:
    """End-to-end walk-forward evaluation output."""

    model_run_id: str
    model_name: str
    window_type: WindowType
    fold_count: int
    predictions_count: int
    predictions_path: Path
    report_path: Path
    aggregate_log_loss: float
    aggregate_brier_score: float
    aggregate_hit_rate: float
    aggregate_calibration: CalibrationSummary
    fold_metrics: tuple[FoldMetrics, ...]


@dataclass(frozen=True)
class _ModelRow:
    """Scored row persisted to prediction output."""

    model_run_id: str
    model_name: str
    model_version: str
    window_type: WindowType
    fold_index: int
    train_start_utc: str
    train_end_utc: str
    test_start_utc: str
    test_end_utc: str
    bout_id: str
    event_id: str
    bout_datetime_utc: str
    fighter_id: str
    opponent_id: str
    label: float
    predicted_probability: float
    feature_cutoff_utc: str


@dataclass
class _OneFeatureLogisticModel:
    """One-feature logistic regression model."""

    mean: float
    std: float
    weight: float
    bias: float

    def predict_probability(self, value: float) -> float:
        standardized = (value - self.mean) / self.std
        return _clamp_probability(_sigmoid(self.bias + (self.weight * standardized)))


def run_walk_forward_evaluation(
    *,
    db_path: Path,
    promotion: str,
    window_type: WindowType,
    min_train_bouts: int,
    test_bouts: int,
    rolling_train_bouts: int | None,
    model_version: str,
    ratings_output_path: Path,
    predictions_output_path: Path,
    report_output_path: Path,
    calibration_bins: int = 10,
    model_run_id: str | None = None,
) -> WalkForwardEvaluationResult:
    """Run strict walk-forward evaluation with fold-level and aggregate metrics."""

    if min_train_bouts < 1:
        raise ValueError("min_train_bouts must be at least 1.")
    if test_bouts < 1:
        raise ValueError("test_bouts must be at least 1.")
    if calibration_bins < 2:
        raise ValueError("calibration_bins must be at least 2.")
    if window_type not in {"expanding", "rolling"}:
        raise ValueError("window_type must be 'expanding' or 'rolling'.")
    if window_type == "rolling" and rolling_train_bouts is None:
        raise ValueError("rolling_train_bouts is required when window_type='rolling'.")
    if rolling_train_bouts is not None and rolling_train_bouts < 1:
        raise ValueError("rolling_train_bouts must be at least 1 when provided.")

    generate_elo_ratings_history(
        db_path=db_path,
        output_path=ratings_output_path,
        config=EloConfig(promotion=promotion),
    )

    elo_by_key = _load_elo_rows(ratings_output_path)
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        bout_rows = _load_bout_rows(connection=connection, promotion=promotion)

    examples = _build_examples(bout_rows=bout_rows, elo_by_key=elo_by_key)
    timeline = _build_timeline(examples=examples)
    folds = build_walk_forward_folds(
        timeline=timeline,
        window_type=window_type,
        min_train_bouts=min_train_bouts,
        test_bouts=test_bouts,
        rolling_train_bouts=rolling_train_bouts,
    )

    all_predictions: list[_ModelRow] = []
    all_probabilities: list[float] = []
    all_labels: list[float] = []
    fold_metrics: list[FoldMetrics] = []
    resolved_run_id = model_run_id or _build_model_run_id()

    for fold in folds:
        train_lookup = set(fold.train_bout_ids)
        test_lookup = set(fold.test_bout_ids)

        train_examples = [
            row for row in examples if row.bout_id in train_lookup and row.label is not None and row.elo_diff is not None
        ]
        test_examples = [
            row for row in examples if row.bout_id in test_lookup and row.label is not None and row.elo_diff is not None
        ]

        if len(train_examples) == 0:
            raise ValueError(f"Fold {fold.fold_index} has no trainable rows with Elo priors.")
        if len(test_examples) == 0:
            raise ValueError(f"Fold {fold.fold_index} has no test rows with Elo priors.")

        model = _fit_one_feature_logistic(
            features=[float(row.elo_diff) for row in train_examples],
            labels=[float(row.label) for row in train_examples],
        )

        fold_probabilities: list[float] = []
        fold_labels: list[float] = []

        for row in test_examples:
            probability = model.predict_probability(float(row.elo_diff))
            label = float(row.label)
            fold_probabilities.append(probability)
            fold_labels.append(label)
            all_probabilities.append(probability)
            all_labels.append(label)
            all_predictions.append(
                _ModelRow(
                    model_run_id=resolved_run_id,
                    model_name="elo_logistic",
                    model_version=model_version,
                    window_type=window_type,
                    fold_index=fold.fold_index,
                    train_start_utc=fold.train_start_utc,
                    train_end_utc=fold.train_end_utc,
                    test_start_utc=fold.test_start_utc,
                    test_end_utc=fold.test_end_utc,
                    bout_id=row.bout_id,
                    event_id=row.event_id,
                    bout_datetime_utc=row.bout_datetime_utc,
                    fighter_id=row.fighter_id,
                    opponent_id=row.opponent_id,
                    label=label,
                    predicted_probability=probability,
                    feature_cutoff_utc=row.feature_cutoff_utc,
                )
            )

        fold_log_loss = _log_loss(fold_labels, fold_probabilities)
        fold_brier = _brier_score(fold_labels, fold_probabilities)
        fold_hit_rate = _hit_rate(fold_labels, fold_probabilities)
        fold_calibration = _build_calibration_summary(
            labels=fold_labels,
            probabilities=fold_probabilities,
            num_bins=calibration_bins,
        )

        fold_metrics.append(
            FoldMetrics(
                fold_index=fold.fold_index,
                train_bouts=len(fold.train_bout_ids),
                test_bouts=len(fold.test_bout_ids),
                train_start_utc=fold.train_start_utc,
                train_end_utc=fold.train_end_utc,
                test_start_utc=fold.test_start_utc,
                test_end_utc=fold.test_end_utc,
                log_loss=fold_log_loss,
                brier_score=fold_brier,
                hit_rate=fold_hit_rate,
                calibration_ece=fold_calibration.expected_calibration_error,
                calibration_mce=fold_calibration.maximum_calibration_error,
            )
        )

    aggregate_calibration = _build_calibration_summary(
        labels=all_labels,
        probabilities=all_probabilities,
        num_bins=calibration_bins,
    )

    _write_predictions_csv(output_path=predictions_output_path, rows=all_predictions)
    _write_report(
        output_path=report_output_path,
        model_name="elo_logistic",
        window_type=window_type,
        fold_metrics=fold_metrics,
        predictions_count=len(all_predictions),
        aggregate_log_loss=_log_loss(all_labels, all_probabilities),
        aggregate_brier_score=_brier_score(all_labels, all_probabilities),
        aggregate_hit_rate=_hit_rate(all_labels, all_probabilities),
        aggregate_calibration=aggregate_calibration,
    )

    return WalkForwardEvaluationResult(
        model_run_id=resolved_run_id,
        model_name="elo_logistic",
        window_type=window_type,
        fold_count=len(fold_metrics),
        predictions_count=len(all_predictions),
        predictions_path=predictions_output_path,
        report_path=report_output_path,
        aggregate_log_loss=_log_loss(all_labels, all_probabilities),
        aggregate_brier_score=_brier_score(all_labels, all_probabilities),
        aggregate_hit_rate=_hit_rate(all_labels, all_probabilities),
        aggregate_calibration=aggregate_calibration,
        fold_metrics=tuple(fold_metrics),
    )


def build_walk_forward_folds(
    *,
    timeline: Sequence[BoutTimelineEntry],
    window_type: WindowType,
    min_train_bouts: int,
    test_bouts: int,
    rolling_train_bouts: int | None,
) -> list[WalkForwardFold]:
    """Build time-ordered folds with no overlap and strict forward progression."""

    if len(timeline) < min_train_bouts + 1:
        raise ValueError("Not enough completed bouts to build walk-forward folds.")

    folds: list[WalkForwardFold] = []
    timeline_sorted = sorted(timeline, key=lambda row: (_parse_utc(row.bout_datetime_utc), row.bout_order, row.bout_id))

    cursor = min_train_bouts
    fold_index = 1

    while cursor < len(timeline_sorted):
        test_end = min(cursor + test_bouts, len(timeline_sorted))
        if window_type == "expanding":
            train_start = 0
        else:
            assert rolling_train_bouts is not None
            train_start = max(0, cursor - rolling_train_bouts)

        train_slice = timeline_sorted[train_start:cursor]
        test_slice = timeline_sorted[cursor:test_end]

        if len(train_slice) == 0 or len(test_slice) == 0:
            break

        train_end_time = _parse_utc(train_slice[-1].bout_datetime_utc)
        test_start_time = _parse_utc(test_slice[0].bout_datetime_utc)
        if train_end_time > test_start_time:
            raise ValueError("Detected invalid fold ordering: train period extends into test period.")

        folds.append(
            WalkForwardFold(
                fold_index=fold_index,
                train_bout_ids=tuple(row.bout_id for row in train_slice),
                test_bout_ids=tuple(row.bout_id for row in test_slice),
                train_start_utc=train_slice[0].bout_datetime_utc,
                train_end_utc=train_slice[-1].bout_datetime_utc,
                test_start_utc=test_slice[0].bout_datetime_utc,
                test_end_utc=test_slice[-1].bout_datetime_utc,
            )
        )

        fold_index += 1
        cursor = test_end

    if len(folds) == 0:
        raise ValueError("No valid walk-forward folds were created.")
    return folds


def _load_bout_rows(*, connection: sqlite3.Connection, promotion: str) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT
            b.bout_id,
            b.event_id,
            COALESCE(b.bout_order, 0) AS bout_order,
            COALESCE(b.bout_start_time_utc, e.event_date_utc) AS bout_datetime_utc,
            b.fighter_red_id,
            b.fighter_blue_id,
            b.winner_fighter_id
        FROM bouts AS b
        JOIN events AS e ON e.event_id = b.event_id
        WHERE UPPER(e.promotion) = UPPER(?)
        ORDER BY
            COALESCE(b.bout_start_time_utc, e.event_date_utc) ASC,
            e.event_date_utc ASC,
            COALESCE(b.bout_order, 0) ASC,
            b.bout_id ASC
        """,
        (promotion,),
    ).fetchall()


def _load_elo_rows(path: Path) -> dict[tuple[str, str], float]:
    by_key: dict[tuple[str, str], float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            bout_id = str(row.get("bout_id", "")).strip()
            fighter_id = str(row.get("fighter_id", "")).strip()
            if bout_id == "" or fighter_id == "":
                continue
            pre_overall_elo = _as_optional_float(row.get("pre_overall_elo"))
            if pre_overall_elo is None:
                continue
            by_key[(bout_id, fighter_id)] = pre_overall_elo
    return by_key


def _build_examples(
    *,
    bout_rows: Sequence[sqlite3.Row],
    elo_by_key: dict[tuple[str, str], float],
) -> list[BoutExample]:
    examples: list[BoutExample] = []

    for row in bout_rows:
        bout_id = str(row["bout_id"])
        event_id = str(row["event_id"])
        bout_datetime = str(row["bout_datetime_utc"])
        bout_order = int(row["bout_order"])
        red_id = str(row["fighter_red_id"])
        blue_id = str(row["fighter_blue_id"])
        winner_id = _clean_text(row["winner_fighter_id"])

        red_label = _resolve_label(winner_id=winner_id, fighter_id=red_id, opponent_id=blue_id)
        blue_label = _resolve_label(winner_id=winner_id, fighter_id=blue_id, opponent_id=red_id)

        examples.append(
            BoutExample(
                bout_id=bout_id,
                event_id=event_id,
                bout_datetime_utc=bout_datetime,
                bout_order=bout_order,
                fighter_id=red_id,
                opponent_id=blue_id,
                label=red_label,
                feature_cutoff_utc=bout_datetime,
                pre_overall_elo=elo_by_key.get((bout_id, red_id)),
                opponent_pre_overall_elo=elo_by_key.get((bout_id, blue_id)),
            )
        )
        examples.append(
            BoutExample(
                bout_id=bout_id,
                event_id=event_id,
                bout_datetime_utc=bout_datetime,
                bout_order=bout_order,
                fighter_id=blue_id,
                opponent_id=red_id,
                label=blue_label,
                feature_cutoff_utc=bout_datetime,
                pre_overall_elo=elo_by_key.get((bout_id, blue_id)),
                opponent_pre_overall_elo=elo_by_key.get((bout_id, red_id)),
            )
        )

    return examples


def _build_timeline(*, examples: Sequence[BoutExample]) -> list[BoutTimelineEntry]:
    by_bout: dict[str, tuple[str, int]] = {}
    for row in examples:
        if row.label is None:
            continue
        existing = by_bout.get(row.bout_id)
        candidate = (row.bout_datetime_utc, row.bout_order)
        if existing is None or candidate < existing:
            by_bout[row.bout_id] = candidate

    timeline = [
        BoutTimelineEntry(
            bout_id=bout_id,
            bout_datetime_utc=metadata[0],
            bout_order=metadata[1],
        )
        for bout_id, metadata in by_bout.items()
    ]
    timeline.sort(key=lambda row: (_parse_utc(row.bout_datetime_utc), row.bout_order, row.bout_id))
    return timeline


def _fit_one_feature_logistic(
    *,
    features: Sequence[float],
    labels: Sequence[float],
    learning_rate: float = 0.05,
    iterations: int = 600,
    l2_penalty: float = 0.001,
) -> _OneFeatureLogisticModel:
    if len(features) == 0:
        raise ValueError("Logistic training requires at least one row.")
    if len(features) != len(labels):
        raise ValueError("Features and labels length mismatch.")

    mean = sum(features) / float(len(features))
    variance = sum((value - mean) ** 2 for value in features) / float(len(features))
    std = math.sqrt(variance)
    std = std if std > 1e-8 else 1.0

    positive_rate = sum(labels) / float(len(labels))
    if positive_rate <= 1e-6 or positive_rate >= 1.0 - 1e-6:
        return _OneFeatureLogisticModel(mean=mean, std=std, weight=0.0, bias=_logit(_clamp_probability(positive_rate)))

    weight = 0.0
    bias = 0.0
    row_count = float(len(features))

    for _ in range(iterations):
        grad_w = 0.0
        grad_b = 0.0
        for index, raw_feature in enumerate(features):
            x = (raw_feature - mean) / std
            prediction = _sigmoid(bias + (weight * x))
            error = prediction - labels[index]
            grad_w += error * x
            grad_b += error

        grad_w = (grad_w / row_count) + (l2_penalty * weight)
        grad_b = grad_b / row_count
        weight -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    return _OneFeatureLogisticModel(mean=mean, std=std, weight=weight, bias=bias)


def _build_calibration_summary(
    *,
    labels: Sequence[float],
    probabilities: Sequence[float],
    num_bins: int,
) -> CalibrationSummary:
    if len(labels) == 0 or len(probabilities) == 0:
        raise ValueError("Calibration summary requires at least one prediction.")
    if len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities length mismatch.")

    bucket_probs: list[list[float]] = [[] for _ in range(num_bins)]
    bucket_labels: list[list[float]] = [[] for _ in range(num_bins)]

    for label, probability in zip(labels, probabilities):
        p = _clamp_probability(probability)
        index = min(int(p * num_bins), num_bins - 1)
        bucket_probs[index].append(p)
        bucket_labels[index].append(label)

    bins: list[CalibrationBin] = []
    total_count = len(probabilities)
    ece = 0.0
    mce = 0.0

    for index in range(num_bins):
        lower = index / num_bins
        upper = (index + 1) / num_bins
        probs = bucket_probs[index]
        labels_for_bin = bucket_labels[index]
        count = len(probs)

        if count == 0:
            bins.append(
                CalibrationBin(
                    bin_index=index,
                    lower_bound=lower,
                    upper_bound=upper,
                    count=0,
                    avg_predicted_probability=None,
                    observed_win_rate=None,
                    absolute_error=None,
                )
            )
            continue

        avg_probability = sum(probs) / float(count)
        observed_rate = sum(labels_for_bin) / float(count)
        abs_error = abs(avg_probability - observed_rate)
        ece += (count / float(total_count)) * abs_error
        if abs_error > mce:
            mce = abs_error

        bins.append(
            CalibrationBin(
                bin_index=index,
                lower_bound=lower,
                upper_bound=upper,
                count=count,
                avg_predicted_probability=avg_probability,
                observed_win_rate=observed_rate,
                absolute_error=abs_error,
            )
        )

    return CalibrationSummary(
        bins=tuple(bins),
        expected_calibration_error=ece,
        maximum_calibration_error=mce,
    )


def _log_loss(labels: Sequence[float], probabilities: Sequence[float]) -> float:
    if len(labels) == 0 or len(probabilities) == 0:
        raise ValueError("log_loss requires at least one prediction.")
    if len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities length mismatch.")

    total = 0.0
    for label, probability in zip(labels, probabilities):
        p = _clamp_probability(probability)
        total += -(label * math.log(p) + ((1.0 - label) * math.log(1.0 - p)))
    return total / float(len(labels))


def _brier_score(labels: Sequence[float], probabilities: Sequence[float]) -> float:
    if len(labels) == 0 or len(probabilities) == 0:
        raise ValueError("brier_score requires at least one prediction.")
    if len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities length mismatch.")

    total = 0.0
    for label, probability in zip(labels, probabilities):
        p = _clamp_probability(probability)
        total += (p - label) ** 2
    return total / float(len(labels))


def _hit_rate(labels: Sequence[float], probabilities: Sequence[float]) -> float:
    if len(labels) == 0 or len(probabilities) == 0:
        raise ValueError("hit_rate requires at least one prediction.")
    if len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities length mismatch.")

    hits = 0
    for label, probability in zip(labels, probabilities):
        predicted_class = 1.0 if probability >= 0.5 else 0.0
        if predicted_class == label:
            hits += 1
    return hits / float(len(labels))


def _write_predictions_csv(*, output_path: Path, rows: Sequence[_ModelRow]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "model_run_id",
        "model_name",
        "model_version",
        "window_type",
        "fold_index",
        "train_start_utc",
        "train_end_utc",
        "test_start_utc",
        "test_end_utc",
        "bout_id",
        "event_id",
        "bout_datetime_utc",
        "fighter_id",
        "opponent_id",
        "label",
        "predicted_probability",
        "feature_cutoff_utc",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model_run_id": row.model_run_id,
                    "model_name": row.model_name,
                    "model_version": row.model_version,
                    "window_type": row.window_type,
                    "fold_index": row.fold_index,
                    "train_start_utc": row.train_start_utc,
                    "train_end_utc": row.train_end_utc,
                    "test_start_utc": row.test_start_utc,
                    "test_end_utc": row.test_end_utc,
                    "bout_id": row.bout_id,
                    "event_id": row.event_id,
                    "bout_datetime_utc": row.bout_datetime_utc,
                    "fighter_id": row.fighter_id,
                    "opponent_id": row.opponent_id,
                    "label": f"{row.label:.6f}",
                    "predicted_probability": f"{row.predicted_probability:.8f}",
                    "feature_cutoff_utc": row.feature_cutoff_utc,
                }
            )


def _write_report(
    *,
    output_path: Path,
    model_name: str,
    window_type: WindowType,
    fold_metrics: Sequence[FoldMetrics],
    predictions_count: int,
    aggregate_log_loss: float,
    aggregate_brier_score: float,
    aggregate_hit_rate: float,
    aggregate_calibration: CalibrationSummary,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "walk_forward_evaluation_report",
        f"model_name={model_name}",
        f"window_type={window_type}",
        f"fold_count={len(fold_metrics)}",
        f"predictions={predictions_count}",
        "aggregate_metrics",
        f"log_loss={aggregate_log_loss:.8f}",
        f"brier_score={aggregate_brier_score:.8f}",
        f"hit_rate={aggregate_hit_rate:.8f}",
        "aggregate_calibration",
        f"ece={aggregate_calibration.expected_calibration_error:.8f}",
        f"mce={aggregate_calibration.maximum_calibration_error:.8f}",
        "period_performance",
    ]

    for fold in fold_metrics:
        lines.append(
            f"fold={fold.fold_index} "
            f"train_start_utc={fold.train_start_utc} train_end_utc={fold.train_end_utc} "
            f"test_start_utc={fold.test_start_utc} test_end_utc={fold.test_end_utc} "
            f"train_bouts={fold.train_bouts} test_bouts={fold.test_bouts} "
            f"log_loss={fold.log_loss:.8f} brier_score={fold.brier_score:.8f} "
            f"hit_rate={fold.hit_rate:.8f} calibration_ece={fold.calibration_ece:.8f} "
            f"calibration_mce={fold.calibration_mce:.8f}"
        )

    lines.append("calibration_bins")
    for calibration_bin in aggregate_calibration.bins:
        avg_pred = "n/a" if calibration_bin.avg_predicted_probability is None else f"{calibration_bin.avg_predicted_probability:.8f}"
        observed = "n/a" if calibration_bin.observed_win_rate is None else f"{calibration_bin.observed_win_rate:.8f}"
        abs_error = "n/a" if calibration_bin.absolute_error is None else f"{calibration_bin.absolute_error:.8f}"
        lines.append(
            f"bin={calibration_bin.bin_index} "
            f"range=[{calibration_bin.lower_bound:.2f},{calibration_bin.upper_bound:.2f}) "
            f"count={calibration_bin.count} avg_pred={avg_pred} observed={observed} abs_error={abs_error}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_label(*, winner_id: str | None, fighter_id: str, opponent_id: str) -> float | None:
    if winner_id is None:
        return None
    if winner_id == fighter_id:
        return 1.0
    if winner_id == opponent_id:
        return 0.0
    return None


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


def _clean_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text != "" else None


def _clamp_probability(value: float) -> float:
    if value < 1e-6:
        return 1e-6
    if value > 1.0 - 1e-6:
        return 1.0 - 1e-6
    return value


def _logit(probability: float) -> float:
    p = _clamp_probability(probability)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        exp_value = math.exp(-z)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(z)
    return exp_value / (1.0 + exp_value)


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _build_model_run_id() -> str:
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return f"walk_forward_{timestamp}"
