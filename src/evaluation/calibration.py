"""Calibration diagnostics for chronologically valid model predictions."""

from __future__ import annotations

import csv
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class CalibrationBinSummary:
    """Single probability-bin summary row."""

    bin_index: int
    lower_bound: float
    upper_bound: float
    count: int
    wins: int
    losses: int
    avg_predicted_probability: float | None
    observed_win_rate: float | None
    absolute_error: float | None


@dataclass(frozen=True)
class CalibrationMetrics:
    """Aggregate calibration summary metrics."""

    bins: tuple[CalibrationBinSummary, ...]
    expected_calibration_error: float
    maximum_calibration_error: float


@dataclass(frozen=True)
class PerformanceBreakdownRow:
    """Performance summary for a segment."""

    segment_type: str
    segment_value: str
    count: int
    wins: int
    losses: int
    avg_predicted_probability: float
    observed_win_rate: float
    hit_rate: float
    log_loss: float
    brier_score: float
    calibration_ece: float
    calibration_mce: float


@dataclass(frozen=True)
class CalibrationDiagnosticsResult:
    """Result metadata and artifact locations for diagnostics runs."""

    model_name: str
    model_run_id: str
    predictions_loaded: int
    predictions_used: int
    dropped_missing_metadata: int
    dropped_chronology_invalid: int
    dropped_invalid_label: int
    dropped_missing_probability: int
    aggregate_log_loss: float
    aggregate_brier_score: float
    aggregate_hit_rate: float
    aggregate_calibration: CalibrationMetrics
    calibration_table_path: Path
    reliability_curve_path: Path
    probability_bucket_summary_path: Path
    performance_breakdown_path: Path
    report_path: Path


@dataclass(frozen=True)
class _PredictionRow:
    model_run_id: str
    model_name: str
    bout_id: str
    fighter_id: str
    opponent_id: str
    bout_datetime_utc: str
    feature_cutoff_utc: str
    predicted_probability: float
    label: float


@dataclass(frozen=True)
class _EnrichedPredictionRow:
    model_run_id: str
    model_name: str
    bout_id: str
    fighter_id: str
    opponent_id: str
    bout_datetime_utc: str
    feature_cutoff_utc: str
    predicted_probability: float
    label: float
    weight_class: str
    favourite_status: str


def run_calibration_diagnostics(
    *,
    db_path: Path,
    predictions_path: Path,
    output_dir: Path,
    report_output_path: Path,
    promotion: str,
    model_name: str,
    model_run_id: str | None,
    calibration_bins: int,
) -> CalibrationDiagnosticsResult:
    """Generate calibration diagnostics artifacts from chronology-safe predictions."""

    if calibration_bins < 2:
        raise ValueError("calibration_bins must be at least 2.")

    loaded_rows, dropped_invalid_label, dropped_missing_probability = _load_prediction_rows(
        predictions_path=predictions_path,
        model_name=model_name,
        model_run_id=model_run_id,
    )
    resolved_model_run_id = _resolve_model_run_id(rows=loaded_rows, requested_run_id=model_run_id)
    if model_run_id is None and resolved_model_run_id != "unknown":
        loaded_rows = [row for row in loaded_rows if row.model_run_id == resolved_model_run_id]

    if len(loaded_rows) == 0:
        requested = "" if model_run_id is None else f" for model_run_id={model_run_id}"
        raise ValueError(f"No predictions available for model={model_name}{requested}.")

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        bout_metadata = _load_bout_metadata(connection=connection, promotion=promotion)

    enriched_rows, dropped_missing_metadata, dropped_chronology_invalid = _enrich_predictions(
        rows=loaded_rows,
        bout_metadata=bout_metadata,
    )
    if len(enriched_rows) == 0:
        raise ValueError("No chronologically valid predictions remained after filtering.")

    labels = [row.label for row in enriched_rows]
    probabilities = [row.predicted_probability for row in enriched_rows]
    calibration = compute_calibration_metrics(
        labels=labels,
        probabilities=probabilities,
        num_bins=calibration_bins,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_table_path = output_dir / "calibration_table.csv"
    reliability_curve_path = output_dir / "reliability_curve.csv"
    probability_bucket_summary_path = output_dir / "probability_bucket_summary.csv"
    performance_breakdown_path = output_dir / "performance_breakdowns.csv"

    _write_calibration_table(calibration_table_path=calibration_table_path, bins=calibration.bins)
    _write_reliability_curve(reliability_curve_path=reliability_curve_path, bins=calibration.bins)
    _write_probability_bucket_summary(
        probability_bucket_summary_path=probability_bucket_summary_path,
        bins=calibration.bins,
        total_count=len(enriched_rows),
    )

    breakdown_rows = _build_performance_breakdowns(rows=enriched_rows, calibration_bins=calibration_bins)
    _write_performance_breakdowns(
        performance_breakdown_path=performance_breakdown_path,
        rows=breakdown_rows,
    )

    _write_report(
        report_output_path=report_output_path,
        model_name=model_name,
        model_run_id=resolved_model_run_id,
        promotion=promotion,
        predictions_loaded=len(loaded_rows),
        predictions_used=len(enriched_rows),
        dropped_missing_metadata=dropped_missing_metadata,
        dropped_chronology_invalid=dropped_chronology_invalid,
        dropped_invalid_label=dropped_invalid_label,
        dropped_missing_probability=dropped_missing_probability,
        calibration=calibration,
        aggregate_log_loss=_log_loss(labels=labels, probabilities=probabilities),
        aggregate_brier_score=_brier_score(labels=labels, probabilities=probabilities),
        aggregate_hit_rate=_hit_rate(labels=labels, probabilities=probabilities),
        breakdown_rows=breakdown_rows,
        calibration_table_path=calibration_table_path,
        reliability_curve_path=reliability_curve_path,
        probability_bucket_summary_path=probability_bucket_summary_path,
        performance_breakdown_path=performance_breakdown_path,
    )

    return CalibrationDiagnosticsResult(
        model_name=model_name,
        model_run_id=resolved_model_run_id,
        predictions_loaded=len(loaded_rows),
        predictions_used=len(enriched_rows),
        dropped_missing_metadata=dropped_missing_metadata,
        dropped_chronology_invalid=dropped_chronology_invalid,
        dropped_invalid_label=dropped_invalid_label,
        dropped_missing_probability=dropped_missing_probability,
        aggregate_log_loss=_log_loss(labels=labels, probabilities=probabilities),
        aggregate_brier_score=_brier_score(labels=labels, probabilities=probabilities),
        aggregate_hit_rate=_hit_rate(labels=labels, probabilities=probabilities),
        aggregate_calibration=calibration,
        calibration_table_path=calibration_table_path,
        reliability_curve_path=reliability_curve_path,
        probability_bucket_summary_path=probability_bucket_summary_path,
        performance_breakdown_path=performance_breakdown_path,
        report_path=report_output_path,
    )


def compute_calibration_metrics(
    *,
    labels: Sequence[float],
    probabilities: Sequence[float],
    num_bins: int,
) -> CalibrationMetrics:
    """Compute bin-level summaries plus ECE and MCE."""

    if len(labels) == 0 or len(probabilities) == 0:
        raise ValueError("Calibration metrics require at least one prediction.")
    if len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities length mismatch.")
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2.")

    bucket_probs: list[list[float]] = [[] for _ in range(num_bins)]
    bucket_labels: list[list[float]] = [[] for _ in range(num_bins)]

    for label, probability in zip(labels, probabilities):
        p = _clamp_probability(probability)
        index = min(int(p * num_bins), num_bins - 1)
        bucket_probs[index].append(p)
        bucket_labels[index].append(label)

    total_count = len(probabilities)
    ece = 0.0
    mce = 0.0
    bins: list[CalibrationBinSummary] = []

    for index in range(num_bins):
        lower = index / num_bins
        upper = (index + 1) / num_bins
        probs = bucket_probs[index]
        labels_for_bin = bucket_labels[index]
        count = len(probs)

        if count == 0:
            bins.append(
                CalibrationBinSummary(
                    bin_index=index,
                    lower_bound=lower,
                    upper_bound=upper,
                    count=0,
                    wins=0,
                    losses=0,
                    avg_predicted_probability=None,
                    observed_win_rate=None,
                    absolute_error=None,
                )
            )
            continue

        avg_probability = sum(probs) / float(count)
        observed_rate = sum(labels_for_bin) / float(count)
        wins = int(round(sum(labels_for_bin)))
        losses = count - wins
        absolute_error = abs(avg_probability - observed_rate)
        ece += (count / float(total_count)) * absolute_error
        mce = max(mce, absolute_error)

        bins.append(
            CalibrationBinSummary(
                bin_index=index,
                lower_bound=lower,
                upper_bound=upper,
                count=count,
                wins=wins,
                losses=losses,
                avg_predicted_probability=avg_probability,
                observed_win_rate=observed_rate,
                absolute_error=absolute_error,
            )
        )

    return CalibrationMetrics(
        bins=tuple(bins),
        expected_calibration_error=ece,
        maximum_calibration_error=mce,
    )


def _build_performance_breakdowns(
    *,
    rows: Sequence[_EnrichedPredictionRow],
    calibration_bins: int,
) -> list[PerformanceBreakdownRow]:
    segmented: dict[tuple[str, str], list[_EnrichedPredictionRow]] = {}

    def add_row(segment_type: str, segment_value: str, row: _EnrichedPredictionRow) -> None:
        segmented.setdefault((segment_type, segment_value), []).append(row)

    for row in rows:
        add_row("overall", "all", row)
        add_row("favourite_status", row.favourite_status, row)
        add_row("weight_class", row.weight_class, row)
        add_row("favourite_status_weight_class", f"{row.favourite_status}|{row.weight_class}", row)

    output: list[PerformanceBreakdownRow] = []
    for (segment_type, segment_value), segment_rows in segmented.items():
        labels = [item.label for item in segment_rows]
        probabilities = [item.predicted_probability for item in segment_rows]
        calibration = compute_calibration_metrics(
            labels=labels,
            probabilities=probabilities,
            num_bins=calibration_bins,
        )
        count = len(segment_rows)
        wins = int(round(sum(labels)))
        losses = count - wins

        output.append(
            PerformanceBreakdownRow(
                segment_type=segment_type,
                segment_value=segment_value,
                count=count,
                wins=wins,
                losses=losses,
                avg_predicted_probability=sum(probabilities) / float(count),
                observed_win_rate=sum(labels) / float(count),
                hit_rate=_hit_rate(labels=labels, probabilities=probabilities),
                log_loss=_log_loss(labels=labels, probabilities=probabilities),
                brier_score=_brier_score(labels=labels, probabilities=probabilities),
                calibration_ece=calibration.expected_calibration_error,
                calibration_mce=calibration.maximum_calibration_error,
            )
        )

    output.sort(key=lambda row: (row.segment_type, row.segment_value))
    return output


def _load_prediction_rows(
    *,
    predictions_path: Path,
    model_name: str,
    model_run_id: str | None,
) -> tuple[list[_PredictionRow], int, int]:
    rows: list[_PredictionRow] = []
    dropped_invalid_label = 0
    dropped_missing_probability = 0

    with predictions_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            if str(raw.get("model_name", "")).strip() != model_name:
                continue

            split_value = str(raw.get("split", "test")).strip()
            if split_value != "test":
                continue

            raw_run_id = str(raw.get("model_run_id", "")).strip()
            if model_run_id is not None and raw_run_id != model_run_id:
                continue

            raw_probability = _as_optional_float(raw.get("predicted_probability"))
            if raw_probability is None:
                dropped_missing_probability += 1
                continue

            raw_label = _as_optional_float(raw.get("label"))
            if raw_label is None or raw_label not in {0.0, 1.0}:
                dropped_invalid_label += 1
                continue

            bout_id = str(raw.get("bout_id", "")).strip()
            fighter_id = str(raw.get("fighter_id", "")).strip()
            opponent_id = str(raw.get("opponent_id", "")).strip()
            bout_datetime_utc = str(raw.get("bout_datetime_utc", "")).strip()
            feature_cutoff_utc = str(raw.get("feature_cutoff_utc", "")).strip()
            if (
                bout_id == ""
                or fighter_id == ""
                or opponent_id == ""
                or bout_datetime_utc == ""
                or feature_cutoff_utc == ""
            ):
                continue

            rows.append(
                _PredictionRow(
                    model_run_id=raw_run_id,
                    model_name=model_name,
                    bout_id=bout_id,
                    fighter_id=fighter_id,
                    opponent_id=opponent_id,
                    bout_datetime_utc=bout_datetime_utc,
                    feature_cutoff_utc=feature_cutoff_utc,
                    predicted_probability=_clamp_probability(raw_probability),
                    label=raw_label,
                )
            )

    if len(rows) == 0:
        requested = "" if model_run_id is None else f" for model_run_id={model_run_id}"
        raise ValueError(f"No prediction rows found for model={model_name}{requested}.")

    return rows, dropped_invalid_label, dropped_missing_probability


def _load_bout_metadata(
    *,
    connection: sqlite3.Connection,
    promotion: str,
) -> dict[str, tuple[str, datetime]]:
    rows = connection.execute(
        """
        SELECT
            b.bout_id,
            COALESCE(NULLIF(TRIM(b.weight_class), ''), 'UNKNOWN') AS weight_class,
            COALESCE(b.bout_start_time_utc, e.event_date_utc) AS bout_datetime_utc
        FROM bouts b
        JOIN events e ON e.event_id = b.event_id
        WHERE UPPER(e.promotion) = UPPER(?)
          AND COALESCE(b.bout_start_time_utc, e.event_date_utc) IS NOT NULL
        """,
        (promotion,),
    ).fetchall()

    metadata: dict[str, tuple[str, datetime]] = {}
    for row in rows:
        bout_id = str(row["bout_id"]).strip()
        start_utc = _parse_utc(str(row["bout_datetime_utc"]).strip())
        weight_class = str(row["weight_class"]).strip()
        metadata[bout_id] = (weight_class, start_utc)
    return metadata


def _enrich_predictions(
    *,
    rows: Sequence[_PredictionRow],
    bout_metadata: dict[str, tuple[str, datetime]],
) -> tuple[list[_EnrichedPredictionRow], int, int]:
    prelim: list[tuple[_PredictionRow, str]] = []
    dropped_missing_metadata = 0
    dropped_chronology_invalid = 0

    for row in rows:
        metadata = bout_metadata.get(row.bout_id)
        if metadata is None:
            dropped_missing_metadata += 1
            continue

        weight_class, bout_start = metadata
        feature_cutoff = _parse_utc(row.feature_cutoff_utc)
        predicted_bout_time = _parse_utc(row.bout_datetime_utc)

        if feature_cutoff > bout_start or feature_cutoff > predicted_bout_time:
            dropped_chronology_invalid += 1
            continue

        prelim.append((row, weight_class))

    probs_by_key: dict[tuple[str, str], float] = {}
    for row, _ in prelim:
        probs_by_key[(row.bout_id, row.fighter_id)] = row.predicted_probability

    enriched: list[_EnrichedPredictionRow] = []
    for row, weight_class in prelim:
        opponent_probability = probs_by_key.get((row.bout_id, row.opponent_id))
        favourite_status = _derive_favourite_status(
            predicted_probability=row.predicted_probability,
            opponent_probability=opponent_probability,
        )
        enriched.append(
            _EnrichedPredictionRow(
                model_run_id=row.model_run_id,
                model_name=row.model_name,
                bout_id=row.bout_id,
                fighter_id=row.fighter_id,
                opponent_id=row.opponent_id,
                bout_datetime_utc=row.bout_datetime_utc,
                feature_cutoff_utc=row.feature_cutoff_utc,
                predicted_probability=row.predicted_probability,
                label=row.label,
                weight_class=weight_class,
                favourite_status=favourite_status,
            )
        )

    return enriched, dropped_missing_metadata, dropped_chronology_invalid


def _derive_favourite_status(*, predicted_probability: float, opponent_probability: float | None) -> str:
    if opponent_probability is not None:
        if predicted_probability > opponent_probability:
            return "favourite"
        if predicted_probability < opponent_probability:
            return "underdog"
        return "pickem"

    if predicted_probability > 0.5:
        return "favourite"
    if predicted_probability < 0.5:
        return "underdog"
    return "pickem"


def _resolve_model_run_id(*, rows: Sequence[_PredictionRow], requested_run_id: str | None) -> str:
    if requested_run_id is not None:
        return requested_run_id

    run_ids = sorted({row.model_run_id for row in rows if row.model_run_id != ""})
    if len(run_ids) == 0:
        return "unknown"
    return run_ids[-1]


def _write_calibration_table(*, calibration_table_path: Path, bins: Sequence[CalibrationBinSummary]) -> None:
    with calibration_table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "bin_index",
                "lower_bound",
                "upper_bound",
                "count",
                "wins",
                "losses",
                "avg_predicted_probability",
                "observed_win_rate",
                "absolute_error",
            ],
        )
        writer.writeheader()
        for calibration_bin in bins:
            writer.writerow(
                {
                    "bin_index": calibration_bin.bin_index,
                    "lower_bound": f"{calibration_bin.lower_bound:.6f}",
                    "upper_bound": f"{calibration_bin.upper_bound:.6f}",
                    "count": calibration_bin.count,
                    "wins": calibration_bin.wins,
                    "losses": calibration_bin.losses,
                    "avg_predicted_probability": ""
                    if calibration_bin.avg_predicted_probability is None
                    else f"{calibration_bin.avg_predicted_probability:.8f}",
                    "observed_win_rate": ""
                    if calibration_bin.observed_win_rate is None
                    else f"{calibration_bin.observed_win_rate:.8f}",
                    "absolute_error": ""
                    if calibration_bin.absolute_error is None
                    else f"{calibration_bin.absolute_error:.8f}",
                }
            )


def _write_reliability_curve(*, reliability_curve_path: Path, bins: Sequence[CalibrationBinSummary]) -> None:
    with reliability_curve_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "point_type",
                "bin_index",
                "predicted_probability",
                "observed_win_rate",
                "count",
            ],
        )
        writer.writeheader()

        writer.writerow(
            {
                "point_type": "reference",
                "bin_index": "",
                "predicted_probability": "0.00000000",
                "observed_win_rate": "0.00000000",
                "count": "",
            }
        )
        writer.writerow(
            {
                "point_type": "reference",
                "bin_index": "",
                "predicted_probability": "1.00000000",
                "observed_win_rate": "1.00000000",
                "count": "",
            }
        )

        for calibration_bin in bins:
            if calibration_bin.count == 0:
                continue
            assert calibration_bin.avg_predicted_probability is not None
            assert calibration_bin.observed_win_rate is not None
            writer.writerow(
                {
                    "point_type": "model",
                    "bin_index": calibration_bin.bin_index,
                    "predicted_probability": f"{calibration_bin.avg_predicted_probability:.8f}",
                    "observed_win_rate": f"{calibration_bin.observed_win_rate:.8f}",
                    "count": calibration_bin.count,
                }
            )


def _write_probability_bucket_summary(
    *,
    probability_bucket_summary_path: Path,
    bins: Sequence[CalibrationBinSummary],
    total_count: int,
) -> None:
    with probability_bucket_summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "bucket_label",
                "count",
                "share_of_predictions",
                "wins",
                "losses",
                "win_rate",
                "avg_predicted_probability",
                "absolute_error",
            ],
        )
        writer.writeheader()
        for calibration_bin in bins:
            bucket_label = f"[{calibration_bin.lower_bound:.2f},{calibration_bin.upper_bound:.2f})"
            share = 0.0 if total_count == 0 else calibration_bin.count / float(total_count)
            writer.writerow(
                {
                    "bucket_label": bucket_label,
                    "count": calibration_bin.count,
                    "share_of_predictions": f"{share:.8f}",
                    "wins": calibration_bin.wins,
                    "losses": calibration_bin.losses,
                    "win_rate": ""
                    if calibration_bin.observed_win_rate is None
                    else f"{calibration_bin.observed_win_rate:.8f}",
                    "avg_predicted_probability": ""
                    if calibration_bin.avg_predicted_probability is None
                    else f"{calibration_bin.avg_predicted_probability:.8f}",
                    "absolute_error": ""
                    if calibration_bin.absolute_error is None
                    else f"{calibration_bin.absolute_error:.8f}",
                }
            )


def _write_performance_breakdowns(
    *,
    performance_breakdown_path: Path,
    rows: Sequence[PerformanceBreakdownRow],
) -> None:
    with performance_breakdown_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "segment_type",
                "segment_value",
                "count",
                "wins",
                "losses",
                "avg_predicted_probability",
                "observed_win_rate",
                "hit_rate",
                "log_loss",
                "brier_score",
                "calibration_ece",
                "calibration_mce",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "segment_type": row.segment_type,
                    "segment_value": row.segment_value,
                    "count": row.count,
                    "wins": row.wins,
                    "losses": row.losses,
                    "avg_predicted_probability": f"{row.avg_predicted_probability:.8f}",
                    "observed_win_rate": f"{row.observed_win_rate:.8f}",
                    "hit_rate": f"{row.hit_rate:.8f}",
                    "log_loss": f"{row.log_loss:.8f}",
                    "brier_score": f"{row.brier_score:.8f}",
                    "calibration_ece": f"{row.calibration_ece:.8f}",
                    "calibration_mce": f"{row.calibration_mce:.8f}",
                }
            )


def _write_report(
    *,
    report_output_path: Path,
    model_name: str,
    model_run_id: str,
    promotion: str,
    predictions_loaded: int,
    predictions_used: int,
    dropped_missing_metadata: int,
    dropped_chronology_invalid: int,
    dropped_invalid_label: int,
    dropped_missing_probability: int,
    calibration: CalibrationMetrics,
    aggregate_log_loss: float,
    aggregate_brier_score: float,
    aggregate_hit_rate: float,
    breakdown_rows: Sequence[PerformanceBreakdownRow],
    calibration_table_path: Path,
    reliability_curve_path: Path,
    probability_bucket_summary_path: Path,
    performance_breakdown_path: Path,
) -> None:
    report_output_path.parent.mkdir(parents=True, exist_ok=True)

    quality = _calibration_quality_label(calibration.expected_calibration_error)
    worst_bins = [row for row in calibration.bins if row.absolute_error is not None]
    worst_bins.sort(key=lambda row: float(row.absolute_error or 0.0), reverse=True)

    lines: list[str] = [
        "calibration_diagnostics_report",
        f"model_name={model_name}",
        f"model_run_id={model_run_id}",
        f"promotion={promotion}",
        "prediction_filtering",
        f"predictions_loaded={predictions_loaded}",
        f"predictions_used={predictions_used}",
        f"dropped_missing_metadata={dropped_missing_metadata}",
        f"dropped_chronology_invalid={dropped_chronology_invalid}",
        f"dropped_invalid_label={dropped_invalid_label}",
        f"dropped_missing_probability={dropped_missing_probability}",
        "aggregate_calibration_metrics",
        f"log_loss={aggregate_log_loss:.8f}",
        f"brier_score={aggregate_brier_score:.8f}",
        f"hit_rate={aggregate_hit_rate:.8f}",
        f"ece={calibration.expected_calibration_error:.8f}",
        f"mce={calibration.maximum_calibration_error:.8f}",
        f"calibration_quality={quality}",
        "artifact_paths",
        f"calibration_table={calibration_table_path.as_posix()}",
        f"reliability_curve={reliability_curve_path.as_posix()}",
        f"probability_bucket_summary={probability_bucket_summary_path.as_posix()}",
        f"performance_breakdowns={performance_breakdown_path.as_posix()}",
        "worst_bins_by_abs_error",
    ]

    for row in worst_bins[:3]:
        assert row.observed_win_rate is not None
        assert row.avg_predicted_probability is not None
        assert row.absolute_error is not None
        lines.append(
            f"bin={row.bin_index} range=[{row.lower_bound:.2f},{row.upper_bound:.2f}) "
            f"count={row.count} avg_pred={row.avg_predicted_probability:.8f} "
            f"observed={row.observed_win_rate:.8f} abs_error={row.absolute_error:.8f}"
        )

    lines.append("performance_breakdown_highlights")
    favourite_rows = [row for row in breakdown_rows if row.segment_type == "favourite_status"]
    for row in sorted(favourite_rows, key=lambda item: item.segment_value):
        lines.append(
            f"segment=favourite_status value={row.segment_value} count={row.count} "
            f"hit_rate={row.hit_rate:.8f} log_loss={row.log_loss:.8f} "
            f"brier={row.brier_score:.8f} ece={row.calibration_ece:.8f}"
        )

    weight_rows = [row for row in breakdown_rows if row.segment_type == "weight_class"]
    for row in sorted(weight_rows, key=lambda item: item.count, reverse=True)[:5]:
        lines.append(
            f"segment=weight_class value={row.segment_value} count={row.count} "
            f"hit_rate={row.hit_rate:.8f} log_loss={row.log_loss:.8f} "
            f"brier={row.brier_score:.8f} ece={row.calibration_ece:.8f}"
        )

    report_output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _calibration_quality_label(ece: float) -> str:
    if ece <= 0.03:
        return "strong"
    if ece <= 0.06:
        return "acceptable"
    return "weak"


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


def _hit_rate(*, labels: Sequence[float], probabilities: Sequence[float]) -> float:
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


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
