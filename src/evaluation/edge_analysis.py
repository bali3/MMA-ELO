"""Segmented model-vs-market performance and betting-edge analysis."""

from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from backtest import PlacedBet, apply_vig_adjustment, run_betting_backtest
from features import DEFAULT_KEY_FEATURES


MIN_CREDIBLE_PREDICTIONS = 100
MIN_CREDIBLE_BETS = 30
MIN_CREDIBLE_BOUTS = 50


@dataclass(frozen=True)
class SegmentComparisonRow:
    """Comparable model-vs-market metrics for one segment."""

    segment_type: str
    segment_value: str
    prediction_count: int
    bout_count: int
    model_bets_count: int
    baseline_bets_count: int
    credible: bool
    model_log_loss: float
    baseline_log_loss: float
    delta_log_loss: float
    model_brier_score: float
    baseline_brier_score: float
    delta_brier_score: float
    model_hit_rate: float
    baseline_hit_rate: float
    delta_hit_rate: float
    model_roi: float | None
    baseline_roi: float | None
    delta_roi: float | None
    model_drawdown: float | None
    baseline_drawdown: float | None
    delta_drawdown: float | None


@dataclass(frozen=True)
class SegmentedEdgeAnalysisResult:
    """Persisted report metadata and top segment summaries."""

    report_path: Path
    segment_metrics_path: Path
    target_model_name: str
    target_model_run_id: str
    baseline_model_name: str
    baseline_model_run_id: str
    segments_evaluated: int
    credible_segments: int
    best_log_loss_segment: str
    best_brier_segment: str
    best_roi_segment: str
    worst_roi_segment: str
    worst_drawdown_segment: str


@dataclass(frozen=True)
class _PredictionRow:
    model_name: str
    model_run_id: str
    bout_id: str
    event_id: str
    bout_datetime_utc: str
    fighter_id: str
    opponent_id: str
    label: float
    predicted_probability: float
    feature_cutoff_utc: str


@dataclass(frozen=True)
class _FeatureRow:
    completeness_ratio: float
    feature_completeness_bucket: str
    prior_fight_count: int | None
    fighter_experience_bucket: str
    days_since_last_fight: int | None
    layoff_bucket: str


@dataclass(frozen=True)
class _BoutMeta:
    year: str
    weight_class: str
    bout_datetime_utc: str
    bout_start: datetime


@dataclass(frozen=True)
class _MarketMeta:
    sportsbook: str
    implied_probability: float
    fair_probability: float
    market_timestamp_utc: str


@dataclass(frozen=True)
class _ComparisonRow:
    bout_id: str
    fighter_id: str
    bout_datetime_utc: str
    label: float
    target_probability: float
    baseline_probability: float
    year: str
    weight_class: str
    favourite_status: str
    feature_completeness_bucket: str
    fighter_experience_bucket: str
    layoff_bucket: str
    bookmaker: str
    edge_threshold_bucket: str


def run_segmented_edge_analysis(
    *,
    db_path: Path,
    features_path: Path,
    predictions_path: Path,
    report_output_path: Path,
    segment_metrics_output_path: Path,
    promotion: str,
    target_model_name: str,
    target_model_run_id: str | None,
    baseline_model_name: str,
    baseline_model_run_id: str | None,
    min_edge: float,
    flat_stake: float,
    kelly_fraction: float | None,
    initial_bankroll: float,
    vig_adjustment: str,
    one_bet_per_bout: bool,
    min_credible_predictions: int = MIN_CREDIBLE_PREDICTIONS,
    min_credible_bets: int = MIN_CREDIBLE_BETS,
    min_credible_bouts: int = MIN_CREDIBLE_BOUTS,
) -> SegmentedEdgeAnalysisResult:
    """Build segment-level quality and betting comparisons against the market baseline."""

    target_rows, resolved_target_run_id = _load_prediction_rows(
        predictions_path=predictions_path,
        model_name=target_model_name,
        model_run_id=target_model_run_id,
    )
    baseline_rows, resolved_baseline_run_id = _load_prediction_rows(
        predictions_path=predictions_path,
        model_name=baseline_model_name,
        model_run_id=baseline_model_run_id,
    )

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        bout_meta = _load_bout_meta(connection=connection, promotion=promotion)
        market_meta = _load_market_meta(connection=connection, promotion=promotion, vig_adjustment=vig_adjustment)

    feature_rows = _load_feature_rows(features_path=features_path, key_features=DEFAULT_KEY_FEATURES)
    comparison_rows = _build_comparison_rows(
        target_rows=target_rows,
        baseline_rows=baseline_rows,
        bout_meta=bout_meta,
        feature_rows=feature_rows,
        market_meta=market_meta,
    )
    if len(comparison_rows) == 0:
        raise ValueError("No matched chronology-safe target/baseline prediction rows were available.")

    model_backtest_tmp = report_output_path.parent / "_tmp_model_backtest.txt"
    baseline_backtest_tmp = report_output_path.parent / "_tmp_market_baseline_backtest.txt"
    try:
        model_backtest = run_betting_backtest(
            db_path=db_path,
            predictions_path=predictions_path,
            report_output_path=model_backtest_tmp,
            promotion=promotion,
            model_name=target_model_name,
            model_run_id=resolved_target_run_id,
            min_edge=min_edge,
            flat_stake=flat_stake,
            kelly_fraction=kelly_fraction,
            initial_bankroll=initial_bankroll,
            vig_adjustment=vig_adjustment,
            one_bet_per_bout=one_bet_per_bout,
        )
        baseline_backtest = run_betting_backtest(
            db_path=db_path,
            predictions_path=predictions_path,
            report_output_path=baseline_backtest_tmp,
            promotion=promotion,
            model_name=baseline_model_name,
            model_run_id=resolved_baseline_run_id,
            min_edge=min_edge,
            flat_stake=flat_stake,
            kelly_fraction=kelly_fraction,
            initial_bankroll=initial_bankroll,
            vig_adjustment=vig_adjustment,
            one_bet_per_bout=one_bet_per_bout,
        )
    finally:
        for temporary_path in (model_backtest_tmp, baseline_backtest_tmp):
            if temporary_path.exists():
                temporary_path.unlink()

    model_bets = {(bet.bout_id, bet.fighter_id): bet for bet in model_backtest.bets}
    baseline_bets = {(bet.bout_id, bet.fighter_id): bet for bet in baseline_backtest.bets}
    segment_rows = _build_segment_rows(
        comparison_rows=comparison_rows,
        model_bets=model_bets,
        baseline_bets=baseline_bets,
        min_credible_predictions=min_credible_predictions,
        min_credible_bets=min_credible_bets,
        min_credible_bouts=min_credible_bouts,
    )

    segment_metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_segment_metrics_csv(output_path=segment_metrics_output_path, rows=segment_rows)

    credible_rows = [row for row in segment_rows if row.credible and row.segment_type != "overall"]
    ranking_rows = credible_rows if len(credible_rows) > 0 else [row for row in segment_rows if row.segment_type != "overall"]
    if len(ranking_rows) == 0:
        ranking_rows = list(segment_rows)

    best_log_loss = min(ranking_rows, key=lambda row: row.model_log_loss)
    best_brier = min(ranking_rows, key=lambda row: row.model_brier_score)
    best_roi = max(ranking_rows, key=lambda row: float("-inf") if row.model_roi is None else row.model_roi)
    worst_roi = min(ranking_rows, key=lambda row: float("inf") if row.model_roi is None else row.model_roi)
    worst_drawdown = max(
        ranking_rows,
        key=lambda row: float("-inf") if row.model_drawdown is None else row.model_drawdown,
    )

    _write_report(
        report_output_path=report_output_path,
        target_model_name=target_model_name,
        target_model_run_id=resolved_target_run_id,
        baseline_model_name=baseline_model_name,
        baseline_model_run_id=resolved_baseline_run_id,
        segment_metrics_path=segment_metrics_output_path,
        rows=segment_rows,
        best_log_loss=best_log_loss,
        best_brier=best_brier,
        best_roi=best_roi,
        worst_roi=worst_roi,
        worst_drawdown=worst_drawdown,
    )

    return SegmentedEdgeAnalysisResult(
        report_path=report_output_path,
        segment_metrics_path=segment_metrics_output_path,
        target_model_name=target_model_name,
        target_model_run_id=resolved_target_run_id,
        baseline_model_name=baseline_model_name,
        baseline_model_run_id=resolved_baseline_run_id,
        segments_evaluated=len(segment_rows),
        credible_segments=sum(1 for row in segment_rows if row.credible),
        best_log_loss_segment=_segment_label(best_log_loss),
        best_brier_segment=_segment_label(best_brier),
        best_roi_segment=_segment_label(best_roi),
        worst_roi_segment=_segment_label(worst_roi),
        worst_drawdown_segment=_segment_label(worst_drawdown),
    )


def _load_prediction_rows(
    *,
    predictions_path: Path,
    model_name: str,
    model_run_id: str | None,
) -> tuple[list[_PredictionRow], str]:
    rows: list[_PredictionRow] = []
    with predictions_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            if str(raw.get("model_name", "")).strip() != model_name:
                continue
            if str(raw.get("split", "test")).strip() != "test":
                continue

            raw_run_id = str(raw.get("model_run_id", "")).strip()
            if model_run_id is not None and raw_run_id != model_run_id:
                continue

            label = _as_optional_float(raw.get("label"))
            probability = _as_optional_float(raw.get("predicted_probability"))
            if label not in {0.0, 1.0} or probability is None:
                continue

            bout_id = str(raw.get("bout_id", "")).strip()
            fighter_id = str(raw.get("fighter_id", "")).strip()
            opponent_id = str(raw.get("opponent_id", "")).strip()
            event_id = str(raw.get("event_id", "")).strip()
            bout_datetime_utc = str(raw.get("bout_datetime_utc", "")).strip()
            feature_cutoff_utc = str(raw.get("feature_cutoff_utc", "")).strip()
            if "" in {bout_id, fighter_id, opponent_id, event_id, bout_datetime_utc, feature_cutoff_utc}:
                continue

            rows.append(
                _PredictionRow(
                    model_name=model_name,
                    model_run_id=raw_run_id,
                    bout_id=bout_id,
                    event_id=event_id,
                    bout_datetime_utc=bout_datetime_utc,
                    fighter_id=fighter_id,
                    opponent_id=opponent_id,
                    label=label,
                    predicted_probability=_clamp_probability(probability),
                    feature_cutoff_utc=feature_cutoff_utc,
                )
            )

    if len(rows) == 0:
        requested = "" if model_run_id is None else f" for model_run_id={model_run_id}"
        raise ValueError(f"No prediction rows found for model={model_name}{requested}.")

    resolved_run_id = model_run_id or sorted({row.model_run_id for row in rows if row.model_run_id != ""})[-1]
    filtered_rows = [row for row in rows if row.model_run_id == resolved_run_id]
    if len(filtered_rows) == 0:
        raise ValueError(f"No prediction rows found for resolved model_run_id={resolved_run_id}.")
    return filtered_rows, resolved_run_id


def _load_feature_rows(
    *,
    features_path: Path,
    key_features: Sequence[str],
) -> dict[tuple[str, str], _FeatureRow]:
    rows: dict[tuple[str, str], _FeatureRow] = {}
    with features_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            bout_id = str(raw.get("bout_id", "")).strip()
            fighter_id = str(raw.get("fighter_id", "")).strip()
            if bout_id == "" or fighter_id == "":
                continue

            available = 0
            for feature_name in key_features:
                if str(raw.get(feature_name, "")).strip() != "":
                    available += 1
            completeness_ratio = available / float(len(key_features))

            prior_count = _as_optional_int(raw.get("prior_fight_count"))
            layoff_days = _as_optional_int(raw.get("days_since_last_fight"))
            rows[(bout_id, fighter_id)] = _FeatureRow(
                completeness_ratio=completeness_ratio,
                feature_completeness_bucket=_feature_completeness_bucket(completeness_ratio),
                prior_fight_count=prior_count,
                fighter_experience_bucket=_fighter_experience_bucket(prior_count),
                days_since_last_fight=layoff_days,
                layoff_bucket=_layoff_bucket(layoff_days),
            )
    return rows


def _load_bout_meta(
    *,
    connection: sqlite3.Connection,
    promotion: str,
) -> dict[str, _BoutMeta]:
    rows = connection.execute(
        """
        SELECT
            b.bout_id,
            COALESCE(NULLIF(TRIM(b.weight_class), ''), 'UNKNOWN') AS weight_class,
            COALESCE(b.bout_start_time_utc, e.event_date_utc) AS bout_datetime_utc
        FROM bouts AS b
        JOIN events AS e ON e.event_id = b.event_id
        WHERE UPPER(e.promotion) = UPPER(?)
          AND COALESCE(b.bout_start_time_utc, e.event_date_utc) IS NOT NULL
        """,
        (promotion,),
    ).fetchall()

    meta: dict[str, _BoutMeta] = {}
    for row in rows:
        bout_datetime_utc = str(row["bout_datetime_utc"]).strip()
        meta[str(row["bout_id"]).strip()] = _BoutMeta(
            year=bout_datetime_utc[:4],
            weight_class=str(row["weight_class"]).strip(),
            bout_datetime_utc=bout_datetime_utc,
            bout_start=_parse_utc(bout_datetime_utc),
        )
    return meta


def _load_market_meta(
    *,
    connection: sqlite3.Connection,
    promotion: str,
    vig_adjustment: str,
) -> dict[tuple[str, str], _MarketMeta]:
    rows = connection.execute(
        """
        SELECT
            b.bout_id,
            m.selection_fighter_id,
            m.sportsbook,
            m.implied_probability,
            m.market_timestamp_utc,
            m.market_id
        FROM markets AS m
        JOIN bouts AS b ON b.bout_id = m.bout_id
        JOIN events AS e ON e.event_id = b.event_id
        WHERE UPPER(e.promotion) = UPPER(?)
          AND m.selection_fighter_id IS NOT NULL
          AND m.implied_probability IS NOT NULL
        ORDER BY
            b.bout_id ASC,
            m.selection_fighter_id ASC,
            m.market_timestamp_utc ASC,
            m.market_id ASC
        """,
        (promotion,),
    ).fetchall()

    latest_by_key: dict[tuple[str, str], tuple[str, float, str]] = {}
    raw_by_bout: dict[str, dict[str, float]] = {}
    for row in rows:
        bout_id = str(row["bout_id"]).strip()
        fighter_id = str(row["selection_fighter_id"]).strip()
        sportsbook = str(row["sportsbook"]).strip() or "missing"
        implied = _clamp_probability(float(row["implied_probability"]))
        timestamp = str(row["market_timestamp_utc"]).strip()
        key = (bout_id, fighter_id)
        existing = latest_by_key.get(key)
        if existing is None or timestamp > existing[2]:
            latest_by_key[key] = (sportsbook, implied, timestamp)
        raw_by_bout.setdefault(bout_id, {})[fighter_id] = latest_by_key[key][1]

    fair_by_bout = {
        bout_id: apply_vig_adjustment(implied_probabilities=probs, method=vig_adjustment)
        for bout_id, probs in raw_by_bout.items()
    }

    output: dict[tuple[str, str], _MarketMeta] = {}
    for (bout_id, fighter_id), (sportsbook, implied, timestamp) in latest_by_key.items():
        fair_probability = fair_by_bout.get(bout_id, {}).get(fighter_id)
        if fair_probability is None:
            continue
        output[(bout_id, fighter_id)] = _MarketMeta(
            sportsbook=sportsbook,
            implied_probability=implied,
            fair_probability=_clamp_probability(fair_probability),
            market_timestamp_utc=timestamp,
        )
    return output


def _build_comparison_rows(
    *,
    target_rows: Sequence[_PredictionRow],
    baseline_rows: Sequence[_PredictionRow],
    bout_meta: dict[str, _BoutMeta],
    feature_rows: dict[tuple[str, str], _FeatureRow],
    market_meta: dict[tuple[str, str], _MarketMeta],
) -> list[_ComparisonRow]:
    target_by_key = {(row.bout_id, row.fighter_id): row for row in target_rows}
    baseline_by_key = {(row.bout_id, row.fighter_id): row for row in baseline_rows}
    baseline_probs = {(row.bout_id, row.fighter_id): row.predicted_probability for row in baseline_rows}

    matched_keys = sorted(set(target_by_key) & set(baseline_by_key))
    output: list[_ComparisonRow] = []
    for key in matched_keys:
        target = target_by_key[key]
        baseline = baseline_by_key[key]
        meta = bout_meta.get(target.bout_id)
        if meta is None:
            continue

        target_cutoff = _parse_utc(target.feature_cutoff_utc)
        baseline_cutoff = _parse_utc(baseline.feature_cutoff_utc)
        target_bout_time = _parse_utc(target.bout_datetime_utc)
        baseline_bout_time = _parse_utc(baseline.bout_datetime_utc)
        if (
            target_cutoff > meta.bout_start
            or baseline_cutoff > meta.bout_start
            or target_cutoff > target_bout_time
            or baseline_cutoff > baseline_bout_time
        ):
            continue

        feature = feature_rows.get(
            key,
            _FeatureRow(
                completeness_ratio=0.0,
                feature_completeness_bucket="missing_row",
                prior_fight_count=None,
                fighter_experience_bucket="unknown",
                days_since_last_fight=None,
                layoff_bucket="missing",
            ),
        )
        market = market_meta.get(key)
        opponent_baseline_probability = baseline_probs.get((baseline.bout_id, baseline.opponent_id))
        favourite_status = _favourite_status(
            predicted_probability=baseline.predicted_probability,
            opponent_probability=opponent_baseline_probability,
        )
        edge_threshold_bucket = "missing"
        bookmaker = "missing"
        if market is not None and market.market_timestamp_utc <= target.feature_cutoff_utc:
            bookmaker = market.sportsbook
            edge = target.predicted_probability - market.fair_probability
            edge_threshold_bucket = _edge_threshold_bucket(edge)

        output.append(
            _ComparisonRow(
                bout_id=target.bout_id,
                fighter_id=target.fighter_id,
                bout_datetime_utc=target.bout_datetime_utc,
                label=target.label,
                target_probability=target.predicted_probability,
                baseline_probability=baseline.predicted_probability,
                year=meta.year,
                weight_class=meta.weight_class,
                favourite_status=favourite_status,
                feature_completeness_bucket=feature.feature_completeness_bucket,
                fighter_experience_bucket=feature.fighter_experience_bucket,
                layoff_bucket=feature.layoff_bucket,
                bookmaker=bookmaker,
                edge_threshold_bucket=edge_threshold_bucket,
            )
        )

    return output


def _build_segment_rows(
    *,
    comparison_rows: Sequence[_ComparisonRow],
    model_bets: dict[tuple[str, str], PlacedBet],
    baseline_bets: dict[tuple[str, str], PlacedBet],
    min_credible_predictions: int,
    min_credible_bets: int,
    min_credible_bouts: int,
) -> list[SegmentComparisonRow]:
    segments: dict[tuple[str, str], list[_ComparisonRow]] = {}
    for row in comparison_rows:
        _append_segment(segments, "overall", "all", row)
        _append_segment(segments, "year", row.year, row)
        _append_segment(segments, "weight_class", row.weight_class, row)
        _append_segment(segments, "favourite_vs_underdog", row.favourite_status, row)
        _append_segment(segments, "feature_completeness_bucket", row.feature_completeness_bucket, row)
        _append_segment(segments, "fighter_experience_bucket", row.fighter_experience_bucket, row)
        _append_segment(segments, "layoff_bucket", row.layoff_bucket, row)
        _append_segment(segments, "bookmaker", row.bookmaker, row)
        _append_segment(segments, "edge_threshold_bucket", row.edge_threshold_bucket, row)

    output: list[SegmentComparisonRow] = []
    for (segment_type, segment_value), rows in segments.items():
        labels = [row.label for row in rows]
        target_probs = [row.target_probability for row in rows]
        baseline_probs = [row.baseline_probability for row in rows]
        unique_bouts = {row.bout_id for row in rows}
        keys = [(row.bout_id, row.fighter_id) for row in rows]
        segment_model_bets = _ordered_bets(model_bets, keys)
        segment_baseline_bets = _ordered_bets(baseline_bets, keys)
        model_betting = _betting_metrics(segment_model_bets)
        baseline_betting = _betting_metrics(segment_baseline_bets)

        credible = (
            len(rows) >= min_credible_predictions
            and len(unique_bouts) >= min_credible_bouts
            and max(len(segment_model_bets), len(segment_baseline_bets)) >= min_credible_bets
        )
        output.append(
            SegmentComparisonRow(
                segment_type=segment_type,
                segment_value=segment_value,
                prediction_count=len(rows),
                bout_count=len(unique_bouts),
                model_bets_count=len(segment_model_bets),
                baseline_bets_count=len(segment_baseline_bets),
                credible=credible,
                model_log_loss=_log_loss(labels=labels, probabilities=target_probs),
                baseline_log_loss=_log_loss(labels=labels, probabilities=baseline_probs),
                delta_log_loss=_log_loss(labels=labels, probabilities=baseline_probs)
                - _log_loss(labels=labels, probabilities=target_probs),
                model_brier_score=_brier_score(labels=labels, probabilities=target_probs),
                baseline_brier_score=_brier_score(labels=labels, probabilities=baseline_probs),
                delta_brier_score=_brier_score(labels=labels, probabilities=baseline_probs)
                - _brier_score(labels=labels, probabilities=target_probs),
                model_hit_rate=_hit_rate(labels=labels, probabilities=target_probs),
                baseline_hit_rate=_hit_rate(labels=labels, probabilities=baseline_probs),
                delta_hit_rate=_hit_rate(labels=labels, probabilities=target_probs)
                - _hit_rate(labels=labels, probabilities=baseline_probs),
                model_roi=model_betting["roi"],
                baseline_roi=baseline_betting["roi"],
                delta_roi=_subtract_optional(model_betting["roi"], baseline_betting["roi"]),
                model_drawdown=model_betting["max_drawdown"],
                baseline_drawdown=baseline_betting["max_drawdown"],
                delta_drawdown=_subtract_optional(baseline_betting["max_drawdown"], model_betting["max_drawdown"]),
            )
        )

    output.sort(key=lambda row: (row.segment_type, row.segment_value))
    return output


def _append_segment(
    segments: dict[tuple[str, str], list[_ComparisonRow]],
    segment_type: str,
    segment_value: str,
    row: _ComparisonRow,
) -> None:
    segments.setdefault((segment_type, segment_value), []).append(row)


def _ordered_bets(
    bet_map: dict[tuple[str, str], PlacedBet],
    keys: Iterable[tuple[str, str]],
) -> list[PlacedBet]:
    bets = [bet_map[key] for key in keys if key in bet_map]
    bets.sort(key=lambda row: (_parse_utc(row.bout_datetime_utc), row.bout_id, row.fighter_id))
    return bets


def _betting_metrics(bets: Sequence[PlacedBet]) -> dict[str, float | None]:
    if len(bets) == 0:
        return {"roi": None, "max_drawdown": None}

    total_staked = sum(bet.stake for bet in bets)
    total_pnl = sum(bet.pnl for bet in bets)
    roi = 0.0 if total_staked == 0.0 else total_pnl / total_staked

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for bet in bets:
        equity += bet.pnl
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    return {"roi": roi, "max_drawdown": max_drawdown}


def _write_segment_metrics_csv(*, output_path: Path, rows: Sequence[SegmentComparisonRow]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "segment_type",
                "segment_value",
                "prediction_count",
                "bout_count",
                "model_bets_count",
                "baseline_bets_count",
                "credible",
                "model_log_loss",
                "baseline_log_loss",
                "delta_log_loss",
                "model_brier_score",
                "baseline_brier_score",
                "delta_brier_score",
                "model_hit_rate",
                "baseline_hit_rate",
                "delta_hit_rate",
                "model_roi",
                "baseline_roi",
                "delta_roi",
                "model_drawdown",
                "baseline_drawdown",
                "delta_drawdown",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "segment_type": row.segment_type,
                    "segment_value": row.segment_value,
                    "prediction_count": row.prediction_count,
                    "bout_count": row.bout_count,
                    "model_bets_count": row.model_bets_count,
                    "baseline_bets_count": row.baseline_bets_count,
                    "credible": int(row.credible),
                    "model_log_loss": f"{row.model_log_loss:.8f}",
                    "baseline_log_loss": f"{row.baseline_log_loss:.8f}",
                    "delta_log_loss": f"{row.delta_log_loss:.8f}",
                    "model_brier_score": f"{row.model_brier_score:.8f}",
                    "baseline_brier_score": f"{row.baseline_brier_score:.8f}",
                    "delta_brier_score": f"{row.delta_brier_score:.8f}",
                    "model_hit_rate": f"{row.model_hit_rate:.8f}",
                    "baseline_hit_rate": f"{row.baseline_hit_rate:.8f}",
                    "delta_hit_rate": f"{row.delta_hit_rate:.8f}",
                    "model_roi": _format_optional(row.model_roi),
                    "baseline_roi": _format_optional(row.baseline_roi),
                    "delta_roi": _format_optional(row.delta_roi),
                    "model_drawdown": _format_optional(row.model_drawdown),
                    "baseline_drawdown": _format_optional(row.baseline_drawdown),
                    "delta_drawdown": _format_optional(row.delta_drawdown),
                }
            )


def _write_report(
    *,
    report_output_path: Path,
    target_model_name: str,
    target_model_run_id: str,
    baseline_model_name: str,
    baseline_model_run_id: str,
    segment_metrics_path: Path,
    rows: Sequence[SegmentComparisonRow],
    best_log_loss: SegmentComparisonRow,
    best_brier: SegmentComparisonRow,
    best_roi: SegmentComparisonRow,
    worst_roi: SegmentComparisonRow,
    worst_drawdown: SegmentComparisonRow,
) -> None:
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    credible_rows = [row for row in rows if row.credible]
    lines = [
        "segmented_edge_analysis_report",
        f"target_model_name={target_model_name}",
        f"target_model_run_id={target_model_run_id}",
        f"baseline_model_name={baseline_model_name}",
        f"baseline_model_run_id={baseline_model_run_id}",
        f"segment_metrics_path={segment_metrics_path.as_posix()}",
        "summary",
        f"segments_evaluated={len(rows)}",
        f"credible_segments={len(credible_rows)}",
        f"best_log_loss_segment={_segment_label(best_log_loss)} model_log_loss={best_log_loss.model_log_loss:.8f} baseline_log_loss={best_log_loss.baseline_log_loss:.8f}",
        f"best_brier_segment={_segment_label(best_brier)} model_brier_score={best_brier.model_brier_score:.8f} baseline_brier_score={best_brier.baseline_brier_score:.8f}",
        f"best_roi_segment={_segment_label(best_roi)} model_roi={_format_optional(best_roi.model_roi)} baseline_roi={_format_optional(best_roi.baseline_roi)}",
        f"worst_roi_segment={_segment_label(worst_roi)} model_roi={_format_optional(worst_roi.model_roi)} baseline_roi={_format_optional(worst_roi.baseline_roi)}",
        f"worst_drawdown_segment={_segment_label(worst_drawdown)} model_drawdown={_format_optional(worst_drawdown.model_drawdown)} baseline_drawdown={_format_optional(worst_drawdown.baseline_drawdown)}",
        "credible_segments",
    ]
    if len(credible_rows) == 0:
        lines.append("none")
    else:
        for row in credible_rows:
            lines.append(
                "segment "
                f"type={row.segment_type} value={row.segment_value} predictions={row.prediction_count} bouts={row.bout_count} "
                f"model_bets={row.model_bets_count} baseline_bets={row.baseline_bets_count} "
                f"model_log_loss={row.model_log_loss:.8f} baseline_log_loss={row.baseline_log_loss:.8f} "
                f"model_brier={row.model_brier_score:.8f} baseline_brier={row.baseline_brier_score:.8f} "
                f"model_roi={_format_optional(row.model_roi)} baseline_roi={_format_optional(row.baseline_roi)} "
                f"model_drawdown={_format_optional(row.model_drawdown)} baseline_drawdown={_format_optional(row.baseline_drawdown)}"
            )

    lines.append("segment_comparisons")
    for row in rows:
        lines.append(
            "segment "
            f"type={row.segment_type} value={row.segment_value} credible={int(row.credible)} "
            f"predictions={row.prediction_count} bouts={row.bout_count} "
            f"model_bets={row.model_bets_count} baseline_bets={row.baseline_bets_count} "
            f"model_log_loss={row.model_log_loss:.8f} baseline_log_loss={row.baseline_log_loss:.8f} delta_log_loss={row.delta_log_loss:.8f} "
            f"model_brier={row.model_brier_score:.8f} baseline_brier={row.baseline_brier_score:.8f} delta_brier={row.delta_brier_score:.8f} "
            f"model_roi={_format_optional(row.model_roi)} baseline_roi={_format_optional(row.baseline_roi)} delta_roi={_format_optional(row.delta_roi)} "
            f"model_drawdown={_format_optional(row.model_drawdown)} baseline_drawdown={_format_optional(row.baseline_drawdown)} delta_drawdown={_format_optional(row.delta_drawdown)}"
        )

    report_output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _segment_label(row: SegmentComparisonRow) -> str:
    return f"{row.segment_type}={row.segment_value}"


def _feature_completeness_bucket(ratio: float) -> str:
    if ratio >= 0.95:
        return "complete_95_plus"
    if ratio >= 0.80:
        return "strong_80_94"
    if ratio >= 0.60:
        return "partial_60_79"
    return "thin_below_60"


def _fighter_experience_bucket(prior_fight_count: int | None) -> str:
    if prior_fight_count is None:
        return "unknown"
    if prior_fight_count <= 0:
        return "debut"
    if prior_fight_count <= 3:
        return "1_to_3"
    if prior_fight_count <= 8:
        return "4_to_8"
    return "9_plus"


def _layoff_bucket(days_since_last_fight: int | None) -> str:
    if days_since_last_fight is None:
        return "missing"
    if days_since_last_fight <= 60:
        return "0_to_60_days"
    if days_since_last_fight <= 180:
        return "61_to_180_days"
    if days_since_last_fight <= 365:
        return "181_to_365_days"
    return "366_plus_days"


def _edge_threshold_bucket(edge: float) -> str:
    if edge < 0.0:
        return "negative"
    if edge < 0.02:
        return "0.00_to_0.02"
    if edge < 0.05:
        return "0.02_to_0.05"
    if edge < 0.10:
        return "0.05_to_0.10"
    if edge < 0.15:
        return "0.10_to_0.15"
    return "0.15_plus"


def _favourite_status(*, predicted_probability: float, opponent_probability: float | None) -> str:
    if opponent_probability is None:
        if predicted_probability > 0.5:
            return "favourite"
        if predicted_probability < 0.5:
            return "underdog"
        return "pickem"
    if predicted_probability > opponent_probability:
        return "favourite"
    if predicted_probability < opponent_probability:
        return "underdog"
    return "pickem"


def _log_loss(*, labels: Sequence[float], probabilities: Sequence[float]) -> float:
    eps = 1e-15
    total = 0.0
    for label, probability in zip(labels, probabilities):
        clipped = min(max(probability, eps), 1.0 - eps)
        total += -(label * _safe_log(clipped) + (1.0 - label) * _safe_log(1.0 - clipped))
    return total / float(len(labels))


def _brier_score(*, labels: Sequence[float], probabilities: Sequence[float]) -> float:
    return sum((label - probability) ** 2 for label, probability in zip(labels, probabilities)) / float(len(labels))


def _hit_rate(*, labels: Sequence[float], probabilities: Sequence[float]) -> float:
    hits = 0
    for label, probability in zip(labels, probabilities):
        prediction = 1.0 if probability >= 0.5 else 0.0
        if prediction == label:
            hits += 1
    return hits / float(len(labels))


def _safe_log(value: float) -> float:
    import math

    return math.log(value)


def _clamp_probability(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _parse_utc(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_optional(value: float | None) -> str:
    return "" if value is None else f"{value:.8f}"


def _subtract_optional(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _as_optional_float(value: object) -> float | None:
    text = str(value or "").strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _as_optional_int(value: object) -> int | None:
    raw = _as_optional_float(value)
    if raw is None:
        return None
    return int(raw)
