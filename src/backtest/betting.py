"""Chronology-safe betting simulation and edge backtesting."""

from __future__ import annotations

import csv
import itertools
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal, Sequence

VigAdjustment = Literal["none", "proportional"]


@dataclass(frozen=True)
class PredictionRow:
    """Single prediction record loaded from a model prediction CSV."""

    model_run_id: str
    model_name: str
    bout_id: str
    event_id: str
    bout_datetime_utc: str
    fighter_id: str
    opponent_id: str
    label: float | None
    predicted_probability: float
    feature_cutoff_utc: str


@dataclass(frozen=True)
class MarketSnapshot:
    """Latest pre-fight market snapshot for a fighter-side selection."""

    bout_id: str
    fighter_id: str
    implied_probability: float
    decimal_odds: float
    market_timestamp_utc: str
    bout_datetime_utc: str
    market_liquidity: float | None = None


@dataclass(frozen=True)
class BetCandidate:
    """Candidate bet derived from model and market comparison."""

    bout_id: str
    event_id: str
    bout_datetime_utc: str
    fighter_id: str
    opponent_id: str
    label: float
    model_probability: float
    market_probability_raw: float
    market_probability_fair: float
    decimal_odds: float
    edge: float
    confidence: float = 0.0
    fighter_pre_bout_count: int | None = None
    opponent_pre_bout_count: int | None = None
    market_liquidity: float | None = None


@dataclass(frozen=True)
class PlacedBet:
    """Executed simulated bet with stake and realized PnL."""

    bout_id: str
    fighter_id: str
    bout_datetime_utc: str
    label: float
    edge: float
    model_probability: float
    market_probability_raw: float
    market_probability_fair: float
    decimal_odds: float
    stake: float
    pnl: float
    event_id: str = ""


@dataclass(frozen=True)
class DecisionRules:
    """Configurable guardrails for bet selection and stake/exposure control."""

    min_edge: float = 0.02
    min_confidence: float = 0.0
    min_pre_bout_fights: int = 0
    require_pre_bout_data: bool = False
    max_stake_per_bet: float | None = None
    max_event_exposure: float | None = None
    max_day_exposure: float | None = None
    min_liquidity: float | None = None
    require_liquidity: bool = False


@dataclass(frozen=True)
class BettingBacktestResult:
    """Summary output for a complete betting simulation run."""

    model_name: str
    model_run_id: str
    predictions_considered: int
    candidates_count: int
    bets_count: int
    total_staked: float
    total_pnl: float
    roi: float
    max_drawdown: float
    average_edge: float
    average_model_probability: float
    average_market_probability_raw: float
    average_market_probability_fair: float
    vig_adjustment: VigAdjustment
    min_edge: float
    flat_stake: float
    kelly_fraction: float | None
    report_path: Path
    bets: tuple[PlacedBet, ...]
    decision_rules: DecisionRules


@dataclass(frozen=True)
class BacktestSensitivityRow:
    """Single scenario output for rule-sensitivity analysis."""

    min_edge: float
    min_confidence: float
    min_pre_bout_fights: int
    max_stake_per_bet: float | None
    max_event_exposure: float | None
    max_day_exposure: float | None
    min_liquidity: float | None
    bets_count: int
    total_staked: float
    roi: float
    max_drawdown: float


@dataclass(frozen=True)
class BacktestSensitivityResult:
    """Sensitivity-analysis summary output."""

    model_name: str
    model_run_id: str
    scenarios: tuple[BacktestSensitivityRow, ...]
    report_path: Path


VigAdjuster = Callable[[dict[str, float]], dict[str, float]]
KellySizer = Callable[[BetCandidate, float, float], float]


def run_betting_backtest(
    *,
    db_path: Path,
    predictions_path: Path,
    report_output_path: Path,
    promotion: str,
    model_name: str,
    model_run_id: str | None = None,
    min_edge: float = 0.02,
    min_confidence: float = 0.0,
    min_pre_bout_fights: int = 0,
    require_pre_bout_data: bool = False,
    max_stake_per_bet: float | None = None,
    max_event_exposure: float | None = None,
    max_day_exposure: float | None = None,
    min_liquidity: float | None = None,
    require_liquidity: bool = False,
    flat_stake: float = 1.0,
    kelly_fraction: float | None = None,
    initial_bankroll: float = 100.0,
    vig_adjustment: VigAdjustment = "proportional",
    one_bet_per_bout: bool = True,
    decision_rules: DecisionRules | None = None,
    vig_adjuster: VigAdjuster | None = None,
    kelly_sizer: KellySizer | None = None,
) -> BettingBacktestResult:
    """Run a chronology-safe betting simulation from precomputed model outputs."""

    if flat_stake <= 0.0:
        raise ValueError("flat_stake must be positive.")
    if initial_bankroll <= 0.0:
        raise ValueError("initial_bankroll must be positive.")
    if kelly_fraction is not None and (kelly_fraction < 0.0 or kelly_fraction > 1.0):
        raise ValueError("kelly_fraction must be between 0 and 1 when provided.")
    if vig_adjustment not in {"none", "proportional"}:
        raise ValueError("vig_adjustment must be either 'none' or 'proportional'.")

    rules = _resolve_decision_rules(
        decision_rules=decision_rules,
        min_edge=min_edge,
        min_confidence=min_confidence,
        min_pre_bout_fights=min_pre_bout_fights,
        require_pre_bout_data=require_pre_bout_data,
        max_stake_per_bet=max_stake_per_bet,
        max_event_exposure=max_event_exposure,
        max_day_exposure=max_day_exposure,
        min_liquidity=min_liquidity,
        require_liquidity=require_liquidity,
    )

    rows = _load_predictions(
        predictions_path=predictions_path,
        model_name=model_name,
        model_run_id=model_run_id,
    )
    resolved_run_id = _resolve_model_run_id(rows=rows, requested_run_id=model_run_id)
    if model_run_id is None and resolved_run_id != "unknown":
        rows = [row for row in rows if row.model_run_id == resolved_run_id]
        if len(rows) == 0:
            raise ValueError(f"No prediction rows found for inferred model_run_id={resolved_run_id}.")

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        market_by_key = _load_market_snapshots(connection=connection, promotion=promotion)
        pre_bout_counts_by_key = _load_pre_bout_fight_counts(connection=connection, promotion=promotion)

    candidates = build_bet_candidates(
        predictions=rows,
        market_by_key=market_by_key,
        pre_bout_counts_by_key=pre_bout_counts_by_key,
        vig_adjustment=vig_adjustment,
        vig_adjuster=vig_adjuster,
    )

    selected = select_bets(
        candidates=candidates,
        min_edge=rules.min_edge,
        one_bet_per_bout=one_bet_per_bout,
        decision_rules=rules,
    )

    placed_bets = _simulate_bets(
        candidates=selected,
        flat_stake=flat_stake,
        initial_bankroll=initial_bankroll,
        kelly_fraction=kelly_fraction,
        kelly_sizer=kelly_sizer,
        decision_rules=rules,
    )

    metrics = _compute_metrics(placed_bets)
    _write_report(
        output_path=report_output_path,
        model_name=model_name,
        model_run_id=resolved_run_id,
        vig_adjustment=vig_adjustment,
        decision_rules=rules,
        flat_stake=flat_stake,
        kelly_fraction=kelly_fraction,
        predictions_considered=len(rows),
        candidates_count=len(candidates),
        bets=placed_bets,
        metrics=metrics,
    )

    return BettingBacktestResult(
        model_name=model_name,
        model_run_id=resolved_run_id,
        predictions_considered=len(rows),
        candidates_count=len(candidates),
        bets_count=len(placed_bets),
        total_staked=metrics["total_staked"],
        total_pnl=metrics["total_pnl"],
        roi=metrics["roi"],
        max_drawdown=metrics["max_drawdown"],
        average_edge=metrics["average_edge"],
        average_model_probability=metrics["average_model_probability"],
        average_market_probability_raw=metrics["average_market_probability_raw"],
        average_market_probability_fair=metrics["average_market_probability_fair"],
        vig_adjustment=vig_adjustment,
        min_edge=rules.min_edge,
        flat_stake=flat_stake,
        kelly_fraction=kelly_fraction,
        report_path=report_output_path,
        bets=tuple(placed_bets),
        decision_rules=rules,
    )


def run_betting_sensitivity_analysis(
    *,
    db_path: Path,
    predictions_path: Path,
    report_output_path: Path,
    promotion: str,
    model_name: str,
    model_run_id: str | None = None,
    min_edge_values: Sequence[float] = (0.02,),
    min_confidence_values: Sequence[float] = (0.0,),
    min_pre_bout_fights_values: Sequence[int] = (0,),
    max_stake_per_bet_values: Sequence[float | None] = (None,),
    max_event_exposure_values: Sequence[float | None] = (None,),
    max_day_exposure_values: Sequence[float | None] = (None,),
    min_liquidity_values: Sequence[float | None] = (None,),
    require_pre_bout_data: bool = False,
    require_liquidity: bool = False,
    flat_stake: float = 1.0,
    kelly_fraction: float | None = None,
    initial_bankroll: float = 100.0,
    vig_adjustment: VigAdjustment = "proportional",
    one_bet_per_bout: bool = True,
    vig_adjuster: VigAdjuster | None = None,
    kelly_sizer: KellySizer | None = None,
) -> BacktestSensitivityResult:
    """Run rule-parameter sensitivity analysis and write a plain-text report."""

    if len(min_edge_values) == 0:
        raise ValueError("min_edge_values must not be empty.")
    if len(min_confidence_values) == 0:
        raise ValueError("min_confidence_values must not be empty.")
    if len(min_pre_bout_fights_values) == 0:
        raise ValueError("min_pre_bout_fights_values must not be empty.")
    if len(max_stake_per_bet_values) == 0:
        raise ValueError("max_stake_per_bet_values must not be empty.")
    if len(max_event_exposure_values) == 0:
        raise ValueError("max_event_exposure_values must not be empty.")
    if len(max_day_exposure_values) == 0:
        raise ValueError("max_day_exposure_values must not be empty.")
    if len(min_liquidity_values) == 0:
        raise ValueError("min_liquidity_values must not be empty.")

    rows = _load_predictions(
        predictions_path=predictions_path,
        model_name=model_name,
        model_run_id=model_run_id,
    )
    resolved_run_id = _resolve_model_run_id(rows=rows, requested_run_id=model_run_id)
    if model_run_id is None and resolved_run_id != "unknown":
        rows = [row for row in rows if row.model_run_id == resolved_run_id]
        if len(rows) == 0:
            raise ValueError(f"No prediction rows found for inferred model_run_id={resolved_run_id}.")

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        market_by_key = _load_market_snapshots(connection=connection, promotion=promotion)
        pre_bout_counts_by_key = _load_pre_bout_fight_counts(connection=connection, promotion=promotion)

    candidates = build_bet_candidates(
        predictions=rows,
        market_by_key=market_by_key,
        pre_bout_counts_by_key=pre_bout_counts_by_key,
        vig_adjustment=vig_adjustment,
        vig_adjuster=vig_adjuster,
    )

    sensitivity_rows: list[BacktestSensitivityRow] = []
    for (
        min_edge,
        min_confidence,
        min_pre_bout_fights,
        max_stake_per_bet,
        max_event_exposure,
        max_day_exposure,
        min_liquidity,
    ) in itertools.product(
        min_edge_values,
        min_confidence_values,
        min_pre_bout_fights_values,
        max_stake_per_bet_values,
        max_event_exposure_values,
        max_day_exposure_values,
        min_liquidity_values,
    ):
        rules = _resolve_decision_rules(
            decision_rules=None,
            min_edge=min_edge,
            min_confidence=min_confidence,
            min_pre_bout_fights=min_pre_bout_fights,
            require_pre_bout_data=require_pre_bout_data,
            max_stake_per_bet=max_stake_per_bet,
            max_event_exposure=max_event_exposure,
            max_day_exposure=max_day_exposure,
            min_liquidity=min_liquidity,
            require_liquidity=require_liquidity,
        )
        selected = select_bets(
            candidates=candidates,
            min_edge=rules.min_edge,
            one_bet_per_bout=one_bet_per_bout,
            decision_rules=rules,
        )
        placed = _simulate_bets(
            candidates=selected,
            flat_stake=flat_stake,
            initial_bankroll=initial_bankroll,
            kelly_fraction=kelly_fraction,
            kelly_sizer=kelly_sizer,
            decision_rules=rules,
        )
        metrics = _compute_metrics(placed)
        sensitivity_rows.append(
            BacktestSensitivityRow(
                min_edge=rules.min_edge,
                min_confidence=rules.min_confidence,
                min_pre_bout_fights=rules.min_pre_bout_fights,
                max_stake_per_bet=rules.max_stake_per_bet,
                max_event_exposure=rules.max_event_exposure,
                max_day_exposure=rules.max_day_exposure,
                min_liquidity=rules.min_liquidity,
                bets_count=len(placed),
                total_staked=metrics["total_staked"],
                roi=metrics["roi"],
                max_drawdown=metrics["max_drawdown"],
            )
        )

    _write_sensitivity_report(
        output_path=report_output_path,
        model_name=model_name,
        model_run_id=resolved_run_id,
        scenarios=sensitivity_rows,
    )

    return BacktestSensitivityResult(
        model_name=model_name,
        model_run_id=resolved_run_id,
        scenarios=tuple(sensitivity_rows),
        report_path=report_output_path,
    )


def calculate_edge(*, model_probability: float, market_probability: float) -> float:
    """Return model edge over market probability."""

    return model_probability - market_probability


def build_bet_candidates(
    *,
    predictions: Sequence[PredictionRow],
    market_by_key: dict[tuple[str, str], MarketSnapshot],
    pre_bout_counts_by_key: dict[tuple[str, str], int] | None = None,
    vig_adjustment: VigAdjustment,
    vig_adjuster: VigAdjuster | None = None,
) -> list[BetCandidate]:
    """Join model outputs with valid market snapshots and compute edge."""

    market_probs_by_bout: dict[str, dict[str, float]] = {}
    for (bout_id, fighter_id), snapshot in market_by_key.items():
        market_probs_by_bout.setdefault(bout_id, {})[fighter_id] = snapshot.implied_probability

    adjusted_market_by_bout: dict[str, dict[str, float]] = {}
    for bout_id, probs in market_probs_by_bout.items():
        adjusted_market_by_bout[bout_id] = apply_vig_adjustment(
            implied_probabilities=probs,
            method=vig_adjustment,
            adjuster=vig_adjuster,
        )

    counts = pre_bout_counts_by_key or {}

    candidates: list[BetCandidate] = []
    for row in predictions:
        if row.label is None:
            continue

        market = market_by_key.get((row.bout_id, row.fighter_id))
        if market is None:
            continue

        # Chronology guard: only compare against prices available before model cutoff.
        if market.market_timestamp_utc > row.feature_cutoff_utc:
            continue

        fair_market = adjusted_market_by_bout.get(row.bout_id, {}).get(row.fighter_id)
        if fair_market is None:
            continue

        edge = calculate_edge(model_probability=row.predicted_probability, market_probability=fair_market)
        confidence = abs(row.predicted_probability - 0.5)

        candidates.append(
            BetCandidate(
                bout_id=row.bout_id,
                event_id=row.event_id,
                bout_datetime_utc=row.bout_datetime_utc,
                fighter_id=row.fighter_id,
                opponent_id=row.opponent_id,
                label=float(row.label),
                model_probability=row.predicted_probability,
                market_probability_raw=market.implied_probability,
                market_probability_fair=fair_market,
                decimal_odds=market.decimal_odds,
                edge=edge,
                confidence=confidence,
                fighter_pre_bout_count=counts.get((row.bout_id, row.fighter_id)),
                opponent_pre_bout_count=counts.get((row.bout_id, row.opponent_id)),
                market_liquidity=market.market_liquidity,
            )
        )

    candidates.sort(key=lambda row: (_parse_utc(row.bout_datetime_utc), row.bout_id, row.fighter_id))
    return candidates


def apply_vig_adjustment(
    *,
    implied_probabilities: dict[str, float],
    method: VigAdjustment,
    adjuster: VigAdjuster | None = None,
) -> dict[str, float]:
    """Apply vig-adjustment policy to fighter-side implied probabilities."""

    if adjuster is not None:
        adjusted = adjuster(dict(implied_probabilities))
        return {key: _clamp_probability(value) for key, value in adjusted.items()}

    if method == "none":
        return {key: _clamp_probability(value) for key, value in implied_probabilities.items()}

    total = sum(max(value, 0.0) for value in implied_probabilities.values())
    if total <= 0.0:
        return {key: _clamp_probability(value) for key, value in implied_probabilities.items()}

    return {
        key: _clamp_probability(max(value, 0.0) / total)
        for key, value in implied_probabilities.items()
    }


def select_bets(
    *,
    candidates: Sequence[BetCandidate],
    min_edge: float,
    one_bet_per_bout: bool,
    decision_rules: DecisionRules | None = None,
) -> list[BetCandidate]:
    """Apply configurable no-bet and threshold rules before stake sizing."""

    rules = decision_rules or _resolve_decision_rules(
        decision_rules=None,
        min_edge=min_edge,
        min_confidence=0.0,
        min_pre_bout_fights=0,
        require_pre_bout_data=False,
        max_stake_per_bet=None,
        max_event_exposure=None,
        max_day_exposure=None,
        min_liquidity=None,
        require_liquidity=False,
    )

    filtered = [row for row in candidates if _passes_decision_rules(candidate=row, rules=rules)]
    filtered.sort(key=lambda row: (_parse_utc(row.bout_datetime_utc), row.bout_id, -row.edge, row.fighter_id))

    if not one_bet_per_bout:
        return filtered

    by_bout: dict[str, BetCandidate] = {}
    for row in filtered:
        existing = by_bout.get(row.bout_id)
        if existing is None or row.edge > existing.edge:
            by_bout[row.bout_id] = row

    selected = list(by_bout.values())
    selected.sort(key=lambda row: (_parse_utc(row.bout_datetime_utc), row.bout_id, row.fighter_id))
    return selected


def _passes_decision_rules(*, candidate: BetCandidate, rules: DecisionRules) -> bool:
    if candidate.edge < rules.min_edge:
        return False

    if candidate.confidence < rules.min_confidence:
        return False

    if rules.min_pre_bout_fights > 0:
        if candidate.fighter_pre_bout_count is None or candidate.opponent_pre_bout_count is None:
            if rules.require_pre_bout_data:
                return False
        else:
            if min(candidate.fighter_pre_bout_count, candidate.opponent_pre_bout_count) < rules.min_pre_bout_fights:
                return False

    if rules.min_liquidity is not None:
        if candidate.market_liquidity is None:
            if rules.require_liquidity:
                return False
        elif candidate.market_liquidity < rules.min_liquidity:
            return False

    return True


def calculate_stake(
    *,
    candidate: BetCandidate,
    bankroll: float,
    flat_stake: float,
    kelly_fraction: float | None,
    kelly_sizer: KellySizer | None,
) -> float:
    """Determine stake for a candidate bet (flat by default, fractional Kelly optional)."""

    if kelly_fraction is None or kelly_fraction <= 0.0:
        return flat_stake

    if kelly_sizer is not None:
        stake = kelly_sizer(candidate, bankroll, kelly_fraction)
        return max(0.0, min(stake, bankroll))

    b = candidate.decimal_odds - 1.0
    if b <= 0.0:
        return 0.0

    p = _clamp_probability(candidate.model_probability)
    q = 1.0 - p
    full_kelly = ((b * p) - q) / b
    if full_kelly <= 0.0:
        return 0.0

    stake = bankroll * (kelly_fraction * full_kelly)
    return max(0.0, min(stake, bankroll))


def _simulate_bets(
    *,
    candidates: Sequence[BetCandidate],
    flat_stake: float,
    initial_bankroll: float,
    kelly_fraction: float | None,
    kelly_sizer: KellySizer | None,
    decision_rules: DecisionRules,
) -> list[PlacedBet]:
    bankroll = initial_bankroll
    placed: list[PlacedBet] = []

    event_exposure: dict[str, float] = {}
    day_exposure: dict[str, float] = {}

    ordered = sorted(candidates, key=lambda row: (_parse_utc(row.bout_datetime_utc), row.bout_id, row.fighter_id))
    for candidate in ordered:
        requested_stake = calculate_stake(
            candidate=candidate,
            bankroll=bankroll,
            flat_stake=flat_stake,
            kelly_fraction=kelly_fraction,
            kelly_sizer=kelly_sizer,
        )
        if requested_stake <= 0.0:
            continue

        stake = requested_stake
        if decision_rules.max_stake_per_bet is not None:
            stake = min(stake, decision_rules.max_stake_per_bet)

        if decision_rules.max_event_exposure is not None:
            current_event_exposure = event_exposure.get(candidate.event_id, 0.0)
            remaining_event = max(0.0, decision_rules.max_event_exposure - current_event_exposure)
            stake = min(stake, remaining_event)

        day_key = _parse_utc(candidate.bout_datetime_utc).date().isoformat()
        if decision_rules.max_day_exposure is not None:
            current_day_exposure = day_exposure.get(day_key, 0.0)
            remaining_day = max(0.0, decision_rules.max_day_exposure - current_day_exposure)
            stake = min(stake, remaining_day)

        stake = min(stake, bankroll)
        if stake <= 0.0:
            continue

        if candidate.label >= 0.5:
            pnl = stake * (candidate.decimal_odds - 1.0)
        else:
            pnl = -stake

        bankroll += pnl
        event_exposure[candidate.event_id] = event_exposure.get(candidate.event_id, 0.0) + stake
        day_exposure[day_key] = day_exposure.get(day_key, 0.0) + stake
        placed.append(
            PlacedBet(
                bout_id=candidate.bout_id,
                fighter_id=candidate.fighter_id,
                bout_datetime_utc=candidate.bout_datetime_utc,
                label=candidate.label,
                edge=candidate.edge,
                model_probability=candidate.model_probability,
                market_probability_raw=candidate.market_probability_raw,
                market_probability_fair=candidate.market_probability_fair,
                decimal_odds=candidate.decimal_odds,
                stake=stake,
                pnl=pnl,
                event_id=candidate.event_id,
            )
        )

    return placed


def _compute_metrics(bets: Sequence[PlacedBet]) -> dict[str, float]:
    if len(bets) == 0:
        return {
            "total_staked": 0.0,
            "total_pnl": 0.0,
            "roi": 0.0,
            "max_drawdown": 0.0,
            "average_edge": 0.0,
            "average_model_probability": 0.0,
            "average_market_probability_raw": 0.0,
            "average_market_probability_fair": 0.0,
        }

    total_staked = sum(row.stake for row in bets)
    total_pnl = sum(row.pnl for row in bets)
    roi = 0.0 if total_staked == 0.0 else total_pnl / total_staked

    average_edge = sum(row.edge for row in bets) / float(len(bets))
    avg_model_probability = sum(row.model_probability for row in bets) / float(len(bets))
    avg_market_probability_raw = sum(row.market_probability_raw for row in bets) / float(len(bets))
    avg_market_probability_fair = sum(row.market_probability_fair for row in bets) / float(len(bets))

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for row in bets:
        equity += row.pnl
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    return {
        "total_staked": total_staked,
        "total_pnl": total_pnl,
        "roi": roi,
        "max_drawdown": max_drawdown,
        "average_edge": average_edge,
        "average_model_probability": avg_model_probability,
        "average_market_probability_raw": avg_market_probability_raw,
        "average_market_probability_fair": avg_market_probability_fair,
    }


def _load_predictions(
    *,
    predictions_path: Path,
    model_name: str,
    model_run_id: str | None,
) -> list[PredictionRow]:
    rows: list[PredictionRow] = []
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

            bout_datetime = str(raw.get("bout_datetime_utc", "")).strip()
            feature_cutoff = str(raw.get("feature_cutoff_utc", "")).strip()
            if bout_datetime == "" or feature_cutoff == "":
                continue
            if feature_cutoff > bout_datetime:
                raise ValueError(
                    "Chronology violation in predictions CSV: feature_cutoff_utc is after bout_datetime_utc."
                )

            label = _as_optional_float(raw.get("label"))
            probability = _as_optional_float(raw.get("predicted_probability"))
            if probability is None:
                continue

            rows.append(
                PredictionRow(
                    model_run_id=raw_run_id,
                    model_name=model_name,
                    bout_id=str(raw.get("bout_id", "")).strip(),
                    event_id=str(raw.get("event_id", "")).strip(),
                    bout_datetime_utc=bout_datetime,
                    fighter_id=str(raw.get("fighter_id", "")).strip(),
                    opponent_id=str(raw.get("opponent_id", "")).strip(),
                    label=label,
                    predicted_probability=_clamp_probability(probability),
                    feature_cutoff_utc=feature_cutoff,
                )
            )

    if len(rows) == 0:
        requested = "" if model_run_id is None else f" for model_run_id={model_run_id}"
        raise ValueError(f"No prediction rows found for model={model_name}{requested}.")

    return rows


def _resolve_model_run_id(*, rows: Sequence[PredictionRow], requested_run_id: str | None) -> str:
    if requested_run_id is not None:
        return requested_run_id

    run_ids = sorted({row.model_run_id for row in rows if row.model_run_id != ""})
    if len(run_ids) == 0:
        return "unknown"
    return run_ids[-1]


def _load_market_snapshots(
    *,
    connection: sqlite3.Connection,
    promotion: str,
) -> dict[tuple[str, str], MarketSnapshot]:
    rows = connection.execute(
        """
        SELECT
            b.bout_id,
            COALESCE(b.bout_start_time_utc, e.event_date_utc) AS bout_datetime_utc,
            m.selection_fighter_id,
            m.implied_probability,
            m.odds_decimal,
            m.line_value,
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

    chosen: dict[tuple[str, str], MarketSnapshot] = {}
    for row in rows:
        bout_id = str(row["bout_id"])
        fighter_id = str(row["selection_fighter_id"])
        timestamp = str(row["market_timestamp_utc"])
        bout_datetime = str(row["bout_datetime_utc"])
        if timestamp > bout_datetime:
            continue

        implied = _as_optional_float(row["implied_probability"])
        if implied is None:
            continue

        decimal_odds = _as_optional_float(row["odds_decimal"])
        resolved_decimal = decimal_odds if decimal_odds is not None and decimal_odds > 1.0 else _implied_to_decimal(implied)

        snapshot = MarketSnapshot(
            bout_id=bout_id,
            fighter_id=fighter_id,
            implied_probability=_clamp_probability(implied),
            decimal_odds=resolved_decimal,
            market_timestamp_utc=timestamp,
            bout_datetime_utc=bout_datetime,
            market_liquidity=_as_optional_float(row["line_value"]),
        )

        key = (bout_id, fighter_id)
        existing = chosen.get(key)
        if existing is None or snapshot.market_timestamp_utc > existing.market_timestamp_utc:
            chosen[key] = snapshot

    return chosen


def _load_pre_bout_fight_counts(
    *,
    connection: sqlite3.Connection,
    promotion: str,
) -> dict[tuple[str, str], int]:
    rows = connection.execute(
        """
        SELECT
            b.bout_id,
            b.fighter_red_id,
            b.fighter_blue_id,
            b.winner_fighter_id,
            COALESCE(b.bout_start_time_utc, e.event_date_utc) AS bout_datetime_utc
        FROM bouts AS b
        JOIN events AS e ON e.event_id = b.event_id
        WHERE UPPER(e.promotion) = UPPER(?)
          AND b.fighter_red_id IS NOT NULL
          AND b.fighter_blue_id IS NOT NULL
          AND COALESCE(b.bout_start_time_utc, e.event_date_utc) IS NOT NULL
        ORDER BY
            COALESCE(b.bout_start_time_utc, e.event_date_utc) ASC,
            b.bout_id ASC
        """,
        (promotion,),
    ).fetchall()

    fighter_counts: dict[str, int] = {}
    pre_bout_counts: dict[tuple[str, str], int] = {}

    index = 0
    while index < len(rows):
        bout_datetime = str(rows[index]["bout_datetime_utc"])
        same_time_rows: list[sqlite3.Row] = []
        while index < len(rows) and str(rows[index]["bout_datetime_utc"]) == bout_datetime:
            same_time_rows.append(rows[index])
            index += 1

        # Chronology guard: assign all pre-bout counts before updating this timestamp bucket.
        for row in same_time_rows:
            bout_id = str(row["bout_id"])
            red_id = str(row["fighter_red_id"])
            blue_id = str(row["fighter_blue_id"])
            pre_bout_counts[(bout_id, red_id)] = fighter_counts.get(red_id, 0)
            pre_bout_counts[(bout_id, blue_id)] = fighter_counts.get(blue_id, 0)

        for row in same_time_rows:
            if str(row["winner_fighter_id"] or "").strip() == "":
                continue
            red_id = str(row["fighter_red_id"])
            blue_id = str(row["fighter_blue_id"])
            fighter_counts[red_id] = fighter_counts.get(red_id, 0) + 1
            fighter_counts[blue_id] = fighter_counts.get(blue_id, 0) + 1

    return pre_bout_counts


def _write_report(
    *,
    output_path: Path,
    model_name: str,
    model_run_id: str,
    vig_adjustment: VigAdjustment,
    decision_rules: DecisionRules,
    flat_stake: float,
    kelly_fraction: float | None,
    predictions_considered: int,
    candidates_count: int,
    bets: Sequence[PlacedBet],
    metrics: dict[str, float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "betting_backtest_report",
        f"model_name={model_name}",
        f"model_run_id={model_run_id}",
        f"vig_adjustment={vig_adjustment}",
        "decision_rules",
        f"min_edge={decision_rules.min_edge:.6f}",
        f"min_confidence={decision_rules.min_confidence:.6f}",
        f"min_pre_bout_fights={decision_rules.min_pre_bout_fights}",
        f"require_pre_bout_data={int(decision_rules.require_pre_bout_data)}",
        f"max_stake_per_bet={'' if decision_rules.max_stake_per_bet is None else f'{decision_rules.max_stake_per_bet:.6f}'}",
        f"max_event_exposure={'' if decision_rules.max_event_exposure is None else f'{decision_rules.max_event_exposure:.6f}'}",
        f"max_day_exposure={'' if decision_rules.max_day_exposure is None else f'{decision_rules.max_day_exposure:.6f}'}",
        f"min_liquidity={'' if decision_rules.min_liquidity is None else f'{decision_rules.min_liquidity:.6f}'}",
        f"require_liquidity={int(decision_rules.require_liquidity)}",
        f"flat_stake={flat_stake:.6f}",
        f"kelly_fraction={'' if kelly_fraction is None else f'{kelly_fraction:.6f}'}",
        "sample_sizes",
        f"predictions_considered={predictions_considered}",
        f"candidates_count={candidates_count}",
        f"bets_count={len(bets)}",
        "performance",
        f"roi={metrics['roi']:.6f}",
        f"total_staked={metrics['total_staked']:.6f}",
        f"total_pnl={metrics['total_pnl']:.6f}",
        f"max_drawdown={metrics['max_drawdown']:.6f}",
        f"average_edge={metrics['average_edge']:.6f}",
        "market_implied_probability_comparison",
        f"average_model_probability={metrics['average_model_probability']:.6f}",
        f"average_market_probability_raw={metrics['average_market_probability_raw']:.6f}",
        f"average_market_probability_fair={metrics['average_market_probability_fair']:.6f}",
        "bet_log",
    ]

    for bet in bets:
        lines.append(
            "bet "
            f"bout_id={bet.bout_id} fighter_id={bet.fighter_id} event_id={bet.event_id} "
            f"stake={bet.stake:.6f} pnl={bet.pnl:.6f} edge={bet.edge:.6f} "
            f"model_probability={bet.model_probability:.6f} "
            f"market_probability_raw={bet.market_probability_raw:.6f} "
            f"market_probability_fair={bet.market_probability_fair:.6f}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_sensitivity_report(
    *,
    output_path: Path,
    model_name: str,
    model_run_id: str,
    scenarios: Sequence[BacktestSensitivityRow],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "betting_rule_sensitivity_report",
        f"model_name={model_name}",
        f"model_run_id={model_run_id}",
        f"scenario_count={len(scenarios)}",
        "scenarios",
    ]

    for index, row in enumerate(scenarios, start=1):
        lines.append(
            "scenario "
            f"id={index} "
            f"min_edge={row.min_edge:.6f} "
            f"min_confidence={row.min_confidence:.6f} "
            f"min_pre_bout_fights={row.min_pre_bout_fights} "
            f"max_stake_per_bet={'' if row.max_stake_per_bet is None else f'{row.max_stake_per_bet:.6f}'} "
            f"max_event_exposure={'' if row.max_event_exposure is None else f'{row.max_event_exposure:.6f}'} "
            f"max_day_exposure={'' if row.max_day_exposure is None else f'{row.max_day_exposure:.6f}'} "
            f"min_liquidity={'' if row.min_liquidity is None else f'{row.min_liquidity:.6f}'} "
            f"bets={row.bets_count} "
            f"total_staked={row.total_staked:.6f} "
            f"roi={row.roi:.6f} "
            f"drawdown={row.max_drawdown:.6f}"
        )

    if len(scenarios) > 0:
        best_roi = max(scenarios, key=lambda row: row.roi)
        best_drawdown = min(scenarios, key=lambda row: row.max_drawdown)
        lines.append("summary")
        lines.append(
            "best_roi "
            f"roi={best_roi.roi:.6f} drawdown={best_roi.max_drawdown:.6f} bets={best_roi.bets_count}"
        )
        lines.append(
            "best_drawdown "
            f"roi={best_drawdown.roi:.6f} drawdown={best_drawdown.max_drawdown:.6f} bets={best_drawdown.bets_count}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_decision_rules(
    *,
    decision_rules: DecisionRules | None,
    min_edge: float,
    min_confidence: float,
    min_pre_bout_fights: int,
    require_pre_bout_data: bool,
    max_stake_per_bet: float | None,
    max_event_exposure: float | None,
    max_day_exposure: float | None,
    min_liquidity: float | None,
    require_liquidity: bool,
) -> DecisionRules:
    rules = decision_rules or DecisionRules(
        min_edge=min_edge,
        min_confidence=min_confidence,
        min_pre_bout_fights=min_pre_bout_fights,
        require_pre_bout_data=require_pre_bout_data,
        max_stake_per_bet=max_stake_per_bet,
        max_event_exposure=max_event_exposure,
        max_day_exposure=max_day_exposure,
        min_liquidity=min_liquidity,
        require_liquidity=require_liquidity,
    )

    if rules.min_edge < 0.0:
        raise ValueError("min_edge must be non-negative.")
    if rules.min_confidence < 0.0 or rules.min_confidence > 0.5:
        raise ValueError("min_confidence must be between 0 and 0.5.")
    if rules.min_pre_bout_fights < 0:
        raise ValueError("min_pre_bout_fights must be non-negative.")
    if rules.max_stake_per_bet is not None and rules.max_stake_per_bet <= 0.0:
        raise ValueError("max_stake_per_bet must be positive when provided.")
    if rules.max_event_exposure is not None and rules.max_event_exposure <= 0.0:
        raise ValueError("max_event_exposure must be positive when provided.")
    if rules.max_day_exposure is not None and rules.max_day_exposure <= 0.0:
        raise ValueError("max_day_exposure must be positive when provided.")
    if rules.min_liquidity is not None and rules.min_liquidity < 0.0:
        raise ValueError("min_liquidity must be non-negative when provided.")

    return rules


def _implied_to_decimal(implied_probability: float) -> float:
    implied = _clamp_probability(implied_probability)
    return 1.0 / implied


def _clamp_probability(value: float) -> float:
    if value < 1e-6:
        return 1e-6
    if value > 1.0 - 1e-6:
        return 1.0 - 1e-6
    return value


def _as_optional_float(value: object | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        numeric = float(text)
    except ValueError:
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _parse_utc(value: str) -> datetime:
    if value.endswith("Z"):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return datetime.fromisoformat(value)
