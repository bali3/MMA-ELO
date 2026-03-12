"""Sequential, leakage-safe pre-fight feature generation."""

from __future__ import annotations

import csv
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

WINDOWS: tuple[int, ...] = (3, 5)
DEFAULT_AGE_YEARS = 30.0
DEFAULT_KEY_FEATURES: tuple[str, ...] = (
    "fighter_age_years",
    "height_diff_cm",
    "reach_diff_cm",
    "days_since_last_fight",
    "recent_form_win_rate_l3",
    "opponent_strength_avg_l3",
    "finish_rate_all",
    "decision_rate_all",
    "sig_strikes_landed_per_min_all",
    "sig_striking_accuracy_all",
    "sig_striking_defense_all",
    "takedown_attempts_per_15m_all",
    "takedown_accuracy_all",
    "takedown_defense_all",
    "submission_attempts_per_15m_all",
    "avg_fight_duration_seconds_all",
    "rematch_flag",
    "weight_class_experience",
)


@dataclass(frozen=True)
class BoutRow:
    """Single bout with red/blue fighter identity and resolved timestamp."""

    bout_id: str
    event_id: str
    bout_order: int
    weight_class: str | None
    winner_fighter_id: str | None
    result_method: str | None
    result_round: int | None
    result_time_seconds: int | None
    fighter_red_id: str
    fighter_blue_id: str
    event_date_utc: str
    bout_start_time_utc: str | None


@dataclass(frozen=True)
class FighterMeta:
    """Static fighter metadata used for pre-fight lookups."""

    date_of_birth_utc: str | None
    stance: str | None
    height_cm: float | None
    reach_cm: float | None


@dataclass(frozen=True)
class BoutStatsLine:
    """Per-fighter stat line for a specific bout."""

    sig_strikes_landed: int | None
    sig_strikes_attempted: int | None
    takedowns_landed: int | None
    takedowns_attempted: int | None
    submission_attempts: int | None


@dataclass(frozen=True)
class HistoryRecord:
    """Historical fighter bout record persisted after features for that bout are emitted."""

    bout_datetime: datetime
    bout_id: str
    opponent_id: str
    weight_class: str | None
    won: bool | None
    finish: bool
    decision: bool
    duration_seconds: int | None
    opponent_pre_fight_win_rate: float | None
    sig_landed: int | None
    sig_attempted: int | None
    opp_sig_landed: int | None
    opp_sig_attempted: int | None
    td_landed: int | None
    td_attempted: int | None
    opp_td_landed: int | None
    opp_td_attempted: int | None
    sub_attempts: int | None


@dataclass(frozen=True)
class FeatureReport:
    """Simple feature coverage and span report for generated output."""

    bout_count: int
    row_count: int
    earliest_bout_date_utc: str | None
    latest_bout_date_utc: str | None
    coverage_rates: dict[str, float]
    missing_counts: dict[str, int]


def generate_pre_fight_features(
    *,
    db_path: Path,
    output_path: Path,
    promotion: str = "UFC",
) -> tuple[int, Path]:
    """Generate a leakage-safe pre-fight feature dataset and persist it as CSV."""

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        bouts = _load_bouts(connection=connection, promotion=promotion)
        fighter_meta = _load_fighter_meta(connection)
        stats_lines = _load_stats_lines(connection)

    rows = _build_feature_rows(bouts=bouts, fighter_meta=fighter_meta, stats_lines=stats_lines)
    _write_rows_to_csv(output_path=output_path, rows=rows)
    return len(rows), output_path


def build_feature_report(*, features_path: Path, key_features: Iterable[str] = DEFAULT_KEY_FEATURES) -> FeatureReport:
    """Build a lightweight summary report from a persisted feature CSV."""

    with features_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        records = list(reader)

    row_count = len(records)
    bout_ids = {row["bout_id"] for row in records}
    bout_dates = [row.get("bout_datetime_utc") for row in records if row.get("bout_datetime_utc")]

    coverage_rates: dict[str, float] = {}
    missing_counts: dict[str, int] = {}

    for feature in key_features:
        present_count = 0
        missing_count = 0
        for row in records:
            value = row.get(feature)
            if value is None or value == "":
                missing_count += 1
            else:
                present_count += 1
        missing_counts[feature] = missing_count
        coverage_rates[feature] = 0.0 if row_count == 0 else present_count / row_count

    earliest = min(bout_dates) if bout_dates else None
    latest = max(bout_dates) if bout_dates else None

    return FeatureReport(
        bout_count=len(bout_ids),
        row_count=row_count,
        earliest_bout_date_utc=earliest,
        latest_bout_date_utc=latest,
        coverage_rates=coverage_rates,
        missing_counts=missing_counts,
    )


def _load_bouts(*, connection: sqlite3.Connection, promotion: str) -> list[BoutRow]:
    rows = connection.execute(
        """
        SELECT
            b.bout_id,
            b.event_id,
            COALESCE(b.bout_order, 0) AS bout_order,
            b.weight_class,
            b.winner_fighter_id,
            b.result_method,
            b.result_round,
            b.result_time_seconds,
            b.fighter_red_id,
            b.fighter_blue_id,
            e.event_date_utc,
            b.bout_start_time_utc
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

    return [
        BoutRow(
            bout_id=str(row["bout_id"]),
            event_id=str(row["event_id"]),
            bout_order=int(row["bout_order"]),
            weight_class=_clean_str(row["weight_class"]),
            winner_fighter_id=_clean_str(row["winner_fighter_id"]),
            result_method=_clean_str(row["result_method"]),
            result_round=_as_int(row["result_round"]),
            result_time_seconds=_as_int(row["result_time_seconds"]),
            fighter_red_id=str(row["fighter_red_id"]),
            fighter_blue_id=str(row["fighter_blue_id"]),
            event_date_utc=str(row["event_date_utc"]),
            bout_start_time_utc=_clean_str(row["bout_start_time_utc"]),
        )
        for row in rows
    ]


def _load_fighter_meta(connection: sqlite3.Connection) -> dict[str, FighterMeta]:
    rows = connection.execute(
        """
        SELECT fighter_id, date_of_birth_utc, stance, height_cm, reach_cm
        FROM fighters
        """
    ).fetchall()
    return {
        str(row["fighter_id"]): FighterMeta(
            date_of_birth_utc=_clean_str(row["date_of_birth_utc"]),
            stance=_clean_str(row["stance"]),
            height_cm=_as_float(row["height_cm"]),
            reach_cm=_as_float(row["reach_cm"]),
        )
        for row in rows
    }


def _load_stats_lines(connection: sqlite3.Connection) -> dict[tuple[str, str], BoutStatsLine]:
    rows = connection.execute(
        """
        SELECT
            bout_id,
            fighter_id,
            sig_strikes_landed,
            sig_strikes_attempted,
            takedowns_landed,
            takedowns_attempted,
            submission_attempts
        FROM fighter_bout_stats
        """
    ).fetchall()
    return {
        (str(row["bout_id"]), str(row["fighter_id"])): BoutStatsLine(
            sig_strikes_landed=_as_int(row["sig_strikes_landed"]),
            sig_strikes_attempted=_as_int(row["sig_strikes_attempted"]),
            takedowns_landed=_as_int(row["takedowns_landed"]),
            takedowns_attempted=_as_int(row["takedowns_attempted"]),
            submission_attempts=_as_int(row["submission_attempts"]),
        )
        for row in rows
    }


def _build_feature_rows(
    *,
    bouts: list[BoutRow],
    fighter_meta: dict[str, FighterMeta],
    stats_lines: dict[tuple[str, str], BoutStatsLine],
) -> list[dict[str, object | None]]:
    history: dict[str, list[HistoryRecord]] = defaultdict(list)
    output: list[dict[str, object | None]] = []
    metadata_defaults = _metadata_defaults(fighter_meta)

    for bout in bouts:
        bout_datetime = _parse_utc_timestamp(bout.bout_start_time_utc or bout.event_date_utc)
        if bout_datetime is None:
            continue

        red_id = bout.fighter_red_id
        blue_id = bout.fighter_blue_id

        red_pre = _compute_fighter_snapshot(
            fighter_id=red_id,
            opponent_id=blue_id,
            bout_datetime=bout_datetime,
            weight_class=bout.weight_class,
            history=history,
            fighter_meta=fighter_meta,
            default_age_years=metadata_defaults["age_years"],
        )
        blue_pre = _compute_fighter_snapshot(
            fighter_id=blue_id,
            opponent_id=red_id,
            bout_datetime=bout_datetime,
            weight_class=bout.weight_class,
            history=history,
            fighter_meta=fighter_meta,
            default_age_years=metadata_defaults["age_years"],
        )

        output.append(
            _merge_row(
                bout=bout,
                bout_datetime=bout_datetime,
                fighter_id=red_id,
                opponent_id=blue_id,
                corner="red",
                fighter_snapshot=red_pre,
                opponent_snapshot=blue_pre,
                fighter_meta=fighter_meta,
                metadata_defaults=metadata_defaults,
            )
        )
        output.append(
            _merge_row(
                bout=bout,
                bout_datetime=bout_datetime,
                fighter_id=blue_id,
                opponent_id=red_id,
                corner="blue",
                fighter_snapshot=blue_pre,
                opponent_snapshot=red_pre,
                fighter_meta=fighter_meta,
                metadata_defaults=metadata_defaults,
            )
        )

        red_stats = stats_lines.get((bout.bout_id, red_id), BoutStatsLine(None, None, None, None, None))
        blue_stats = stats_lines.get((bout.bout_id, blue_id), BoutStatsLine(None, None, None, None, None))

        red_won = bout.winner_fighter_id == red_id if bout.winner_fighter_id is not None else None
        blue_won = bout.winner_fighter_id == blue_id if bout.winner_fighter_id is not None else None

        finish = _is_finish_method(bout.result_method)
        decision = _is_decision_method(bout.result_method)
        duration_seconds = _fight_duration_seconds(
            result_round=bout.result_round,
            result_time_seconds=bout.result_time_seconds,
        )

        history[red_id].append(
            HistoryRecord(
                bout_datetime=bout_datetime,
                bout_id=bout.bout_id,
                opponent_id=blue_id,
                weight_class=bout.weight_class,
                won=red_won,
                finish=finish,
                decision=decision,
                duration_seconds=duration_seconds,
                opponent_pre_fight_win_rate=blue_pre.get("win_rate_all"),
                sig_landed=red_stats.sig_strikes_landed,
                sig_attempted=red_stats.sig_strikes_attempted,
                opp_sig_landed=blue_stats.sig_strikes_landed,
                opp_sig_attempted=blue_stats.sig_strikes_attempted,
                td_landed=red_stats.takedowns_landed,
                td_attempted=red_stats.takedowns_attempted,
                opp_td_landed=blue_stats.takedowns_landed,
                opp_td_attempted=blue_stats.takedowns_attempted,
                sub_attempts=red_stats.submission_attempts,
            )
        )
        history[blue_id].append(
            HistoryRecord(
                bout_datetime=bout_datetime,
                bout_id=bout.bout_id,
                opponent_id=red_id,
                weight_class=bout.weight_class,
                won=blue_won,
                finish=finish,
                decision=decision,
                duration_seconds=duration_seconds,
                opponent_pre_fight_win_rate=red_pre.get("win_rate_all"),
                sig_landed=blue_stats.sig_strikes_landed,
                sig_attempted=blue_stats.sig_strikes_attempted,
                opp_sig_landed=red_stats.sig_strikes_landed,
                opp_sig_attempted=red_stats.sig_strikes_attempted,
                td_landed=blue_stats.takedowns_landed,
                td_attempted=blue_stats.takedowns_attempted,
                opp_td_landed=red_stats.takedowns_landed,
                opp_td_attempted=red_stats.takedowns_attempted,
                sub_attempts=blue_stats.submission_attempts,
            )
        )

    return output


def _compute_fighter_snapshot(
    *,
    fighter_id: str,
    opponent_id: str,
    bout_datetime: datetime,
    weight_class: str | None,
    history: dict[str, list[HistoryRecord]],
    fighter_meta: dict[str, FighterMeta],
    default_age_years: float,
) -> dict[str, float | int | str | None]:
    records = history.get(fighter_id, [])

    stance = _normalize_stance(fighter_meta.get(fighter_id, FighterMeta(None, None, None, None)).stance)
    dob = fighter_meta.get(fighter_id, FighterMeta(None, None, None, None)).date_of_birth_utc
    age_years = _age_years(date_of_birth_utc=dob, at_time=bout_datetime)
    if age_years is None:
        age_years = default_age_years

    last_fight = records[-1].bout_datetime if records else None
    days_since = None
    if last_fight is not None:
        days_since = (bout_datetime - last_fight).days

    rematch_flag = 1 if any(rec.opponent_id == opponent_id for rec in records) else 0
    weight_class_exp = 0
    if weight_class is not None:
        weight_class_exp = sum(1 for rec in records if rec.weight_class == weight_class)

    snapshot: dict[str, float | int | str | None] = {
        "fighter_age_years": age_years,
        "fighter_stance": stance,
        "prior_fight_count": len(records),
        "days_since_last_fight": days_since,
        "rematch_flag": rematch_flag,
        "weight_class_experience": weight_class_exp,
    }

    all_agg = _aggregate(records)
    snapshot.update(
        {
            "win_rate_all": all_agg["win_rate"],
            "finish_rate_all": all_agg["finish_rate"],
            "decision_rate_all": all_agg["decision_rate"],
            "sig_strikes_landed_per_min_all": all_agg["sig_volume"],
            "sig_striking_accuracy_all": all_agg["sig_accuracy"],
            "sig_striking_defense_all": all_agg["sig_defense"],
            "takedown_attempts_per_15m_all": all_agg["td_attempt_rate"],
            "takedown_accuracy_all": all_agg["td_accuracy"],
            "takedown_defense_all": all_agg["td_defense"],
            "submission_attempts_per_15m_all": all_agg["sub_attempt_rate"],
            "avg_fight_duration_seconds_all": all_agg["avg_duration"],
            "opponent_strength_avg_all": all_agg["opp_strength"],
        }
    )

    for window in WINDOWS:
        agg = _aggregate(records[-window:])
        snapshot[f"recent_form_win_rate_l{window}"] = agg["win_rate"]
        snapshot[f"opponent_strength_avg_l{window}"] = agg["opp_strength"]

    return snapshot


def _merge_row(
    *,
    bout: BoutRow,
    bout_datetime: datetime,
    fighter_id: str,
    opponent_id: str,
    corner: str,
    fighter_snapshot: dict[str, float | int | str | None],
    opponent_snapshot: dict[str, float | int | str | None],
    fighter_meta: dict[str, FighterMeta],
    metadata_defaults: dict[str, float],
) -> dict[str, object | None]:
    fighter = fighter_meta.get(fighter_id, FighterMeta(None, None, None, None))
    opponent = fighter_meta.get(opponent_id, FighterMeta(None, None, None, None))

    fighter_height = fighter.height_cm if fighter.height_cm is not None else metadata_defaults["height_cm"]
    opponent_height = opponent.height_cm if opponent.height_cm is not None else metadata_defaults["height_cm"]
    fighter_reach = fighter.reach_cm if fighter.reach_cm is not None else metadata_defaults["reach_cm"]
    opponent_reach = opponent.reach_cm if opponent.reach_cm is not None else metadata_defaults["reach_cm"]

    fighter_stance = fighter_snapshot.get("fighter_stance")
    opponent_stance = opponent_snapshot.get("fighter_stance")
    stance_matchup = f"{fighter_stance}_vs_{opponent_stance}"

    return {
        "bout_id": bout.bout_id,
        "event_id": bout.event_id,
        "bout_order": bout.bout_order,
        "bout_datetime_utc": bout_datetime.isoformat().replace("+00:00", "Z"),
        "fighter_id": fighter_id,
        "opponent_id": opponent_id,
        "corner": corner,
        "weight_class": bout.weight_class,
        "fighter_age_years": fighter_snapshot.get("fighter_age_years"),
        "height_diff_cm": _safe_subtract(fighter_height, opponent_height),
        "reach_diff_cm": _safe_subtract(fighter_reach, opponent_reach),
        "stance_matchup": stance_matchup,
        "days_since_last_fight": fighter_snapshot.get("days_since_last_fight"),
        "prior_fight_count": fighter_snapshot.get("prior_fight_count"),
        "recent_form_win_rate_l3": fighter_snapshot.get("recent_form_win_rate_l3"),
        "recent_form_win_rate_l5": fighter_snapshot.get("recent_form_win_rate_l5"),
        "opponent_strength_avg_l3": fighter_snapshot.get("opponent_strength_avg_l3"),
        "opponent_strength_avg_l5": fighter_snapshot.get("opponent_strength_avg_l5"),
        "finish_rate_all": fighter_snapshot.get("finish_rate_all"),
        "decision_rate_all": fighter_snapshot.get("decision_rate_all"),
        "sig_strikes_landed_per_min_all": fighter_snapshot.get("sig_strikes_landed_per_min_all"),
        "sig_striking_accuracy_all": fighter_snapshot.get("sig_striking_accuracy_all"),
        "sig_striking_defense_all": fighter_snapshot.get("sig_striking_defense_all"),
        "takedown_attempts_per_15m_all": fighter_snapshot.get("takedown_attempts_per_15m_all"),
        "takedown_accuracy_all": fighter_snapshot.get("takedown_accuracy_all"),
        "takedown_defense_all": fighter_snapshot.get("takedown_defense_all"),
        "submission_attempts_per_15m_all": fighter_snapshot.get("submission_attempts_per_15m_all"),
        "avg_fight_duration_seconds_all": fighter_snapshot.get("avg_fight_duration_seconds_all"),
        "rematch_flag": fighter_snapshot.get("rematch_flag"),
        "weight_class_experience": fighter_snapshot.get("weight_class_experience"),
    }


def _aggregate(records: list[HistoryRecord]) -> dict[str, float | None]:
    count = len(records)
    if count == 0:
        return {
            "win_rate": None,
            "finish_rate": None,
            "decision_rate": None,
            "opp_strength": None,
            "sig_volume": None,
            "sig_accuracy": None,
            "sig_defense": None,
            "td_attempt_rate": None,
            "td_accuracy": None,
            "td_defense": None,
            "sub_attempt_rate": None,
            "avg_duration": None,
        }

    wins = sum(1 for rec in records if rec.won is True)
    finishes = sum(1 for rec in records if rec.finish)
    decisions = sum(1 for rec in records if rec.decision)

    duration_total = sum(rec.duration_seconds for rec in records if rec.duration_seconds is not None)
    duration_count = sum(1 for rec in records if rec.duration_seconds is not None)

    sig_landed = sum(rec.sig_landed for rec in records if rec.sig_landed is not None)
    sig_attempted = sum(rec.sig_attempted for rec in records if rec.sig_attempted is not None)
    opp_sig_landed = sum(rec.opp_sig_landed for rec in records if rec.opp_sig_landed is not None)
    opp_sig_attempted = sum(rec.opp_sig_attempted for rec in records if rec.opp_sig_attempted is not None)

    td_landed = sum(rec.td_landed for rec in records if rec.td_landed is not None)
    td_attempted = sum(rec.td_attempted for rec in records if rec.td_attempted is not None)
    opp_td_landed = sum(rec.opp_td_landed for rec in records if rec.opp_td_landed is not None)
    opp_td_attempted = sum(rec.opp_td_attempted for rec in records if rec.opp_td_attempted is not None)

    sub_attempts = sum(rec.sub_attempts for rec in records if rec.sub_attempts is not None)

    opp_strength_values = [
        rec.opponent_pre_fight_win_rate
        for rec in records
        if rec.opponent_pre_fight_win_rate is not None
    ]
    opp_strength = (
        sum(opp_strength_values) / len(opp_strength_values) if opp_strength_values else None
    )

    minutes_total = duration_total / 60.0 if duration_total > 0 else None
    fifteen_min_units = duration_total / 900.0 if duration_total > 0 else None

    return {
        "win_rate": wins / count,
        "finish_rate": finishes / count,
        "decision_rate": decisions / count,
        "opp_strength": opp_strength,
        "sig_volume": _safe_divide(sig_landed, minutes_total),
        "sig_accuracy": _safe_divide_or_zero(sig_landed, sig_attempted),
        "sig_defense": _safe_one_minus_ratio_or_one(opp_sig_landed, opp_sig_attempted),
        "td_attempt_rate": _safe_divide(td_attempted, fifteen_min_units),
        "td_accuracy": _safe_divide_or_zero(td_landed, td_attempted),
        "td_defense": _safe_one_minus_ratio_or_one(opp_td_landed, opp_td_attempted),
        "sub_attempt_rate": _safe_divide(sub_attempts, fifteen_min_units),
        "avg_duration": _safe_divide(duration_total, duration_count),
    }


def _write_rows_to_csv(*, output_path: Path, rows: list[dict[str, object | None]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _coerce_for_csv(value) for key, value in row.items()})


def _coerce_for_csv(value: object | None) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, 8)
    return value


def _safe_divide(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _safe_divide_or_zero(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _safe_one_minus_ratio_or_one(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return 1.0
    return 1.0 - (float(numerator) / float(denominator))


def _safe_subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _metadata_defaults(fighter_meta: dict[str, FighterMeta]) -> dict[str, float]:
    heights = [item.height_cm for item in fighter_meta.values() if item.height_cm is not None]
    reaches = [item.reach_cm for item in fighter_meta.values() if item.reach_cm is not None]
    return {
        "height_cm": _median(heights, default=175.0),
        "reach_cm": _median(reaches, default=180.0),
        "age_years": DEFAULT_AGE_YEARS,
    }


def _median(values: list[float], *, default: float) -> float:
    if len(values) == 0:
        return default
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _fight_duration_seconds(*, result_round: int | None, result_time_seconds: int | None) -> int | None:
    if result_round is None or result_time_seconds is None:
        return None
    if result_round <= 0 or result_time_seconds < 0:
        return None
    return ((result_round - 1) * 300) + result_time_seconds


def _is_finish_method(method: str | None) -> bool:
    if method is None:
        return False
    lowered = method.lower()
    return "decision" not in lowered and lowered != ""


def _is_decision_method(method: str | None) -> bool:
    if method is None:
        return False
    return "decision" in method.lower()


def _normalize_stance(stance: str | None) -> str:
    if stance is None or stance.strip() == "":
        return "Unknown"
    return stance.strip().title()


def _age_years(*, date_of_birth_utc: str | None, at_time: datetime) -> float | None:
    birth_dt = _parse_utc_timestamp(date_of_birth_utc)
    if birth_dt is None:
        return None
    if birth_dt > at_time:
        return None
    age_days = (at_time - birth_dt).days
    return age_days / 365.25


def _parse_utc_timestamp(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    clean = raw.strip()
    if clean == "":
        return None

    text = clean.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _clean_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _as_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
