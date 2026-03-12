"""Sequential Elo rating generation for MMA bouts."""

from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class EloConfig:
    """Configurable Elo parameters."""

    initial_rating: float = 1500.0
    k_factor: float = 32.0
    promotion: str = "UFC"


@dataclass(frozen=True)
class RatingsHistoryResult:
    """Summary of persisted rating history output."""

    row_count: int
    bout_count: int
    output_path: Path


@dataclass(frozen=True)
class TopRatedFighter:
    """Current top-rated fighter entry for reporting."""

    fighter_id: str
    full_name: str
    overall_elo: float
    completed_bouts: int


@dataclass(frozen=True)
class WeightClassCoverage:
    """Per-weight-class rating coverage from completed bouts."""

    weight_class: str
    rated_fighters: int
    share_of_rated_fighters: float


@dataclass(frozen=True)
class RatingsReport:
    """Coverage and ranking summary from persisted Elo history."""

    total_fighters: int
    rated_fighters: int
    coverage_rate: float
    earliest_rating_datetime_utc: str | None
    latest_rating_datetime_utc: str | None
    weight_class_coverage: list[WeightClassCoverage]
    top_fighters: list[TopRatedFighter]


@dataclass(frozen=True)
class BoutRow:
    """Bout with resolved chronological ordering fields."""

    bout_id: str
    event_id: str
    promotion: str
    bout_order: int
    bout_datetime_utc: str
    weight_class: str | None
    fighter_red_id: str
    fighter_blue_id: str
    winner_fighter_id: str | None
    result_method: str | None


class InactivityDecayHook(Protocol):
    """Hook for optional pre-fight inactivity adjustment."""

    def adjust_rating(
        self,
        *,
        fighter_id: str,
        dimension: str,
        weight_class: str | None,
        current_rating: float,
        days_inactive: int | None,
    ) -> float:
        """Return the effective pre-fight rating after inactivity handling."""


class NoOpInactivityDecayHook:
    """Default inactivity hook that leaves ratings unchanged."""

    def adjust_rating(
        self,
        *,
        fighter_id: str,
        dimension: str,
        weight_class: str | None,
        current_rating: float,
        days_inactive: int | None,
    ) -> float:
        del fighter_id, dimension, weight_class, days_inactive
        return current_rating


HISTORY_COLUMNS: tuple[str, ...] = (
    "bout_id",
    "event_id",
    "promotion",
    "bout_order",
    "bout_datetime_utc",
    "bout_completed",
    "weight_class",
    "winner_fighter_id",
    "fighter_id",
    "opponent_fighter_id",
    "corner",
    "result_score",
    "k_factor",
    "days_since_last_fight",
    "days_since_last_weight_class_fight",
    "pre_overall_elo",
    "post_overall_elo",
    "pre_weight_class_elo",
    "post_weight_class_elo",
)


def generate_elo_ratings_history(
    *,
    db_path: Path,
    output_path: Path,
    config: EloConfig | None = None,
    inactivity_decay_hook: InactivityDecayHook | None = None,
) -> RatingsHistoryResult:
    """Generate per-fighter Elo history with strict chronological updates."""

    resolved_config = config or EloConfig()
    decay_hook = inactivity_decay_hook or NoOpInactivityDecayHook()

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        bouts = _load_bouts(connection=connection, promotion=resolved_config.promotion)

    rows = _build_history_rows(bouts=bouts, config=resolved_config, inactivity_decay_hook=decay_hook)
    _write_history_rows(output_path=output_path, rows=rows)
    return RatingsHistoryResult(
        row_count=len(rows),
        bout_count=len(bouts),
        output_path=output_path,
    )


def build_ratings_report(
    *,
    ratings_history_path: Path,
    db_path: Path | None = None,
    top_n: int = 10,
) -> RatingsReport:
    """Build a compact report of current top ratings and fighter coverage."""

    with ratings_history_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    latest_by_fighter: dict[str, dict[str, str]] = {}
    latest_key_by_fighter: dict[str, tuple[str, int, str, int]] = {}
    completed_counts: dict[str, int] = {}
    weight_class_fighters: dict[str, set[str]] = {}
    earliest_rating_datetime_utc: str | None = None
    latest_rating_datetime_utc: str | None = None

    for index, row in enumerate(rows):
        fighter_id = row["fighter_id"]
        if row.get("bout_completed") != "1":
            continue

        bout_datetime = row["bout_datetime_utc"]
        if earliest_rating_datetime_utc is None or bout_datetime < earliest_rating_datetime_utc:
            earliest_rating_datetime_utc = bout_datetime
        if latest_rating_datetime_utc is None or bout_datetime > latest_rating_datetime_utc:
            latest_rating_datetime_utc = bout_datetime

        completed_counts[fighter_id] = completed_counts.get(fighter_id, 0) + 1
        weight_class = row.get("weight_class", "").strip() or "Unknown"
        weight_class_fighters.setdefault(weight_class, set()).add(fighter_id)

        sort_key = (row["bout_datetime_utc"], int(row["bout_order"]), row["bout_id"], index)
        if fighter_id not in latest_by_fighter:
            latest_by_fighter[fighter_id] = row
            latest_key_by_fighter[fighter_id] = sort_key
            continue
        existing_key = latest_key_by_fighter[fighter_id]
        if sort_key >= existing_key:
            latest_by_fighter[fighter_id] = row
            latest_key_by_fighter[fighter_id] = sort_key

    fighter_names: dict[str, str] = {}
    total_fighters = len(latest_by_fighter)
    if db_path is not None:
        with sqlite3.connect(db_path) as connection:
            db_rows = connection.execute("SELECT fighter_id, full_name FROM fighters").fetchall()
        fighter_names = {str(row[0]): str(row[1]) for row in db_rows}
        total_fighters = len(db_rows)

    top_fighters = sorted(
        (
            TopRatedFighter(
                fighter_id=fighter_id,
                full_name=fighter_names.get(fighter_id, fighter_id),
                overall_elo=float(row["post_overall_elo"]),
                completed_bouts=completed_counts.get(fighter_id, 0),
            )
            for fighter_id, row in latest_by_fighter.items()
        ),
        key=lambda fighter: (-fighter.overall_elo, fighter.fighter_id),
    )[:top_n]

    rated_fighters = len(latest_by_fighter)
    coverage_rate = 0.0 if total_fighters == 0 else rated_fighters / total_fighters
    weight_class_coverage = sorted(
        (
            WeightClassCoverage(
                weight_class=weight_class,
                rated_fighters=len(fighter_ids),
                share_of_rated_fighters=(
                    0.0 if rated_fighters == 0 else len(fighter_ids) / rated_fighters
                ),
            )
            for weight_class, fighter_ids in weight_class_fighters.items()
        ),
        key=lambda entry: (-entry.rated_fighters, entry.weight_class),
    )
    return RatingsReport(
        total_fighters=total_fighters,
        rated_fighters=rated_fighters,
        coverage_rate=coverage_rate,
        earliest_rating_datetime_utc=earliest_rating_datetime_utc,
        latest_rating_datetime_utc=latest_rating_datetime_utc,
        weight_class_coverage=weight_class_coverage,
        top_fighters=top_fighters,
    )


def _load_bouts(*, connection: sqlite3.Connection, promotion: str) -> list[BoutRow]:
    rows = connection.execute(
        """
        SELECT
            b.bout_id,
            b.event_id,
            e.promotion,
            COALESCE(b.bout_order, 0) AS bout_order,
            COALESCE(b.bout_start_time_utc, e.event_date_utc) AS bout_datetime_utc,
            b.weight_class,
            b.fighter_red_id,
            b.fighter_blue_id,
            b.winner_fighter_id,
            b.result_method
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
            promotion=str(row["promotion"]),
            bout_order=int(row["bout_order"]),
            bout_datetime_utc=str(row["bout_datetime_utc"]),
            weight_class=_clean_text(row["weight_class"]),
            fighter_red_id=str(row["fighter_red_id"]),
            fighter_blue_id=str(row["fighter_blue_id"]),
            winner_fighter_id=_clean_text(row["winner_fighter_id"]),
            result_method=_clean_text(row["result_method"]),
        )
        for row in rows
    ]


def _build_history_rows(
    *,
    bouts: list[BoutRow],
    config: EloConfig,
    inactivity_decay_hook: InactivityDecayHook,
) -> list[dict[str, str]]:
    overall_ratings: dict[str, float] = {}
    weight_class_ratings: dict[tuple[str, str], float] = {}

    last_completed_fight_datetime: dict[str, datetime] = {}
    last_completed_weight_class_datetime: dict[tuple[str, str], datetime] = {}

    rows: list[dict[str, str]] = []

    for bout in bouts:
        bout_datetime = _parse_utc_datetime(bout.bout_datetime_utc)

        red_days_since_last = _days_since(last_completed_fight_datetime.get(bout.fighter_red_id), bout_datetime)
        blue_days_since_last = _days_since(last_completed_fight_datetime.get(bout.fighter_blue_id), bout_datetime)

        red_pre_overall = inactivity_decay_hook.adjust_rating(
            fighter_id=bout.fighter_red_id,
            dimension="overall",
            weight_class=None,
            current_rating=overall_ratings.get(bout.fighter_red_id, config.initial_rating),
            days_inactive=red_days_since_last,
        )
        blue_pre_overall = inactivity_decay_hook.adjust_rating(
            fighter_id=bout.fighter_blue_id,
            dimension="overall",
            weight_class=None,
            current_rating=overall_ratings.get(bout.fighter_blue_id, config.initial_rating),
            days_inactive=blue_days_since_last,
        )

        red_weight_days = _days_since_weight_class(
            fighter_id=bout.fighter_red_id,
            weight_class=bout.weight_class,
            reference_datetime=bout_datetime,
            last_completed_weight_class_datetime=last_completed_weight_class_datetime,
        )
        blue_weight_days = _days_since_weight_class(
            fighter_id=bout.fighter_blue_id,
            weight_class=bout.weight_class,
            reference_datetime=bout_datetime,
            last_completed_weight_class_datetime=last_completed_weight_class_datetime,
        )

        red_pre_weight = inactivity_decay_hook.adjust_rating(
            fighter_id=bout.fighter_red_id,
            dimension="weight_class",
            weight_class=bout.weight_class,
            current_rating=_get_weight_rating(
                weight_class_ratings=weight_class_ratings,
                fighter_id=bout.fighter_red_id,
                weight_class=bout.weight_class,
                initial_rating=config.initial_rating,
            ),
            days_inactive=red_weight_days,
        )
        blue_pre_weight = inactivity_decay_hook.adjust_rating(
            fighter_id=bout.fighter_blue_id,
            dimension="weight_class",
            weight_class=bout.weight_class,
            current_rating=_get_weight_rating(
                weight_class_ratings=weight_class_ratings,
                fighter_id=bout.fighter_blue_id,
                weight_class=bout.weight_class,
                initial_rating=config.initial_rating,
            ),
            days_inactive=blue_weight_days,
        )

        result_score_red = _resolve_red_outcome_score(bout)
        bout_completed = result_score_red is not None

        red_post_overall = red_pre_overall
        blue_post_overall = blue_pre_overall
        red_post_weight = red_pre_weight
        blue_post_weight = blue_pre_weight

        if bout_completed:
            red_expected_overall = _elo_expected_score(red_pre_overall, blue_pre_overall)
            blue_expected_overall = _elo_expected_score(blue_pre_overall, red_pre_overall)
            red_post_overall = red_pre_overall + config.k_factor * (result_score_red - red_expected_overall)
            blue_post_overall = blue_pre_overall + config.k_factor * ((1.0 - result_score_red) - blue_expected_overall)

            red_expected_weight = _elo_expected_score(red_pre_weight, blue_pre_weight)
            blue_expected_weight = _elo_expected_score(blue_pre_weight, red_pre_weight)
            red_post_weight = red_pre_weight + config.k_factor * (result_score_red - red_expected_weight)
            blue_post_weight = blue_pre_weight + config.k_factor * ((1.0 - result_score_red) - blue_expected_weight)

            overall_ratings[bout.fighter_red_id] = red_post_overall
            overall_ratings[bout.fighter_blue_id] = blue_post_overall

            if bout.weight_class is not None:
                weight_class_ratings[(bout.fighter_red_id, bout.weight_class)] = red_post_weight
                weight_class_ratings[(bout.fighter_blue_id, bout.weight_class)] = blue_post_weight
                last_completed_weight_class_datetime[(bout.fighter_red_id, bout.weight_class)] = bout_datetime
                last_completed_weight_class_datetime[(bout.fighter_blue_id, bout.weight_class)] = bout_datetime

            last_completed_fight_datetime[bout.fighter_red_id] = bout_datetime
            last_completed_fight_datetime[bout.fighter_blue_id] = bout_datetime

        rows.append(
            _make_history_row(
                bout=bout,
                fighter_id=bout.fighter_red_id,
                opponent_fighter_id=bout.fighter_blue_id,
                corner="red",
                result_score=result_score_red,
                bout_completed=bout_completed,
                days_since_last_fight=red_days_since_last,
                days_since_last_weight_class_fight=red_weight_days,
                config=config,
                pre_overall=red_pre_overall,
                post_overall=red_post_overall,
                pre_weight=red_pre_weight,
                post_weight=red_post_weight,
            )
        )
        rows.append(
            _make_history_row(
                bout=bout,
                fighter_id=bout.fighter_blue_id,
                opponent_fighter_id=bout.fighter_red_id,
                corner="blue",
                result_score=None if result_score_red is None else 1.0 - result_score_red,
                bout_completed=bout_completed,
                days_since_last_fight=blue_days_since_last,
                days_since_last_weight_class_fight=blue_weight_days,
                config=config,
                pre_overall=blue_pre_overall,
                post_overall=blue_post_overall,
                pre_weight=blue_pre_weight,
                post_weight=blue_post_weight,
            )
        )

    return rows


def _make_history_row(
    *,
    bout: BoutRow,
    fighter_id: str,
    opponent_fighter_id: str,
    corner: str,
    result_score: float | None,
    bout_completed: bool,
    days_since_last_fight: int | None,
    days_since_last_weight_class_fight: int | None,
    config: EloConfig,
    pre_overall: float,
    post_overall: float,
    pre_weight: float,
    post_weight: float,
) -> dict[str, str]:
    return {
        "bout_id": bout.bout_id,
        "event_id": bout.event_id,
        "promotion": bout.promotion,
        "bout_order": str(bout.bout_order),
        "bout_datetime_utc": bout.bout_datetime_utc,
        "bout_completed": "1" if bout_completed else "0",
        "weight_class": bout.weight_class or "",
        "winner_fighter_id": bout.winner_fighter_id or "",
        "fighter_id": fighter_id,
        "opponent_fighter_id": opponent_fighter_id,
        "corner": corner,
        "result_score": "" if result_score is None else f"{result_score:.6f}",
        "k_factor": f"{config.k_factor:.6f}",
        "days_since_last_fight": "" if days_since_last_fight is None else str(days_since_last_fight),
        "days_since_last_weight_class_fight": (
            "" if days_since_last_weight_class_fight is None else str(days_since_last_weight_class_fight)
        ),
        "pre_overall_elo": f"{pre_overall:.6f}",
        "post_overall_elo": f"{post_overall:.6f}",
        "pre_weight_class_elo": f"{pre_weight:.6f}",
        "post_weight_class_elo": f"{post_weight:.6f}",
    }


def _write_history_rows(*, output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(HISTORY_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _resolve_red_outcome_score(bout: BoutRow) -> float | None:
    if bout.winner_fighter_id == bout.fighter_red_id:
        return 1.0
    if bout.winner_fighter_id == bout.fighter_blue_id:
        return 0.0
    if bout.result_method is None:
        return None
    if "draw" in bout.result_method.lower():
        return 0.5
    return None


def _elo_expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _days_since(previous: datetime | None, current: datetime) -> int | None:
    if previous is None:
        return None
    days = (current - previous).days
    return days if days >= 0 else 0


def _days_since_weight_class(
    *,
    fighter_id: str,
    weight_class: str | None,
    reference_datetime: datetime,
    last_completed_weight_class_datetime: dict[tuple[str, str], datetime],
) -> int | None:
    if weight_class is None:
        return None
    last_datetime = last_completed_weight_class_datetime.get((fighter_id, weight_class))
    return _days_since(last_datetime, reference_datetime)


def _get_weight_rating(
    *,
    weight_class_ratings: dict[tuple[str, str], float],
    fighter_id: str,
    weight_class: str | None,
    initial_rating: float,
) -> float:
    if weight_class is None:
        return initial_rating
    return weight_class_ratings.get((fighter_id, weight_class), initial_rating)


def _parse_utc_datetime(value: str) -> datetime:
    if value.endswith("Z"):
        return datetime.fromisoformat(value[:-1]).replace(tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None
