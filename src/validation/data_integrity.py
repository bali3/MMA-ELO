"""Data integrity and chronology safeguards for MMA domain tables.

Scope:
- duplicate entity detection
- chronology consistency checks across events/bouts/markets/predictions
- missing critical field checks

These checks are read-only and designed to prevent future-data leakage.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable


@dataclass(frozen=True)
class ValidationIssue:
    """Single validation issue produced by a data-integrity check."""

    check_name: str
    table_name: str
    row_ref: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    """Aggregate validation report for one database snapshot."""

    issues: tuple[ValidationIssue, ...]

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    @property
    def is_valid(self) -> bool:
        return self.issue_count == 0


CRITICAL_FIELDS_BY_TABLE: dict[str, tuple[str, ...]] = {
    "fighters": (
        "fighter_id",
        "full_name",
        "source_system",
        "source_record_id",
        "ingested_at_utc",
    ),
    "events": (
        "event_id",
        "event_name",
        "event_date_utc",
        "source_system",
        "source_record_id",
        "ingested_at_utc",
    ),
    "bouts": (
        "bout_id",
        "event_id",
        "fighter_red_id",
        "fighter_blue_id",
        "source_system",
        "source_record_id",
        "ingested_at_utc",
    ),
    "fighter_bout_stats": (
        "fighter_bout_stats_id",
        "bout_id",
        "fighter_id",
        "opponent_fighter_id",
        "corner",
        "source_system",
        "source_record_id",
        "ingested_at_utc",
    ),
    "markets": (
        "market_id",
        "bout_id",
        "sportsbook",
        "market_type",
        "market_timestamp_utc",
        "source_system",
        "source_record_id",
        "ingested_at_utc",
    ),
    "model_predictions": (
        "model_prediction_id",
        "bout_id",
        "target_fighter_id",
        "model_name",
        "model_version",
        "model_run_id",
        "prediction_type",
        "predicted_probability",
        "generated_at_utc",
        "feature_cutoff_utc",
        "source_system",
        "source_record_id",
    ),
}


def run_data_validations(connection: sqlite3.Connection) -> ValidationReport:
    """Execute all data-integrity checks against an open SQLite connection."""

    all_issues: list[ValidationIssue] = []
    all_issues.extend(check_duplicate_fighters(connection))
    all_issues.extend(check_duplicate_events(connection))
    all_issues.extend(check_duplicate_bouts(connection))
    all_issues.extend(check_chronology_consistency(connection))
    all_issues.extend(check_missing_critical_fields(connection))

    all_issues.sort(key=lambda issue: (issue.check_name, issue.table_name, issue.row_ref))
    return ValidationReport(issues=tuple(all_issues))


def check_duplicate_fighters(connection: sqlite3.Connection) -> tuple[ValidationIssue, ...]:
    """Flag likely duplicate fighters by normalized name and date of birth."""

    rows = connection.execute(
        """
        SELECT
            LOWER(TRIM(full_name)) AS normalized_name,
            date_of_birth_utc AS dob,
            COUNT(*) AS duplicate_count,
            GROUP_CONCAT(fighter_id, ',') AS fighter_ids
        FROM fighters
        WHERE date_of_birth_utc IS NOT NULL
          AND TRIM(date_of_birth_utc) != ''
        GROUP BY normalized_name, dob
        HAVING duplicate_count > 1
        """
    ).fetchall()

    return tuple(
        ValidationIssue(
            check_name="duplicate_fighters",
            table_name="fighters",
            row_ref=str(row[3]),
            message=(
                "Multiple fighters share normalized full_name/date_of_birth_utc "
                f"(name={row[0]!r}, dob={row[1]!r}, count={row[2]})."
            ),
        )
        for row in rows
    )


def check_duplicate_events(connection: sqlite3.Connection) -> tuple[ValidationIssue, ...]:
    """Flag likely duplicate events by normalized event name and event date."""

    rows = connection.execute(
        """
        SELECT
            LOWER(TRIM(event_name)) AS normalized_name,
            event_date_utc,
            COUNT(*) AS duplicate_count,
            GROUP_CONCAT(event_id, ',') AS event_ids
        FROM events
        GROUP BY normalized_name, event_date_utc
        HAVING duplicate_count > 1
        """
    ).fetchall()

    return tuple(
        ValidationIssue(
            check_name="duplicate_events",
            table_name="events",
            row_ref=str(row[3]),
            message=(
                "Multiple events share normalized event_name/event_date_utc "
                f"(name={row[0]!r}, date={row[1]!r}, count={row[2]})."
            ),
        )
        for row in rows
    )


def check_duplicate_bouts(connection: sqlite3.Connection) -> tuple[ValidationIssue, ...]:
    """Flag duplicate bouts within an event regardless of corner assignment."""

    rows = connection.execute(
        """
        SELECT
            event_id,
            CASE
                WHEN fighter_red_id < fighter_blue_id THEN fighter_red_id
                ELSE fighter_blue_id
            END AS fighter_a,
            CASE
                WHEN fighter_red_id < fighter_blue_id THEN fighter_blue_id
                ELSE fighter_red_id
            END AS fighter_b,
            COUNT(*) AS duplicate_count,
            GROUP_CONCAT(bout_id, ',') AS bout_ids
        FROM bouts
        GROUP BY event_id, fighter_a, fighter_b
        HAVING duplicate_count > 1
        """
    ).fetchall()

    return tuple(
        ValidationIssue(
            check_name="duplicate_bouts",
            table_name="bouts",
            row_ref=str(row[4]),
            message=(
                "Multiple bouts share the same event and fighter pair "
                f"(event_id={row[0]!r}, fighter_a={row[1]!r}, fighter_b={row[2]!r}, count={row[3]})."
            ),
        )
        for row in rows
    )


def check_chronology_consistency(connection: sqlite3.Connection) -> tuple[ValidationIssue, ...]:
    """Run chronology and leakage-prevention checks across related tables."""

    issues: list[ValidationIssue] = []
    issues.extend(_check_invalid_event_datetimes(connection))
    issues.extend(_check_orphan_bout_events(connection))
    issues.extend(_check_bout_start_vs_event_date(connection))
    issues.extend(_check_fighter_bout_stats_alignment(connection))
    issues.extend(_check_market_timestamps_before_bout_start(connection))
    issues.extend(_check_prediction_timestamps_before_bout_start(connection))
    issues.extend(_check_prediction_feature_cutoff_order(connection))
    return tuple(issues)


def check_missing_critical_fields(connection: sqlite3.Connection) -> tuple[ValidationIssue, ...]:
    """Validate required non-empty fields for critical domain tables."""

    issues: list[ValidationIssue] = []
    for table_name, critical_fields in CRITICAL_FIELDS_BY_TABLE.items():
        row_id_column = _row_id_column(table_name)
        for column in critical_fields:
            condition = (
                f"{column} IS NULL OR (typeof({column}) = 'text' AND TRIM({column}) = '')"
            )
            query = (
                f"SELECT {row_id_column} AS row_id FROM {table_name} "
                f"WHERE {condition} ORDER BY {row_id_column}"
            )
            rows = connection.execute(query).fetchall()
            for row in rows:
                row_id = str(row[0]) if row[0] is not None else "<null-row-id>"
                issues.append(
                    ValidationIssue(
                        check_name="missing_critical_fields",
                        table_name=table_name,
                        row_ref=row_id,
                        message=f"Missing critical field {column!r}.",
                    )
                )
    return tuple(issues)


def _check_invalid_event_datetimes(connection: sqlite3.Connection) -> Iterable[ValidationIssue]:
    rows = connection.execute("SELECT event_id, event_date_utc FROM events").fetchall()
    issues: list[ValidationIssue] = []
    for event_id, event_date_utc in rows:
        if _parse_utc_timestamp(event_date_utc) is None:
            issues.append(
                ValidationIssue(
                    check_name="chronology_invalid_event_date",
                    table_name="events",
                    row_ref=str(event_id),
                    message=f"event_date_utc is missing or invalid: {event_date_utc!r}.",
                )
            )
    return issues


def _check_orphan_bout_events(connection: sqlite3.Connection) -> Iterable[ValidationIssue]:
    rows = connection.execute(
        """
        SELECT b.bout_id, b.event_id
        FROM bouts AS b
        LEFT JOIN events AS e ON e.event_id = b.event_id
        WHERE e.event_id IS NULL
        """
    ).fetchall()

    return (
        ValidationIssue(
            check_name="chronology_orphan_bout_event",
            table_name="bouts",
            row_ref=str(row[0]),
            message=f"Bout references missing event_id={row[1]!r}.",
        )
        for row in rows
    )


def _check_bout_start_vs_event_date(connection: sqlite3.Connection) -> Iterable[ValidationIssue]:
    rows = connection.execute(
        """
        SELECT b.bout_id, b.bout_start_time_utc, e.event_date_utc
        FROM bouts AS b
        JOIN events AS e ON e.event_id = b.event_id
        """
    ).fetchall()

    issues: list[ValidationIssue] = []
    for bout_id, bout_start_time_utc, event_date_utc in rows:
        event_dt = _parse_utc_timestamp(event_date_utc)
        if event_dt is None:
            continue

        if bout_start_time_utc is None:
            continue

        bout_dt = _parse_utc_timestamp(bout_start_time_utc)
        if bout_dt is None:
            issues.append(
                ValidationIssue(
                    check_name="chronology_invalid_bout_start",
                    table_name="bouts",
                    row_ref=str(bout_id),
                    message=f"bout_start_time_utc is invalid: {bout_start_time_utc!r}.",
                )
            )
            continue

        if bout_dt < event_dt:
            issues.append(
                ValidationIssue(
                    check_name="chronology_bout_before_event_date",
                    table_name="bouts",
                    row_ref=str(bout_id),
                    message=(
                        "bout_start_time_utc occurs before event_date_utc "
                        f"({bout_start_time_utc!r} < {event_date_utc!r})."
                    ),
                )
            )

        if bout_dt > event_dt + timedelta(hours=72):
            issues.append(
                ValidationIssue(
                    check_name="chronology_bout_far_after_event_date",
                    table_name="bouts",
                    row_ref=str(bout_id),
                    message=(
                        "bout_start_time_utc is more than 72 hours after event_date_utc "
                        f"({bout_start_time_utc!r} vs {event_date_utc!r})."
                    ),
                )
            )

    return issues


def _check_fighter_bout_stats_alignment(connection: sqlite3.Connection) -> Iterable[ValidationIssue]:
    rows = connection.execute(
        """
        SELECT
            s.fighter_bout_stats_id,
            s.bout_id,
            s.fighter_id,
            s.opponent_fighter_id,
            b.fighter_red_id,
            b.fighter_blue_id
        FROM fighter_bout_stats AS s
        JOIN bouts AS b ON b.bout_id = s.bout_id
        """
    ).fetchall()

    issues: list[ValidationIssue] = []
    for row in rows:
        stats_id, bout_id, fighter_id, opponent_id, red_id, blue_id = row
        valid_pair = (fighter_id == red_id and opponent_id == blue_id) or (
            fighter_id == blue_id and opponent_id == red_id
        )
        if valid_pair:
            continue

        issues.append(
            ValidationIssue(
                check_name="chronology_stats_fighter_mismatch",
                table_name="fighter_bout_stats",
                row_ref=str(stats_id),
                message=(
                    "fighter_bout_stats fighter/opponent IDs do not match bout fighter pair "
                    f"(bout_id={bout_id!r})."
                ),
            )
        )

    return issues


def _check_market_timestamps_before_bout_start(connection: sqlite3.Connection) -> Iterable[ValidationIssue]:
    rows = connection.execute(
        """
        SELECT m.market_id, m.market_timestamp_utc, b.bout_start_time_utc
        FROM markets AS m
        JOIN bouts AS b ON b.bout_id = m.bout_id
        WHERE b.bout_start_time_utc IS NOT NULL
        """
    ).fetchall()

    issues: list[ValidationIssue] = []
    for market_id, market_ts, bout_start_ts in rows:
        market_dt = _parse_utc_timestamp(market_ts)
        bout_dt = _parse_utc_timestamp(bout_start_ts)

        if market_dt is None:
            issues.append(
                ValidationIssue(
                    check_name="chronology_invalid_market_timestamp",
                    table_name="markets",
                    row_ref=str(market_id),
                    message=f"market_timestamp_utc is missing or invalid: {market_ts!r}.",
                )
            )
            continue

        if bout_dt is None:
            continue

        if market_dt > bout_dt:
            issues.append(
                ValidationIssue(
                    check_name="chronology_market_after_bout_start",
                    table_name="markets",
                    row_ref=str(market_id),
                    message=(
                        "Market snapshot occurs after bout_start_time_utc, which is leakage-prone "
                        f"({market_ts!r} > {bout_start_ts!r})."
                    ),
                )
            )

    return issues


def _check_prediction_timestamps_before_bout_start(connection: sqlite3.Connection) -> Iterable[ValidationIssue]:
    rows = connection.execute(
        """
        SELECT p.model_prediction_id, p.generated_at_utc, b.bout_start_time_utc
        FROM model_predictions AS p
        JOIN bouts AS b ON b.bout_id = p.bout_id
        WHERE b.bout_start_time_utc IS NOT NULL
        """
    ).fetchall()

    issues: list[ValidationIssue] = []
    for prediction_id, generated_at_utc, bout_start_time_utc in rows:
        generated_dt = _parse_utc_timestamp(generated_at_utc)
        bout_dt = _parse_utc_timestamp(bout_start_time_utc)

        if generated_dt is None:
            issues.append(
                ValidationIssue(
                    check_name="chronology_invalid_prediction_generated_at",
                    table_name="model_predictions",
                    row_ref=str(prediction_id),
                    message=f"generated_at_utc is missing or invalid: {generated_at_utc!r}.",
                )
            )
            continue

        if bout_dt is None:
            continue

        if generated_dt > bout_dt:
            issues.append(
                ValidationIssue(
                    check_name="chronology_prediction_after_bout_start",
                    table_name="model_predictions",
                    row_ref=str(prediction_id),
                    message=(
                        "Prediction generated_at_utc is after bout_start_time_utc, which leaks outcome timing "
                        f"({generated_at_utc!r} > {bout_start_time_utc!r})."
                    ),
                )
            )

    return issues


def _check_prediction_feature_cutoff_order(connection: sqlite3.Connection) -> Iterable[ValidationIssue]:
    rows = connection.execute(
        """
        SELECT p.model_prediction_id, p.feature_cutoff_utc, p.generated_at_utc, b.bout_start_time_utc
        FROM model_predictions AS p
        JOIN bouts AS b ON b.bout_id = p.bout_id
        """
    ).fetchall()

    issues: list[ValidationIssue] = []
    for prediction_id, feature_cutoff_utc, generated_at_utc, bout_start_time_utc in rows:
        cutoff_dt = _parse_utc_timestamp(feature_cutoff_utc)
        generated_dt = _parse_utc_timestamp(generated_at_utc)
        bout_dt = _parse_utc_timestamp(bout_start_time_utc) if bout_start_time_utc is not None else None

        if cutoff_dt is None:
            issues.append(
                ValidationIssue(
                    check_name="chronology_invalid_feature_cutoff",
                    table_name="model_predictions",
                    row_ref=str(prediction_id),
                    message=f"feature_cutoff_utc is missing or invalid: {feature_cutoff_utc!r}.",
                )
            )
            continue

        if generated_dt is not None and cutoff_dt > generated_dt:
            issues.append(
                ValidationIssue(
                    check_name="chronology_feature_cutoff_after_generated_at",
                    table_name="model_predictions",
                    row_ref=str(prediction_id),
                    message=(
                        "feature_cutoff_utc occurs after generated_at_utc "
                        f"({feature_cutoff_utc!r} > {generated_at_utc!r})."
                    ),
                )
            )

        if bout_dt is not None and cutoff_dt > bout_dt:
            issues.append(
                ValidationIssue(
                    check_name="chronology_feature_cutoff_after_bout_start",
                    table_name="model_predictions",
                    row_ref=str(prediction_id),
                    message=(
                        "feature_cutoff_utc occurs after bout_start_time_utc, which is leakage-prone "
                        f"({feature_cutoff_utc!r} > {bout_start_time_utc!r})."
                    ),
                )
            )

    return issues


def _row_id_column(table_name: str) -> str:
    row_id_map = {
        "fighters": "fighter_id",
        "events": "event_id",
        "bouts": "bout_id",
        "fighter_bout_stats": "fighter_bout_stats_id",
        "markets": "market_id",
        "model_predictions": "model_prediction_id",
    }
    return row_id_map[table_name]


def _parse_utc_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None

    cleaned = value.strip()
    if not cleaned:
        return None

    normalized = cleaned.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)
