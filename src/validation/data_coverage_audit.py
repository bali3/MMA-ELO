"""Data coverage and feature completeness audit reporting."""

from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from features import DEFAULT_KEY_FEATURES, build_feature_report


@dataclass(frozen=True)
class DataCoverageAuditResult:
    """Persisted audit report summary."""

    report_path: Path
    total_events: int
    total_completed_bouts: int
    total_fighters: int
    total_fighter_bout_stats_rows: int
    completed_bouts_with_two_stats_rows: int
    completed_bouts_with_one_stats_row: int
    completed_bouts_with_zero_stats_rows: int


def run_data_coverage_audit(
    *,
    db_path: Path,
    promotion: str,
    features_path: Path,
    report_output_path: Path,
    key_features: Sequence[str] = DEFAULT_KEY_FEATURES,
) -> DataCoverageAuditResult:
    """Build a plain-text audit of coverage, missingness, and join failures."""

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        counts = _load_counts(connection=connection, promotion=promotion)
        bout_stats_coverage = _load_bout_stats_coverage(connection=connection, promotion=promotion)
        bout_stats_coverage_by_year = _load_bout_stats_coverage_by_year(connection=connection, promotion=promotion)
        parser_path_proxy = _load_bout_stats_parser_path_proxy(connection=connection, promotion=promotion)
        date_spans = _load_date_spans(connection=connection, promotion=promotion)
        join_failures = dict(_load_join_failures(connection=connection, promotion=promotion))
        fighter_meta = _load_fighter_meta_flags(connection=connection)
        valid_feature_keys = _load_valid_feature_join_keys(connection=connection, promotion=promotion)

    feature_report = build_feature_report(features_path=features_path, key_features=key_features)
    feature_rows = _load_feature_rows(features_path=features_path)
    join_failures.update(_feature_join_failures(feature_rows=feature_rows, valid_keys=valid_feature_keys))
    missingness_reasons = _build_missingness_reasons(
        feature_rows=feature_rows,
        key_features=key_features,
        fighter_meta=fighter_meta,
    )

    lines = [
        "data_coverage_audit_report",
        f"promotion={promotion}",
        "coverage_totals",
        f"total_events={counts['total_events']}",
        f"total_completed_bouts={counts['total_completed_bouts']}",
        f"total_fighters={counts['total_fighters']}",
        f"total_fighter_bout_stats_rows={counts['total_fighter_bout_stats_rows']}",
        "bout_stats_coverage_totals",
        f"completed_bouts={bout_stats_coverage['completed_bouts']}",
        f"bouts_with_two_stats_rows={bout_stats_coverage['bouts_with_two_stats_rows']}",
        f"bouts_with_one_stats_row={bout_stats_coverage['bouts_with_one_stats_row']}",
        f"bouts_with_zero_stats_rows={bout_stats_coverage['bouts_with_zero_stats_rows']}",
        "bout_stats_coverage_by_year",
    ]
    if not bout_stats_coverage_by_year:
        lines.append("year=n/a completed_bouts=0 bouts_with_two_stats_rows=0 bouts_with_one_stats_row=0 bouts_with_zero_stats_rows=0")
    else:
        for row in bout_stats_coverage_by_year:
            lines.append(
                (
                    f"year={row['year']} completed_bouts={row['completed_bouts']} "
                    f"bouts_with_two_stats_rows={row['bouts_with_two_stats_rows']} "
                    f"bouts_with_one_stats_row={row['bouts_with_one_stats_row']} "
                    f"bouts_with_zero_stats_rows={row['bouts_with_zero_stats_rows']}"
                )
            )

    lines.extend(
        [
            "bout_stats_parser_path_proxy",
            *[f"{key}={value}" for key, value in sorted(parser_path_proxy.items())],
            "feature_coverage",
        ]
    )
    for feature_name in key_features:
        coverage = feature_report.coverage_rates.get(feature_name, 0.0)
        missing = feature_report.missing_counts.get(feature_name, 0)
        lines.append(f"feature={feature_name} coverage={coverage:.8f} missing={missing}")

    lines.append("feature_missingness_reasons")
    for feature_name in key_features:
        reasons = missingness_reasons.get(feature_name, {})
        if len(reasons) == 0:
            lines.append(f"feature={feature_name} reason=none count=0")
            continue
        for reason, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"feature={feature_name} reason={reason} count={count}")

    lines.append("earliest_latest_dates")
    for table_name, span in date_spans.items():
        lines.append(f"table={table_name} earliest={span[0]} latest={span[1]}")

    lines.append("join_matching_failures")
    for key, value in sorted(join_failures.items()):
        lines.append(f"{key}={value}")

    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return DataCoverageAuditResult(
        report_path=report_output_path,
        total_events=int(counts["total_events"]),
        total_completed_bouts=int(counts["total_completed_bouts"]),
        total_fighters=int(counts["total_fighters"]),
        total_fighter_bout_stats_rows=int(counts["total_fighter_bout_stats_rows"]),
        completed_bouts_with_two_stats_rows=int(bout_stats_coverage["bouts_with_two_stats_rows"]),
        completed_bouts_with_one_stats_row=int(bout_stats_coverage["bouts_with_one_stats_row"]),
        completed_bouts_with_zero_stats_rows=int(bout_stats_coverage["bouts_with_zero_stats_rows"]),
    )


def _load_counts(*, connection: sqlite3.Connection, promotion: str) -> Mapping[str, str]:
    cursor = connection.cursor()
    return {
        "total_events": str(
            _single_count(cursor, "SELECT COUNT(*) FROM events WHERE UPPER(promotion) = UPPER(?)", promotion)
        ),
        "total_completed_bouts": str(
            _single_count(
                cursor,
                """
                SELECT COUNT(*)
                FROM bouts AS b
                JOIN events AS e ON e.event_id = b.event_id
                WHERE UPPER(e.promotion) = UPPER(?)
                  AND b.winner_fighter_id IS NOT NULL
                """,
                promotion,
            )
        ),
        "total_fighters": str(_single_count(cursor, "SELECT COUNT(*) FROM fighters")),
        "total_fighter_bout_stats_rows": str(_single_count(cursor, "SELECT COUNT(*) FROM fighter_bout_stats")),
    }


def _load_date_spans(*, connection: sqlite3.Connection, promotion: str) -> dict[str, tuple[str, str]]:
    spans: dict[str, tuple[str, str]] = {}
    cursor = connection.cursor()
    rows = {
        "events": cursor.execute(
            "SELECT MIN(event_date_utc), MAX(event_date_utc) FROM events WHERE UPPER(promotion) = UPPER(?)",
            (promotion,),
        ).fetchone(),
        "bouts": cursor.execute(
            """
            SELECT
                MIN(COALESCE(b.bout_start_time_utc, e.event_date_utc)),
                MAX(COALESCE(b.bout_start_time_utc, e.event_date_utc))
            FROM bouts AS b
            JOIN events AS e ON e.event_id = b.event_id
            WHERE UPPER(e.promotion) = UPPER(?)
            """,
            (promotion,),
        ).fetchone(),
        "fighter_bout_stats": cursor.execute(
            """
            SELECT
                MIN(COALESCE(b.bout_start_time_utc, e.event_date_utc)),
                MAX(COALESCE(b.bout_start_time_utc, e.event_date_utc))
            FROM fighter_bout_stats AS s
            JOIN bouts AS b ON b.bout_id = s.bout_id
            JOIN events AS e ON e.event_id = b.event_id
            WHERE UPPER(e.promotion) = UPPER(?)
            """,
            (promotion,),
        ).fetchone(),
        "markets": cursor.execute(
            """
            SELECT MIN(m.market_timestamp_utc), MAX(m.market_timestamp_utc)
            FROM markets AS m
            JOIN bouts AS b ON b.bout_id = m.bout_id
            JOIN events AS e ON e.event_id = b.event_id
            WHERE UPPER(e.promotion) = UPPER(?)
            """,
            (promotion,),
        ).fetchone(),
        "model_predictions": cursor.execute(
            """
            SELECT MIN(p.generated_at_utc), MAX(p.generated_at_utc)
            FROM model_predictions AS p
            JOIN bouts AS b ON b.bout_id = p.bout_id
            JOIN events AS e ON e.event_id = b.event_id
            WHERE UPPER(e.promotion) = UPPER(?)
            """,
            (promotion,),
        ).fetchone(),
    }
    for table_name, row in rows.items():
        earliest = "n/a" if row is None or row[0] is None else str(row[0])
        latest = "n/a" if row is None or row[1] is None else str(row[1])
        spans[table_name] = (earliest, latest)
    return spans


def _load_bout_stats_coverage(*, connection: sqlite3.Connection, promotion: str) -> dict[str, str]:
    row = connection.execute(
        """
        WITH completed AS (
            SELECT b.bout_id, COUNT(s.fighter_bout_stats_id) AS stat_rows
            FROM bouts AS b
            JOIN events AS e ON e.event_id = b.event_id
            LEFT JOIN fighter_bout_stats AS s ON s.bout_id = b.bout_id
            WHERE UPPER(e.promotion) = UPPER(?)
              AND b.winner_fighter_id IS NOT NULL
            GROUP BY b.bout_id
        )
        SELECT
            COUNT(*) AS completed_bouts,
            SUM(CASE WHEN stat_rows = 2 THEN 1 ELSE 0 END) AS bouts_with_two_stats_rows,
            SUM(CASE WHEN stat_rows = 1 THEN 1 ELSE 0 END) AS bouts_with_one_stats_row,
            SUM(CASE WHEN stat_rows = 0 THEN 1 ELSE 0 END) AS bouts_with_zero_stats_rows
        FROM completed
        """,
        (promotion,),
    ).fetchone()
    if row is None:
        return {
            "completed_bouts": "0",
            "bouts_with_two_stats_rows": "0",
            "bouts_with_one_stats_row": "0",
            "bouts_with_zero_stats_rows": "0",
        }
    return {
        "completed_bouts": str(int(row[0] or 0)),
        "bouts_with_two_stats_rows": str(int(row[1] or 0)),
        "bouts_with_one_stats_row": str(int(row[2] or 0)),
        "bouts_with_zero_stats_rows": str(int(row[3] or 0)),
    }


def _load_bout_stats_coverage_by_year(*, connection: sqlite3.Connection, promotion: str) -> list[dict[str, str]]:
    rows = connection.execute(
        """
        WITH completed AS (
            SELECT
                b.bout_id,
                strftime('%Y', COALESCE(b.bout_start_time_utc, e.event_date_utc)) AS year,
                COUNT(s.fighter_bout_stats_id) AS stat_rows
            FROM bouts AS b
            JOIN events AS e ON e.event_id = b.event_id
            LEFT JOIN fighter_bout_stats AS s ON s.bout_id = b.bout_id
            WHERE UPPER(e.promotion) = UPPER(?)
              AND b.winner_fighter_id IS NOT NULL
            GROUP BY b.bout_id, year
        )
        SELECT
            year,
            COUNT(*) AS completed_bouts,
            SUM(CASE WHEN stat_rows = 2 THEN 1 ELSE 0 END) AS bouts_with_two_stats_rows,
            SUM(CASE WHEN stat_rows = 1 THEN 1 ELSE 0 END) AS bouts_with_one_stats_row,
            SUM(CASE WHEN stat_rows = 0 THEN 1 ELSE 0 END) AS bouts_with_zero_stats_rows
        FROM completed
        GROUP BY year
        ORDER BY year
        """,
        (promotion,),
    ).fetchall()
    return [
        {
            "year": str(row[0]),
            "completed_bouts": str(int(row[1] or 0)),
            "bouts_with_two_stats_rows": str(int(row[2] or 0)),
            "bouts_with_one_stats_row": str(int(row[3] or 0)),
            "bouts_with_zero_stats_rows": str(int(row[4] or 0)),
        }
        for row in rows
    ]


def _load_bout_stats_parser_path_proxy(*, connection: sqlite3.Connection, promotion: str) -> dict[str, str]:
    row = connection.execute(
        """
        SELECT
            SUM(
                CASE WHEN
                    s.sig_strikes_attempted IS NOT NULL
                    OR s.total_strikes_landed IS NOT NULL
                    OR s.total_strikes_attempted IS NOT NULL
                    OR s.takedowns_attempted IS NOT NULL
                    OR s.reversals IS NOT NULL
                    OR s.control_time_seconds IS NOT NULL
                THEN 1 ELSE 0 END
            ) AS rows_with_fight_details_shape,
            SUM(
                CASE WHEN
                    s.sig_strikes_attempted IS NULL
                    AND s.total_strikes_landed IS NULL
                    AND s.total_strikes_attempted IS NULL
                    AND s.takedowns_attempted IS NULL
                    AND s.reversals IS NULL
                    AND s.control_time_seconds IS NULL
                    AND (
                        s.knockdowns IS NOT NULL
                        OR s.sig_strikes_landed IS NOT NULL
                        OR s.takedowns_landed IS NOT NULL
                        OR s.submission_attempts IS NOT NULL
                    )
                THEN 1 ELSE 0 END
            ) AS rows_with_event_details_fallback_shape,
            SUM(
                CASE WHEN
                    s.knockdowns IS NULL
                    AND s.sig_strikes_landed IS NULL
                    AND s.sig_strikes_attempted IS NULL
                    AND s.total_strikes_landed IS NULL
                    AND s.total_strikes_attempted IS NULL
                    AND s.takedowns_landed IS NULL
                    AND s.takedowns_attempted IS NULL
                    AND s.submission_attempts IS NULL
                    AND s.reversals IS NULL
                    AND s.control_time_seconds IS NULL
                THEN 1 ELSE 0 END
            ) AS rows_with_empty_stats_shape
        FROM fighter_bout_stats AS s
        JOIN bouts AS b ON b.bout_id = s.bout_id
        JOIN events AS e ON e.event_id = b.event_id
        WHERE UPPER(e.promotion) = UPPER(?)
        """,
        (promotion,),
    ).fetchone()
    if row is None:
        return {
            "rows_with_fight_details_shape": "0",
            "rows_with_event_details_fallback_shape": "0",
            "rows_with_empty_stats_shape": "0",
        }
    return {
        "rows_with_fight_details_shape": str(int(row[0] or 0)),
        "rows_with_event_details_fallback_shape": str(int(row[1] or 0)),
        "rows_with_empty_stats_shape": str(int(row[2] or 0)),
    }


def _load_join_failures(*, connection: sqlite3.Connection, promotion: str) -> Mapping[str, str]:
    cursor = connection.cursor()
    total_bouts = _single_count(
        cursor,
        """
        SELECT COUNT(*)
        FROM bouts AS b
        JOIN events AS e ON e.event_id = b.event_id
        WHERE UPPER(e.promotion) = UPPER(?)
        """,
        promotion,
    )
    return {
        "bouts_missing_red_fighter": str(
            _single_count(
                cursor,
                """
                SELECT COUNT(*)
                FROM bouts AS b
                JOIN events AS e ON e.event_id = b.event_id
                LEFT JOIN fighters AS f ON f.fighter_id = b.fighter_red_id
                WHERE UPPER(e.promotion) = UPPER(?)
                  AND f.fighter_id IS NULL
                """,
                promotion,
            )
        ),
        "bouts_missing_blue_fighter": str(
            _single_count(
                cursor,
                """
                SELECT COUNT(*)
                FROM bouts AS b
                JOIN events AS e ON e.event_id = b.event_id
                LEFT JOIN fighters AS f ON f.fighter_id = b.fighter_blue_id
                WHERE UPPER(e.promotion) = UPPER(?)
                  AND f.fighter_id IS NULL
                """,
                promotion,
            )
        ),
        "bouts_without_two_stats_rows": str(
            _single_count(
                cursor,
                """
                SELECT COUNT(*)
                FROM (
                    SELECT b.bout_id, COUNT(s.fighter_bout_stats_id) AS stat_rows
                    FROM bouts AS b
                    JOIN events AS e ON e.event_id = b.event_id
                    LEFT JOIN fighter_bout_stats AS s ON s.bout_id = b.bout_id
                    WHERE UPPER(e.promotion) = UPPER(?)
                    GROUP BY b.bout_id
                    HAVING stat_rows < 2
                )
                """,
                promotion,
            )
        ),
        "fighter_bout_stats_missing_fighter": str(
            _single_count(
                cursor,
                """
                SELECT COUNT(*)
                FROM fighter_bout_stats AS s
                LEFT JOIN fighters AS f ON f.fighter_id = s.fighter_id
                WHERE f.fighter_id IS NULL
                """,
            )
        ),
        "fighter_bout_stats_missing_opponent": str(
            _single_count(
                cursor,
                """
                SELECT COUNT(*)
                FROM fighter_bout_stats AS s
                LEFT JOIN fighters AS f ON f.fighter_id = s.opponent_fighter_id
                WHERE f.fighter_id IS NULL
                """,
            )
        ),
        "expected_feature_rows_from_bouts": str(total_bouts * 2),
    }


def _load_valid_feature_join_keys(*, connection: sqlite3.Connection, promotion: str) -> set[tuple[str, str, str]]:
    rows = connection.execute(
        """
        SELECT b.bout_id, b.fighter_red_id, b.fighter_blue_id
        FROM bouts AS b
        JOIN events AS e ON e.event_id = b.event_id
        WHERE UPPER(e.promotion) = UPPER(?)
        """,
        (promotion,),
    ).fetchall()
    keys: set[tuple[str, str, str]] = set()
    for row in rows:
        bout_id = str(row[0])
        red = str(row[1])
        blue = str(row[2])
        keys.add((bout_id, red, blue))
        keys.add((bout_id, blue, red))
    return keys


def _feature_join_failures(
    *,
    feature_rows: Sequence[Mapping[str, str]],
    valid_keys: set[tuple[str, str, str]],
) -> Mapping[str, str]:
    total = len(feature_rows)
    missing = 0
    for row in feature_rows:
        key = (
            str(row.get("bout_id", "")).strip(),
            str(row.get("fighter_id", "")).strip(),
            str(row.get("opponent_id", "")).strip(),
        )
        if key not in valid_keys:
            missing += 1
    return {
        "feature_rows_total": str(total),
        "feature_rows_without_bout_fighter_match": str(missing),
    }


def _load_feature_rows(*, features_path: Path) -> list[dict[str, str]]:
    with features_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_fighter_meta_flags(*, connection: sqlite3.Connection) -> dict[str, tuple[bool, bool, bool, bool]]:
    rows = connection.execute(
        "SELECT fighter_id, date_of_birth_utc, height_cm, reach_cm, stance FROM fighters"
    ).fetchall()
    result: dict[str, tuple[bool, bool, bool, bool]] = {}
    for row in rows:
        fighter_id = str(row[0])
        result[fighter_id] = (
            row[1] not in (None, ""),
            row[2] is not None,
            row[3] is not None,
            row[4] not in (None, ""),
        )
    return result


def _build_missingness_reasons(
    *,
    feature_rows: Sequence[Mapping[str, str]],
    key_features: Sequence[str],
    fighter_meta: Mapping[str, tuple[bool, bool, bool, bool]],
) -> dict[str, dict[str, int]]:
    reasons: dict[str, dict[str, int]] = {feature: {} for feature in key_features}
    for row in feature_rows:
        fighter_id = str(row.get("fighter_id", "")).strip()
        opponent_id = str(row.get("opponent_id", "")).strip()
        prior_count = _as_int(row.get("prior_fight_count"))
        fighter_flags = fighter_meta.get(fighter_id, (False, False, False, False))
        opponent_flags = fighter_meta.get(opponent_id, (False, False, False, False))

        for feature in key_features:
            value = str(row.get(feature, "")).strip()
            if value != "":
                continue
            reason = _reason_for_missing_feature(
                feature=feature,
                prior_count=prior_count,
                fighter_flags=fighter_flags,
                opponent_flags=opponent_flags,
            )
            bucket = reasons.setdefault(feature, {})
            bucket[reason] = bucket.get(reason, 0) + 1
    return reasons


def _reason_for_missing_feature(
    *,
    feature: str,
    prior_count: int | None,
    fighter_flags: tuple[bool, bool, bool, bool],
    opponent_flags: tuple[bool, bool, bool, bool],
) -> str:
    fighter_has_dob, fighter_has_height, fighter_has_reach, _fighter_has_stance = fighter_flags
    _, opponent_has_height, opponent_has_reach, _opponent_has_stance = opponent_flags

    if feature == "fighter_age_years":
        return "missing_fighter_dob" if not fighter_has_dob else "unknown"
    if feature == "height_diff_cm":
        return "missing_height_metadata" if not (fighter_has_height and opponent_has_height) else "unknown"
    if feature == "reach_diff_cm":
        return "missing_reach_metadata" if not (fighter_has_reach and opponent_has_reach) else "unknown"
    if feature in {"days_since_last_fight", "prior_fight_count", "finish_rate_all", "decision_rate_all", "rematch_flag"}:
        return "no_prior_fights" if (prior_count is None or prior_count < 1) else "missing_history_row"
    if feature in {"recent_form_win_rate_l3", "opponent_strength_avg_l3"}:
        if prior_count is None or prior_count < 1:
            return "no_prior_fights"
        if prior_count < 3:
            return "insufficient_prior_fights_l3"
        return "missing_recent_window_stats"
    if feature in {"recent_form_win_rate_l5", "opponent_strength_avg_l5"}:
        if prior_count is None or prior_count < 1:
            return "no_prior_fights"
        if prior_count < 5:
            return "insufficient_prior_fights_l5"
        return "missing_recent_window_stats"
    if feature in {
        "sig_strikes_landed_per_min_all",
        "sig_striking_accuracy_all",
        "sig_striking_defense_all",
        "takedown_attempts_per_15m_all",
        "takedown_accuracy_all",
        "takedown_defense_all",
        "submission_attempts_per_15m_all",
        "avg_fight_duration_seconds_all",
    }:
        return "no_prior_fights" if (prior_count is None or prior_count < 1) else "missing_historical_bout_stats"
    if feature == "weight_class_experience":
        return "no_prior_fights_or_weight_class_history" if (prior_count is None or prior_count < 1) else "missing_weight_class_history"
    return "unknown"


def _as_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def _single_count(cursor: sqlite3.Cursor, query: str, *params: object) -> int:
    row = cursor.execute(query, params).fetchone()
    return int(row[0] if row is not None else 0)
