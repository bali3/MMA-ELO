"""Market coverage and matching validation report generation."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class MarketCoverageReportResult:
    """Summary metrics for market coverage over UFC fights."""

    report_path: Path
    market_rows: int
    matched_fights: int
    unmatched_fights: int
    match_rate: float
    completed_fights: int
    opening_odds_covered_fights: int
    closing_odds_covered_fights: int | None


def run_market_coverage_report(
    *,
    db_path: Path,
    promotion: str,
    report_output_path: Path,
    source_system: str | None = None,
    unmatched_reasons: Mapping[str, int] | None = None,
) -> MarketCoverageReportResult:
    """Write market-matching coverage and readiness report to disk."""

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        completed_fights = _count_completed_fights(connection=connection, promotion=promotion)
        market_rows = _count_market_rows(
            connection=connection,
            promotion=promotion,
            source_system=source_system,
        )
        matched_fights = _count_matched_fights(
            connection=connection,
            promotion=promotion,
            source_system=source_system,
        )
        unmatched_fights = max(0, completed_fights - matched_fights)
        match_rate = 0.0 if completed_fights == 0 else matched_fights / float(completed_fights)
        opening_covered = _count_opening_covered_fights(
            connection=connection,
            promotion=promotion,
            source_system=source_system,
        )
        closing_counts = _closing_coverage_counts(
            connection=connection,
            promotion=promotion,
            source_system=source_system,
        )
        coverage_rows = _coverage_by_year_rows(
            connection=connection,
            promotion=promotion,
            source_system=source_system,
        )
        opening_by_year = _opening_coverage_by_year_rows(
            connection=connection,
            promotion=promotion,
            source_system=source_system,
        )
        closing_by_year = _closing_coverage_by_year_rows(
            connection=connection,
            promotion=promotion,
            source_system=source_system,
        )

    lines = [
        "market_data_validation_report",
        f"promotion={promotion}",
        f"source_system={source_system or 'all'}",
        f"market_rows={market_rows}",
        f"completed_fights={completed_fights}",
        f"matched_fights={matched_fights}",
        f"unmatched_fights={unmatched_fights}",
        f"match_rate={match_rate:.8f}",
        "coverage_by_year",
    ]

    if len(coverage_rows) == 0:
        lines.append("year=n/a completed_fights=0 matched_fights=0 coverage=0.00000000")
    else:
        for row in coverage_rows:
            completed_year = int(row["completed_fights"])
            matched_year = int(row["matched_fights"])
            coverage = 0.0 if completed_year == 0 else matched_year / float(completed_year)
            lines.append(
                (
                    f"year={row['year']} completed_fights={completed_year} "
                    f"matched_fights={matched_year} coverage={coverage:.8f}"
                )
            )

    lines.append("opening_odds_coverage_by_year")
    if len(opening_by_year) == 0:
        lines.append("year=n/a covered_fights=0 total_fights=0 coverage=0.00000000")
    else:
        for row in opening_by_year:
            covered = int(row["opening_covered_fights"])
            total = int(row["completed_fights"])
            coverage = 0.0 if total == 0 else covered / float(total)
            lines.append(
                f"year={row['year']} covered_fights={covered} total_fights={total} coverage={coverage:.8f}"
            )

    opening_rate = 0.0 if completed_fights == 0 else opening_covered / float(completed_fights)
    lines.append(
        (
            "opening_odds_coverage "
            f"covered_fights={opening_covered} total_fights={completed_fights} coverage={opening_rate:.8f}"
        )
    )

    closing_available = int(closing_counts["closing_available_fights"])
    closing_covered = int(closing_counts["closing_covered_fights"])
    if closing_available == 0:
        lines.append("closing_odds_coverage not_available")
        closing_value: int | None = None
    else:
        closing_rate = closing_covered / float(completed_fights) if completed_fights > 0 else 0.0
        lines.append(
            (
                "closing_odds_coverage "
                f"covered_fights={closing_covered} total_fights={completed_fights} "
                f"coverage={closing_rate:.8f} closing_available_fights={closing_available}"
            )
        )
        closing_value = closing_covered
    lines.append("closing_odds_coverage_by_year")
    if len(closing_by_year) == 0:
        lines.append("year=n/a covered_fights=0 total_fights=0 coverage=0.00000000 closing_available_fights=0")
    else:
        for row in closing_by_year:
            total = int(row["completed_fights"])
            covered = int(row["closing_covered_fights"])
            available = int(row["closing_available_fights"])
            coverage = 0.0 if total == 0 else covered / float(total)
            lines.append(
                (
                    f"year={row['year']} covered_fights={covered} total_fights={total} "
                    f"coverage={coverage:.8f} closing_available_fights={available}"
                )
            )

    lines.append("unmatched_rows_by_reason")
    if unmatched_reasons is None:
        lines.append("not_available")
    elif len(unmatched_reasons) == 0:
        lines.append("reason=none count=0")
    else:
        for reason, count in sorted(unmatched_reasons.items(), key=lambda pair: (-pair[1], pair[0])):
            lines.append(f"reason={reason} count={count}")

    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return MarketCoverageReportResult(
        report_path=report_output_path,
        market_rows=market_rows,
        matched_fights=matched_fights,
        unmatched_fights=unmatched_fights,
        match_rate=match_rate,
        completed_fights=completed_fights,
        opening_odds_covered_fights=opening_covered,
        closing_odds_covered_fights=closing_value,
    )


def _count_completed_fights(*, connection: sqlite3.Connection, promotion: str) -> int:
    row = connection.execute(
        """
        SELECT COUNT(*)
        FROM bouts AS b
        JOIN events AS e ON e.event_id = b.event_id
        WHERE UPPER(e.promotion) = UPPER(?)
          AND b.winner_fighter_id IS NOT NULL
        """,
        (promotion,),
    ).fetchone()
    return int(row[0] if row is not None else 0)


def _count_market_rows(
    *,
    connection: sqlite3.Connection,
    promotion: str,
    source_system: str | None,
) -> int:
    clause, params = _source_filter_clause(source_system=source_system)
    row = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM markets AS m
        JOIN bouts AS b ON b.bout_id = m.bout_id
        JOIN events AS e ON e.event_id = b.event_id
        WHERE UPPER(e.promotion) = UPPER(?)
          {clause}
        """,
        (promotion, *params),
    ).fetchone()
    return int(row[0] if row is not None else 0)


def _count_matched_fights(
    *,
    connection: sqlite3.Connection,
    promotion: str,
    source_system: str | None,
) -> int:
    clause, params = _source_filter_clause(source_system=source_system)
    row = connection.execute(
        f"""
        SELECT COUNT(DISTINCT b.bout_id)
        FROM bouts AS b
        JOIN events AS e ON e.event_id = b.event_id
        JOIN markets AS m ON m.bout_id = b.bout_id
        WHERE UPPER(e.promotion) = UPPER(?)
          AND b.winner_fighter_id IS NOT NULL
          {clause}
        """,
        (promotion, *params),
    ).fetchone()
    return int(row[0] if row is not None else 0)


def _count_opening_covered_fights(
    *,
    connection: sqlite3.Connection,
    promotion: str,
    source_system: str | None,
) -> int:
    clause, params = _source_filter_clause(source_system=source_system)
    row = connection.execute(
        f"""
        WITH sides AS (
            SELECT b.bout_id, COUNT(DISTINCT m.selection_fighter_id) AS side_count
            FROM bouts AS b
            JOIN events AS e ON e.event_id = b.event_id
            LEFT JOIN markets AS m
              ON m.bout_id = b.bout_id
             AND m.selection_fighter_id IN (b.fighter_red_id, b.fighter_blue_id)
             {clause}
            WHERE UPPER(e.promotion) = UPPER(?)
              AND b.winner_fighter_id IS NOT NULL
            GROUP BY b.bout_id
        )
        SELECT COUNT(*) FROM sides WHERE side_count = 2
        """,
        (*params, promotion),
    ).fetchone()
    return int(row[0] if row is not None else 0)


def _closing_coverage_counts(
    *,
    connection: sqlite3.Connection,
    promotion: str,
    source_system: str | None,
) -> dict[str, int]:
    clause, params = _source_filter_clause(source_system=source_system)
    row = connection.execute(
        f"""
        WITH side_snapshots AS (
            SELECT
                b.bout_id,
                m.selection_fighter_id,
                COUNT(DISTINCT m.market_timestamp_utc) AS ts_count
            FROM bouts AS b
            JOIN events AS e ON e.event_id = b.event_id
            JOIN markets AS m
              ON m.bout_id = b.bout_id
             AND m.selection_fighter_id IN (b.fighter_red_id, b.fighter_blue_id)
             {clause}
            WHERE UPPER(e.promotion) = UPPER(?)
              AND b.winner_fighter_id IS NOT NULL
            GROUP BY b.bout_id, m.selection_fighter_id
        ),
        bout_rollup AS (
            SELECT
                bout_id,
                SUM(CASE WHEN ts_count >= 2 THEN 1 ELSE 0 END) AS sides_with_closing_proxy
            FROM side_snapshots
            GROUP BY bout_id
        )
        SELECT
            SUM(CASE WHEN sides_with_closing_proxy >= 1 THEN 1 ELSE 0 END) AS closing_available_fights,
            SUM(CASE WHEN sides_with_closing_proxy = 2 THEN 1 ELSE 0 END) AS closing_covered_fights
        FROM bout_rollup
        """,
        (*params, promotion),
    ).fetchone()
    if row is None:
        return {"closing_available_fights": 0, "closing_covered_fights": 0}
    return {
        "closing_available_fights": int(row[0] or 0),
        "closing_covered_fights": int(row[1] or 0),
    }


def _coverage_by_year_rows(
    *,
    connection: sqlite3.Connection,
    promotion: str,
    source_system: str | None,
) -> Sequence[sqlite3.Row]:
    clause, params = _source_filter_clause(source_system=source_system)
    rows = connection.execute(
        f"""
        WITH completed AS (
            SELECT
                b.bout_id,
                strftime('%Y', COALESCE(b.bout_start_time_utc, e.event_date_utc)) AS year
            FROM bouts AS b
            JOIN events AS e ON e.event_id = b.event_id
            WHERE UPPER(e.promotion) = UPPER(?)
              AND b.winner_fighter_id IS NOT NULL
        ),
        matched AS (
            SELECT DISTINCT m.bout_id
            FROM markets AS m
            JOIN bouts AS b ON b.bout_id = m.bout_id
            JOIN events AS e ON e.event_id = b.event_id
            WHERE UPPER(e.promotion) = UPPER(?)
              {clause}
        )
        SELECT
            c.year AS year,
            COUNT(*) AS completed_fights,
            SUM(CASE WHEN m.bout_id IS NOT NULL THEN 1 ELSE 0 END) AS matched_fights
        FROM completed AS c
        LEFT JOIN matched AS m ON m.bout_id = c.bout_id
        GROUP BY c.year
        ORDER BY c.year ASC
        """,
        (promotion, promotion, *params),
    ).fetchall()
    return rows


def _source_filter_clause(*, source_system: str | None) -> tuple[str, tuple[str, ...]]:
    if source_system is None or source_system.strip() == "":
        return ("", ())
    return ("AND m.source_system = ?", (source_system.strip(),))


def _opening_coverage_by_year_rows(
    *,
    connection: sqlite3.Connection,
    promotion: str,
    source_system: str | None,
) -> Sequence[sqlite3.Row]:
    clause, params = _source_filter_clause(source_system=source_system)
    rows = connection.execute(
        f"""
        WITH completed AS (
            SELECT
                b.bout_id,
                b.fighter_red_id,
                b.fighter_blue_id,
                strftime('%Y', COALESCE(b.bout_start_time_utc, e.event_date_utc)) AS year
            FROM bouts AS b
            JOIN events AS e ON e.event_id = b.event_id
            WHERE UPPER(e.promotion) = UPPER(?)
              AND b.winner_fighter_id IS NOT NULL
        ),
        sides AS (
            SELECT
                c.year,
                c.bout_id,
                COUNT(DISTINCT m.selection_fighter_id) AS side_count
            FROM completed AS c
            LEFT JOIN markets AS m
              ON m.bout_id = c.bout_id
             AND m.selection_fighter_id IN (c.fighter_red_id, c.fighter_blue_id)
             {clause}
            GROUP BY c.year, c.bout_id
        )
        SELECT
            year,
            COUNT(*) AS completed_fights,
            SUM(CASE WHEN side_count = 2 THEN 1 ELSE 0 END) AS opening_covered_fights
        FROM sides
        GROUP BY year
        ORDER BY year ASC
        """,
        (promotion, *params),
    ).fetchall()
    return rows


def _closing_coverage_by_year_rows(
    *,
    connection: sqlite3.Connection,
    promotion: str,
    source_system: str | None,
) -> Sequence[sqlite3.Row]:
    clause, params = _source_filter_clause(source_system=source_system)
    rows = connection.execute(
        f"""
        WITH completed AS (
            SELECT
                b.bout_id,
                b.fighter_red_id,
                b.fighter_blue_id,
                strftime('%Y', COALESCE(b.bout_start_time_utc, e.event_date_utc)) AS year
            FROM bouts AS b
            JOIN events AS e ON e.event_id = b.event_id
            WHERE UPPER(e.promotion) = UPPER(?)
              AND b.winner_fighter_id IS NOT NULL
        ),
        side_snapshots AS (
            SELECT
                c.year,
                c.bout_id,
                m.selection_fighter_id,
                COUNT(DISTINCT m.market_timestamp_utc) AS ts_count
            FROM completed AS c
            LEFT JOIN markets AS m
              ON m.bout_id = c.bout_id
             AND m.selection_fighter_id IN (c.fighter_red_id, c.fighter_blue_id)
             {clause}
            GROUP BY c.year, c.bout_id, m.selection_fighter_id
        ),
        bout_rollup AS (
            SELECT
                year,
                bout_id,
                SUM(CASE WHEN ts_count >= 2 THEN 1 ELSE 0 END) AS sides_with_closing_proxy
            FROM side_snapshots
            GROUP BY year, bout_id
        )
        SELECT
            year,
            COUNT(*) AS completed_fights,
            SUM(CASE WHEN sides_with_closing_proxy >= 1 THEN 1 ELSE 0 END) AS closing_available_fights,
            SUM(CASE WHEN sides_with_closing_proxy = 2 THEN 1 ELSE 0 END) AS closing_covered_fights
        FROM bout_rollup
        GROUP BY year
        ORDER BY year ASC
        """,
        (promotion, *params),
    ).fetchall()
    return rows
