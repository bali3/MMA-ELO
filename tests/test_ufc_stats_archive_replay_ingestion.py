"""Tests for archive-replay UFC Stats ingestion paths."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ingestion.db_init import initialize_sqlite_database
from ingestion.sources.ufc_stats import (
    run_ufc_stats_bout_stats_ingestion_from_archives,
    run_ufc_stats_events_ingestion_from_archives,
)


def test_events_archive_replay_prefers_latest_file_per_event_id(tmp_path: Path) -> None:
    db_path = tmp_path / "mma.sqlite3"
    raw_root = tmp_path / "raw"
    initialize_sqlite_database(db_path)

    source_dir = raw_root / "ufc_stats" / "2026" / "03" / "11"
    source_dir.mkdir(parents=True, exist_ok=True)

    old_html = """
    <span class="b-content__title-highlight"> UFC Old Name </span>
    <li class="b-list__box-list-item">Date: March 01, 2024</li>
    <li class="b-list__box-list-item">Location: Las Vegas, Nevada, USA</li>
    """
    new_html = """
    <span class="b-content__title-highlight"> UFC New Name </span>
    <li class="b-list__box-list-item">Date: March 01, 2024</li>
    <li class="b-list__box-list-item">Location: Las Vegas, Nevada, USA</li>
    """
    second_html = """
    <span class="b-content__title-highlight"> UFC Another Event </span>
    <li class="b-list__box-list-item">Date: April 01, 2024</li>
    <li class="b-list__box-list-item">Location: Miami, Florida, USA</li>
    """

    (source_dir / "010000_http_ufcstats_com_event_details_abc123_aaaa.html").write_text(old_html, encoding="utf-8")
    (source_dir / "020000_http_ufcstats_com_event_details_abc123_bbbb.html").write_text(new_html, encoding="utf-8")
    (source_dir / "030000_http_ufcstats_com_event_details_def456_cccc.html").write_text(second_html, encoding="utf-8")

    result = run_ufc_stats_events_ingestion_from_archives(db_path=db_path, raw_root=raw_root)
    assert result.inserted_count >= 2

    with sqlite3.connect(db_path) as connection:
        names = {
            row[0]: row[1]
            for row in connection.execute("SELECT event_id, event_name FROM events ORDER BY event_id")
        }

    assert names["abc123"] == "UFC New Name"
    assert names["def456"] == "UFC Another Event"


def test_bout_stats_archive_replay_uses_event_fallback_when_fight_page_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "mma.sqlite3"
    raw_root = tmp_path / "raw"
    initialize_sqlite_database(db_path)

    source_dir = raw_root / "ufc_stats" / "2026" / "03" / "11"
    source_dir.mkdir(parents=True, exist_ok=True)

    event_html = """
    <span class="b-content__title-highlight"> UFC Test Card </span>
    <li class="b-list__box-list-item">Date: April 13, 2024</li>
    <li class="b-list__box-list-item">Location: Las Vegas, Nevada, USA</li>

    <tr class="b-fight-details__table-row" data-link="http://ufcstats.com/fight-details/fight001">
      <td>W</td>
      <td>
        <a href="http://ufcstats.com/fighter-details/red001">Red Fighter</a>
        <a href="http://ufcstats.com/fighter-details/blue001">Blue Fighter</a>
      </td>
      <td><p>1</p><p>0</p></td>
      <td><p>30</p><p>10</p></td>
      <td><p>2</p><p>1</p></td>
      <td><p>0</p><p>0</p></td>
      <td>Lightweight</td>
      <td>Decision</td>
      <td>3</td>
      <td>5:00</td>
    </tr>
    """
    (source_dir / "010000_http_ufcstats_com_event_details_abc123_aaaa.html").write_text(event_html, encoding="utf-8")

    run_ufc_stats_events_ingestion_from_archives(db_path=db_path, raw_root=raw_root)
    result = run_ufc_stats_bout_stats_ingestion_from_archives(db_path=db_path, raw_root=raw_root)
    assert result.inserted_count >= 1

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT fighter_id, opponent_fighter_id, corner, knockdowns, sig_strikes_landed, takedowns_landed
            FROM fighter_bout_stats
            ORDER BY fighter_id
            """
        ).fetchall()

    assert rows == [
        ("blue001", "red001", "blue", 0, 10, 1),
        ("red001", "blue001", "red", 1, 30, 2),
    ]
