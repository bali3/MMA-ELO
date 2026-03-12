"""Tests for UFC Stats fight-details parsing into fighter-bout stats."""

from __future__ import annotations

from datetime import datetime, timezone

from ingestion.contracts import RawIngestionRecord
from ingestion.sources.ufc_stats import UFCStatsBoutStatsParser, parse_fight_details_page


def test_parse_fight_details_page_maps_red_and_blue_stats() -> None:
    """Parser should map stats values by fighter row order and parse safely."""

    html = """
    <table class="b-fight-details__table">
      <thead>
        <tr>
          <th>Round</th>
          <th>Fighter</th>
          <th>KD</th>
          <th>SIG. STR.</th>
          <th>SIG. STR. %</th>
          <th>TOTAL STR.</th>
          <th>TD</th>
          <th>TD %</th>
          <th>SUB. ATT</th>
          <th>REV.</th>
          <th>CTRL</th>
        </tr>
      </thead>
      <tbody>
        <tr class="b-fight-details__table-row">
          <td><p>Total</p><p>Total</p></td>
          <td><p>Red Fighter</p><p>Blue Fighter</p></td>
          <td><p>1</p><p>0</p></td>
          <td><p>10 of 20</p><p>15 of 30</p></td>
          <td><p>50%</p><p>50%</p></td>
          <td><p>20 of 40</p><p>30 of 60</p></td>
          <td><p>2 of 5</p><p>1 of 4</p></td>
          <td><p>40%</p><p>25%</p></td>
          <td><p>1</p><p>0</p></td>
          <td><p>0</p><p>2</p></td>
          <td><p>3:12</p><p>--</p></td>
        </tr>
      </tbody>
    </table>
    """

    parsed = parse_fight_details_page(
        bout_id="bout001",
        fight_url="http://ufcstats.com/fight-details/bout001",
        fight_html=html,
        fighter_red_id="red001",
        fighter_blue_id="blue001",
    )

    rows = parsed["fighter_bout_stats"]
    assert isinstance(rows, tuple)
    assert len(rows) == 2

    red = rows[0]
    blue = rows[1]

    assert red["fighter_id"] == "red001"
    assert red["opponent_fighter_id"] == "blue001"
    assert red["corner"] == "red"
    assert red["knockdowns"] == 1
    assert red["sig_strikes_landed"] == 10
    assert red["sig_strikes_attempted"] == 20
    assert red["total_strikes_landed"] == 20
    assert red["total_strikes_attempted"] == 40
    assert red["takedowns_landed"] == 2
    assert red["takedowns_attempted"] == 5
    assert red["submission_attempts"] == 1
    assert red["reversals"] == 0
    assert red["control_time_seconds"] == 192

    assert blue["fighter_id"] == "blue001"
    assert blue["opponent_fighter_id"] == "red001"
    assert blue["corner"] == "blue"
    assert blue["knockdowns"] == 0
    assert blue["sig_strikes_landed"] == 15
    assert blue["sig_strikes_attempted"] == 30
    assert blue["total_strikes_landed"] == 30
    assert blue["total_strikes_attempted"] == 60
    assert blue["takedowns_landed"] == 1
    assert blue["takedowns_attempted"] == 4
    assert blue["submission_attempts"] == 0
    assert blue["reversals"] == 2
    assert blue["control_time_seconds"] is None


def test_bout_stats_parser_uses_event_row_fallback_when_fight_html_missing() -> None:
    parser = UFCStatsBoutStatsParser()
    raw_record = RawIngestionRecord(
        source_system="ufc_stats",
        source_record_id="fight_page:bout001",
        fetched_at_utc=datetime.now(timezone.utc),
        payload={
            "event_id": "event001",
            "event_url": "http://ufcstats.com/event-details/event001",
            "event_name": "UFC Test Night",
            "event_date_utc": "2024-01-20T00:00:00Z",
            "bout_id": "bout001",
            "fight_url": "http://ufcstats.com/fight-details/bout001",
            "fighter_red_id": "red001",
            "fighter_red_name": "Red Fighter",
            "fighter_blue_id": "blue001",
            "fighter_blue_name": "Blue Fighter",
            "winner_fighter_id": "red001",
            "result_method": "Decision",
            "result_round": 3,
            "result_time": "5:00",
            "scheduled_rounds": 3,
            "weight_class": "Lightweight",
            "html": None,
            "event_row_fighter_bout_stats": (
                {
                    "fighter_id": "red001",
                    "opponent_fighter_id": "blue001",
                    "corner": "red",
                    "knockdowns": 1,
                    "sig_strikes_landed": 45,
                    "takedowns_landed": 2,
                    "submission_attempts": 1,
                },
                {
                    "fighter_id": "blue001",
                    "opponent_fighter_id": "red001",
                    "corner": "blue",
                    "knockdowns": 0,
                    "sig_strikes_landed": 40,
                    "takedowns_landed": 1,
                    "submission_attempts": 0,
                },
            ),
        },
        payload_sha256="sha",
        idempotency_key="idem",
    )

    parsed_records = parser.parse((raw_record,))
    assert len(parsed_records) == 1
    parsed = parsed_records[0].parsed_payload
    rows = parsed["fighter_bout_stats"]
    assert len(rows) == 2
    assert rows[0]["fighter_id"] == "red001"
    assert rows[0]["knockdowns"] == 1
    assert rows[0]["sig_strikes_landed"] == 45
    assert rows[0]["sig_strikes_attempted"] is None
    assert rows[1]["fighter_id"] == "blue001"
    assert rows[1]["takedowns_landed"] == 1
