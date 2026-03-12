"""Tests for deterministic UFC Stats HTML parsing."""

from __future__ import annotations

from ingestion.sources.ufc_stats import parse_completed_events_index, parse_event_page


def test_parse_completed_events_index_extracts_event_references() -> None:
    """Completed events index parser should return deterministic event references."""

    html = """
    <table>
      <tr>
        <td><a href="http://ufcstats.com/event-details/abc123">UFC 300: Test Card</a></td>
        <td><span class="b-statistics__date">April 13, 2024</span></td>
      </tr>
      <tr>
        <td><a href="http://ufcstats.com/event-details/def456">UFC Fight Night: Example</a></td>
        <td><span class="b-statistics__date">May 01, 2024</span></td>
      </tr>
    </table>
    """

    refs = parse_completed_events_index(html)

    assert [ref.event_id for ref in refs] == ["abc123", "def456"]
    assert refs[0].event_name == "UFC 300: Test Card"
    assert refs[0].event_date_text == "April 13, 2024"


def test_parse_event_page_extracts_event_metadata_and_bout_results() -> None:
    """Event page parser should extract event and bout result fields."""

    html = """
    <span class="b-content__title-highlight"> UFC 300: Test Card </span>
    <li class="b-list__box-list-item">Date: April 13, 2024</li>
    <li class="b-list__box-list-item">Location: Las Vegas, Nevada, USA</li>

    <tr class="b-fight-details__table-row" data-link="http://ufcstats.com/fight-details/fight001">
      <td>W</td>
      <td>
        <a href="http://ufcstats.com/fighter-details/red001">Red Fighter</a>
        <a href="http://ufcstats.com/fighter-details/blue001">Blue Fighter</a>
      </td>
      <td><p>2</p><p>0</p></td>
      <td><p>25</p><p>8</p></td>
      <td><p>1</p><p>0</p></td>
      <td><p>0</p><p>0</p></td>
      <td></td><td></td>
      <td>KO/TKO</td>
      <td>2</td>
      <td>1:30</td>
    </tr>
    """

    parsed = parse_event_page(
        event_id="abc123",
        event_url="http://ufcstats.com/event-details/abc123",
        event_html=html,
        fallback_event_name=None,
        fallback_event_date=None,
    )

    assert parsed["event_id"] == "abc123"
    assert parsed["event_name"] == "UFC 300: Test Card"
    assert parsed["event_date_utc"] == "2024-04-13T00:00:00Z"
    assert parsed["city"] == "Las Vegas"
    assert parsed["region"] == "Nevada"
    assert parsed["country"] == "USA"

    fighters = parsed["fighters"]
    assert isinstance(fighters, list)
    assert len(fighters) == 2

    bouts = parsed["bouts"]
    assert isinstance(bouts, list)
    assert len(bouts) == 1

    bout = bouts[0]
    assert bout["bout_id"] == "fight001"
    assert bout["fighter_red_id"] == "red001"
    assert bout["fighter_blue_id"] == "blue001"
    assert bout["winner_fighter_id"] == "red001"
    assert bout["result_method"] == "KO/TKO"
    assert bout["result_round"] == 2
    assert bout["result_time"] == "1:30"
    event_row_stats = bout["event_row_fighter_bout_stats"]
    assert isinstance(event_row_stats, tuple)
    assert event_row_stats[0]["fighter_id"] == "red001"
    assert event_row_stats[0]["knockdowns"] == 2
    assert event_row_stats[0]["sig_strikes_landed"] == 25
    assert event_row_stats[0]["takedowns_landed"] == 1
    assert event_row_stats[0]["submission_attempts"] == 0
    assert event_row_stats[1]["fighter_id"] == "blue001"
    assert event_row_stats[1]["knockdowns"] == 0
    assert event_row_stats[1]["sig_strikes_landed"] == 8


def test_parse_event_page_extracts_date_from_icon_label_markup() -> None:
    html = """
    <span class="b-content__title-highlight"> UFC Icon Label Card </span>
    <li class="b-list__box-list-item"><i class="b-list__box-item-title">Date:</i> April 13, 2024</li>
    <li class="b-list__box-list-item"><i class="b-list__box-item-title">Location:</i> Las Vegas, Nevada, USA</li>
    """

    parsed = parse_event_page(
        event_id="abc123",
        event_url="http://ufcstats.com/event-details/abc123",
        event_html=html,
        fallback_event_name=None,
        fallback_event_date=None,
    )

    assert parsed["event_date_utc"] == "2024-04-13T00:00:00Z"
    assert parsed["city"] == "Las Vegas"
