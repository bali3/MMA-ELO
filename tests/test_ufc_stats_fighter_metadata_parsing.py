"""Tests for deterministic UFC Stats fighter metadata parsing."""

from __future__ import annotations

from ingestion.sources.ufc_stats import parse_fighter_profile_page, parse_fighters_index_page


def test_parse_fighters_index_page_extracts_fighter_references() -> None:
    """Fighter index parser should return deterministic fighter references."""

    html = """
    <table>
      <tr class="b-statistics__table-row">
        <td><a href="http://ufcstats.com/fighter-details/abc123">Conor</a></td>
        <td><a href="http://ufcstats.com/fighter-details/abc123">McGregor</a></td>
      </tr>
      <tr class="b-statistics__table-row">
        <td><a href="http://ufcstats.com/fighter-details/def456">Zhang</a></td>
        <td><a href="http://ufcstats.com/fighter-details/def456">Weili</a></td>
      </tr>
    </table>
    """

    refs = parse_fighters_index_page(html)

    assert [ref.fighter_id for ref in refs] == ["abc123", "def456"]
    assert refs[0].fighter_url == "http://ufcstats.com/fighter-details/abc123"
    assert refs[0].full_name == "Conor McGregor"
    assert refs[1].full_name == "Zhang Weili"


def test_parse_fighter_profile_page_extracts_metadata_fields() -> None:
    """Fighter profile parser should extract pre-fight metadata fields."""

    html = """
    <span class="b-content__title-highlight"> Conor McGregor </span>
    <li class="b-list__box-list-item">
      <i class="b-list__box-item-title">Height:</i>
      5' 9"
    </li>
    <li class="b-list__box-list-item">
      <i class="b-list__box-item-title">Reach:</i>
      74"
    </li>
    <li class="b-list__box-list-item">
      <i class="b-list__box-item-title">STANCE:</i>
      Southpaw
    </li>
    <li class="b-list__box-list-item">
      <i class="b-list__box-item-title">DOB:</i>
      Jul 14, 1988
    </li>
    """

    parsed = parse_fighter_profile_page(
        fighter_id="abc123",
        fighter_url="http://ufcstats.com/fighter-details/abc123",
        fighter_html=html,
        fallback_full_name=None,
    )

    assert parsed["fighter_id"] == "abc123"
    assert parsed["full_name"] == "Conor McGregor"
    assert parsed["stance"] == "Southpaw"
    assert parsed["date_of_birth_utc"] == "1988-07-14T00:00:00Z"
    assert parsed["height_cm"] == 175.26
    assert parsed["reach_cm"] == 187.96
