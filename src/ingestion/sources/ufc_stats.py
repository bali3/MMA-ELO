"""Live UFC Stats ingestion for fighter metadata, events, and bout stats.

Scope constraints:
- UFC Stats source only
- ingestion only (no features, ratings, or modeling)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence
from urllib.request import Request, urlopen

from ingestion.contracts import (
    Fetcher,
    NormalizedRow,
    Normalizer,
    ParsedIngestionRecord,
    Parser,
    PersistResult,
    Persister,
    RawIngestionRecord,
)
from ingestion.db_init import initialize_sqlite_database

LOGGER = logging.getLogger(__name__)

UFC_STATS_SOURCE = "ufc_stats"
UFC_STATS_COMPLETED_EVENTS_URL = "http://ufcstats.com/statistics/events/completed?page=all"
UFC_STATS_FIGHTERS_INDEX_TEMPLATE = "http://ufcstats.com/statistics/fighters?char={letter}&page=all"
UFC_STATS_FIGHTER_LIST_LETTERS = tuple("abcdefghijklmnopqrstuvwxyz")

_EVENT_LINK_RE = re.compile(
    r'<a[^>]+href="(?P<url>http://ufcstats\.com/event-details/(?P<id>[a-z0-9]+))"[^>]*>'
    r'(?P<name>.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_EVENT_DATE_NEAR_LINK_RE = re.compile(
    r'event-details/(?P<id>[a-z0-9]+).*?b-statistics__date">\s*(?P<date>[^<]+?)\s*</span>',
    re.IGNORECASE | re.DOTALL,
)
_EVENT_NAME_RE = re.compile(
    r'b-content__title-highlight[^>]*>\s*(?P<name>[^<]+?)\s*</span>',
    re.IGNORECASE | re.DOTALL,
)
_EVENT_DETAIL_ITEM_RE = re.compile(
    r'<li[^>]*b-list__box-list-item[^>]*>\s*(?P<label>[A-Za-z ]+):\s*(?P<value>.*?)\s*</li>',
    re.IGNORECASE | re.DOTALL,
)
_BOUT_ROW_RE = re.compile(
    r'<tr[^>]*b-fight-details__table-row[^>]*data-link="(?P<url>http://ufcstats\.com/fight-details/(?P<id>[a-z0-9]+))"[^>]*>'
    r'(?P<body>.*?)</tr>',
    re.IGNORECASE | re.DOTALL,
)
_TD_RE = re.compile(r"<td[^>]*>(?P<value>.*?)</td>", re.IGNORECASE | re.DOTALL)
_FIGHTER_LINK_RE = re.compile(
    r'<a[^>]+href="(?P<url>http://ufcstats\.com/fighter-details/(?P<id>[a-z0-9]+))"[^>]*>'
    r'(?P<name>.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_FIGHTER_TABLE_ROW_RE = re.compile(
    r'<tr[^>]*b-statistics__table-row[^>]*>(?P<body>.*?)</tr>',
    re.IGNORECASE | re.DOTALL,
)
_FIGHTER_PROFILE_NAME_RE = re.compile(
    r'b-content__title-highlight[^>]*>\s*(?P<name>.*?)\s*</span>',
    re.IGNORECASE | re.DOTALL,
)
_FIGHTER_PROFILE_ITEM_RE = re.compile(
    r'<li[^>]*b-list__box-list-item[^>]*>.*?'
    r'<i[^>]*>\s*(?P<label>[A-Za-z ]+):\s*</i>\s*(?P<value>.*?)\s*</li>',
    re.IGNORECASE | re.DOTALL,
)
_FIGHT_DETAILS_TABLE_RE = re.compile(
    r'<table[^>]*class="[^"]*b-fight-details__table[^"]*"[^>]*>(?P<body>.*?)</table>',
    re.IGNORECASE | re.DOTALL,
)
_TH_RE = re.compile(r"<th[^>]*>(?P<value>.*?)</th>", re.IGNORECASE | re.DOTALL)
_TR_RE = re.compile(r"<tr[^>]*>(?P<body>.*?)</tr>", re.IGNORECASE | re.DOTALL)
_P_RE = re.compile(r"<p[^>]*>(?P<value>.*?)</p>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class CompletedEventReference:
    """Reference for an event listed on UFC Stats completed events page."""

    event_id: str
    event_url: str
    event_name: str
    event_date_text: str | None


@dataclass(frozen=True)
class FighterIndexReference:
    """Reference for a fighter listed on UFC Stats fighter index pages."""

    fighter_id: str
    fighter_url: str
    full_name: str | None


@dataclass(frozen=True)
class ArchivedResponse:
    """Metadata for an archived HTTP response body."""

    archive_path: Path
    payload_sha256: str


class UFCStatsFighterMetadataFetcher(Fetcher):
    """Fetch UFC Stats fighter index + profile pages and archive raw HTML payloads."""

    source_system = UFC_STATS_SOURCE

    def __init__(
        self,
        *,
        raw_root: Path,
        timeout_seconds: int = 30,
        user_agent: str = "mma-elo-ingestion/0.1",
        fighter_limit: int | None = None,
        index_letters: Sequence[str] = UFC_STATS_FIGHTER_LIST_LETTERS,
    ) -> None:
        self._raw_root = raw_root
        self._timeout_seconds = timeout_seconds
        self._user_agent = user_agent
        self._fighter_limit = fighter_limit
        self._index_letters = tuple(letter.lower() for letter in index_letters if letter.strip())

    def fetch(self, *, as_of_utc: datetime | None = None) -> Sequence[RawIngestionRecord]:
        """Fetch and archive fighter index and profile pages."""

        del as_of_utc
        fetched_at = datetime.now(timezone.utc)
        refs_by_id: dict[str, FighterIndexReference] = {}

        for letter in self._index_letters:
            index_url = UFC_STATS_FIGHTERS_INDEX_TEMPLATE.format(letter=letter)
            index_html = self._http_get_text(index_url)
            archived_index = archive_raw_response(
                raw_root=self._raw_root,
                fetched_at_utc=fetched_at,
                source_system=self.source_system,
                url=index_url,
                payload_text=index_html,
            )

            page_refs = parse_fighters_index_page(index_html)
            for ref in page_refs:
                if ref.fighter_id not in refs_by_id:
                    refs_by_id[ref.fighter_id] = ref

            LOGGER.info(
                "ufc_stats.fighter_metadata.fetch.index_page",
                extra={
                    "letter": letter,
                    "fighters_found": len(page_refs),
                    "archive_path": str(archived_index.archive_path),
                },
            )

        fighter_refs = sorted(refs_by_id.values(), key=lambda value: value.fighter_id)
        if self._fighter_limit is not None:
            fighter_refs = fighter_refs[: self._fighter_limit]

        records: list[RawIngestionRecord] = []
        for ref in fighter_refs:
            profile_html = self._http_get_text(ref.fighter_url)
            archived_profile = archive_raw_response(
                raw_root=self._raw_root,
                fetched_at_utc=fetched_at,
                source_system=self.source_system,
                url=ref.fighter_url,
                payload_text=profile_html,
            )
            payload = {
                "fighter_id": ref.fighter_id,
                "fighter_url": ref.fighter_url,
                "full_name_from_index": ref.full_name,
                "archive_path": archived_profile.archive_path.as_posix(),
                "html": profile_html,
            }
            records.append(
                RawIngestionRecord(
                    source_system=self.source_system,
                    source_record_id=f"fighter_profile:{ref.fighter_id}",
                    fetched_at_utc=fetched_at,
                    payload=payload,
                    payload_sha256=archived_profile.payload_sha256,
                    idempotency_key=_stable_sha256(
                        f"{self.source_system}|fighter_profile|{ref.fighter_id}|{archived_profile.payload_sha256}"
                    ),
                )
            )

        LOGGER.info(
            "ufc_stats.fighter_metadata.fetch.fighter_profiles",
            extra={"records_fetched": len(records)},
        )
        return tuple(records)

    def _http_get_text(self, url: str) -> str:
        request = Request(url=url, headers={"User-Agent": self._user_agent})
        with urlopen(request, timeout=self._timeout_seconds) as response:
            payload = response.read()
        return payload.decode("utf-8", errors="replace")


class UFCStatsFighterMetadataParser(Parser):
    """Parse raw UFC Stats fighter profile pages into structured fighter metadata."""

    def parse(self, records: Sequence[RawIngestionRecord]) -> Sequence[ParsedIngestionRecord]:
        parsed: list[ParsedIngestionRecord] = []
        for record in records:
            html = str(record.payload.get("html", ""))
            fighter_id = str(record.payload.get("fighter_id", "")).strip().lower()
            fighter_url = str(record.payload.get("fighter_url", "")).strip()
            if not fighter_id or not html:
                continue

            parsed_payload = parse_fighter_profile_page(
                fighter_id=fighter_id,
                fighter_url=fighter_url,
                fighter_html=html,
                fallback_full_name=_as_optional_str(record.payload.get("full_name_from_index")),
            )
            parsed.append(
                ParsedIngestionRecord(
                    source_system=record.source_system,
                    source_record_id=record.source_record_id,
                    parsed_payload=parsed_payload,
                    payload_sha256=record.payload_sha256,
                    idempotency_key=record.idempotency_key,
                )
            )

        LOGGER.info(
            "ufc_stats.fighter_metadata.parse.fighter_profiles",
            extra={"records_parsed": len(parsed)},
        )
        return tuple(parsed)


class UFCStatsFighterMetadataNormalizer(Normalizer):
    """Normalize parsed fighter metadata into canonical fighters rows."""

    def normalize(self, records: Sequence[ParsedIngestionRecord]) -> Sequence[NormalizedRow]:
        normalized_rows: list[NormalizedRow] = []
        now_utc = _utc_now_iso()

        for record in records:
            parsed = record.parsed_payload
            fighter_id = _as_optional_str(parsed.get("fighter_id"))
            full_name = _as_optional_str(parsed.get("full_name"))
            if not fighter_id or not full_name:
                continue

            fighter_payload = {
                "fighter_id": fighter_id,
                "full_name": full_name,
                "date_of_birth_utc": _as_optional_str(parsed.get("date_of_birth_utc")),
                "nationality": _as_optional_str(parsed.get("nationality")),
                "stance": _as_optional_str(parsed.get("stance")),
                "height_cm": _as_optional_float(parsed.get("height_cm")),
                "reach_cm": _as_optional_float(parsed.get("reach_cm")),
            }
            payload_sha = _stable_sha256(json.dumps(fighter_payload, sort_keys=True))
            source_record_id = f"fighter:{fighter_id}"

            normalized_rows.append(
                NormalizedRow(
                    table_name="fighters",
                    primary_key=fighter_id,
                    data={
                        **fighter_payload,
                        "source_system": record.source_system,
                        "source_record_id": source_record_id,
                        "source_updated_at_utc": None,
                        "source_payload_sha256": payload_sha,
                        "ingested_at_utc": now_utc,
                        "created_at_utc": now_utc,
                        "updated_at_utc": now_utc,
                    },
                    source_system=record.source_system,
                    source_record_id=source_record_id,
                    payload_sha256=payload_sha,
                    idempotency_key=_stable_sha256(
                        f"{record.source_system}|fighters|{source_record_id}|{payload_sha}"
                    ),
                )
            )

        normalized_rows.sort(key=lambda row: row.primary_key)
        LOGGER.info(
            "ufc_stats.fighter_metadata.normalize.rows",
            extra={"normalized_rows": len(normalized_rows)},
        )
        return tuple(normalized_rows)


class UFCStatsFetcher(Fetcher):
    """Fetch completed UFC Stats event pages and archive raw HTML payloads."""

    source_system = UFC_STATS_SOURCE

    def __init__(
        self,
        *,
        raw_root: Path,
        completed_events_url: str = UFC_STATS_COMPLETED_EVENTS_URL,
        timeout_seconds: int = 30,
        user_agent: str = "mma-elo-ingestion/0.1",
        event_limit: int | None = None,
    ) -> None:
        self._raw_root = raw_root
        self._completed_events_url = completed_events_url
        self._timeout_seconds = timeout_seconds
        self._user_agent = user_agent
        self._event_limit = event_limit

    def fetch(self, *, as_of_utc: datetime | None = None) -> Sequence[RawIngestionRecord]:
        """Fetch and archive completed events list + event detail pages."""

        fetched_at = datetime.now(timezone.utc)
        completed_html = self._http_get_text(self._completed_events_url)
        completed_archive = archive_raw_response(
            raw_root=self._raw_root,
            fetched_at_utc=fetched_at,
            source_system=self.source_system,
            url=self._completed_events_url,
            payload_text=completed_html,
        )

        event_refs = parse_completed_events_index(completed_html)
        if as_of_utc is not None:
            event_refs = [ref for ref in event_refs if _event_is_not_after(ref.event_date_text, as_of_utc)]

        if self._event_limit is not None:
            event_refs = event_refs[: self._event_limit]

        LOGGER.info(
            "ufc_stats.fetch.completed_index",
            extra={
                "events_found": len(event_refs),
                "completed_events_archive": str(completed_archive.archive_path),
            },
        )

        records: list[RawIngestionRecord] = []
        for event_ref in event_refs:
            event_html = self._http_get_text(event_ref.event_url)
            archived = archive_raw_response(
                raw_root=self._raw_root,
                fetched_at_utc=fetched_at,
                source_system=self.source_system,
                url=event_ref.event_url,
                payload_text=event_html,
            )
            payload = {
                "event_url": event_ref.event_url,
                "event_id": event_ref.event_id,
                "event_name_from_index": event_ref.event_name,
                "event_date_from_index": event_ref.event_date_text,
                "archive_path": archived.archive_path.as_posix(),
                "html": event_html,
            }
            records.append(
                RawIngestionRecord(
                    source_system=self.source_system,
                    source_record_id=f"event_page:{event_ref.event_id}",
                    fetched_at_utc=fetched_at,
                    payload=payload,
                    payload_sha256=archived.payload_sha256,
                    idempotency_key=_stable_sha256(
                        f"{self.source_system}|event_page|{event_ref.event_id}|{archived.payload_sha256}"
                    ),
                )
            )

        LOGGER.info(
            "ufc_stats.fetch.completed_events",
            extra={"records_fetched": len(records), "source": self.source_system},
        )
        return tuple(records)

    def _http_get_text(self, url: str) -> str:
        request = Request(url=url, headers={"User-Agent": self._user_agent})
        with urlopen(request, timeout=self._timeout_seconds) as response:
            payload = response.read()
        return payload.decode("utf-8", errors="replace")


class UFCStatsParser(Parser):
    """Parse raw UFC Stats event pages into deterministic event+bout records."""

    def parse(self, records: Sequence[RawIngestionRecord]) -> Sequence[ParsedIngestionRecord]:
        parsed: list[ParsedIngestionRecord] = []
        for record in records:
            html = str(record.payload.get("html", ""))
            event_url = str(record.payload.get("event_url", ""))
            event_id = str(record.payload.get("event_id", "")).strip().lower()
            if not event_id or not html:
                LOGGER.warning(
                    "ufc_stats.parse.skip_invalid_raw_record",
                    extra={"source_record_id": record.source_record_id},
                )
                continue

            parsed_payload = parse_event_page(
                event_id=event_id,
                event_url=event_url,
                event_html=html,
                fallback_event_name=_as_optional_str(record.payload.get("event_name_from_index")),
                fallback_event_date=_as_optional_str(record.payload.get("event_date_from_index")),
            )
            parsed.append(
                ParsedIngestionRecord(
                    source_system=record.source_system,
                    source_record_id=record.source_record_id,
                    parsed_payload=parsed_payload,
                    payload_sha256=record.payload_sha256,
                    idempotency_key=record.idempotency_key,
                )
            )

        LOGGER.info("ufc_stats.parse.events", extra={"records_parsed": len(parsed)})
        return tuple(parsed)


class UFCStatsBoutStatsFetcher(Fetcher):
    """Fetch completed UFC event pages and linked fight-details pages."""

    source_system = UFC_STATS_SOURCE

    def __init__(
        self,
        *,
        raw_root: Path,
        completed_events_url: str = UFC_STATS_COMPLETED_EVENTS_URL,
        timeout_seconds: int = 30,
        user_agent: str = "mma-elo-ingestion/0.1",
        event_limit: int | None = None,
    ) -> None:
        self._raw_root = raw_root
        self._completed_events_url = completed_events_url
        self._timeout_seconds = timeout_seconds
        self._user_agent = user_agent
        self._event_limit = event_limit

    def fetch(self, *, as_of_utc: datetime | None = None) -> Sequence[RawIngestionRecord]:
        """Fetch and archive fight-details pages with event+bout provenance."""

        fetched_at = datetime.now(timezone.utc)
        completed_html = self._http_get_text(self._completed_events_url)
        completed_archive = archive_raw_response(
            raw_root=self._raw_root,
            fetched_at_utc=fetched_at,
            source_system=self.source_system,
            url=self._completed_events_url,
            payload_text=completed_html,
        )

        event_refs = parse_completed_events_index(completed_html)
        if as_of_utc is not None:
            event_refs = [ref for ref in event_refs if _event_is_not_after(ref.event_date_text, as_of_utc)]
        if self._event_limit is not None:
            event_refs = event_refs[: self._event_limit]

        LOGGER.info(
            "ufc_stats.bout_stats.fetch.completed_index",
            extra={
                "events_found": len(event_refs),
                "completed_events_archive": str(completed_archive.archive_path),
            },
        )

        raw_records: list[RawIngestionRecord] = []
        for event_ref in event_refs:
            event_html = self._http_get_text(event_ref.event_url)
            archived_event = archive_raw_response(
                raw_root=self._raw_root,
                fetched_at_utc=fetched_at,
                source_system=self.source_system,
                url=event_ref.event_url,
                payload_text=event_html,
            )
            parsed_event = parse_event_page(
                event_id=event_ref.event_id,
                event_url=event_ref.event_url,
                event_html=event_html,
                fallback_event_name=event_ref.event_name,
                fallback_event_date=event_ref.event_date_text,
            )
            bouts = parsed_event.get("bouts")
            if not isinstance(bouts, Sequence):
                continue

            for bout_obj in bouts:
                if not isinstance(bout_obj, Mapping):
                    continue
                bout_id = _as_optional_str(bout_obj.get("bout_id"))
                fight_url = _as_optional_str(bout_obj.get("fight_url"))
                red_id = _as_optional_str(bout_obj.get("fighter_red_id"))
                blue_id = _as_optional_str(bout_obj.get("fighter_blue_id"))
                if not bout_id or not fight_url or not red_id or not blue_id or red_id == blue_id:
                    continue
                fight_html: str | None = None
                archive_path: str | None = None
                payload_sha: str | None = None
                source_page_type = "event_details_fallback"
                try:
                    fight_html = self._http_get_text(fight_url)
                    archived_fight = archive_raw_response(
                        raw_root=self._raw_root,
                        fetched_at_utc=fetched_at,
                        source_system=self.source_system,
                        url=fight_url,
                        payload_text=fight_html,
                    )
                    archive_path = archived_fight.archive_path.as_posix()
                    payload_sha = archived_fight.payload_sha256
                    source_page_type = "fight_details"
                except Exception as exc:  # noqa: BLE001 - network/parser fallback boundary
                    LOGGER.warning(
                        "ufc_stats.bout_stats.fetch.fight_page_failed",
                        extra={"bout_id": bout_id, "fight_url": fight_url, "error": str(exc)},
                    )

                payload = {
                    "event_id": parsed_event.get("event_id"),
                    "event_url": parsed_event.get("event_url"),
                    "event_name": parsed_event.get("event_name"),
                    "event_date_utc": parsed_event.get("event_date_utc"),
                    "venue": parsed_event.get("venue"),
                    "city": parsed_event.get("city"),
                    "region": parsed_event.get("region"),
                    "country": parsed_event.get("country"),
                    "event_archive_path": archived_event.archive_path.as_posix(),
                    "bout_id": bout_id,
                    "fight_url": fight_url,
                    "bout_order": bout_obj.get("bout_order"),
                    "fighter_red_id": red_id,
                    "fighter_red_name": bout_obj.get("fighter_red_name"),
                    "fighter_blue_id": blue_id,
                    "fighter_blue_name": bout_obj.get("fighter_blue_name"),
                    "winner_fighter_id": bout_obj.get("winner_fighter_id"),
                    "result_method": bout_obj.get("result_method"),
                    "result_round": bout_obj.get("result_round"),
                    "result_time": bout_obj.get("result_time"),
                    "scheduled_rounds": bout_obj.get("scheduled_rounds"),
                    "weight_class": bout_obj.get("weight_class"),
                    "event_row_fighter_bout_stats": bout_obj.get("event_row_fighter_bout_stats"),
                    "source_page_type": source_page_type,
                    "archive_path": archive_path,
                    "html": fight_html,
                }
                payload_text = (
                    fight_html
                    if fight_html is not None
                    else json.dumps(
                        {
                            "event_id": payload["event_id"],
                            "bout_id": payload["bout_id"],
                            "event_row_fighter_bout_stats": payload["event_row_fighter_bout_stats"],
                            "source_page_type": source_page_type,
                        },
                        sort_keys=True,
                    )
                )
                raw_records.append(
                    RawIngestionRecord(
                        source_system=self.source_system,
                        source_record_id=f"fight_page:{bout_id}",
                        fetched_at_utc=fetched_at,
                        payload=payload,
                        payload_sha256=payload_sha or _stable_sha256(payload_text),
                        idempotency_key=_stable_sha256(
                            f"{self.source_system}|fight_page|{bout_id}|{payload_sha or _stable_sha256(payload_text)}"
                        ),
                    )
                )

        LOGGER.info(
            "ufc_stats.bout_stats.fetch.fight_pages",
            extra={"records_fetched": len(raw_records)},
        )
        return tuple(raw_records)

    def _http_get_text(self, url: str) -> str:
        request = Request(url=url, headers={"User-Agent": self._user_agent})
        with urlopen(request, timeout=self._timeout_seconds) as response:
            payload = response.read()
        return payload.decode("utf-8", errors="replace")


class UFCStatsBoutStatsParser(Parser):
    """Parse fight-details pages into per-fighter per-bout stat rows."""

    def parse(self, records: Sequence[RawIngestionRecord]) -> Sequence[ParsedIngestionRecord]:
        parsed: list[ParsedIngestionRecord] = []
        for record in records:
            html_value = record.payload.get("html")
            html = str(html_value) if html_value is not None else ""
            bout_id = _as_optional_str(record.payload.get("bout_id"))
            fight_url = _as_optional_str(record.payload.get("fight_url"))
            red_id = _as_optional_str(record.payload.get("fighter_red_id"))
            blue_id = _as_optional_str(record.payload.get("fighter_blue_id"))
            if not bout_id or not red_id or not blue_id:
                continue

            fallback_event_stats = _coerce_event_row_fighter_stats(
                value=record.payload.get("event_row_fighter_bout_stats"),
                red_id=red_id,
                blue_id=blue_id,
            )
            if html and fight_url:
                parsed_fight = parse_fight_details_page(
                    bout_id=bout_id,
                    fight_url=fight_url,
                    fight_html=html,
                    fighter_red_id=red_id,
                    fighter_blue_id=blue_id,
                )
                stats_source_type = "fight_details"
            elif fallback_event_stats is not None:
                parsed_fight = {
                    "bout_id": bout_id,
                    "fight_url": fight_url,
                    "fighter_bout_stats": fallback_event_stats,
                }
                stats_source_type = "event_details_fallback"
            else:
                parsed_fight = {
                    "bout_id": bout_id,
                    "fight_url": fight_url,
                    "fighter_bout_stats": (
                        _empty_fighter_bout_stats(fighter_id=red_id, opponent_id=blue_id, corner="red"),
                        _empty_fighter_bout_stats(fighter_id=blue_id, opponent_id=red_id, corner="blue"),
                    ),
                }
                stats_source_type = "empty_stats_fallback"
            parsed_payload = {
                "event_id": _as_optional_str(record.payload.get("event_id")),
                "event_url": _as_optional_str(record.payload.get("event_url")),
                "event_name": _as_optional_str(record.payload.get("event_name")),
                "event_date_utc": _as_optional_str(record.payload.get("event_date_utc")),
                "venue": _as_optional_str(record.payload.get("venue")),
                "city": _as_optional_str(record.payload.get("city")),
                "region": _as_optional_str(record.payload.get("region")),
                "country": _as_optional_str(record.payload.get("country")),
                "fighters": [
                    {
                        "fighter_id": red_id,
                        "full_name": _as_optional_str(record.payload.get("fighter_red_name")),
                    },
                    {
                        "fighter_id": blue_id,
                        "full_name": _as_optional_str(record.payload.get("fighter_blue_name")),
                    },
                ],
                "bout": {
                    "bout_id": bout_id,
                    "fight_url": fight_url,
                    "bout_order": _as_optional_int(record.payload.get("bout_order")),
                    "fighter_red_id": red_id,
                    "fighter_blue_id": blue_id,
                    "winner_fighter_id": _as_optional_str(record.payload.get("winner_fighter_id")),
                    "result_method": _as_optional_str(record.payload.get("result_method")),
                    "result_round": _as_optional_int(record.payload.get("result_round")),
                    "result_time": _as_optional_str(record.payload.get("result_time")),
                    "scheduled_rounds": _as_optional_int(record.payload.get("scheduled_rounds")),
                    "weight_class": _as_optional_str(record.payload.get("weight_class")),
                    "stats_source_type": _as_optional_str(record.payload.get("source_page_type")) or stats_source_type,
                },
                "fighter_bout_stats": parsed_fight["fighter_bout_stats"],
            }
            parsed.append(
                ParsedIngestionRecord(
                    source_system=record.source_system,
                    source_record_id=record.source_record_id,
                    parsed_payload=parsed_payload,
                    payload_sha256=record.payload_sha256,
                    idempotency_key=record.idempotency_key,
                )
            )

        LOGGER.info("ufc_stats.bout_stats.parse.fight_pages", extra={"records_parsed": len(parsed)})
        return tuple(parsed)


class UFCStatsBoutStatsNormalizer(Normalizer):
    """Normalize parsed fight stats into fighters/events/bouts/fighter_bout_stats rows."""

    def normalize(self, records: Sequence[ParsedIngestionRecord]) -> Sequence[NormalizedRow]:
        normalized_rows: list[NormalizedRow] = []
        now_utc = _utc_now_iso()

        for record in records:
            parsed = record.parsed_payload
            event_id = _as_optional_str(parsed.get("event_id"))
            bout = parsed.get("bout")
            if not event_id or not isinstance(bout, Mapping):
                continue

            bout_id = _as_optional_str(bout.get("bout_id"))
            red_id = _as_optional_str(bout.get("fighter_red_id"))
            blue_id = _as_optional_str(bout.get("fighter_blue_id"))
            if not bout_id or not red_id or not blue_id or red_id == blue_id:
                continue

            fighters = parsed.get("fighters")
            if isinstance(fighters, Sequence):
                for fighter_obj in fighters:
                    if not isinstance(fighter_obj, Mapping):
                        continue
                    fighter_id = _as_optional_str(fighter_obj.get("fighter_id"))
                    full_name = _as_optional_str(fighter_obj.get("full_name"))
                    if not fighter_id or not full_name:
                        continue
                    fighter_source_record_id = f"fighter:{fighter_id}"
                    fighter_payload = {
                        "fighter_id": fighter_id,
                        "full_name": full_name,
                        "date_of_birth_utc": None,
                        "nationality": None,
                        "stance": None,
                        "height_cm": None,
                        "reach_cm": None,
                    }
                    fighter_payload_sha = _stable_sha256(json.dumps(fighter_payload, sort_keys=True))
                    normalized_rows.append(
                        NormalizedRow(
                            table_name="fighters",
                            primary_key=fighter_id,
                            data={
                                **fighter_payload,
                                "source_system": record.source_system,
                                "source_record_id": fighter_source_record_id,
                                "source_updated_at_utc": None,
                                "source_payload_sha256": fighter_payload_sha,
                                "ingested_at_utc": now_utc,
                                "created_at_utc": now_utc,
                                "updated_at_utc": now_utc,
                            },
                            source_system=record.source_system,
                            source_record_id=fighter_source_record_id,
                            payload_sha256=fighter_payload_sha,
                            idempotency_key=_stable_sha256(
                                f"{record.source_system}|fighters|{fighter_source_record_id}|{fighter_payload_sha}"
                            ),
                        )
                    )

            event_payload = {
                "event_id": event_id,
                "event_name": _coalesce(_as_optional_str(parsed.get("event_name")), event_id),
                "event_date_utc": _coalesce(
                    _as_optional_str(parsed.get("event_date_utc")),
                    "1970-01-01T00:00:00Z",
                ),
                "venue": _as_optional_str(parsed.get("venue")),
                "city": _as_optional_str(parsed.get("city")),
                "region": _as_optional_str(parsed.get("region")),
                "country": _as_optional_str(parsed.get("country")),
            }
            event_payload_sha = _stable_sha256(json.dumps(event_payload, sort_keys=True))
            normalized_rows.append(
                NormalizedRow(
                    table_name="events",
                    primary_key=event_id,
                    data={
                        "event_id": event_payload["event_id"],
                        "promotion": "UFC",
                        "event_name": event_payload["event_name"],
                        "event_date_utc": event_payload["event_date_utc"],
                        "venue": event_payload["venue"],
                        "city": event_payload["city"],
                        "region": event_payload["region"],
                        "country": event_payload["country"],
                        "timezone_name": "UTC",
                        "source_system": record.source_system,
                        "source_record_id": f"event:{event_id}",
                        "source_updated_at_utc": None,
                        "source_payload_sha256": event_payload_sha,
                        "ingested_at_utc": now_utc,
                        "created_at_utc": now_utc,
                        "updated_at_utc": now_utc,
                    },
                    source_system=record.source_system,
                    source_record_id=f"event:{event_id}",
                    payload_sha256=event_payload_sha,
                    idempotency_key=_stable_sha256(
                        f"{record.source_system}|events|event:{event_id}|{event_payload_sha}"
                    ),
                )
            )

            bout_payload = {
                "bout_id": bout_id,
                "event_id": event_id,
                "bout_order": _as_optional_int(bout.get("bout_order")),
                "fighter_red_id": red_id,
                "fighter_blue_id": blue_id,
                "weight_class": _as_optional_str(bout.get("weight_class")),
                "result_method": _as_optional_str(bout.get("result_method")),
                "result_round": _as_optional_int(bout.get("result_round")),
                "result_time_seconds": _duration_to_seconds(_as_optional_str(bout.get("result_time"))),
                "winner_fighter_id": _as_optional_str(bout.get("winner_fighter_id")),
            }
            bout_payload_sha = _stable_sha256(json.dumps(bout_payload, sort_keys=True))
            bout_source_record_id = f"bout:{bout_id}"
            normalized_rows.append(
                NormalizedRow(
                    table_name="bouts",
                    primary_key=bout_id,
                    data={
                        "bout_id": bout_id,
                        "event_id": event_id,
                        "bout_order": bout_payload["bout_order"],
                        "fighter_red_id": red_id,
                        "fighter_blue_id": blue_id,
                        "weight_class": bout_payload["weight_class"],
                        "gender": None,
                        "is_title_fight": 0,
                        "scheduled_rounds": _as_optional_int(bout.get("scheduled_rounds")),
                        "bout_start_time_utc": None,
                        "result_method": bout_payload["result_method"],
                        "result_round": bout_payload["result_round"],
                        "result_time_seconds": bout_payload["result_time_seconds"],
                        "winner_fighter_id": bout_payload["winner_fighter_id"],
                        "source_system": record.source_system,
                        "source_record_id": bout_source_record_id,
                        "source_updated_at_utc": None,
                        "source_payload_sha256": bout_payload_sha,
                        "ingested_at_utc": now_utc,
                        "created_at_utc": now_utc,
                        "updated_at_utc": now_utc,
                    },
                    source_system=record.source_system,
                    source_record_id=bout_source_record_id,
                    payload_sha256=bout_payload_sha,
                    idempotency_key=_stable_sha256(
                        f"{record.source_system}|bouts|{bout_source_record_id}|{bout_payload_sha}"
                    ),
                )
            )

            fighter_bout_stats = parsed.get("fighter_bout_stats")
            if not isinstance(fighter_bout_stats, Sequence):
                continue
            for stats_obj in fighter_bout_stats:
                if not isinstance(stats_obj, Mapping):
                    continue
                fighter_id = _as_optional_str(stats_obj.get("fighter_id"))
                opponent_id = _as_optional_str(stats_obj.get("opponent_fighter_id"))
                corner = _as_optional_str(stats_obj.get("corner"))
                if not fighter_id or not opponent_id or fighter_id == opponent_id:
                    continue

                stats_payload = {
                    "fighter_bout_stats_id": f"{bout_id}:{fighter_id}",
                    "bout_id": bout_id,
                    "fighter_id": fighter_id,
                    "opponent_fighter_id": opponent_id,
                    "corner": corner,
                    "knockdowns": _as_optional_int(stats_obj.get("knockdowns")),
                    "sig_strikes_landed": _as_optional_int(stats_obj.get("sig_strikes_landed")),
                    "sig_strikes_attempted": _as_optional_int(stats_obj.get("sig_strikes_attempted")),
                    "total_strikes_landed": _as_optional_int(stats_obj.get("total_strikes_landed")),
                    "total_strikes_attempted": _as_optional_int(stats_obj.get("total_strikes_attempted")),
                    "takedowns_landed": _as_optional_int(stats_obj.get("takedowns_landed")),
                    "takedowns_attempted": _as_optional_int(stats_obj.get("takedowns_attempted")),
                    "submission_attempts": _as_optional_int(stats_obj.get("submission_attempts")),
                    "reversals": _as_optional_int(stats_obj.get("reversals")),
                    "control_time_seconds": _as_optional_int(stats_obj.get("control_time_seconds")),
                }
                stats_payload_sha = _stable_sha256(json.dumps(stats_payload, sort_keys=True))
                source_record_id = f"fighter_bout_stats:{bout_id}:{fighter_id}"
                normalized_rows.append(
                    NormalizedRow(
                        table_name="fighter_bout_stats",
                        primary_key=stats_payload["fighter_bout_stats_id"],
                        data={
                            **stats_payload,
                            "source_system": record.source_system,
                            "source_record_id": source_record_id,
                            "source_updated_at_utc": None,
                            "source_payload_sha256": stats_payload_sha,
                            "ingested_at_utc": now_utc,
                            "created_at_utc": now_utc,
                            "updated_at_utc": now_utc,
                        },
                        source_system=record.source_system,
                        source_record_id=source_record_id,
                        payload_sha256=stats_payload_sha,
                        idempotency_key=_stable_sha256(
                            f"{record.source_system}|fighter_bout_stats|{source_record_id}|{stats_payload_sha}"
                        ),
                    )
                )

        table_order = {"fighters": 0, "events": 1, "bouts": 2, "fighter_bout_stats": 3}
        normalized_rows.sort(key=lambda row: (table_order.get(row.table_name, 99), row.primary_key))
        LOGGER.info(
            "ufc_stats.bout_stats.normalize.rows",
            extra={"normalized_rows": len(normalized_rows)},
        )
        return tuple(normalized_rows)


class UFCStatsNormalizer(Normalizer):
    """Normalize parsed UFC Stats payloads into fighters/events/bouts rows."""

    def normalize(self, records: Sequence[ParsedIngestionRecord]) -> Sequence[NormalizedRow]:
        normalized_rows: list[NormalizedRow] = []
        now_utc = _utc_now_iso()

        for record in records:
            parsed = record.parsed_payload
            event_id = _as_optional_str(parsed.get("event_id"))
            if not event_id:
                continue

            fighters = parsed.get("fighters", [])
            if isinstance(fighters, Sequence):
                for fighter_obj in fighters:
                    if not isinstance(fighter_obj, Mapping):
                        continue
                    fighter_id = _as_optional_str(fighter_obj.get("fighter_id"))
                    full_name = _as_optional_str(fighter_obj.get("full_name"))
                    if not fighter_id or not full_name:
                        continue
                    fighter_source_record_id = f"fighter:{fighter_id}"
                    fighter_payload = {
                        "fighter_id": fighter_id,
                        "full_name": full_name,
                        "date_of_birth_utc": None,
                        "nationality": None,
                        "stance": None,
                        "height_cm": None,
                        "reach_cm": None,
                    }
                    fighter_payload_sha = _stable_sha256(json.dumps(fighter_payload, sort_keys=True))

                    normalized_rows.append(
                        NormalizedRow(
                            table_name="fighters",
                            primary_key=fighter_id,
                            data={
                                **fighter_payload,
                                "source_system": record.source_system,
                                "source_record_id": fighter_source_record_id,
                                "source_updated_at_utc": None,
                                "source_payload_sha256": fighter_payload_sha,
                                "ingested_at_utc": now_utc,
                                "created_at_utc": now_utc,
                                "updated_at_utc": now_utc,
                            },
                            source_system=record.source_system,
                            source_record_id=fighter_source_record_id,
                            payload_sha256=fighter_payload_sha,
                            idempotency_key=_stable_sha256(
                                f"{record.source_system}|fighters|{fighter_source_record_id}|{fighter_payload_sha}"
                            ),
                        )
                    )

            event_payload = {
                "event_id": event_id,
                "event_name": _coalesce(_as_optional_str(parsed.get("event_name")), event_id),
                "event_date_utc": _coalesce(
                    _as_optional_str(parsed.get("event_date_utc")),
                    "1970-01-01T00:00:00Z",
                ),
                "venue": _as_optional_str(parsed.get("venue")),
                "city": _as_optional_str(parsed.get("city")),
                "region": _as_optional_str(parsed.get("region")),
                "country": _as_optional_str(parsed.get("country")),
            }
            event_payload_sha = _stable_sha256(json.dumps(event_payload, sort_keys=True))
            normalized_rows.append(
                NormalizedRow(
                    table_name="events",
                    primary_key=event_id,
                    data={
                        "event_id": event_payload["event_id"],
                        "promotion": "UFC",
                        "event_name": event_payload["event_name"],
                        "event_date_utc": event_payload["event_date_utc"],
                        "venue": event_payload["venue"],
                        "city": event_payload["city"],
                        "region": event_payload["region"],
                        "country": event_payload["country"],
                        "timezone_name": "UTC",
                        "source_system": record.source_system,
                        "source_record_id": f"event:{event_id}",
                        "source_updated_at_utc": None,
                        "source_payload_sha256": event_payload_sha,
                        "ingested_at_utc": now_utc,
                        "created_at_utc": now_utc,
                        "updated_at_utc": now_utc,
                    },
                    source_system=record.source_system,
                    source_record_id=f"event:{event_id}",
                    payload_sha256=event_payload_sha,
                    idempotency_key=_stable_sha256(
                        f"{record.source_system}|events|event:{event_id}|{event_payload_sha}"
                    ),
                )
            )

            bouts = parsed.get("bouts", [])
            if isinstance(bouts, Sequence):
                for bout in bouts:
                    if not isinstance(bout, Mapping):
                        continue
                    bout_id = _as_optional_str(bout.get("bout_id"))
                    red_id = _as_optional_str(bout.get("fighter_red_id"))
                    blue_id = _as_optional_str(bout.get("fighter_blue_id"))
                    if not bout_id or not red_id or not blue_id or red_id == blue_id:
                        continue

                    bout_payload = {
                        "bout_id": bout_id,
                        "event_id": event_id,
                        "bout_order": _as_optional_int(bout.get("bout_order")),
                        "fighter_red_id": red_id,
                        "fighter_blue_id": blue_id,
                        "weight_class": _as_optional_str(bout.get("weight_class")),
                        "result_method": _as_optional_str(bout.get("result_method")),
                        "result_round": _as_optional_int(bout.get("result_round")),
                        "result_time_seconds": _duration_to_seconds(_as_optional_str(bout.get("result_time"))),
                        "winner_fighter_id": _as_optional_str(bout.get("winner_fighter_id")),
                    }
                    bout_payload_sha = _stable_sha256(json.dumps(bout_payload, sort_keys=True))
                    source_record_id = f"bout:{bout_id}"

                    normalized_rows.append(
                        NormalizedRow(
                            table_name="bouts",
                            primary_key=bout_id,
                            data={
                                "bout_id": bout_id,
                                "event_id": event_id,
                                "bout_order": bout_payload["bout_order"],
                                "fighter_red_id": red_id,
                                "fighter_blue_id": blue_id,
                                "weight_class": bout_payload["weight_class"],
                                "gender": None,
                                "is_title_fight": 0,
                                "scheduled_rounds": _as_optional_int(bout.get("scheduled_rounds")),
                                "bout_start_time_utc": None,
                                "result_method": bout_payload["result_method"],
                                "result_round": bout_payload["result_round"],
                                "result_time_seconds": bout_payload["result_time_seconds"],
                                "winner_fighter_id": bout_payload["winner_fighter_id"],
                                "source_system": record.source_system,
                                "source_record_id": source_record_id,
                                "source_updated_at_utc": None,
                                "source_payload_sha256": bout_payload_sha,
                                "ingested_at_utc": now_utc,
                                "created_at_utc": now_utc,
                                "updated_at_utc": now_utc,
                            },
                            source_system=record.source_system,
                            source_record_id=source_record_id,
                            payload_sha256=bout_payload_sha,
                            idempotency_key=_stable_sha256(
                                f"{record.source_system}|bouts|{source_record_id}|{bout_payload_sha}"
                            ),
                        )
                    )

        table_order = {"fighters": 0, "events": 1, "bouts": 2}
        normalized_rows.sort(key=lambda row: (table_order.get(row.table_name, 99), row.primary_key))

        LOGGER.info(
            "ufc_stats.normalize.records",
            extra={"normalized_rows": len(normalized_rows)},
        )
        return tuple(normalized_rows)


class UFCStatsPersister(Persister):
    """Persist normalized UFC Stats rows into the local SQLite schema."""

    def __init__(self, *, db_path: Path) -> None:
        self._db_path = db_path

    def persist(self, rows: Sequence[NormalizedRow]) -> PersistResult:
        """Persist normalized rows with idempotent upsert behavior."""

        initialize_sqlite_database(self._db_path)
        inserted_count = 0
        updated_count = 0
        skipped_count = 0

        with sqlite3.connect(self._db_path) as connection:
            connection.execute("PRAGMA foreign_keys = ON;")
            for row in rows:
                existing_sha = self._get_existing_payload_sha(connection, row)
                if existing_sha is None:
                    self._insert_row(connection, row)
                    inserted_count += 1
                    continue

                if existing_sha == row.payload_sha256:
                    skipped_count += 1
                    continue

                self._update_row_by_source_key(connection, row)
                updated_count += 1

            connection.commit()

        LOGGER.info(
            "ufc_stats.persist.rows",
            extra={
                "inserted_count": inserted_count,
                "updated_count": updated_count,
                "skipped_count": skipped_count,
            },
        )
        return PersistResult(
            inserted_count=inserted_count,
            updated_count=updated_count,
            skipped_count=skipped_count,
        )

    def _get_existing_payload_sha(self, connection: sqlite3.Connection, row: NormalizedRow) -> str | None:
        result = connection.execute(
            f"SELECT source_payload_sha256 FROM {row.table_name} "
            "WHERE source_system = ? AND source_record_id = ?",
            (row.source_system, row.source_record_id),
        ).fetchone()
        if result is None:
            return None
        return str(result[0]) if result[0] is not None else None

    def _insert_row(self, connection: sqlite3.Connection, row: NormalizedRow) -> None:
        columns = list(row.data.keys())
        placeholders = ", ".join("?" for _ in columns)
        column_sql = ", ".join(columns)
        values = [row.data[column] for column in columns]
        connection.execute(
            f"INSERT INTO {row.table_name} ({column_sql}) VALUES ({placeholders})",
            values,
        )

    def _update_row_by_source_key(self, connection: sqlite3.Connection, row: NormalizedRow) -> None:
        update_columns = [
            column
            for column in row.data.keys()
            if column not in {"created_at_utc", "source_system", "source_record_id"}
        ]
        set_sql = ", ".join(f"{column} = ?" for column in update_columns)
        values = [row.data[column] for column in update_columns]
        values.extend([row.source_system, row.source_record_id])

        connection.execute(
            f"UPDATE {row.table_name} SET {set_sql} WHERE source_system = ? AND source_record_id = ?",
            values,
        )


class UFCStatsFighterMetadataIngestionService:
    """Coordinate UFC Stats fighter metadata fetch, parse, normalize, persistence."""

    def __init__(
        self,
        *,
        fetcher: UFCStatsFighterMetadataFetcher,
        parser: UFCStatsFighterMetadataParser,
        normalizer: UFCStatsFighterMetadataNormalizer,
        persister: UFCStatsPersister,
    ) -> None:
        self._fetcher = fetcher
        self._parser = parser
        self._normalizer = normalizer
        self._persister = persister

    def run(self, *, as_of_utc: datetime | None = None) -> PersistResult:
        raw_records = self._fetcher.fetch(as_of_utc=as_of_utc)
        parsed = self._parser.parse(raw_records)
        normalized = self._normalizer.normalize(parsed)
        return self._persister.persist(normalized)


class UFCStatsIngestionService:
    """Coordinate UFC Stats event fetch, parse, normalize, persistence."""

    def __init__(
        self,
        *,
        fetcher: UFCStatsFetcher,
        parser: UFCStatsParser,
        normalizer: UFCStatsNormalizer,
        persister: UFCStatsPersister,
    ) -> None:
        self._fetcher = fetcher
        self._parser = parser
        self._normalizer = normalizer
        self._persister = persister

    def run(self, *, as_of_utc: datetime | None = None) -> PersistResult:
        raw_records = self._fetcher.fetch(as_of_utc=as_of_utc)
        parsed = self._parser.parse(raw_records)
        normalized = self._normalizer.normalize(parsed)
        return self._persister.persist(normalized)


class UFCStatsBoutStatsIngestionService:
    """Coordinate UFC Stats bout-level fighter stats ingestion."""

    def __init__(
        self,
        *,
        fetcher: UFCStatsBoutStatsFetcher,
        parser: UFCStatsBoutStatsParser,
        normalizer: UFCStatsBoutStatsNormalizer,
        persister: UFCStatsPersister,
    ) -> None:
        self._fetcher = fetcher
        self._parser = parser
        self._normalizer = normalizer
        self._persister = persister

    def run(self, *, as_of_utc: datetime | None = None) -> PersistResult:
        raw_records = self._fetcher.fetch(as_of_utc=as_of_utc)
        parsed = self._parser.parse(raw_records)
        normalized = self._normalizer.normalize(parsed)
        return self._persister.persist(normalized)


def run_ufc_stats_events_ingestion(
    *,
    db_path: Path,
    raw_root: Path,
    as_of_utc: datetime | None = None,
    event_limit: int | None = None,
) -> PersistResult:
    """Entry point for UFC Stats completed events + bout result ingestion."""

    service = UFCStatsIngestionService(
        fetcher=UFCStatsFetcher(raw_root=raw_root, event_limit=event_limit),
        parser=UFCStatsParser(),
        normalizer=UFCStatsNormalizer(),
        persister=UFCStatsPersister(db_path=db_path),
    )
    return service.run(as_of_utc=as_of_utc)


def run_ufc_stats_events_ingestion_from_archives(
    *,
    db_path: Path,
    raw_root: Path,
    event_limit: int | None = None,
) -> PersistResult:
    """Replay archived event-details pages into events/bouts/fighters tables."""

    parser = UFCStatsParser()
    normalizer = UFCStatsNormalizer()
    persister = UFCStatsPersister(db_path=db_path)
    fetched_at = datetime.now(timezone.utc)

    event_pages = _discover_latest_archived_html_by_id(
        raw_root=raw_root,
        id_pattern=re.compile(r"_event_details_(?P<id>[a-z0-9]+)_", re.IGNORECASE),
    )
    event_ids = sorted(event_pages)
    if event_limit is not None:
        event_ids = event_ids[:event_limit]

    raw_records: list[RawIngestionRecord] = []
    for event_id in event_ids:
        archive_path = event_pages[event_id]
        html = archive_path.read_text(encoding="utf-8", errors="replace")
        payload_sha = _stable_sha256(html)
        raw_records.append(
            RawIngestionRecord(
                source_system=UFC_STATS_SOURCE,
                source_record_id=f"event_page:{event_id}",
                fetched_at_utc=fetched_at,
                payload={
                    "event_url": f"http://ufcstats.com/event-details/{event_id}",
                    "event_id": event_id,
                    "event_name_from_index": None,
                    "event_date_from_index": None,
                    "archive_path": archive_path.as_posix(),
                    "html": html,
                },
                payload_sha256=payload_sha,
                idempotency_key=_stable_sha256(f"{UFC_STATS_SOURCE}|event_page|{event_id}|{payload_sha}"),
            )
        )

    parsed = parser.parse(raw_records)
    normalized = normalizer.normalize(parsed)
    LOGGER.info(
        "ufc_stats.events.archive_replay",
        extra={"event_pages": len(raw_records), "normalized_rows": len(normalized)},
    )
    return persister.persist(normalized)


def run_ufc_stats_bout_stats_ingestion(
    *,
    db_path: Path,
    raw_root: Path,
    as_of_utc: datetime | None = None,
    event_limit: int | None = None,
) -> PersistResult:
    """Entry point for UFC Stats per-bout fighter statistics ingestion."""

    service = UFCStatsBoutStatsIngestionService(
        fetcher=UFCStatsBoutStatsFetcher(raw_root=raw_root, event_limit=event_limit),
        parser=UFCStatsBoutStatsParser(),
        normalizer=UFCStatsBoutStatsNormalizer(),
        persister=UFCStatsPersister(db_path=db_path),
    )
    return service.run(as_of_utc=as_of_utc)


def run_ufc_stats_bout_stats_ingestion_from_archives(
    *,
    db_path: Path,
    raw_root: Path,
    event_limit: int | None = None,
) -> PersistResult:
    """Replay archived event-details + fight-details pages into bout stats tables."""

    parser = UFCStatsBoutStatsParser()
    normalizer = UFCStatsBoutStatsNormalizer()
    persister = UFCStatsPersister(db_path=db_path)
    fetched_at = datetime.now(timezone.utc)

    event_pages = _discover_latest_archived_html_by_id(
        raw_root=raw_root,
        id_pattern=re.compile(r"_event_details_(?P<id>[a-z0-9]+)_", re.IGNORECASE),
    )
    fight_pages = _discover_latest_archived_html_by_id(
        raw_root=raw_root,
        id_pattern=re.compile(r"_fight_details_(?P<id>[a-z0-9]+)_", re.IGNORECASE),
    )
    event_ids = sorted(event_pages)
    if event_limit is not None:
        event_ids = event_ids[:event_limit]

    raw_records: list[RawIngestionRecord] = []
    fight_pages_used = 0
    event_fallback_rows = 0
    for event_id in event_ids:
        event_archive_path = event_pages[event_id]
        event_html = event_archive_path.read_text(encoding="utf-8", errors="replace")
        parsed_event = parse_event_page(
            event_id=event_id,
            event_url=f"http://ufcstats.com/event-details/{event_id}",
            event_html=event_html,
            fallback_event_name=None,
            fallback_event_date=None,
        )
        bouts = parsed_event.get("bouts")
        if not isinstance(bouts, Sequence):
            continue

        for bout_obj in bouts:
            if not isinstance(bout_obj, Mapping):
                continue
            bout_id = _as_optional_str(bout_obj.get("bout_id"))
            red_id = _as_optional_str(bout_obj.get("fighter_red_id"))
            blue_id = _as_optional_str(bout_obj.get("fighter_blue_id"))
            if not bout_id or not red_id or not blue_id:
                continue
            fight_archive_path = fight_pages.get(bout_id)
            source_page_type = "event_details_fallback"
            if fight_archive_path is not None:
                fight_html = fight_archive_path.read_text(encoding="utf-8", errors="replace")
                payload_sha = _stable_sha256(fight_html)
                archive_path = fight_archive_path.as_posix()
                source_page_type = "fight_details"
                fight_pages_used += 1
            else:
                fight_html = None
                payload_sha = _stable_sha256(
                    json.dumps(
                        {
                            "event_id": parsed_event.get("event_id"),
                            "bout_id": bout_id,
                            "event_row_fighter_bout_stats": bout_obj.get("event_row_fighter_bout_stats"),
                        },
                        sort_keys=True,
                    )
                )
                archive_path = None
                event_fallback_rows += 1
            raw_records.append(
                RawIngestionRecord(
                    source_system=UFC_STATS_SOURCE,
                    source_record_id=f"fight_page:{bout_id}",
                    fetched_at_utc=fetched_at,
                    payload={
                        "event_id": parsed_event.get("event_id"),
                        "event_url": parsed_event.get("event_url"),
                        "event_name": parsed_event.get("event_name"),
                        "event_date_utc": parsed_event.get("event_date_utc"),
                        "venue": parsed_event.get("venue"),
                        "city": parsed_event.get("city"),
                        "region": parsed_event.get("region"),
                        "country": parsed_event.get("country"),
                        "event_archive_path": event_archive_path.as_posix(),
                        "bout_id": bout_id,
                        "fight_url": bout_obj.get("fight_url"),
                        "bout_order": bout_obj.get("bout_order"),
                        "fighter_red_id": red_id,
                        "fighter_red_name": bout_obj.get("fighter_red_name"),
                        "fighter_blue_id": blue_id,
                        "fighter_blue_name": bout_obj.get("fighter_blue_name"),
                        "winner_fighter_id": bout_obj.get("winner_fighter_id"),
                        "result_method": bout_obj.get("result_method"),
                        "result_round": bout_obj.get("result_round"),
                        "result_time": bout_obj.get("result_time"),
                        "scheduled_rounds": bout_obj.get("scheduled_rounds"),
                        "weight_class": bout_obj.get("weight_class"),
                        "event_row_fighter_bout_stats": bout_obj.get("event_row_fighter_bout_stats"),
                        "source_page_type": source_page_type,
                        "archive_path": archive_path,
                        "html": fight_html,
                    },
                    payload_sha256=payload_sha,
                    idempotency_key=_stable_sha256(f"{UFC_STATS_SOURCE}|fight_page|{bout_id}|{payload_sha}"),
                )
            )

    parsed = parser.parse(raw_records)
    normalized = normalizer.normalize(parsed)
    LOGGER.info(
        "ufc_stats.bout_stats.archive_replay",
        extra={
            "fight_pages_or_fallback_records": len(raw_records),
            "fight_pages_used": fight_pages_used,
            "event_fallback_records": event_fallback_rows,
            "normalized_rows": len(normalized),
        },
    )
    return persister.persist(normalized)


def run_ufc_stats_fighter_metadata_ingestion(
    *,
    db_path: Path,
    raw_root: Path,
    as_of_utc: datetime | None = None,
    fighter_limit: int | None = None,
) -> PersistResult:
    """Entry point for UFC Stats fighter profile metadata ingestion."""

    service = UFCStatsFighterMetadataIngestionService(
        fetcher=UFCStatsFighterMetadataFetcher(raw_root=raw_root, fighter_limit=fighter_limit),
        parser=UFCStatsFighterMetadataParser(),
        normalizer=UFCStatsFighterMetadataNormalizer(),
        persister=UFCStatsPersister(db_path=db_path),
    )
    return service.run(as_of_utc=as_of_utc)


def run_ufc_stats_fighter_metadata_ingestion_from_archives(
    *,
    db_path: Path,
    raw_root: Path,
    fighter_limit: int | None = None,
) -> PersistResult:
    """Replay archived fighter-details pages into fighter metadata table."""

    parser = UFCStatsFighterMetadataParser()
    normalizer = UFCStatsFighterMetadataNormalizer()
    persister = UFCStatsPersister(db_path=db_path)
    fetched_at = datetime.now(timezone.utc)

    fighter_pages = _discover_latest_archived_html_by_id(
        raw_root=raw_root,
        id_pattern=re.compile(r"_fighter_details_(?P<id>[a-z0-9]+)_", re.IGNORECASE),
    )
    fighter_ids = sorted(fighter_pages)
    if fighter_limit is not None:
        fighter_ids = fighter_ids[:fighter_limit]

    raw_records: list[RawIngestionRecord] = []
    for fighter_id in fighter_ids:
        archive_path = fighter_pages[fighter_id]
        html = archive_path.read_text(encoding="utf-8", errors="replace")
        payload_sha = _stable_sha256(html)
        raw_records.append(
            RawIngestionRecord(
                source_system=UFC_STATS_SOURCE,
                source_record_id=f"fighter_profile:{fighter_id}",
                fetched_at_utc=fetched_at,
                payload={
                    "fighter_id": fighter_id,
                    "fighter_url": f"http://ufcstats.com/fighter-details/{fighter_id}",
                    "full_name_from_index": None,
                    "archive_path": archive_path.as_posix(),
                    "html": html,
                },
                payload_sha256=payload_sha,
                idempotency_key=_stable_sha256(f"{UFC_STATS_SOURCE}|fighter_profile|{fighter_id}|{payload_sha}"),
            )
        )

    parsed = parser.parse(raw_records)
    normalized = normalizer.normalize(parsed)
    LOGGER.info(
        "ufc_stats.fighter_metadata.archive_replay",
        extra={"fighter_pages": len(raw_records), "normalized_rows": len(normalized)},
    )
    return persister.persist(normalized)


def archive_raw_response(
    *,
    raw_root: Path,
    fetched_at_utc: datetime,
    source_system: str,
    url: str,
    payload_text: str,
) -> ArchivedResponse:
    """Archive raw response payload to ``data/raw`` with deterministic naming."""

    payload_bytes = payload_text.encode("utf-8", errors="replace")
    payload_sha = hashlib.sha256(payload_bytes).hexdigest()
    endpoint_token = _endpoint_token(url)

    partition_dir = raw_root / source_system / fetched_at_utc.strftime("%Y/%m/%d")
    partition_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{fetched_at_utc.strftime('%H%M%S')}_{endpoint_token}_{payload_sha[:12]}.html"
    archive_path = partition_dir / filename
    archive_path.write_bytes(payload_bytes)

    return ArchivedResponse(archive_path=archive_path, payload_sha256=payload_sha)


def parse_completed_events_index(html: str) -> list[CompletedEventReference]:
    """Extract completed UFC event references from index page HTML."""

    refs: list[CompletedEventReference] = []
    dates_by_id = {
        match.group("id").lower(): _clean_text(match.group("date"))
        for match in _EVENT_DATE_NEAR_LINK_RE.finditer(html)
    }

    seen_ids: set[str] = set()
    for match in _EVENT_LINK_RE.finditer(html):
        event_id = match.group("id").lower()
        if event_id in seen_ids:
            continue
        seen_ids.add(event_id)
        refs.append(
            CompletedEventReference(
                event_id=event_id,
                event_url=match.group("url"),
                event_name=_clean_text(match.group("name")),
                event_date_text=dates_by_id.get(event_id),
            )
        )

    return refs


def parse_fighters_index_page(html: str) -> list[FighterIndexReference]:
    """Extract fighter references from a UFC Stats fighter index page."""

    refs_by_id: dict[str, FighterIndexReference] = {}
    for row in _FIGHTER_TABLE_ROW_RE.finditer(html):
        body = row.group("body")
        links = [
            (match.group("id").lower(), match.group("url"), _clean_text(match.group("name")))
            for match in _FIGHTER_LINK_RE.finditer(body)
        ]
        if not links:
            continue

        fighter_id = links[0][0]
        fighter_url = links[0][1]
        name_parts: list[str] = []
        for link_id, _, name in links:
            if link_id != fighter_id:
                continue
            if name and name not in name_parts:
                name_parts.append(name)
            if len(name_parts) >= 2:
                break

        full_name = " ".join(name_parts).strip() if name_parts else None
        refs_by_id[fighter_id] = FighterIndexReference(
            fighter_id=fighter_id,
            fighter_url=fighter_url,
            full_name=full_name or None,
        )

    return [refs_by_id[key] for key in sorted(refs_by_id)]


def parse_fighter_profile_page(
    *,
    fighter_id: str,
    fighter_url: str,
    fighter_html: str,
    fallback_full_name: str | None,
) -> Mapping[str, object]:
    """Parse UFC Stats fighter profile HTML into deterministic fighter metadata."""

    name_match = _FIGHTER_PROFILE_NAME_RE.search(fighter_html)
    full_name = _clean_text(name_match.group("name")) if name_match else fallback_full_name

    detail_items: dict[str, str] = {}
    for match in _FIGHTER_PROFILE_ITEM_RE.finditer(fighter_html):
        label = _clean_text(match.group("label")).lower()
        value = _clean_text(match.group("value"))
        detail_items[label] = value

    return {
        "fighter_id": fighter_id,
        "fighter_url": fighter_url,
        "full_name": _as_optional_str(full_name),
        "date_of_birth_utc": _parse_date_to_utc_iso(detail_items.get("dob")),
        "nationality": None,
        "stance": _as_optional_str(detail_items.get("stance")),
        "height_cm": _height_text_to_cm(detail_items.get("height")),
        "reach_cm": _reach_text_to_cm(detail_items.get("reach")),
    }


def parse_event_page(
    *,
    event_id: str,
    event_url: str,
    event_html: str,
    fallback_event_name: str | None,
    fallback_event_date: str | None,
) -> Mapping[str, object]:
    """Parse event page HTML into deterministic event + bout structures."""

    event_name_match = _EVENT_NAME_RE.search(event_html)
    event_name = _clean_text(event_name_match.group("name")) if event_name_match else fallback_event_name

    detail_items: dict[str, str] = {}
    for match in _EVENT_DETAIL_ITEM_RE.finditer(event_html):
        detail_items[_clean_text(match.group("label")).lower()] = _clean_text(match.group("value"))

    event_date_text = detail_items.get("date") or _extract_event_detail_text(event_html, "date") or fallback_event_date
    event_date_utc = _parse_date_to_utc_iso(event_date_text)

    location = detail_items.get("location") or _extract_event_detail_text(event_html, "location")
    city, region, country = _parse_location(location)

    fighters_by_id: dict[str, str] = {}
    bouts: list[dict[str, object]] = []

    for bout_order, match in enumerate(_BOUT_ROW_RE.finditer(event_html), start=1):
        body = match.group("body")
        fight_id = match.group("id").lower()
        fighter_entries = [
            (entry.group("id").lower(), _clean_text(entry.group("name")))
            for entry in _FIGHTER_LINK_RE.finditer(body)
        ]
        if len(fighter_entries) < 2:
            continue

        red_id, red_name = fighter_entries[0]
        blue_id, blue_name = fighter_entries[1]
        fighters_by_id[red_id] = red_name
        fighters_by_id[blue_id] = blue_name

        td_cells = [_extract_cell_values(td.group("value")) for td in _TD_RE.finditer(body)]
        td_values = [values[0] if values else "" for values in td_cells]
        winner = _winner_from_row_columns(td_values=td_values, red_id=red_id, blue_id=blue_id)

        method = td_values[-3] if len(td_values) >= 3 else None
        result_round = _safe_int(td_values[-2]) if len(td_values) >= 2 else None
        result_time = td_values[-1] if len(td_values) >= 1 else None

        bouts.append(
            {
                "bout_id": fight_id,
                "fight_url": match.group("url"),
                "bout_order": bout_order,
                "fighter_red_id": red_id,
                "fighter_red_name": red_name,
                "fighter_blue_id": blue_id,
                "fighter_blue_name": blue_name,
                "winner_fighter_id": winner,
                "result_method": _as_optional_str(method),
                "result_round": result_round,
                "result_time": _as_optional_str(result_time),
                "scheduled_rounds": None,
                "weight_class": None,
                "event_row_fighter_bout_stats": _event_row_fighter_bout_stats(
                    td_cells=td_cells,
                    red_id=red_id,
                    blue_id=blue_id,
                ),
            }
        )

    fighters = [
        {"fighter_id": fighter_id, "full_name": full_name}
        for fighter_id, full_name in sorted(fighters_by_id.items())
    ]

    return {
        "event_id": event_id,
        "event_url": event_url,
        "event_name": event_name,
        "event_date_utc": event_date_utc,
        "venue": None,
        "city": city,
        "region": region,
        "country": country,
        "fighters": fighters,
        "bouts": bouts,
    }


def _extract_event_detail_text(event_html: str, label: str) -> str | None:
    pattern = re.compile(
        rf"<li[^>]*>\s*(?:<i[^>]*>)?\s*{re.escape(label)}\s*:\s*(?:</i>)?\s*(?P<value>.*?)\s*</li>",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(event_html)
    if match is None:
        return None
    return _as_optional_str(_clean_text(match.group("value")))


def parse_fight_details_page(
    *,
    bout_id: str,
    fight_url: str,
    fight_html: str,
    fighter_red_id: str,
    fighter_blue_id: str,
) -> Mapping[str, object]:
    """Parse fight-details page HTML into per-fighter stat lines."""

    totals_table = _find_totals_table(fight_html)
    if totals_table is None:
        return {
            "bout_id": bout_id,
            "fight_url": fight_url,
            "fighter_bout_stats": (
                _empty_fighter_bout_stats(fighter_id=fighter_red_id, opponent_id=fighter_blue_id, corner="red"),
                _empty_fighter_bout_stats(fighter_id=fighter_blue_id, opponent_id=fighter_red_id, corner="blue"),
            ),
        }

    headers, row_cells = totals_table
    index_map = _totals_table_index_map(headers)
    red_stats, blue_stats = _extract_fighter_stats_from_totals_row(row_cells=row_cells, index_map=index_map)

    return {
        "bout_id": bout_id,
        "fight_url": fight_url,
        "fighter_bout_stats": (
            {
                "fighter_id": fighter_red_id,
                "opponent_fighter_id": fighter_blue_id,
                "corner": "red",
                **red_stats,
            },
            {
                "fighter_id": fighter_blue_id,
                "opponent_fighter_id": fighter_red_id,
                "corner": "blue",
                **blue_stats,
            },
        ),
    }


def _event_row_fighter_bout_stats(
    *,
    td_cells: Sequence[Sequence[str]],
    red_id: str,
    blue_id: str,
) -> tuple[dict[str, object], dict[str, object]]:
    red_kd, blue_kd = _cell_pair_to_ints(td_cells, 2)
    red_sig_landed, blue_sig_landed = _cell_pair_to_ints(td_cells, 3)
    red_td_landed, blue_td_landed = _cell_pair_to_ints(td_cells, 4)
    red_sub, blue_sub = _cell_pair_to_ints(td_cells, 5)

    red = _empty_fighter_bout_stats(fighter_id=red_id, opponent_id=blue_id, corner="red")
    blue = _empty_fighter_bout_stats(fighter_id=blue_id, opponent_id=red_id, corner="blue")
    red["knockdowns"] = red_kd
    blue["knockdowns"] = blue_kd
    red["sig_strikes_landed"] = red_sig_landed
    blue["sig_strikes_landed"] = blue_sig_landed
    red["takedowns_landed"] = red_td_landed
    blue["takedowns_landed"] = blue_td_landed
    red["submission_attempts"] = red_sub
    blue["submission_attempts"] = blue_sub
    return red, blue


def _coerce_event_row_fighter_stats(
    *,
    value: object,
    red_id: str,
    blue_id: str,
) -> tuple[dict[str, object], dict[str, object]] | None:
    if not isinstance(value, Sequence) or len(value) < 2:
        return None
    rows: list[dict[str, object]] = []
    for item in value[:2]:
        if not isinstance(item, Mapping):
            return None
        fighter_id = _as_optional_str(item.get("fighter_id"))
        opponent_id = _as_optional_str(item.get("opponent_fighter_id"))
        corner = _as_optional_str(item.get("corner"))
        if not fighter_id or not opponent_id or not corner:
            return None
        stats_row = _empty_fighter_bout_stats(fighter_id=fighter_id, opponent_id=opponent_id, corner=corner)
        for key in _empty_stats_metrics():
            stats_row[key] = _as_optional_int(item.get(key))
        rows.append(stats_row)
    if len(rows) != 2:
        return None
    ids = {rows[0]["fighter_id"], rows[1]["fighter_id"]}
    expected = {red_id, blue_id}
    if ids != expected:
        return None
    return rows[0], rows[1]


def _winner_from_row_columns(*, td_values: Sequence[str], red_id: str, blue_id: str) -> str | None:
    if not td_values:
        return None

    status = td_values[0].strip().lower()
    if status.startswith("win") or status == "w":
        return red_id
    if status.startswith("loss") or status == "l":
        return blue_id
    return None


def _event_is_not_after(event_date_text: str | None, as_of_utc: datetime) -> bool:
    parsed = _parse_date_to_datetime(event_date_text)
    if parsed is None:
        return True
    return parsed <= as_of_utc.astimezone(timezone.utc)


def _parse_date_to_datetime(date_text: str | None) -> datetime | None:
    if not date_text:
        return None

    cleaned = _clean_text(date_text)
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _parse_date_to_utc_iso(date_text: str | None) -> str | None:
    parsed = _parse_date_to_datetime(date_text)
    if parsed is None:
        return None
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_location(location: str | None) -> tuple[str | None, str | None, str | None]:
    if not location:
        return (None, None, None)
    parts = [part.strip() for part in location.split(",") if part.strip()]
    if len(parts) == 1:
        return (parts[0], None, None)
    if len(parts) == 2:
        return (parts[0], None, parts[1])
    return (parts[0], parts[1], parts[2])


def _duration_to_seconds(value: str | None) -> int | None:
    if not value:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None

    parts = cleaned.split(":")
    if len(parts) != 2:
        return None

    minutes = _safe_int(parts[0])
    seconds = _safe_int(parts[1])
    if minutes is None or seconds is None:
        return None

    return (minutes * 60) + seconds


def _find_totals_table(html: str) -> tuple[list[str], list[list[str]]] | None:
    for table_match in _FIGHT_DETAILS_TABLE_RE.finditer(html):
        table_body = table_match.group("body")
        headers = [_clean_text(match.group("value")) for match in _TH_RE.finditer(table_body)]
        header_tokens = {_header_token(header) for header in headers}
        if "kd" not in header_tokens or "ctrl" not in header_tokens:
            continue

        rows = [match.group("body") for match in _TR_RE.finditer(table_body)]
        totals_row_cells: list[list[str]] | None = None
        fallback_row_cells: list[list[str]] | None = None
        for row_html in rows:
            cells = [_extract_cell_values(td.group("value")) for td in _TD_RE.finditer(row_html)]
            if not cells:
                continue
            if fallback_row_cells is None:
                fallback_row_cells = cells
            first_values = [value.lower() for value in cells[0]]
            if any(value.startswith("tot") for value in first_values):
                totals_row_cells = cells
                break

        chosen = totals_row_cells or fallback_row_cells
        if chosen is not None:
            return headers, chosen
    return None


def _extract_cell_values(cell_html: str) -> list[str]:
    p_values = [_clean_text(match.group("value")) for match in _P_RE.finditer(cell_html)]
    values = [value for value in p_values if value]
    if values:
        return values
    cleaned = _clean_text(cell_html)
    return [cleaned] if cleaned else []


def _header_token(value: str) -> str:
    return re.sub(r"[^a-z0-9%]+", "", value.lower())


def _totals_table_index_map(headers: Sequence[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for index, header in enumerate(headers):
        token = _header_token(header)
        if token == "kd":
            mapping["knockdowns"] = index
        elif token == "sigstr":
            mapping["sig_strikes"] = index
        elif token == "totalstr":
            mapping["total_strikes"] = index
        elif token == "td":
            mapping["takedowns"] = index
        elif token == "subatt":
            mapping["submission_attempts"] = index
        elif token == "rev":
            mapping["reversals"] = index
        elif token == "ctrl":
            mapping["control_time_seconds"] = index
    return mapping


def _extract_fighter_stats_from_totals_row(
    *,
    row_cells: Sequence[Sequence[str]],
    index_map: Mapping[str, int],
) -> tuple[dict[str, int | None], dict[str, int | None]]:
    red = _empty_stats_metrics()
    blue = _empty_stats_metrics()

    red["knockdowns"], blue["knockdowns"] = _cell_pair_to_ints(row_cells, index_map.get("knockdowns"))

    red_sig, blue_sig = _cell_pair_to_landed_attempted(row_cells, index_map.get("sig_strikes"))
    red["sig_strikes_landed"], red["sig_strikes_attempted"] = red_sig
    blue["sig_strikes_landed"], blue["sig_strikes_attempted"] = blue_sig

    red_total, blue_total = _cell_pair_to_landed_attempted(row_cells, index_map.get("total_strikes"))
    red["total_strikes_landed"], red["total_strikes_attempted"] = red_total
    blue["total_strikes_landed"], blue["total_strikes_attempted"] = blue_total

    red_td, blue_td = _cell_pair_to_landed_attempted(row_cells, index_map.get("takedowns"))
    red["takedowns_landed"], red["takedowns_attempted"] = red_td
    blue["takedowns_landed"], blue["takedowns_attempted"] = blue_td

    red["submission_attempts"], blue["submission_attempts"] = _cell_pair_to_ints(
        row_cells, index_map.get("submission_attempts")
    )
    red["reversals"], blue["reversals"] = _cell_pair_to_ints(row_cells, index_map.get("reversals"))
    red["control_time_seconds"], blue["control_time_seconds"] = _cell_pair_to_durations(
        row_cells, index_map.get("control_time_seconds")
    )

    return red, blue


def _empty_fighter_bout_stats(*, fighter_id: str, opponent_id: str, corner: str) -> dict[str, object]:
    return {
        "fighter_id": fighter_id,
        "opponent_fighter_id": opponent_id,
        "corner": corner,
        **_empty_stats_metrics(),
    }


def _empty_stats_metrics() -> dict[str, int | None]:
    return {
        "knockdowns": None,
        "sig_strikes_landed": None,
        "sig_strikes_attempted": None,
        "total_strikes_landed": None,
        "total_strikes_attempted": None,
        "takedowns_landed": None,
        "takedowns_attempted": None,
        "submission_attempts": None,
        "reversals": None,
        "control_time_seconds": None,
    }


def _cell_pair_to_ints(row_cells: Sequence[Sequence[str]], index: int | None) -> tuple[int | None, int | None]:
    red_text, blue_text = _cell_pair_values(row_cells, index)
    return _safe_int(_extract_first_number_text(red_text)), _safe_int(_extract_first_number_text(blue_text))


def _cell_pair_to_durations(row_cells: Sequence[Sequence[str]], index: int | None) -> tuple[int | None, int | None]:
    red_text, blue_text = _cell_pair_values(row_cells, index)
    return _duration_to_seconds(red_text), _duration_to_seconds(blue_text)


def _cell_pair_to_landed_attempted(
    row_cells: Sequence[Sequence[str]], index: int | None
) -> tuple[tuple[int | None, int | None], tuple[int | None, int | None]]:
    red_text, blue_text = _cell_pair_values(row_cells, index)
    return _parse_landed_attempted(red_text), _parse_landed_attempted(blue_text)


def _cell_pair_values(row_cells: Sequence[Sequence[str]], index: int | None) -> tuple[str | None, str | None]:
    if index is None or index < 0 or index >= len(row_cells):
        return (None, None)
    values = row_cells[index]
    if not values:
        return (None, None)
    if len(values) == 1:
        return (values[0], None)
    return (values[0], values[1])


def _parse_landed_attempted(value: str | None) -> tuple[int | None, int | None]:
    if not value:
        return (None, None)
    cleaned = value.strip()
    if not cleaned or cleaned == "--":
        return (None, None)

    match = re.search(r"(?P<landed>\d+)\s*of\s*(?P<attempted>\d+)", cleaned, re.IGNORECASE)
    if match is None:
        maybe = _safe_int(_extract_first_number_text(cleaned))
        return (maybe, None)
    return (_safe_int(match.group("landed")), _safe_int(match.group("attempted")))


def _extract_first_number_text(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"\d+", value)
    return match.group(0) if match is not None else None


def _height_text_to_cm(value: str | None) -> float | None:
    if not value:
        return None
    cleaned = value.strip()
    if not cleaned or cleaned == "--":
        return None

    match = re.search(r"(?P<feet>\d+)\s*'\s*(?P<inches>\d+)", cleaned)
    if match is None:
        return None

    feet = _safe_int(match.group("feet"))
    inches = _safe_int(match.group("inches"))
    if feet is None or inches is None:
        return None

    return round(((feet * 12) + inches) * 2.54, 2)


def _reach_text_to_cm(value: str | None) -> float | None:
    if not value:
        return None
    cleaned = value.strip()
    if not cleaned or cleaned == "--":
        return None

    match = re.search(r"(?P<inches>\d+(?:\.\d+)?)", cleaned)
    if match is None:
        return None

    try:
        inches = float(match.group("inches"))
    except ValueError:
        return None
    return round(inches * 2.54, 2)


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def _as_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _as_optional_int(value: object) -> int | None:
    return _safe_int(value)


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_text(value: str) -> str:
    without_tags = _TAG_RE.sub(" ", value)
    return _SPACE_RE.sub(" ", without_tags).strip()


def _endpoint_token(url: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", url.strip()).strip("_").lower()
    return cleaned[-60:] if len(cleaned) > 60 else cleaned


def _discover_latest_archived_html_by_id(
    *,
    raw_root: Path,
    id_pattern: re.Pattern[str],
) -> dict[str, Path]:
    source_root = raw_root / UFC_STATS_SOURCE
    if not source_root.exists():
        return {}

    latest_by_id: dict[str, Path] = {}
    for path in source_root.rglob("*.html"):
        match = id_pattern.search(path.name)
        if match is None:
            continue
        entity_id = match.group("id").lower()
        existing = latest_by_id.get(entity_id)
        if existing is None or path.as_posix() > existing.as_posix():
            latest_by_id[entity_id] = path
    return latest_by_id


def _stable_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _coalesce(value: str | None, fallback: str) -> str:
    return value if value is not None else fallback


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
