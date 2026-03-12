"""Market odds ingestion from The Odds API with chronology-safe bout matching."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence
from urllib.parse import urlencode
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
from markets.reporting import MarketCoverageReportResult, run_market_coverage_report

LOGGER = logging.getLogger(__name__)

ODDS_API_SOURCE = "odds_api"
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_API_DEFAULT_SPORT_KEY = "mma_mixed_martial_arts"
ODDS_SOURCE_MODE_UPCOMING = "upcoming"
ODDS_SOURCE_MODE_HISTORICAL = "historical"
OddsSourceMode = Literal["upcoming", "historical"]
MATCH_EXACT = "exact_match"
MATCH_NORMALIZED = "normalized_name_match"
MATCH_REVERSED = "reversed_name_match"
MATCH_AMBIGUOUS = "ambiguous_match"
MATCH_NONE = "no_match"
MatchCategory = Literal[
    "exact_match",
    "normalized_name_match",
    "reversed_name_match",
    "ambiguous_match",
    "no_match",
]


@dataclass(frozen=True)
class MarketIngestionRunResult:
    """End-to-end result for market odds ingestion and coverage reporting."""

    inserted_count: int
    updated_count: int
    skipped_count: int
    matched_rows: int
    unmatched_rows: int
    exact_matches: int
    normalized_matches: int
    reversed_matches: int
    ambiguous_rows: int
    no_match_rows: int
    matching_audit_report_path: Path
    report: MarketCoverageReportResult


@dataclass(frozen=True)
class MatchAuditResult:
    """Per-run matching diagnostics across parsed market rows."""

    total_market_rows: int
    exact_matches: int
    normalized_matches: int
    reversed_matches: int
    ambiguous_rows: int
    no_match_rows: int
    top_unmatched_reasons: tuple[tuple[str, int], ...]
    sample_unmatched_records: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class ParsedMarketRecord:
    """Typed parsed market record before bout matching."""

    source_record_id: str
    sportsbook: str
    market_type: str
    market_timestamp_utc: str
    event_date_utc: str | None
    event_name: str | None
    source_event_id: str
    fighter_name: str
    opponent_name: str
    selection_name: str
    odds_american: int | None
    odds_decimal: float | None
    implied_probability: float | None
    line_value: float | None


@dataclass(frozen=True)
class ArchivedResponse:
    """Metadata for archived raw API responses."""

    archive_path: Path
    payload_sha256: str


@dataclass(frozen=True)
class _FighterRow:
    fighter_id: str
    full_name: str
    normalized_name: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class _BoutRow:
    bout_id: str
    red_fighter_id: str
    blue_fighter_id: str
    event_date_utc: str
    bout_start_utc: str
    event_name: str
    normalized_event_name: str
    matchup_key: tuple[str, str]


@dataclass(frozen=True)
class _CandidateSide:
    fighter_id: str
    opponent_id: str
    score: int


@dataclass(frozen=True)
class _CandidateMatch:
    bout: _BoutRow
    side: _CandidateSide
    event_delta_seconds: float


@dataclass(frozen=True)
class _MatchOutcome:
    category: MatchCategory
    bout: _BoutRow | None
    selection_fighter_id: str | None
    reason: str | None


class OddsFetcher(Fetcher):
    """Fetch The Odds API snapshots and archive raw responses."""

    source_system = ODDS_API_SOURCE

    def __init__(
        self,
        *,
        raw_root: Path | None = None,
        api_key: str | None = None,
        api_base_url: str = ODDS_API_BASE_URL,
        sport_key: str = ODDS_API_DEFAULT_SPORT_KEY,
        regions: str = "us",
        markets: str = "h2h",
        odds_format: str = "american",
        date_format: str = "iso",
        timeout_seconds: int = 30,
        user_agent: str = "mma-elo-ingestion/0.1",
        source_system: str | None = None,
        from_archives: bool = False,
        source_mode: OddsSourceMode = ODDS_SOURCE_MODE_UPCOMING,
        historical_start_utc: datetime | None = None,
        historical_end_utc: datetime | None = None,
        historical_interval_hours: int = 24,
    ) -> None:
        self._raw_root = raw_root
        self._api_key = _as_optional_text(api_key)
        self._api_base_url = api_base_url.rstrip("/")
        self._sport_key = sport_key.strip()
        self._regions = regions.strip()
        self._markets = markets.strip()
        self._odds_format = odds_format.strip()
        self._date_format = date_format.strip()
        self._timeout_seconds = timeout_seconds
        self._user_agent = user_agent
        self._source_system = source_system.strip() if source_system else self.source_system
        self._from_archives = from_archives
        self._source_mode = source_mode
        self._historical_start_utc = historical_start_utc.astimezone(timezone.utc) if historical_start_utc else None
        self._historical_end_utc = historical_end_utc.astimezone(timezone.utc) if historical_end_utc else None
        self._historical_interval_hours = max(1, historical_interval_hours)

    def fetch(self, *, as_of_utc: datetime | None = None) -> Sequence[RawIngestionRecord]:
        """Fetch live API payloads or replay archived snapshots."""

        if self._from_archives:
            return self._fetch_from_archives(as_of_utc=as_of_utc)

        if self._api_key is None:
            return ()

        if self._source_mode == ODDS_SOURCE_MODE_HISTORICAL:
            return self._fetch_historical_live(as_of_utc=as_of_utc)

        fetched_at = datetime.now(timezone.utc)
        request_url = self._build_upcoming_odds_url(api_key=self._api_key)
        response_text = self._http_get_text(request_url)
        payload_sha = _stable_sha256(response_text)

        archive_path: str | None = None
        if self._raw_root is not None:
            archived = archive_raw_response(
                raw_root=self._raw_root,
                fetched_at_utc=fetched_at,
                source_system=self._source_system,
                url=request_url,
                payload_text=response_text,
            )
            archive_path = archived.archive_path.as_posix()

        payload_json = _safe_json_loads(response_text)
        events = _extract_events_from_api_payload(payload_json)

        payload: Mapping[str, Any] = {
            "request_url": request_url,
            "archive_path": archive_path,
            "events": events,
            "source_mode": self._source_mode,
        }
        return (
            RawIngestionRecord(
                source_system=self._source_system,
                source_record_id=f"odds_snapshot:{fetched_at.strftime('%Y%m%dT%H%M%SZ')}",
                fetched_at_utc=fetched_at,
                payload=payload,
                payload_sha256=payload_sha,
                idempotency_key=_stable_sha256(f"{self._source_system}|snapshot|{payload_sha}"),
            ),
        )

    def _fetch_from_archives(self, *, as_of_utc: datetime | None = None) -> Sequence[RawIngestionRecord]:
        if self._raw_root is None:
            return ()

        records: list[RawIngestionRecord] = []
        for archive_path in _discover_archived_raw_json(
            raw_root=self._raw_root,
            source_system=self._source_system,
        ):
            payload_text = archive_path.read_text(encoding="utf-8", errors="replace")
            payload_sha = _stable_sha256(payload_text)
            fetched_at = _parse_fetched_at_from_archive_path(archive_path=archive_path)
            if as_of_utc is not None and fetched_at > as_of_utc.astimezone(timezone.utc):
                continue

            payload_json = _safe_json_loads(payload_text)
            events = _extract_events_from_api_payload(payload_json)
            snapshot_timestamp_utc = _extract_payload_snapshot_timestamp(payload_json)

            records.append(
                RawIngestionRecord(
                    source_system=self._source_system,
                    source_record_id=f"odds_snapshot_archive:{payload_sha[:16]}",
                    fetched_at_utc=fetched_at,
                    payload={
                        "request_url": None,
                        "archive_path": archive_path.as_posix(),
                        "events": events,
                        "historical_snapshot_utc": snapshot_timestamp_utc,
                        "source_mode": self._source_mode,
                    },
                    payload_sha256=payload_sha,
                    idempotency_key=_stable_sha256(f"{self._source_system}|snapshot_archive|{payload_sha}"),
                )
            )

        records.sort(key=lambda record: record.fetched_at_utc)
        return tuple(records)

    def _fetch_historical_live(self, *, as_of_utc: datetime | None) -> Sequence[RawIngestionRecord]:
        if self._api_key is None:
            return ()
        window_end = (as_of_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
        window_start = self._historical_start_utc or window_end
        bounded_end = self._historical_end_utc or window_end
        if bounded_end < window_start:
            window_start, bounded_end = bounded_end, window_start

        records: list[RawIngestionRecord] = []
        for snapshot_dt in _iter_historical_snapshot_times(
            start_utc=window_start,
            end_utc=bounded_end,
            interval_hours=self._historical_interval_hours,
        ):
            request_url = self._build_historical_odds_url(api_key=self._api_key, snapshot_utc=snapshot_dt)
            response_text = self._http_get_text(request_url)
            payload_sha = _stable_sha256(response_text)
            fetched_at = datetime.now(timezone.utc)

            archive_path: str | None = None
            if self._raw_root is not None:
                archived = archive_raw_response(
                    raw_root=self._raw_root,
                    fetched_at_utc=fetched_at,
                    source_system=self._source_system,
                    url=request_url,
                    payload_text=response_text,
                )
                archive_path = archived.archive_path.as_posix()

            payload_json = _safe_json_loads(response_text)
            events = _extract_events_from_api_payload(payload_json)
            snapshot_timestamp_utc = _extract_payload_snapshot_timestamp(payload_json) or _normalize_utc_text(
                snapshot_dt.isoformat()
            )
            payload: Mapping[str, Any] = {
                "request_url": request_url,
                "archive_path": archive_path,
                "events": events,
                "historical_snapshot_utc": snapshot_timestamp_utc,
                "source_mode": self._source_mode,
            }
            records.append(
                RawIngestionRecord(
                    source_system=self._source_system,
                    source_record_id=f"odds_historical_snapshot:{snapshot_dt.strftime('%Y%m%dT%H%M%SZ')}",
                    fetched_at_utc=fetched_at,
                    payload=payload,
                    payload_sha256=payload_sha,
                    idempotency_key=_stable_sha256(
                        f"{self._source_system}|historical_snapshot|{snapshot_timestamp_utc}|{payload_sha}"
                    ),
                )
            )
        records.sort(key=lambda record: record.fetched_at_utc)
        return tuple(records)

    def _build_upcoming_odds_url(self, *, api_key: str) -> str:
        query = urlencode(
            {
                "apiKey": api_key,
                "regions": self._regions,
                "markets": self._markets,
                "oddsFormat": self._odds_format,
                "dateFormat": self._date_format,
            }
        )
        return f"{self._api_base_url}/sports/{self._sport_key}/odds/?{query}"

    def _build_historical_odds_url(self, *, api_key: str, snapshot_utc: datetime) -> str:
        query = urlencode(
            {
                "apiKey": api_key,
                "regions": self._regions,
                "markets": self._markets,
                "oddsFormat": self._odds_format,
                "dateFormat": self._date_format,
                "date": snapshot_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            }
        )
        return f"{self._api_base_url}/historical/sports/{self._sport_key}/odds/?{query}"

    def _http_get_text(self, url: str) -> str:
        request = Request(url=url, headers={"User-Agent": self._user_agent})
        with urlopen(request, timeout=self._timeout_seconds) as response:
            payload = response.read()
        return payload.decode("utf-8", errors="replace")


class OddsParser(Parser):
    """Parse The Odds API snapshots into per-selection market rows."""

    def __init__(self, *, source_mode: OddsSourceMode = ODDS_SOURCE_MODE_UPCOMING) -> None:
        self._source_mode = source_mode

    def parse(self, records: Sequence[RawIngestionRecord]) -> Sequence[ParsedIngestionRecord]:
        parsed: list[ParsedIngestionRecord] = []

        for record in records:
            events = record.payload.get("events")
            if not isinstance(events, list):
                continue
            snapshot_timestamp_utc = _normalize_utc_text(record.payload.get("historical_snapshot_utc"))

            for event in events:
                if not isinstance(event, Mapping):
                    continue
                event_id = _as_optional_text(event.get("id"))
                event_date_utc = _normalize_utc_text(event.get("commence_time"))
                event_name = (
                    _as_optional_text(event.get("event_name"))
                    or _as_optional_text(event.get("name"))
                    or _as_optional_text(event.get("sport_title"))
                    or _event_name_from_teams(event)
                )
                if event_id is None:
                    continue

                bookmakers = event.get("bookmakers")
                if not isinstance(bookmakers, list):
                    continue

                for bookmaker in bookmakers:
                    if not isinstance(bookmaker, Mapping):
                        continue
                    sportsbook = _as_optional_text(bookmaker.get("title")) or (
                        _as_optional_text(bookmaker.get("key")) or "unknown"
                    )
                    bookmaker_key = _as_optional_text(bookmaker.get("key")) or sportsbook.lower()
                    bookmaker_last = _normalize_utc_text(bookmaker.get("last_update"))

                    markets = bookmaker.get("markets")
                    if not isinstance(markets, list):
                        continue

                    for market in markets:
                        if not isinstance(market, Mapping):
                            continue
                        market_key = (_as_optional_text(market.get("key")) or "").lower()
                        if market_key != "h2h":
                            continue

                        market_last = _normalize_utc_text(market.get("last_update"))
                        market_timestamp_utc = (
                            market_last
                            or bookmaker_last
                            or snapshot_timestamp_utc
                            or _normalize_utc_text(record.fetched_at_utc.isoformat())
                        )
                        if market_timestamp_utc is None:
                            continue

                        outcomes_raw = market.get("outcomes")
                        if not isinstance(outcomes_raw, list):
                            continue

                        outcomes: list[tuple[str, int | None]] = []
                        for outcome in outcomes_raw:
                            if not isinstance(outcome, Mapping):
                                continue
                            name = _as_optional_text(outcome.get("name"))
                            if name is None:
                                continue
                            price = _as_optional_int(outcome.get("price"))
                            outcomes.append((name, price))

                        if len(outcomes) != 2:
                            continue

                        fighter_a, odds_a = outcomes[0]
                        fighter_b, odds_b = outcomes[1]
                        rows = (
                            (fighter_a, fighter_b, odds_a),
                            (fighter_b, fighter_a, odds_b),
                        )

                        for fighter_name, opponent_name, odds_american in rows:
                            source_record_id = _build_outcome_source_record_id(
                                event_id=event_id,
                                bookmaker_key=bookmaker_key,
                                market_key=market_key,
                                selection_name=fighter_name,
                                market_timestamp_utc=market_timestamp_utc,
                            )
                            parsed_row = ParsedMarketRecord(
                                source_record_id=source_record_id,
                                sportsbook=sportsbook,
                                market_type="moneyline",
                                market_timestamp_utc=market_timestamp_utc,
                                event_date_utc=event_date_utc,
                                event_name=event_name,
                                source_event_id=event_id,
                                fighter_name=fighter_name,
                                opponent_name=opponent_name,
                                selection_name=fighter_name,
                                odds_american=odds_american,
                                odds_decimal=None,
                                implied_probability=None,
                                line_value=None,
                            )
                            parsed_payload = {
                                "source_record_id": parsed_row.source_record_id,
                                "sportsbook": parsed_row.sportsbook,
                                "market_type": parsed_row.market_type,
                                "market_timestamp_utc": parsed_row.market_timestamp_utc,
                                "event_date_utc": parsed_row.event_date_utc,
                                "event_name": parsed_row.event_name,
                                "source_event_id": parsed_row.source_event_id,
                                "fighter_name": parsed_row.fighter_name,
                                "opponent_name": parsed_row.opponent_name,
                                "selection_name": parsed_row.selection_name,
                                "odds_american": parsed_row.odds_american,
                                "odds_decimal": parsed_row.odds_decimal,
                                "implied_probability": parsed_row.implied_probability,
                                "line_value": parsed_row.line_value,
                            }
                            payload_sha = _stable_sha256(json.dumps(parsed_payload, sort_keys=True))
                            parsed.append(
                                ParsedIngestionRecord(
                                    source_system=record.source_system,
                                    source_record_id=source_record_id,
                                    parsed_payload=parsed_payload,
                                    payload_sha256=payload_sha,
                                    idempotency_key=_stable_sha256(
                                        f"{record.source_system}|market_outcome|{source_record_id}|{payload_sha}"
                                    ),
                                )
                            )

        parsed.sort(key=lambda record: record.source_record_id)
        return tuple(parsed)


class OddsNormalizer(Normalizer):
    """Match parsed market rows to fights/fighters and normalize for `markets` table."""

    def __init__(
        self,
        *,
        db_path: Path | None = None,
        promotion: str = "UFC",
        event_date_tolerance_days: int = 3,
        completed_bouts_only: bool = False,
    ) -> None:
        self._db_path = db_path
        self._promotion = promotion
        self._event_date_tolerance = timedelta(days=max(0, event_date_tolerance_days))
        self._completed_bouts_only = completed_bouts_only
        self.last_matched_rows = 0
        self.last_unmatched_rows = 0
        self.last_audit = MatchAuditResult(
            total_market_rows=0,
            exact_matches=0,
            normalized_matches=0,
            reversed_matches=0,
            ambiguous_rows=0,
            no_match_rows=0,
            top_unmatched_reasons=(),
            sample_unmatched_records=(),
        )

    def normalize(self, records: Sequence[ParsedIngestionRecord]) -> Sequence[NormalizedRow]:
        """Normalize parsed rows and attach bout/fighter ids before persistence."""

        if self._db_path is None:
            self.last_matched_rows = 0
            self.last_unmatched_rows = len(records)
            self.last_audit = MatchAuditResult(
                total_market_rows=len(records),
                exact_matches=0,
                normalized_matches=0,
                reversed_matches=0,
                ambiguous_rows=0,
                no_match_rows=len(records),
                top_unmatched_reasons=(("missing_db_path", len(records)),) if len(records) > 0 else (),
                sample_unmatched_records=(),
            )
            return ()

        exact_name_index, alias_name_index, bout_index = _load_matching_indexes(
            db_path=self._db_path,
            promotion=self._promotion,
            completed_bouts_only=self._completed_bouts_only,
        )
        now_utc = _utc_now_iso()
        rows: list[NormalizedRow] = []
        matched = 0
        unmatched = 0
        category_counts: Counter[str] = Counter()
        unmatched_reasons: Counter[str] = Counter()
        unmatched_samples: list[dict[str, str]] = []
        context_bout_by_source: dict[tuple[str, str], str] = {}

        for record in records:
            parsed = record.parsed_payload
            fighter_name = _as_optional_text(parsed.get("fighter_name"))
            opponent_name = _as_optional_text(parsed.get("opponent_name"))
            market_timestamp = _normalize_utc_text(parsed.get("market_timestamp_utc"))
            if fighter_name is None or opponent_name is None or market_timestamp is None:
                category_counts[MATCH_NONE] += 1
                unmatched_reasons["missing_required_market_fields"] += 1
                if len(unmatched_samples) < 10:
                    unmatched_samples.append(
                        _build_unmatched_sample(
                            parsed=parsed,
                            category=MATCH_NONE,
                            reason="missing_required_market_fields",
                        )
                    )
                unmatched += 1
                continue

            outcome = _match_market_row(
                fighter_name=fighter_name,
                opponent_name=opponent_name,
                event_name=_as_optional_text(parsed.get("event_name")),
                source_event_id=_as_optional_text(parsed.get("source_event_id")),
                sportsbook=_as_optional_text(parsed.get("sportsbook")) or "unknown",
                event_date_utc=_normalize_utc_text(parsed.get("event_date_utc")),
                market_timestamp_utc=market_timestamp,
                exact_name_index=exact_name_index,
                alias_name_index=alias_name_index,
                bout_index=bout_index,
                source_context=context_bout_by_source,
                event_date_tolerance=self._event_date_tolerance,
            )
            category_counts[outcome.category] += 1

            if outcome.bout is None or outcome.selection_fighter_id is None:
                unmatched += 1
                unmatched_reasons[outcome.reason or "no_candidate_bouts"] += 1
                if len(unmatched_samples) < 10:
                    unmatched_samples.append(
                        _build_unmatched_sample(
                            parsed=parsed,
                            category=outcome.category,
                            reason=outcome.reason or "no_candidate_bouts",
                        )
                    )
                continue

            source_event_id = _as_optional_text(parsed.get("source_event_id"))
            sportsbook = _as_optional_text(parsed.get("sportsbook")) or "unknown"
            if source_event_id is not None:
                context_bout_by_source[(source_event_id, sportsbook)] = outcome.bout.bout_id

            market_payload = _build_market_payload(
                source_system=record.source_system,
                source_record_id=record.source_record_id,
                sportsbook=sportsbook,
                market_type=str(parsed.get("market_type") or "moneyline"),
                selection_fighter_id=outcome.selection_fighter_id,
                selection_label=str(parsed.get("selection_name") or fighter_name),
                market_timestamp_utc=market_timestamp,
                bout_id=outcome.bout.bout_id,
                odds_american=_as_optional_int(parsed.get("odds_american")),
                odds_decimal=_as_optional_float(parsed.get("odds_decimal")),
                implied_probability=_as_optional_float(parsed.get("implied_probability")),
                line_value=_as_optional_float(parsed.get("line_value")),
                now_utc=now_utc,
            )
            market_sha = _stable_sha256(json.dumps(market_payload, sort_keys=True))
            market_id = _stable_sha256(
                f"{record.source_system}|{record.source_record_id}|{outcome.bout.bout_id}|{outcome.selection_fighter_id}"
            )[:32]
            rows.append(
                NormalizedRow(
                    table_name="markets",
                    primary_key=market_id,
                    data={
                        "market_id": market_id,
                        **market_payload,
                        "source_payload_sha256": market_sha,
                    },
                    source_system=record.source_system,
                    source_record_id=record.source_record_id,
                    payload_sha256=market_sha,
                    idempotency_key=record.idempotency_key,
                )
            )
            matched += 1

        rows.sort(key=lambda row: row.primary_key)
        self.last_matched_rows = matched
        self.last_unmatched_rows = unmatched
        self.last_audit = MatchAuditResult(
            total_market_rows=len(records),
            exact_matches=int(category_counts[MATCH_EXACT]),
            normalized_matches=int(category_counts[MATCH_NORMALIZED]),
            reversed_matches=int(category_counts[MATCH_REVERSED]),
            ambiguous_rows=int(category_counts[MATCH_AMBIGUOUS]),
            no_match_rows=int(category_counts[MATCH_NONE]),
            top_unmatched_reasons=tuple(unmatched_reasons.most_common(10)),
            sample_unmatched_records=tuple(unmatched_samples),
        )
        return tuple(rows)


class OddsPersister(Persister):
    """Idempotent persistence of normalized market rows."""

    def __init__(self, *, db_path: Path | None = None) -> None:
        self._db_path = db_path

    def persist(self, rows: Sequence[NormalizedRow]) -> PersistResult:
        """Persist market rows into SQLite with upsert-by-source semantics."""

        if self._db_path is None:
            return PersistResult(inserted_count=0, updated_count=0, skipped_count=len(rows))

        initialize_sqlite_database(self._db_path)
        inserted = 0
        updated = 0
        skipped = 0

        with sqlite3.connect(self._db_path) as connection:
            connection.execute("PRAGMA foreign_keys = ON;")
            for row in rows:
                existing = connection.execute(
                    """
                    SELECT source_payload_sha256
                    FROM markets
                    WHERE source_system = ? AND source_record_id = ?
                    """,
                    (row.source_system, row.source_record_id),
                ).fetchone()

                if existing is None:
                    _insert_row(connection=connection, row=row)
                    inserted += 1
                    continue

                existing_sha = str(existing[0]) if existing[0] is not None else None
                if existing_sha == row.payload_sha256:
                    skipped += 1
                    continue

                _update_row_by_source_key(connection=connection, row=row)
                updated += 1

            connection.commit()

        return PersistResult(inserted_count=inserted, updated_count=updated, skipped_count=skipped)


def run_market_odds_ingestion(
    *,
    db_path: Path,
    raw_root: Path,
    report_output_path: Path,
    promotion: str = "UFC",
    source_system: str = ODDS_API_SOURCE,
    source_mode: OddsSourceMode = ODDS_SOURCE_MODE_UPCOMING,
    odds_api_key: str | None = None,
    odds_api_base_url: str = ODDS_API_BASE_URL,
    sport_key: str = ODDS_API_DEFAULT_SPORT_KEY,
    regions: str = "us",
    markets: str = "h2h",
    odds_format: str = "american",
    date_format: str = "iso",
    from_archives: bool = False,
    as_of_utc: datetime | None = None,
    historical_start_utc: datetime | None = None,
    historical_end_utc: datetime | None = None,
    historical_interval_hours: int = 24,
    completed_bouts_only: bool | None = None,
    matching_audit_output_path: Path | None = None,
) -> MarketIngestionRunResult:
    """Run market odds ingestion and write coverage validation report."""

    if not from_archives and _as_optional_text(odds_api_key) is None:
        raise ValueError("missing_odds_api_key env_var=MMA_ELO_ODDS_API_KEY")

    fetcher = OddsFetcher(
        raw_root=raw_root,
        api_key=odds_api_key,
        api_base_url=odds_api_base_url,
        sport_key=sport_key,
        regions=regions,
        markets=markets,
        odds_format=odds_format,
        date_format=date_format,
        source_system=source_system,
        from_archives=from_archives,
        source_mode=source_mode,
        historical_start_utc=historical_start_utc,
        historical_end_utc=historical_end_utc,
        historical_interval_hours=historical_interval_hours,
    )
    parser = OddsParser(source_mode=source_mode)
    normalize_completed_only = (
        source_mode == ODDS_SOURCE_MODE_HISTORICAL if completed_bouts_only is None else completed_bouts_only
    )
    normalizer = OddsNormalizer(
        db_path=db_path,
        promotion=promotion,
        completed_bouts_only=normalize_completed_only,
    )
    persister = OddsPersister(db_path=db_path)

    raw = fetcher.fetch(as_of_utc=as_of_utc)
    parsed = parser.parse(raw)
    normalized = normalizer.normalize(parsed)
    persist_result = persister.persist(normalized)
    report = run_market_coverage_report(
        db_path=db_path,
        promotion=promotion,
        source_system=source_system,
        unmatched_reasons=dict(normalizer.last_audit.top_unmatched_reasons),
        report_output_path=report_output_path,
    )
    audit_path = matching_audit_output_path or report_output_path.with_name("market_matching_audit_report.txt")
    _write_matching_audit_report(path=audit_path, audit=normalizer.last_audit)
    return MarketIngestionRunResult(
        inserted_count=persist_result.inserted_count,
        updated_count=persist_result.updated_count,
        skipped_count=persist_result.skipped_count,
        matched_rows=normalizer.last_matched_rows,
        unmatched_rows=normalizer.last_unmatched_rows,
        exact_matches=normalizer.last_audit.exact_matches,
        normalized_matches=normalizer.last_audit.normalized_matches,
        reversed_matches=normalizer.last_audit.reversed_matches,
        ambiguous_rows=normalizer.last_audit.ambiguous_rows,
        no_match_rows=normalizer.last_audit.no_match_rows,
        matching_audit_report_path=audit_path,
        report=report,
    )


def archive_raw_response(
    *,
    raw_root: Path,
    fetched_at_utc: datetime,
    source_system: str,
    url: str,
    payload_text: str,
) -> ArchivedResponse:
    """Archive raw JSON response payload to `data/raw` with deterministic naming."""

    payload_bytes = payload_text.encode("utf-8", errors="replace")
    payload_sha = hashlib.sha256(payload_bytes).hexdigest()
    endpoint_token = _endpoint_token(url)

    partition_dir = raw_root / source_system / fetched_at_utc.strftime("%Y/%m/%d")
    partition_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{fetched_at_utc.strftime('%H%M%S')}_{endpoint_token}_{payload_sha[:12]}.json"
    archive_path = partition_dir / filename
    archive_path.write_bytes(payload_bytes)

    return ArchivedResponse(archive_path=archive_path, payload_sha256=payload_sha)


def _load_matching_indexes(
    *,
    db_path: Path,
    promotion: str,
    completed_bouts_only: bool,
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[tuple[str, str], list[_BoutRow]]]:
    exact_name_index: dict[str, set[str]] = {}
    alias_name_index: dict[str, set[str]] = {}
    bout_index: dict[tuple[str, str], list[_BoutRow]] = {}

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        fighters = connection.execute(
            """
            SELECT fighter_id, full_name
            FROM fighters
            """
        ).fetchall()
        for row in fighters:
            full_name = str(row["full_name"])
            fighter = _FighterRow(
                fighter_id=str(row["fighter_id"]),
                full_name=full_name,
                normalized_name=_normalize_person_name(full_name),
                aliases=tuple(_fighter_name_aliases(full_name)),
            )
            if fighter.normalized_name != "":
                exact_name_index.setdefault(fighter.normalized_name, set()).add(fighter.fighter_id)
            for alias in fighter.aliases:
                alias_name_index.setdefault(alias, set()).add(fighter.fighter_id)

        bouts = connection.execute(
            """
            SELECT
                b.bout_id,
                b.fighter_red_id,
                b.fighter_blue_id,
                e.event_name,
                e.event_date_utc,
                COALESCE(b.bout_start_time_utc, e.event_date_utc) AS bout_start_utc
            FROM bouts AS b
            JOIN events AS e ON e.event_id = b.event_id
            WHERE UPPER(e.promotion) = UPPER(?)
              AND (? = 0 OR b.winner_fighter_id IS NOT NULL)
            """,
            (promotion, int(completed_bouts_only)),
        ).fetchall()

        for row in bouts:
            red_id = str(row["fighter_red_id"])
            blue_id = str(row["fighter_blue_id"])
            bout = _BoutRow(
                bout_id=str(row["bout_id"]),
                red_fighter_id=red_id,
                blue_fighter_id=blue_id,
                event_name=str(row["event_name"]),
                normalized_event_name=_normalize_event_name(str(row["event_name"])),
                event_date_utc=str(row["event_date_utc"]),
                bout_start_utc=str(row["bout_start_utc"]),
                matchup_key=tuple(sorted((red_id, blue_id))),
            )
            bout_index.setdefault(bout.matchup_key, []).append(bout)

    for matchup_key in list(bout_index.keys()):
        bout_index[matchup_key] = sorted(bout_index[matchup_key], key=lambda row: row.bout_start_utc)
    return exact_name_index, alias_name_index, bout_index


def _match_market_row(
    *,
    fighter_name: str,
    opponent_name: str,
    event_name: str | None,
    source_event_id: str | None,
    sportsbook: str,
    event_date_utc: str | None,
    market_timestamp_utc: str,
    exact_name_index: dict[str, set[str]],
    alias_name_index: dict[str, set[str]],
    bout_index: dict[tuple[str, str], list[_BoutRow]],
    source_context: dict[tuple[str, str], str],
    event_date_tolerance: timedelta,
) -> _MatchOutcome:
    fighter_normalized = _normalize_person_name(fighter_name)
    opponent_normalized = _normalize_person_name(opponent_name)
    fighter_exact_ids = exact_name_index.get(fighter_normalized, set())
    opponent_exact_ids = exact_name_index.get(opponent_normalized, set())
    fighter_alias_ids = _candidate_fighter_ids(name=fighter_name, alias_name_index=alias_name_index)
    opponent_alias_ids = _candidate_fighter_ids(name=opponent_name, alias_name_index=alias_name_index)

    if len(fighter_alias_ids) == 0 or len(opponent_alias_ids) == 0:
        return _MatchOutcome(
            category=MATCH_NONE,
            bout=None,
            selection_fighter_id=None,
            reason="fighter_name_not_found",
        )
    if fighter_alias_ids == opponent_alias_ids and len(fighter_alias_ids) == 1:
        return _MatchOutcome(
            category=MATCH_NONE,
            bout=None,
            selection_fighter_id=None,
            reason="fighter_and_opponent_resolve_to_same_id",
        )

    candidate_sides = _build_candidate_sides(
        fighter_ids=fighter_alias_ids,
        opponent_ids=opponent_alias_ids,
        fighter_exact_ids=fighter_exact_ids,
        opponent_exact_ids=opponent_exact_ids,
    )
    if len(candidate_sides) == 0:
        return _MatchOutcome(
            category=MATCH_NONE,
            bout=None,
            selection_fighter_id=None,
            reason="fighter_pair_not_found",
        )

    market_dt = _parse_utc_timestamp(market_timestamp_utc)
    if market_dt is None:
        return _MatchOutcome(
            category=MATCH_NONE,
            bout=None,
            selection_fighter_id=None,
            reason="invalid_market_timestamp",
        )

    event_dt = _parse_utc_timestamp(event_date_utc)
    normalized_event_name = _normalize_event_name(event_name or "")
    candidate_matches: list[_CandidateMatch] = []
    for side in candidate_sides:
        matchup_key = tuple(sorted((side.fighter_id, side.opponent_id)))
        bouts = bout_index.get(matchup_key, [])
        for bout in bouts:
            bout_start_dt = _parse_utc_timestamp(bout.bout_start_utc)
            if bout_start_dt is None:
                continue
            if market_dt > bout_start_dt:
                continue
            event_delta_seconds = _event_delta_seconds(event_dt=event_dt, bout=bout)
            if event_dt is not None and event_delta_seconds > event_date_tolerance.total_seconds():
                continue
            event_name_bonus = 1 if normalized_event_name != "" and normalized_event_name == bout.normalized_event_name else 0
            context_key = (source_event_id or "", sportsbook)
            context_bonus = 1 if source_context.get(context_key) == bout.bout_id else 0
            candidate_matches.append(
                _CandidateMatch(
                    bout=bout,
                    side=_CandidateSide(
                        fighter_id=side.fighter_id,
                        opponent_id=side.opponent_id,
                        score=side.score + event_name_bonus + context_bonus,
                    ),
                    event_delta_seconds=event_delta_seconds,
                )
            )

    if len(candidate_matches) == 0:
        if event_dt is not None:
            return _MatchOutcome(
                category=MATCH_NONE,
                bout=None,
                selection_fighter_id=None,
                reason="no_pre_fight_candidate_within_date_tolerance",
            )
        return _MatchOutcome(
            category=MATCH_NONE,
            bout=None,
            selection_fighter_id=None,
            reason="no_pre_fight_candidate_bout",
        )

    ordered = sorted(
        candidate_matches,
        key=lambda row: (
            -row.side.score,
            row.event_delta_seconds,
            _absolute_seconds_delta(market_dt, _parse_utc_timestamp(row.bout.bout_start_utc)),
            row.bout.bout_start_utc,
            row.bout.bout_id,
        ),
    )
    best = ordered[0]
    if len(ordered) > 1:
        next_best = ordered[1]
        if (
            best.side.score == next_best.side.score
            and best.event_delta_seconds == next_best.event_delta_seconds
            and best.bout.bout_id != next_best.bout.bout_id
        ):
            return _MatchOutcome(
                category=MATCH_AMBIGUOUS,
                bout=None,
                selection_fighter_id=None,
                reason="multiple_candidate_bouts_same_rank",
            )

    selection_fighter_id = best.side.fighter_id
    if selection_fighter_id == best.bout.blue_fighter_id and best.side.opponent_id == best.bout.red_fighter_id:
        return _MatchOutcome(
            category=MATCH_REVERSED,
            bout=best.bout,
            selection_fighter_id=selection_fighter_id,
            reason=None,
        )
    if best.side.score >= 2:
        return _MatchOutcome(
            category=MATCH_EXACT,
            bout=best.bout,
            selection_fighter_id=selection_fighter_id,
            reason=None,
        )
    return _MatchOutcome(
        category=MATCH_NORMALIZED,
        bout=best.bout,
        selection_fighter_id=selection_fighter_id,
        reason=None,
    )


def _build_candidate_sides(
    *,
    fighter_ids: set[str],
    opponent_ids: set[str],
    fighter_exact_ids: set[str],
    opponent_exact_ids: set[str],
) -> tuple[_CandidateSide, ...]:
    rows: list[_CandidateSide] = []
    for fighter_id in fighter_ids:
        for opponent_id in opponent_ids:
            if fighter_id == opponent_id:
                continue
            score = 0
            if fighter_id in fighter_exact_ids:
                score += 1
            if opponent_id in opponent_exact_ids:
                score += 1
            rows.append(_CandidateSide(fighter_id=fighter_id, opponent_id=opponent_id, score=score))
    return tuple(rows)


def _event_delta_seconds(*, event_dt: datetime | None, bout: _BoutRow) -> float:
    if event_dt is None:
        return 0.0
    bout_event_dt = _parse_utc_timestamp(bout.event_date_utc)
    bout_start_dt = _parse_utc_timestamp(bout.bout_start_utc)
    if bout_event_dt is not None:
        return _absolute_seconds_delta(event_dt, bout_event_dt)
    return _absolute_seconds_delta(event_dt, bout_start_dt)


def _candidate_fighter_ids(*, name: str, alias_name_index: dict[str, set[str]]) -> set[str]:
    ids: set[str] = set()
    for alias in _fighter_name_aliases(name):
        ids.update(alias_name_index.get(alias, set()))
    return ids


def _fighter_name_aliases(name: str) -> set[str]:
    normalized = _normalize_person_name(name)
    if normalized == "":
        return set()
    tokens = normalized.split()
    aliases = {normalized}
    if len(tokens) >= 2:
        aliases.add(f"{tokens[0]} {tokens[-1]}")
    tokens_no_suffix = tuple(token for token in tokens if token not in {"jr", "sr", "ii", "iii", "iv", "v"})
    if len(tokens_no_suffix) >= 2:
        aliases.add(" ".join(tokens_no_suffix))
        aliases.add(f"{tokens_no_suffix[0]} {tokens_no_suffix[-1]}")
    if len(tokens) == 3 and len(tokens[1]) == 1:
        aliases.add(f"{tokens[0]} {tokens[2]}")
    return {alias for alias in aliases if alias != ""}


def _normalize_event_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    compact = "".join(ch if ch.isalnum() else " " for ch in ascii_text.lower())
    collapsed = " ".join(compact.split())
    if collapsed == "":
        return ""
    drop_tokens = {"ufc", "fight", "night", "on", "espn", "abc"}
    tokens = [token for token in collapsed.split() if token not in drop_tokens]
    return " ".join(tokens) if len(tokens) > 0 else collapsed


def _build_unmatched_sample(
    *,
    parsed: Mapping[str, Any],
    category: MatchCategory,
    reason: str,
) -> dict[str, str]:
    return {
        "category": category,
        "reason": reason,
        "source_event_id": str(parsed.get("source_event_id") or "n/a"),
        "sportsbook": str(parsed.get("sportsbook") or "n/a"),
        "event_date_utc": str(parsed.get("event_date_utc") or "n/a"),
        "market_timestamp_utc": str(parsed.get("market_timestamp_utc") or "n/a"),
        "fighter_name": str(parsed.get("fighter_name") or "n/a"),
        "opponent_name": str(parsed.get("opponent_name") or "n/a"),
    }


def _build_market_payload(
    *,
    source_system: str,
    source_record_id: str,
    sportsbook: str,
    market_type: str,
    selection_fighter_id: str,
    selection_label: str,
    market_timestamp_utc: str,
    bout_id: str,
    odds_american: int | None,
    odds_decimal: float | None,
    implied_probability: float | None,
    line_value: float | None,
    now_utc: str,
) -> dict[str, Any]:
    normalized_decimal = odds_decimal
    normalized_implied = implied_probability

    if normalized_decimal is None and odds_american is not None:
        normalized_decimal = _decimal_from_american(odds_american)
    if normalized_implied is None and normalized_decimal is not None and normalized_decimal > 0.0:
        normalized_implied = 1.0 / normalized_decimal

    if normalized_implied is not None:
        normalized_implied = max(0.0, min(1.0, normalized_implied))

    return {
        "bout_id": bout_id,
        "sportsbook": sportsbook,
        "market_type": market_type,
        "selection_fighter_id": selection_fighter_id,
        "selection_label": selection_label,
        "odds_american": odds_american,
        "odds_decimal": normalized_decimal,
        "implied_probability": normalized_implied,
        "line_value": line_value,
        "market_timestamp_utc": market_timestamp_utc,
        "source_system": source_system,
        "source_record_id": source_record_id,
        "source_updated_at_utc": market_timestamp_utc,
        "ingested_at_utc": now_utc,
        "created_at_utc": now_utc,
        "updated_at_utc": now_utc,
    }


def _insert_row(*, connection: sqlite3.Connection, row: NormalizedRow) -> None:
    columns = list(row.data.keys())
    placeholders = ", ".join("?" for _ in columns)
    sql = f"INSERT INTO markets ({', '.join(columns)}) VALUES ({placeholders})"
    values = [row.data[column] for column in columns]
    connection.execute(sql, values)


def _update_row_by_source_key(*, connection: sqlite3.Connection, row: NormalizedRow) -> None:
    update_columns = [
        column for column in row.data.keys() if column not in {"created_at_utc", "source_system", "source_record_id"}
    ]
    sql = f"UPDATE markets SET {', '.join(f'{column} = ?' for column in update_columns)} WHERE source_system = ? AND source_record_id = ?"
    values = [row.data[column] for column in update_columns]
    values.extend([row.source_system, row.source_record_id])
    connection.execute(sql, values)


def _build_outcome_source_record_id(
    *,
    event_id: str,
    bookmaker_key: str,
    market_key: str,
    selection_name: str,
    market_timestamp_utc: str,
) -> str:
    normalized_selection = _normalize_person_name(selection_name)
    return (
        f"event:{event_id}|book:{bookmaker_key}|market:{market_key}|"
        f"selection:{normalized_selection}|ts:{market_timestamp_utc}"
    )


def _normalize_person_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    compact = "".join(ch if ch.isalnum() else " " for ch in ascii_text.lower())
    return " ".join(compact.split())


def _parse_utc_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_utc_text(value: object) -> str | None:
    parsed = _parse_utc_timestamp(_as_optional_text(value))
    if parsed is None:
        return None
    return parsed.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _as_optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return text


def _as_optional_float(value: object) -> float | None:
    text = _as_optional_text(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _as_optional_int(value: object) -> int | None:
    text = _as_optional_text(value)
    if text is None:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _decimal_from_american(odds_american: int) -> float:
    if odds_american > 0:
        return 1.0 + (float(odds_american) / 100.0)
    if odds_american < 0:
        return 1.0 + (100.0 / abs(float(odds_american)))
    return 0.0


def _absolute_seconds_delta(left: datetime | None, right: datetime | None) -> float:
    if left is None or right is None:
        return float("inf")
    return abs((left - right).total_seconds())


def _stable_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _endpoint_token(url: str) -> str:
    compact = "".join(ch if ch.isalnum() else "_" for ch in url.lower())
    return compact[:96].strip("_") or "payload"


def _safe_json_loads(payload_text: str) -> object:
    try:
        return json.loads(payload_text)
    except json.JSONDecodeError:
        LOGGER.warning("odds_api.invalid_json_payload")
        return []


def _extract_events_from_api_payload(payload: object) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [event for event in payload if isinstance(event, Mapping)]
    if isinstance(payload, Mapping):
        data = payload.get("data")
        if isinstance(data, list):
            return [event for event in data if isinstance(event, Mapping)]
        events = payload.get("events")
        if isinstance(events, list):
            return [event for event in events if isinstance(event, Mapping)]
    return []


def _extract_payload_snapshot_timestamp(payload: object) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    return (
        _normalize_utc_text(payload.get("timestamp"))
        or _normalize_utc_text(payload.get("date"))
        or _normalize_utc_text(payload.get("snapshot_time"))
    )


def _event_name_from_teams(event: Mapping[str, Any]) -> str | None:
    home_team = _as_optional_text(event.get("home_team"))
    away_team = _as_optional_text(event.get("away_team"))
    if home_team is None or away_team is None:
        return None
    return f"{home_team} vs {away_team}"


def _iter_historical_snapshot_times(
    *,
    start_utc: datetime,
    end_utc: datetime,
    interval_hours: int,
) -> tuple[datetime, ...]:
    current = start_utc.astimezone(timezone.utc).replace(microsecond=0)
    end_bound = end_utc.astimezone(timezone.utc).replace(microsecond=0)
    step = timedelta(hours=max(1, interval_hours))
    rows: list[datetime] = []
    while current <= end_bound:
        rows.append(current)
        current = current + step
    if len(rows) == 0:
        rows.append(start_utc.astimezone(timezone.utc).replace(microsecond=0))
    return tuple(rows)


def _write_matching_audit_report(*, path: Path, audit: MatchAuditResult) -> None:
    unmatched_total = audit.ambiguous_rows + audit.no_match_rows
    lines = [
        "market_matching_audit_report",
        f"total_market_rows={audit.total_market_rows}",
        f"exact_matches={audit.exact_matches}",
        f"normalized_matches={audit.normalized_matches}",
        f"reversed_matches={audit.reversed_matches}",
        f"ambiguous_rows={audit.ambiguous_rows}",
        f"unmatched_rows={audit.no_match_rows}",
        "top_unmatched_reasons",
    ]
    if len(audit.top_unmatched_reasons) == 0:
        lines.append("reason=none count=0")
    else:
        for reason, count in audit.top_unmatched_reasons:
            lines.append(f"reason={reason} count={count}")

    lines.append("sample_unmatched_records")
    if unmatched_total == 0 or len(audit.sample_unmatched_records) == 0:
        lines.append("none")
    else:
        for index, row in enumerate(audit.sample_unmatched_records, start=1):
            lines.append(
                (
                    f"sample={index} category={row['category']} reason={row['reason']} "
                    f"source_event_id={row['source_event_id']} sportsbook={row['sportsbook']} "
                    f"event_date_utc={row['event_date_utc']} market_timestamp_utc={row['market_timestamp_utc']} "
                    f"fighter_name={row['fighter_name']} opponent_name={row['opponent_name']}"
                )
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _discover_archived_raw_json(*, raw_root: Path, source_system: str) -> tuple[Path, ...]:
    source_root = raw_root / source_system
    if not source_root.exists():
        return ()
    paths = [path for path in source_root.rglob("*.json") if path.is_file()]
    paths.sort()
    return tuple(paths)


def _parse_fetched_at_from_archive_path(*, archive_path: Path) -> datetime:
    parts = archive_path.parts
    # .../<source>/<YYYY>/<MM>/<DD>/<HHMMSS>_...
    try:
        year = int(parts[-4])
        month = int(parts[-3])
        day = int(parts[-2])
    except (IndexError, ValueError):
        return datetime.now(timezone.utc)

    hhmmss_token = archive_path.name.split("_", maxsplit=1)[0]
    hour = 0
    minute = 0
    second = 0
    if len(hhmmss_token) == 6 and hhmmss_token.isdigit():
        hour = int(hhmmss_token[0:2])
        minute = int(hhmmss_token[2:4])
        second = int(hhmmss_token[4:6])

    try:
        return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    except ValueError:
        return datetime(year, month, day, tzinfo=timezone.utc)
