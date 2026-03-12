"""Typed ingestion contracts for source adapters and persistence layers.

These interfaces define a strict ingestion boundary:
- fetch raw source records
- parse source-specific payloads
- normalize records into canonical row payloads
- persist normalized rows with idempotency guarantees
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Protocol, Sequence, Union, runtime_checkable


JSONValue = Union[
    str,
    int,
    float,
    bool,
    None,
    Mapping[str, Any],
    Sequence[Any],
]


@dataclass(frozen=True)
class RawIngestionRecord:
    """Raw record captured from a source system before transformation."""

    source_system: str
    source_record_id: str
    fetched_at_utc: datetime
    payload: Mapping[str, JSONValue]
    payload_sha256: str
    idempotency_key: str


@dataclass(frozen=True)
class ParsedIngestionRecord:
    """Source-shaped parsed record with provenance preserved."""

    source_system: str
    source_record_id: str
    parsed_payload: Mapping[str, JSONValue]
    payload_sha256: str
    idempotency_key: str


@dataclass(frozen=True)
class NormalizedRow:
    """Canonical row payload targeting a specific destination table."""

    table_name: str
    primary_key: str
    data: Mapping[str, JSONValue]
    source_system: str
    source_record_id: str
    payload_sha256: str
    idempotency_key: str


@dataclass(frozen=True)
class PersistResult:
    """Persistence summary for idempotent upsert-style ingestion writes."""

    inserted_count: int
    updated_count: int
    skipped_count: int


@runtime_checkable
class Fetcher(Protocol):
    """Fetches source records without applying transformation logic."""

    def fetch(self, *, as_of_utc: datetime | None = None) -> Sequence[RawIngestionRecord]:
        """Return raw records available up to ``as_of_utc``."""


@runtime_checkable
class Parser(Protocol):
    """Parses raw records into source-specific structured payloads."""

    def parse(self, records: Sequence[RawIngestionRecord]) -> Sequence[ParsedIngestionRecord]:
        """Convert raw payloads into parsed records while preserving provenance."""


@runtime_checkable
class Normalizer(Protocol):
    """Normalizes parsed records into canonical destination rows."""

    def normalize(self, records: Sequence[ParsedIngestionRecord]) -> Sequence[NormalizedRow]:
        """Map parsed payloads into target-table row payloads."""


@runtime_checkable
class Persister(Protocol):
    """Persists normalized rows using idempotent semantics."""

    def persist(self, rows: Sequence[NormalizedRow]) -> PersistResult:
        """Write rows and return insert/update/skip counts."""
