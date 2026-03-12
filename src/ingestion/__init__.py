"""Ingestion domain modules and typed source adapter contracts."""

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

__all__ = [
    "Fetcher",
    "Parser",
    "Normalizer",
    "Persister",
    "RawIngestionRecord",
    "ParsedIngestionRecord",
    "NormalizedRow",
    "PersistResult",
]
