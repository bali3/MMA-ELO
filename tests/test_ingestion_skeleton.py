"""Tests for ingestion module imports and baseline protocol contracts."""

from __future__ import annotations

from pathlib import Path

from ingestion.contracts import Fetcher, Normalizer, Parser, Persister, PersistResult
from ingestion.sources.odds import OddsFetcher, OddsNormalizer, OddsParser, OddsPersister
from ingestion.sources.ufc_stats import (
    UFCStatsBoutStatsFetcher,
    UFCStatsBoutStatsNormalizer,
    UFCStatsBoutStatsParser,
    UFCStatsFetcher,
    UFCStatsFighterMetadataFetcher,
    UFCStatsFighterMetadataNormalizer,
    UFCStatsFighterMetadataParser,
    UFCStatsNormalizer,
    UFCStatsParser,
    UFCStatsPersister,
)


def test_source_modules_import_and_expose_classes() -> None:
    """Source adapter modules should be importable and constructible."""

    assert OddsFetcher is not None
    assert OddsParser is not None
    assert OddsNormalizer is not None
    assert OddsPersister is not None

    assert UFCStatsFetcher is not None
    assert UFCStatsParser is not None
    assert UFCStatsNormalizer is not None
    assert UFCStatsPersister is not None
    assert UFCStatsFighterMetadataFetcher is not None
    assert UFCStatsFighterMetadataParser is not None
    assert UFCStatsFighterMetadataNormalizer is not None
    assert UFCStatsBoutStatsFetcher is not None
    assert UFCStatsBoutStatsParser is not None
    assert UFCStatsBoutStatsNormalizer is not None


def test_source_adapters_satisfy_typed_protocols() -> None:
    """Adapters should satisfy runtime-checkable ingestion protocols."""

    odds_fetcher = OddsFetcher()
    odds_parser = OddsParser()
    odds_normalizer = OddsNormalizer()
    odds_persister = OddsPersister()

    ufc_fetcher = UFCStatsFetcher(raw_root=Path("data/raw"))
    ufc_parser = UFCStatsParser()
    ufc_normalizer = UFCStatsNormalizer()
    ufc_persister = UFCStatsPersister(db_path=Path("data/processed/test.sqlite3"))
    ufc_fighter_fetcher = UFCStatsFighterMetadataFetcher(raw_root=Path("data/raw"))
    ufc_fighter_parser = UFCStatsFighterMetadataParser()
    ufc_fighter_normalizer = UFCStatsFighterMetadataNormalizer()
    ufc_bout_stats_fetcher = UFCStatsBoutStatsFetcher(raw_root=Path("data/raw"))
    ufc_bout_stats_parser = UFCStatsBoutStatsParser()
    ufc_bout_stats_normalizer = UFCStatsBoutStatsNormalizer()

    for fetcher in (odds_fetcher, ufc_fetcher, ufc_fighter_fetcher, ufc_bout_stats_fetcher):
        assert isinstance(fetcher, Fetcher)
    for parser in (odds_parser, ufc_parser, ufc_fighter_parser, ufc_bout_stats_parser):
        assert isinstance(parser, Parser)
    for normalizer in (odds_normalizer, ufc_normalizer, ufc_fighter_normalizer, ufc_bout_stats_normalizer):
        assert isinstance(normalizer, Normalizer)
    for persister in (odds_persister, ufc_persister):
        assert isinstance(persister, Persister)


def test_odds_placeholders_remain_noop_and_idempotent() -> None:
    """Odds placeholder adapters are intentionally no-op in this stage."""

    odds_records = OddsFetcher().fetch()
    assert odds_records == ()

    assert OddsParser().parse(odds_records) == ()
    assert OddsNormalizer().normalize(()) == ()

    odds_result = OddsPersister().persist(())
    assert odds_result == PersistResult(inserted_count=0, updated_count=0, skipped_count=0)
