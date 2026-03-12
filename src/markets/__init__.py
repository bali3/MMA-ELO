"""Betting market ingestion, matching, and coverage reporting modules."""

from markets.reporting import MarketCoverageReportResult, run_market_coverage_report

__all__ = [
    "MarketCoverageReportResult",
    "run_market_coverage_report",
]
