"""Basic runtime configuration.

Assumption: simple environment-variable based settings are enough for scaffold stage.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Application settings."""

    environment: str
    log_level: str
    database_url: str
    timezone: str
    odds_api_key: str | None
    odds_api_base_url: str
    project_root: Path


def load_settings() -> Settings:
    """Load settings from environment variables with conservative defaults."""

    return Settings(
        environment=os.getenv("MMA_ELO_ENV", "dev"),
        log_level=os.getenv("MMA_ELO_LOG_LEVEL", "INFO"),
        # Assumption: local SQLite remains the default until storage requirements expand.
        database_url=os.getenv("MMA_ELO_DATABASE_URL", "sqlite:///data/processed/mma_elo.db"),
        # Assumption: UTC is the canonical timeline for chronology-safe feature generation.
        timezone=os.getenv("MMA_ELO_TIMEZONE", "UTC"),
        odds_api_key=os.getenv("MMA_ELO_ODDS_API_KEY"),
        odds_api_base_url=os.getenv("MMA_ELO_ODDS_API_BASE_URL", "https://api.the-odds-api.com/v4"),
        project_root=Path.cwd(),
    )
