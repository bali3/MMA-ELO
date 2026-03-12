"""SQLite schema definitions for the MMA database layer.

Design assumptions:
- all input rows must preserve source provenance fields
- bout-level facts are separate from model outputs
- no derived feature engineering tables are created at this stage
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TableDef:
    """Simple table DDL container used by schema creation code."""

    name: str
    ddl: str


FIGHTERS = TableDef(
    name="fighters",
    ddl="""
    CREATE TABLE IF NOT EXISTS fighters (
        fighter_id TEXT PRIMARY KEY,
        full_name TEXT NOT NULL,
        date_of_birth_utc TEXT,
        nationality TEXT,
        stance TEXT,
        height_cm REAL,
        reach_cm REAL,
        -- Source provenance fields for reproducibility and lineage audits.
        source_system TEXT NOT NULL,
        source_record_id TEXT NOT NULL,
        source_updated_at_utc TEXT,
        source_payload_sha256 TEXT,
        ingested_at_utc TEXT NOT NULL,
        created_at_utc TEXT NOT NULL,
        updated_at_utc TEXT NOT NULL,
        UNIQUE (source_system, source_record_id)
    );
    """.strip(),
)

EVENTS = TableDef(
    name="events",
    ddl="""
    CREATE TABLE IF NOT EXISTS events (
        event_id TEXT PRIMARY KEY,
        promotion TEXT NOT NULL,
        event_name TEXT NOT NULL,
        event_date_utc TEXT NOT NULL,
        venue TEXT,
        city TEXT,
        region TEXT,
        country TEXT,
        timezone_name TEXT,
        -- Source provenance fields for reproducibility and lineage audits.
        source_system TEXT NOT NULL,
        source_record_id TEXT NOT NULL,
        source_updated_at_utc TEXT,
        source_payload_sha256 TEXT,
        ingested_at_utc TEXT NOT NULL,
        created_at_utc TEXT NOT NULL,
        updated_at_utc TEXT NOT NULL,
        UNIQUE (source_system, source_record_id)
    );
    """.strip(),
)

BOUTS = TableDef(
    name="bouts",
    ddl="""
    CREATE TABLE IF NOT EXISTS bouts (
        bout_id TEXT PRIMARY KEY,
        event_id TEXT NOT NULL,
        bout_order INTEGER,
        fighter_red_id TEXT NOT NULL,
        fighter_blue_id TEXT NOT NULL,
        weight_class TEXT,
        gender TEXT,
        is_title_fight INTEGER NOT NULL DEFAULT 0 CHECK (is_title_fight IN (0, 1)),
        scheduled_rounds INTEGER,
        bout_start_time_utc TEXT,
        result_method TEXT,
        result_round INTEGER,
        result_time_seconds INTEGER,
        winner_fighter_id TEXT,
        -- Source provenance fields for reproducibility and lineage audits.
        source_system TEXT NOT NULL,
        source_record_id TEXT NOT NULL,
        source_updated_at_utc TEXT,
        source_payload_sha256 TEXT,
        ingested_at_utc TEXT NOT NULL,
        created_at_utc TEXT NOT NULL,
        updated_at_utc TEXT NOT NULL,
        FOREIGN KEY (event_id) REFERENCES events(event_id),
        FOREIGN KEY (fighter_red_id) REFERENCES fighters(fighter_id),
        FOREIGN KEY (fighter_blue_id) REFERENCES fighters(fighter_id),
        FOREIGN KEY (winner_fighter_id) REFERENCES fighters(fighter_id),
        UNIQUE (source_system, source_record_id),
        UNIQUE (event_id, bout_order),
        CHECK (fighter_red_id <> fighter_blue_id)
    );
    """.strip(),
)

FIGHTER_BOUT_STATS = TableDef(
    name="fighter_bout_stats",
    ddl="""
    CREATE TABLE IF NOT EXISTS fighter_bout_stats (
        fighter_bout_stats_id TEXT PRIMARY KEY,
        bout_id TEXT NOT NULL,
        fighter_id TEXT NOT NULL,
        opponent_fighter_id TEXT NOT NULL,
        corner TEXT CHECK (corner IN ('red', 'blue')),
        knockdowns INTEGER,
        sig_strikes_landed INTEGER,
        sig_strikes_attempted INTEGER,
        total_strikes_landed INTEGER,
        total_strikes_attempted INTEGER,
        takedowns_landed INTEGER,
        takedowns_attempted INTEGER,
        submission_attempts INTEGER,
        reversals INTEGER,
        control_time_seconds INTEGER,
        -- Source provenance fields for reproducibility and lineage audits.
        source_system TEXT NOT NULL,
        source_record_id TEXT NOT NULL,
        source_updated_at_utc TEXT,
        source_payload_sha256 TEXT,
        ingested_at_utc TEXT NOT NULL,
        created_at_utc TEXT NOT NULL,
        updated_at_utc TEXT NOT NULL,
        FOREIGN KEY (bout_id) REFERENCES bouts(bout_id),
        FOREIGN KEY (fighter_id) REFERENCES fighters(fighter_id),
        FOREIGN KEY (opponent_fighter_id) REFERENCES fighters(fighter_id),
        UNIQUE (source_system, source_record_id),
        UNIQUE (bout_id, fighter_id),
        CHECK (fighter_id <> opponent_fighter_id)
    );
    """.strip(),
)

MARKETS = TableDef(
    name="markets",
    ddl="""
    CREATE TABLE IF NOT EXISTS markets (
        market_id TEXT PRIMARY KEY,
        bout_id TEXT NOT NULL,
        sportsbook TEXT NOT NULL,
        market_type TEXT NOT NULL,
        selection_fighter_id TEXT,
        selection_label TEXT,
        odds_american INTEGER,
        odds_decimal REAL,
        implied_probability REAL CHECK (implied_probability >= 0.0 AND implied_probability <= 1.0),
        line_value REAL,
        market_timestamp_utc TEXT NOT NULL,
        -- Source provenance fields for reproducibility and lineage audits.
        source_system TEXT NOT NULL,
        source_record_id TEXT NOT NULL,
        source_updated_at_utc TEXT,
        source_payload_sha256 TEXT,
        ingested_at_utc TEXT NOT NULL,
        created_at_utc TEXT NOT NULL,
        updated_at_utc TEXT NOT NULL,
        FOREIGN KEY (bout_id) REFERENCES bouts(bout_id),
        FOREIGN KEY (selection_fighter_id) REFERENCES fighters(fighter_id),
        UNIQUE (source_system, source_record_id)
    );
    """.strip(),
)

MODEL_PREDICTIONS = TableDef(
    name="model_predictions",
    ddl="""
    CREATE TABLE IF NOT EXISTS model_predictions (
        model_prediction_id TEXT PRIMARY KEY,
        bout_id TEXT NOT NULL,
        target_fighter_id TEXT NOT NULL,
        model_name TEXT NOT NULL,
        model_version TEXT NOT NULL,
        model_run_id TEXT NOT NULL,
        prediction_type TEXT NOT NULL,
        predicted_probability REAL NOT NULL CHECK (
            predicted_probability >= 0.0 AND predicted_probability <= 1.0
        ),
        generated_at_utc TEXT NOT NULL,
        feature_cutoff_utc TEXT NOT NULL,
        -- Provenance for model artifact + feature snapshot used for this prediction.
        source_system TEXT NOT NULL,
        source_record_id TEXT NOT NULL,
        source_updated_at_utc TEXT,
        source_payload_sha256 TEXT,
        created_at_utc TEXT NOT NULL,
        updated_at_utc TEXT NOT NULL,
        FOREIGN KEY (bout_id) REFERENCES bouts(bout_id),
        FOREIGN KEY (target_fighter_id) REFERENCES fighters(fighter_id),
        UNIQUE (source_system, source_record_id),
        UNIQUE (model_run_id, bout_id, target_fighter_id, prediction_type)
    );
    """.strip(),
)

INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_bouts_event_id ON bouts(event_id);",
    "CREATE INDEX IF NOT EXISTS idx_bouts_start_time ON bouts(bout_start_time_utc);",
    "CREATE INDEX IF NOT EXISTS idx_fighter_bout_stats_bout_id ON fighter_bout_stats(bout_id);",
    "CREATE INDEX IF NOT EXISTS idx_markets_bout_timestamp ON markets(bout_id, market_timestamp_utc);",
    "CREATE INDEX IF NOT EXISTS idx_model_predictions_lookup "
    "ON model_predictions(model_name, model_version, generated_at_utc);",
)

TABLES = (FIGHTERS, EVENTS, BOUTS, FIGHTER_BOUT_STATS, MARKETS, MODEL_PREDICTIONS)
REQUIRED_TABLE_NAMES = tuple(table.name for table in TABLES)

SCHEMA_DDL_STATEMENTS = tuple(table.ddl for table in TABLES) + INDEX_DDL
SCHEMA_DDL = "\n\n".join(SCHEMA_DDL_STATEMENTS)
