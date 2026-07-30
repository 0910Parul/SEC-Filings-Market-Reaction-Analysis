"""
migrate_to_sqlite.py — Phase 1: Migrate CSVs to SQLite Database
Not Yet Priced In · Team 10

Reads master_scored.csv (2023) and backfill CSVs (2021-2022),
harmonizes columns, and writes to a single normalized SQLite database.

Usage:
    python src/migrate_to_sqlite.py
"""

import os
import sys
import sqlite3
import json
from pathlib import Path

import pandas as pd

# ── Setup paths ─────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import config

DB_PATH = config.DATA_DIR / "not_yet_priced_in.db"
log = config.setup_logging("migrate_db")

# ── Step 1: Load and harmonize all CSVs ─────────────────────────────────────
def load_all_data() -> pd.DataFrame:
    """Load 2023 master + backfill years, harmonize columns, return one DataFrame."""
    frames = []

    # 2023 master_scored.csv
    if config.MASTER_SCORED_CSV.exists():
        log.info(f"Loading 2023: {config.MASTER_SCORED_CSV}")
        df = pd.read_csv(config.MASTER_SCORED_CSV)
        df["year"] = 2023
        # Harmonize: master uses 'filename', backfill uses 'accession'
        if "filename" in df.columns and "accession" not in df.columns:
            df["accession"] = df["filename"].str.replace(".txt", "", regex=False)
        if "company_name" in df.columns and "company" not in df.columns:
            df.rename(columns={"company_name": "company"}, inplace=True)
        frames.append(df)
        log.info(f"  → {len(df)} rows")

    # Backfill years
    for year_dir in sorted(config.DATA_DIR.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        labelled = year_dir / "backfill_labelled.csv"
        checkpoint = year_dir / "backfill_checkpoint.csv"
        target = labelled if labelled.exists() else (checkpoint if checkpoint.exists() else None)
        if target:
            log.info(f"Loading {year_dir.name}: {target.name}")
            df = pd.read_csv(target)
            if "year" not in df.columns:
                df["year"] = int(year_dir.name)
            frames.append(df)
            log.info(f"  → {len(df)} rows")

    if not frames:
        log.error("No data found!")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    log.info(f"Combined: {len(combined)} total rows, years: {sorted(combined['year'].unique())}")
    return combined


# ── Step 2: Create SQLite schema ────────────────────────────────────────────
SCHEMA = """
-- Core filing metadata
CREATE TABLE IF NOT EXISTS filings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    accession       TEXT UNIQUE NOT NULL,
    ticker          TEXT NOT NULL,
    company         TEXT,
    filed_date      TEXT,
    year            INTEGER,
    primary_item    TEXT,
    broad_category  TEXT,
    clean_text      TEXT,
    parse_confidence REAL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Extracted features (one row per filing)
CREATE TABLE IF NOT EXISTS features (
    filing_id                INTEGER PRIMARY KEY REFERENCES filings(id),
    numeric_density          REAL,
    forward_looking_density  REAL,
    financial_symbol_density REAL,
    baseline_importance      REAL,
    importance_score         INTEGER,
    llm_event_category       TEXT,
    llm_confidence           REAL,
    grounding_rate           REAL,
    key_signals              TEXT,
    reasoning                TEXT
);

-- Market reaction labels (ground truth)
CREATE TABLE IF NOT EXISTS reactions (
    filing_id        INTEGER PRIMARY KEY REFERENCES filings(id),
    R_short_0_1      REAL,
    R_medium_0_3     REAL,
    R_long_5_20      REAL,
    reaction_class   TEXT,
    MOS_prospective  REAL,
    MOS_retrospective REAL
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_filings_ticker ON filings(ticker);
CREATE INDEX IF NOT EXISTS idx_filings_year ON filings(year);
CREATE INDEX IF NOT EXISTS idx_filings_category ON filings(broad_category);
CREATE INDEX IF NOT EXISTS idx_reactions_class ON reactions(reaction_class);
"""


def create_database(conn: sqlite3.Connection):
    """Create all tables."""
    conn.executescript(SCHEMA)
    conn.commit()
    log.info("Database schema created.")


# ── Step 3: Insert data ─────────────────────────────────────────────────────
def insert_data(conn: sqlite3.Connection, df: pd.DataFrame):
    """Insert harmonized DataFrame into normalized tables."""
    cursor = conn.cursor()

    inserted = 0
    skipped = 0

    for _, row in df.iterrows():
        accession = str(row.get("accession", ""))
        if not accession or accession == "nan":
            skipped += 1
            continue

        # Check for duplicate
        cursor.execute("SELECT id FROM filings WHERE accession = ?", (accession,))
        if cursor.fetchone():
            skipped += 1
            continue

        # --- Insert into filings ---
        clean_text = row.get("clean_text", None)
        if pd.isna(clean_text):
            clean_text = None

        cursor.execute("""
            INSERT INTO filings (accession, ticker, company, filed_date, year, 
                                 primary_item, broad_category, clean_text, parse_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            accession,
            row.get("ticker"),
            row.get("company", row.get("company_name")),
            str(row.get("filed_date", "")),
            int(row.get("year", 0)),
            row.get("primary_item"),
            row.get("broad_category"),
            clean_text,
            row.get("parse_confidence") if pd.notna(row.get("parse_confidence")) else None,
        ))

        filing_id = cursor.lastrowid

        # --- Insert into features ---
        cursor.execute("""
            INSERT INTO features (filing_id, numeric_density, forward_looking_density,
                                  financial_symbol_density, baseline_importance,
                                  importance_score, llm_event_category, llm_confidence,
                                  grounding_rate, key_signals, reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            filing_id,
            row.get("numeric_density") if pd.notna(row.get("numeric_density")) else None,
            row.get("forward_looking_density") if pd.notna(row.get("forward_looking_density")) else None,
            row.get("financial_symbol_density") if pd.notna(row.get("financial_symbol_density")) else None,
            row.get("baseline_importance") if pd.notna(row.get("baseline_importance")) else None,
            int(row.get("importance_score", 3)) if pd.notna(row.get("importance_score")) else None,
            row.get("llm_event_category"),
            row.get("llm_confidence") if pd.notna(row.get("llm_confidence")) else None,
            row.get("grounding_rate") if pd.notna(row.get("grounding_rate")) else None,
            row.get("key_signals"),
            row.get("reasoning"),
        ))

        # --- Insert into reactions ---
        cursor.execute("""
            INSERT INTO reactions (filing_id, R_short_0_1, R_medium_0_3, R_long_5_20,
                                   reaction_class, MOS_prospective, MOS_retrospective)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            filing_id,
            row.get("R_short_0_1") if pd.notna(row.get("R_short_0_1")) else None,
            row.get("R_medium_0_3") if pd.notna(row.get("R_medium_0_3")) else None,
            row.get("R_long_5_20") if pd.notna(row.get("R_long_5_20")) else None,
            row.get("reaction_class"),
            row.get("MOS_prospective") if pd.notna(row.get("MOS_prospective")) else None,
            row.get("MOS_retrospective") if pd.notna(row.get("MOS_retrospective")) else None,
        ))

        inserted += 1

    conn.commit()
    log.info(f"Inserted: {inserted} | Skipped (dupes/empty): {skipped}")


# ── Step 4: Verify ──────────────────────────────────────────────────────────
def verify_database(conn: sqlite3.Connection):
    """Print summary stats."""
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM filings")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT year, COUNT(*) FROM filings GROUP BY year ORDER BY year")
    by_year = cursor.fetchall()

    cursor.execute("SELECT reaction_class, COUNT(*) FROM reactions GROUP BY reaction_class ORDER BY COUNT(*) DESC")
    by_class = cursor.fetchall()

    cursor.execute("SELECT broad_category, COUNT(*) FROM filings GROUP BY broad_category ORDER BY COUNT(*) DESC LIMIT 10")
    by_cat = cursor.fetchall()

    log.info("")
    log.info("=" * 50)
    log.info("  DATABASE SUMMARY")
    log.info("=" * 50)
    log.info(f"  Total filings: {total}")
    log.info("")
    log.info("  By Year:")
    for year, count in by_year:
        log.info(f"    {year}: {count}")
    log.info("")
    log.info("  By Reaction Class:")
    for cls, count in by_class:
        pct = count / total * 100 if total > 0 else 0
        log.info(f"    {cls}: {count} ({pct:.1f}%)")
    log.info("")
    log.info("  Top Categories:")
    for cat, count in by_cat:
        log.info(f"    {cat}: {count}")
    log.info("")
    log.info(f"  Database saved → {DB_PATH}")
    log.info(f"  Size: {DB_PATH.stat().st_size / 1024 / 1024:.2f} MB")


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Phase 1: Migrating CSVs to SQLite...")

    # Load all data
    df = load_all_data()
    if df.empty:
        log.error("No data to migrate. Exiting.")
        sys.exit(1)

    # Create or connect to database
    conn = sqlite3.connect(DB_PATH)
    create_database(conn)

    # Insert data
    insert_data(conn, df)

    # Verify
    verify_database(conn)

    conn.close()
    log.info("Migration complete!")
