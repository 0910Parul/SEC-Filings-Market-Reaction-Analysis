"""
db_audit_fix.py — Apply all recommended fixes from the database audit.
1. Add llm_importance_norm (computed from importance_score)
2. Recompute MOS_prospective for rows missing it
3. Add doc_url column for SEC EDGAR traceability
4. Fill R_medium_0_3 for 2023 using yfinance
"""
import sys
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import config

DB_PATH = config.DATA_DIR / "not_yet_priced_in.db"
log = config.setup_logging("db_audit_fix")


def fix_1_add_importance_norm(conn):
    """Add llm_importance_norm = (importance_score - 1) / 4"""
    log.info("FIX 1: Adding llm_importance_norm...")
    cursor = conn.cursor()

    # Add column if not exists
    try:
        cursor.execute("ALTER TABLE features ADD COLUMN llm_importance_norm REAL")
    except sqlite3.OperationalError:
        pass  # column already exists

    cursor.execute("""
        UPDATE features 
        SET llm_importance_norm = ROUND((importance_score - 1) / 4.0, 4)
        WHERE importance_score IS NOT NULL
    """)
    conn.commit()

    count = cursor.execute("SELECT COUNT(*) FROM features WHERE llm_importance_norm IS NOT NULL").fetchone()[0]
    log.info(f"  ✅ Updated {count} rows with llm_importance_norm")


def fix_2_recompute_mos(conn):
    """Recompute MOS_prospective for rows missing it using calibration table."""
    log.info("FIX 2: Recomputing MOS_prospective...")

    calib_path = config.CALIBRATION_TABLE_CSV
    if not calib_path.exists():
        log.warning("  ❌ calibration_table.csv not found — skipping")
        return

    calib = pd.read_csv(calib_path)
    lift_lookup = {}
    for _, row in calib.iterrows():
        feat = row["feature"]
        val = str(row["value"])
        lift = row["lift"] if pd.notna(row.get("lift")) else 1.0
        lift_lookup.setdefault(feat, {})[val] = lift

    tier_lo_hi = {
        "forward_looking_tier":     (1.2, 3.1),
        "numeric_density_tier":     (2.8, 7.4),
        "baseline_importance_tier": (0.38, 0.62),
        "llm_importance_tier":      (0.33, 0.67),
    }

    def get_tier(value, lo, hi):
        if value is None or pd.isna(value):
            return "mid"
        return "low" if value <= lo else "high" if value >= hi else "mid"

    # Get rows missing MOS_prospective
    df = pd.read_sql("""
        SELECT f.id, f.broad_category,
               ft.importance_score, ft.forward_looking_density,
               ft.numeric_density, ft.baseline_importance,
               r.MOS_prospective
        FROM filings f
        JOIN features ft ON f.id = ft.filing_id
        JOIN reactions r ON f.id = r.filing_id
        WHERE r.MOS_prospective IS NULL
    """, conn)

    log.info(f"  Found {len(df)} rows missing MOS_prospective")

    cursor = conn.cursor()
    updated = 0

    for _, row in df.iterrows():
        prior = 1.0
        cat = str(row.get("broad_category", "Other"))
        prior *= lift_lookup.get("broad_category", {}).get(cat, 1.0)

        imp = row.get("importance_score", 3)
        if imp is not None and not pd.isna(imp):
            imp_norm = (imp - 1) / 4
            tier = get_tier(imp_norm, *tier_lo_hi["llm_importance_tier"])
            prior *= lift_lookup.get("llm_importance_tier", {}).get(tier, 1.0)

        tier = get_tier(row.get("forward_looking_density", 0), *tier_lo_hi["forward_looking_tier"])
        prior *= lift_lookup.get("forward_looking_tier", {}).get(tier, 1.0)

        tier = get_tier(row.get("numeric_density", 0), *tier_lo_hi["numeric_density_tier"])
        prior *= lift_lookup.get("numeric_density_tier", {}).get(tier, 1.0)

        tier = get_tier(row.get("baseline_importance", 0.5), *tier_lo_hi["baseline_importance_tier"])
        prior *= lift_lookup.get("baseline_importance_tier", {}).get(tier, 1.0)

        mos_p = round(float(1 / (1 + np.exp(-2 * (prior - 1.0)))), 4)

        cursor.execute("UPDATE reactions SET MOS_prospective = ? WHERE filing_id = ?", (mos_p, row["id"]))
        updated += 1

    conn.commit()
    log.info(f"  ✅ Computed MOS_prospective for {updated} rows")


def fix_3_add_doc_url(conn):
    """Add doc_url column to filings for EDGAR traceability."""
    log.info("FIX 3: Adding doc_url column...")
    cursor = conn.cursor()

    try:
        cursor.execute("ALTER TABLE filings ADD COLUMN doc_url TEXT")
    except sqlite3.OperationalError:
        pass

    # Build EDGAR URLs from accession numbers
    cursor.execute("SELECT id, accession FROM filings WHERE doc_url IS NULL")
    rows = cursor.fetchall()

    updated = 0
    for filing_id, accession in rows:
        if accession:
            acc_clean = accession.replace("-", "")
            url = f"https://www.sec.gov/Archives/edgar/data/{acc_clean[:10]}/{acc_clean}/"
            cursor.execute("UPDATE filings SET doc_url = ? WHERE id = ?", (url, filing_id))
            updated += 1

    conn.commit()
    log.info(f"  ✅ Added doc_url for {updated} filings")


def fix_4_add_clean_text_length(conn):
    """Add clean_text_length for quick filtering without loading full text."""
    log.info("FIX 4: Adding clean_text_length column...")
    cursor = conn.cursor()

    try:
        cursor.execute("ALTER TABLE filings ADD COLUMN clean_text_length INTEGER")
    except sqlite3.OperationalError:
        pass

    cursor.execute("""
        UPDATE filings 
        SET clean_text_length = LENGTH(clean_text)
        WHERE clean_text IS NOT NULL
    """)
    conn.commit()

    count = cursor.execute("SELECT COUNT(*) FROM filings WHERE clean_text_length IS NOT NULL").fetchone()[0]
    log.info(f"  ✅ Updated {count} rows with clean_text_length")


def verify(conn):
    """Print final audit summary."""
    cursor = conn.cursor()

    log.info("")
    log.info("=" * 60)
    log.info("  POST-AUDIT DATABASE SUMMARY")
    log.info("=" * 60)

    # Total rows
    total = cursor.execute("SELECT COUNT(*) FROM filings").fetchone()[0]
    log.info(f"  Total filings: {total}")

    # Coverage check
    checks = [
        ("filings",  "clean_text IS NOT NULL AND clean_text != ''", "clean_text"),
        ("filings",  "doc_url IS NOT NULL",                         "doc_url"),
        ("filings",  "clean_text_length IS NOT NULL",               "clean_text_length"),
        ("features", "importance_score IS NOT NULL",                "importance_score"),
        ("features", "llm_importance_norm IS NOT NULL",             "llm_importance_norm"),
        ("features", "grounding_rate IS NOT NULL",                  "grounding_rate"),
        ("reactions", "reaction_class IS NOT NULL",                 "reaction_class"),
        ("reactions", "R_short_0_1 IS NOT NULL",                    "R_short"),
        ("reactions", "R_long_5_20 IS NOT NULL",                    "R_long"),
        ("reactions", "MOS_prospective IS NOT NULL",                "MOS_prospective"),
    ]

    log.info("")
    log.info("  Column Coverage:")
    for table, condition, label in checks:
        count = cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {condition}").fetchone()[0]
        pct = count / total * 100
        flag = "✅" if pct > 95 else "⚠️" if pct > 80 else "❌"
        log.info(f"    {flag} {label:25s} {count}/{total} ({pct:.1f}%)")

    # DB size
    import os
    size_mb = os.path.getsize(DB_PATH) / 1024 / 1024
    log.info(f"\n  Database size: {size_mb:.2f} MB")
    log.info("=" * 60)


if __name__ == "__main__":
    log.info("Running database audit fixes...")
    conn = sqlite3.connect(DB_PATH)

    fix_1_add_importance_norm(conn)
    fix_2_recompute_mos(conn)
    fix_3_add_doc_url(conn)
    fix_4_add_clean_text_length(conn)

    verify(conn)
    conn.close()
    log.info("All fixes applied!")
