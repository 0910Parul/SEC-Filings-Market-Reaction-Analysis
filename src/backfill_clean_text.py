"""
backfill_clean_text.py — Populate missing clean_text in SQLite from .txt files on disk.
Reads raw 8-K text files, parses them, and updates the filings table.
"""
import sys
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import config

# Import parser
sys.path.insert(0, str(config.NOTEBOOKS_DIR))
from parse_filing import parse_filing

DB_PATH = config.DATA_DIR / "not_yet_priced_in.db"
log = config.setup_logging("backfill_text")

def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get all filings missing clean_text
    cursor.execute("""
        SELECT id, accession, year, ticker
        FROM filings
        WHERE clean_text IS NULL OR clean_text = ''
    """)
    missing = cursor.fetchall()
    log.info(f"Found {len(missing)} filings missing clean_text")

    updated = 0
    not_found = 0

    for filing_id, accession, year, ticker in missing:
        # Build filename: TICKER_ACCESSION_WITH_UNDERSCORES.txt
        acc_underscored = accession.replace("-", "_")
        filename = f"{ticker}_{acc_underscored}.txt"
        txt_path = config.DATA_DIR / str(year) / filename

        if not txt_path.exists():
            not_found += 1
            continue

        # Read and parse
        try:
            raw_text = txt_path.read_text(encoding="utf-8", errors="replace")
            parsed = parse_filing(raw_text)
            clean_text = parsed.get("clean_text", "")

            if clean_text and len(clean_text) > 10:
                cursor.execute(
                    "UPDATE filings SET clean_text = ? WHERE id = ?",
                    (clean_text, filing_id)
                )
                updated += 1
                if updated % 50 == 0:
                    conn.commit()
                    log.info(f"  Progress: {updated} updated...")
        except Exception as e:
            log.warning(f"  Parse error for {filename}: {e}")

    conn.commit()

    # Final stats
    cursor.execute("SELECT COUNT(*) FROM filings WHERE clean_text IS NOT NULL AND clean_text != ''")
    total_with_text = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM filings")
    total = cursor.fetchone()[0]

    log.info("")
    log.info("=" * 50)
    log.info(f"  Updated:   {updated}")
    log.info(f"  Not found: {not_found}")
    log.info(f"  Coverage:  {total_with_text}/{total} ({total_with_text/total*100:.1f}%)")
    log.info("=" * 50)

    conn.close()

if __name__ == "__main__":
    main()
