"""
embed_to_chroma.py — Phase 2: Create vector embeddings in ChromaDB
Not Yet Priced In · Team 10

Reads clean_text from SQLite, embeds using OpenAI text-embedding-3-small,
and stores in a persistent ChromaDB collection with metadata.

Usage:
    python src/embed_to_chroma.py
    python src/embed_to_chroma.py --reset   # wipe and re-embed everything
"""
import sys
import sqlite3
import argparse
import shutil
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import config

DB_PATH = config.DATA_DIR / "not_yet_priced_in.db"
CHROMA_DIR = config.DATA_DIR / "chroma_db"
COLLECTION_NAME = "sec_8k_filings"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 50  # OpenAI embedding API handles batches efficiently

log = config.setup_logging("embed_chroma")


def get_chroma_client():
    """Initialize persistent ChromaDB client."""
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client


def get_or_create_collection(client):
    """Get or create the filings collection with OpenAI embeddings."""
    from chromadb.utils import embedding_functions

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=config.OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL,
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def load_filings_from_db():
    """Load all filings with clean_text from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    df = conn.execute("""
        SELECT f.id, f.accession, f.ticker, f.company, f.filed_date,
               f.year, f.broad_category, f.primary_item,
               f.clean_text, f.clean_text_length,
               ft.importance_score, ft.llm_importance_norm,
               ft.numeric_density, ft.forward_looking_density,
               ft.grounding_rate, ft.llm_event_category,
               r.reaction_class, r.R_short_0_1, r.R_long_5_20,
               r.MOS_prospective
        FROM filings f
        JOIN features ft ON f.id = ft.filing_id
        JOIN reactions r ON f.id = r.filing_id
        WHERE f.clean_text IS NOT NULL AND f.clean_text != ''
    """).fetchall()

    columns = [
        "id", "accession", "ticker", "company", "filed_date",
        "year", "broad_category", "primary_item",
        "clean_text", "clean_text_length",
        "importance_score", "llm_importance_norm",
        "numeric_density", "forward_looking_density",
        "grounding_rate", "llm_event_category",
        "reaction_class", "R_short_0_1", "R_long_5_20",
        "MOS_prospective"
    ]
    conn.close()
    return df, columns


def embed_filings(collection, filings, columns, existing_ids):
    """Embed filings in batches."""
    # Filter out already-embedded filings
    new_filings = [f for f in filings if f[columns.index("accession")] not in existing_ids]
    log.info(f"New filings to embed: {len(new_filings)} (skipping {len(filings) - len(new_filings)} already embedded)")

    if not new_filings:
        log.info("Nothing new to embed!")
        return 0

    total_embedded = 0

    for i in range(0, len(new_filings), BATCH_SIZE):
        batch = new_filings[i:i + BATCH_SIZE]

        ids = []
        documents = []
        metadatas = []

        for row in batch:
            row_dict = dict(zip(columns, row))

            accession = str(row_dict["accession"])
            clean_text = str(row_dict["clean_text"])

            # Truncate very long texts (embedding model has token limits)
            if len(clean_text) > 30000:
                clean_text = clean_text[:30000]

            # Build metadata (ChromaDB only supports str, int, float, bool)
            metadata = {
                "ticker": str(row_dict["ticker"] or ""),
                "company": str(row_dict["company"] or ""),
                "filed_date": str(row_dict["filed_date"] or ""),
                "year": int(row_dict["year"]) if row_dict["year"] else 0,
                "broad_category": str(row_dict["broad_category"] or ""),
                "primary_item": str(row_dict["primary_item"] or ""),
                "importance_score": int(row_dict["importance_score"]) if row_dict["importance_score"] else 0,
                "reaction_class": str(row_dict["reaction_class"] or ""),
                "MOS_prospective": float(row_dict["MOS_prospective"]) if row_dict["MOS_prospective"] else 0.0,
                "numeric_density": float(row_dict["numeric_density"]) if row_dict["numeric_density"] else 0.0,
                "grounding_rate": float(row_dict["grounding_rate"]) if row_dict["grounding_rate"] else 0.0,
                "clean_text_length": int(row_dict["clean_text_length"]) if row_dict["clean_text_length"] else 0,
            }

            ids.append(accession)
            documents.append(clean_text)
            metadatas.append(metadata)

        # Upsert batch to ChromaDB (handles embedding via OpenAI automatically)
        try:
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            total_embedded += len(batch)
            log.info(f"  Batch {i // BATCH_SIZE + 1}: embedded {len(batch)} filings (total: {total_embedded})")
        except Exception as e:
            log.error(f"  Batch {i // BATCH_SIZE + 1} failed: {e}")
            time.sleep(5)  # Rate limit backoff

        # Small delay to avoid rate limits
        time.sleep(0.5)

    return total_embedded


def verify_collection(collection):
    """Print summary of the ChromaDB collection."""
    count = collection.count()

    log.info("")
    log.info("=" * 60)
    log.info("  CHROMADB COLLECTION SUMMARY")
    log.info("=" * 60)
    log.info(f"  Collection: {COLLECTION_NAME}")
    log.info(f"  Embedding model: {EMBEDDING_MODEL}")
    log.info(f"  Total vectors: {count}")
    log.info(f"  Storage: {CHROMA_DIR}")

    # Test query
    if count > 0:
        log.info("")
        log.info("  Test Query: 'company reports revenue increase and raises guidance'")
        results = collection.query(
            query_texts=["company reports revenue increase and raises guidance"],
            n_results=5,
        )
        log.info("  Top 5 Similar Filings:")
        for j, (doc_id, meta, dist) in enumerate(zip(
            results["ids"][0], results["metadatas"][0], results["distances"][0]
        )):
            log.info(f"    {j+1}. {meta['ticker']} {meta['filed_date']} "
                      f"({meta['broad_category']}) → {meta['reaction_class']} "
                      f"[similarity: {1-dist:.3f}]")

    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Wipe and re-embed everything")
    args = parser.parse_args()

    if not config.OPENAI_API_KEY:
        log.error("OPENAI_API_KEY not found! Set it in .env")
        sys.exit(1)

    # Reset if requested
    if args.reset and CHROMA_DIR.exists():
        log.warning("Resetting ChromaDB — deleting all embeddings...")
        shutil.rmtree(CHROMA_DIR)

    # Initialize ChromaDB
    log.info("Initializing ChromaDB...")
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    # Check existing
    existing_ids = set(collection.get()["ids"]) if collection.count() > 0 else set()
    log.info(f"Existing embeddings: {len(existing_ids)}")

    # Load filings from SQLite
    log.info("Loading filings from SQLite...")
    filings, columns = load_filings_from_db()
    log.info(f"Filings with clean_text: {len(filings)}")

    # Embed
    total = embed_filings(collection, filings, columns, existing_ids)
    log.info(f"Newly embedded: {total}")

    # Verify
    verify_collection(collection)


if __name__ == "__main__":
    main()
