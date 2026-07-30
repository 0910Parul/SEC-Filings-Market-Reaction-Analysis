import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import backfill, config

def test_llm_scoring():
    print("Testing OpenAI Scoring in Backfill...")
    test_text = """
    Item 2.02 Results of Operations and Financial Condition.
    CVS Health Corporation reported total revenues of $89.3 billion for the three months ended March 31, 2024, 
    an increase of 3.7% compared to the prior year. The company revised its full-year 2024 GAAP EPS guidance range 
    to $5.64 to $5.94 from at least $7.06.
    """
    
    result = backfill.llm_score(test_text)
    
    print("\n--- LLM Result ---")
    if result.get("llm_error"):
        print(f"FAILED: {result['llm_error']}")
    else:
        print(f"Event Category: {result['llm_event_category']}")
        print(f"Importance Score: {result['importance_score']}")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Grounding Rate: {result['grounding_rate']}")
        print("SUCCESS")

def test_edgar_fetch():
    print("\nTesting EDGAR Fetch (AAPL, 2023)...")
    filings = backfill.fetch_edgar_filings("AAPL", "0000320193", 2023)
    print(f"Found {len(filings)} filings.")
    if filings:
        print(f"First filing: {filings[0]['accession']} on {filings[0]['filed_date']}")

if __name__ == "__main__":
    if not config.OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not found in environment or .env file.")
    else:
        test_llm_scoring()
        test_edgar_fetch()
