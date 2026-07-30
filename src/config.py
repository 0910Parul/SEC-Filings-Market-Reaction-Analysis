import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from .env if it exists
load_dotenv()

# --- PROJECT ROOT ---
REPO_ROOT = Path(__file__).resolve().parent.parent

# --- FOLDER PATHS ---
DATA_DIR = Path(os.getenv("DATA_DIR", REPO_ROOT / "data"))
FILINGS_DIR = Path(os.getenv("FILINGS_DIR", DATA_DIR / "2023" / "2023"))
LOG_DIR = REPO_ROOT / "logs"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
SRC_DIR = REPO_ROOT / "src"

# --- DATA FILES ---
MASTER_SCORED_CSV = DATA_DIR / "master_scored.csv"
WATCHLIST_FINAL_CSV = DATA_DIR / "watchlist_final.csv"
CALIBRATION_TABLE_CSV = DATA_DIR / "calibration_table.csv"
ALL_FILINGS_ANALYSIS_CSV = DATA_DIR / "all_filings_analysis.csv"

# --- API KEYS ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# --- UI SETTINGS ---
SHOW_CALIBRATION = True #False  # Set to False to hide the calibration tab in the dashboard

# --- ENSURE DIRECTORIES EXIST ---
try:
    for folder in [DATA_DIR, FILINGS_DIR, LOG_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"CRITICAL: Failed to create directories: {e}")

# --- LOGGING SETUP ---
def setup_logging(name=__name__, level=logging.INFO):
    """Initializes and returns a logger with consistent formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if setup_logging is called multiple times
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console Handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File Handler
        fh = logging.FileHandler(LOG_DIR / "app.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

def get_config_summary():
    """Returns a string summary of loaded configuration for debugging."""
    return f"""
    --- Project Config ---
    Root: {REPO_ROOT}
    Data: {DATA_DIR}
    OpenAI Key: {"Set" if OPENAI_API_KEY else "Missing"}
    Anthropic Key: {"Set" if ANTHROPIC_API_KEY else "Missing"}
    ----------------------
    """

if __name__ == "__main__":
    print(get_config_summary())
