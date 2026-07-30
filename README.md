# Not Yet Priced In
### An AI Copilot for Detecting Delayed Market Reactions to Corporate Disclosures

**Parul Chaudhary · Big Data Analytics · Carlson MSBA**

> [!NOTE]
> This project repository is created in partial fulfillment of the requirements for the **Big Data Analytics** course offered by the **Master of Science in Business Analytics** program at the **Carlson School of Management, University of Minnesota**.

---

## 📄 Project Materials
- **[Live Interactive Dashboard](https://not-yet-priced-in-rn9nsszrjvndpz88nuuen3.streamlit.app/)**
- **[Project Flyer (PDF)](assets/flier.pdf)**
- **[Raw Data Filings (Google Drive)](https://drive.google.com/drive/folders/1hqc-C8oFAVq0fYVoFeOYDbTyNhWEKPoG?usp=sharing)**
- **[Dataset (Master Scored CSV)](data/master_scored.csv)**
- **[Watchlist (Final Rankings)](data/watchlist_final.csv)**
- **[Data Explorer Notebook](notebooks/DB_Explorer.ipynb)**

---

## What this project does

Public companies are required by the SEC to file an 8-K report whenever something material happens — a new CEO, an earnings update, a major acquisition, a debt restructuring. The market is supposed to absorb this information and adjust prices immediately. Often it doesn't.

This project builds a system that identifies **delayed market reactions**: cases where a filing contained important information, but the stock barely moved on day 0–1, then drifted significantly in the same direction over the following weeks. We call these "missed opportunities" — moments where the information was public but not yet priced in.

The system produces a **Missed Opportunity Score (MOS)** for every filing, ranks them, and generates a watchlist of the most interesting cases for analyst review. It includes a **production Streamlit dashboard**, a **SQLite + ChromaDB** data platform, and an **ML prediction model**.

---

## Key design choices (and why)

**Three-layer importance scoring.** The LLM is not the only judge. Every filing gets a rule-based category (A1), a text-feature importance score (A2), and an LLM importance score (A3). They cross-check each other.

**Verbatim grounding.** The LLM is required to return its `key_signals` as exact quotes from the filing text. Each quote is programmatically verified to appear in the source. This is the anti-hallucination mechanism.

**Quantile-based thresholds.** Reaction class labels (Immediate / Delayed / Gradual / No Reaction) use thresholds derived from the empirical distribution — not values picked from thin air.

**Vector similarity as a feature.** ChromaDB embeddings enable "find me filings that look like this one" — the RAG-derived `similar_delayed_pct` feature turned out to be a top signal in ML training.

---

## Data

| Dataset | Source | Coverage |
|---|---|---|
| 8-K filings | SEC EDGAR | 942 filings, 50 companies, 2021–2023 |
| Stock prices | Yahoo Finance | 50 tickers + S&P 500 |
| Benchmark | S&P 500 (^GSPC) | Used to compute abnormal returns |
| Vector embeddings | OpenAI text-embedding-3-small | 925 filings embedded in ChromaDB |

**50 companies across 8 sectors:** Technology (AAPL, MSFT, GOOGL, AMZN, META, NVDA, ADBE, CRM), Finance (JPM, BAC, GS, MS, WFC, BLK), Healthcare (JNJ, PFE, MRK, ABBV, UNH, LLY), Consumer (WMT, COST, KO, PEP, NKE, SBUX), Industrials (BA, GE, HON, CAT, UNP), Energy (XOM, CVX, COP, SLB), Media (DIS, NFLX, CMCSA, T, VZ), Materials/Other (MMM, DOW, DD, FCX, TSLA, INTC, ORCL, QCOM, AXP, CVS).

---

## Architecture

```
INPUT: 942 8-K filings  +  50 stocks' daily prices (2021-2023)
         ↓
┌─────────────────────────────────────────┐
│  Stage A — Event Classification         │
│  A0: Parse & clean raw .txt files       │
│  A1: Rule-based SEC Item → category     │
│  A2: Text features (3 signals)          │
│  A3: OpenAI GPT-4o importance scoring   │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  Stage B — Market Reaction              │
│  compute_event_windows() per filing     │
│  4-class labels via quantile thresholds │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  Stage C — Scoring                      │
│  C1: MOS_retrospective → watchlist      │
│  C2: Calibration table (lift ratios)    │
│  C3: MOS_prospective → new filings      │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  Data Platform                          │
│  SQLite (structured) + ChromaDB (RAG)   │
│  ML Model (RF + GradientBoosting)       │
│  Streamlit Dashboard (Live Analyzer)    │
└─────────────────────────────────────────┘
```

---

## Repository structure

```
not-yet-priced-in/
│
├── data/
│   ├── 2021/                        # Backfilled 8-K text files (52 filings)
│   ├── 2022/                        # Backfilled 8-K text files (469 filings)
│   ├── not_yet_priced_in.db         # SQLite: filings + features + reactions
│   ├── chroma_db/                   # ChromaDB vector store (925 embeddings)
│   ├── delayed_reaction_model.pkl   # Trained ML model
│   ├── model_results.json           # ML evaluation metrics
│   ├── calibration_table.csv        # C2 lift ratios with bootstrap CIs
│   ├── watchlist_final.csv          # Top-ranked filings for review
│   └── master_scored.csv            # 2023 data — all stages merged
│
├── src/
│   ├── config.py                    # Centralized configuration & logging
│   ├── backend.py                   # Core pipeline logic (parse, features, LLM)
│   ├── dashboard.py                 # Streamlit app (6 tabs)
│   ├── backfill.py                  # Historical 8-K batch processor (EDGAR → OpenAI)
│   ├── migrate_to_sqlite.py         # Phase 1: CSV → SQLite migration
│   ├── embed_to_chroma.py           # Phase 2: SQLite → ChromaDB embeddings
│   ├── train_model.py               # Phase 3: ML model training
│   ├── backfill_clean_text.py       # Utility: populate clean_text from .txt files
│   └── db_audit_fix.py              # Database audit & gap fixes
│
├── notebooks/
│   ├── NB1_2_8k_Data_Ingestion.ipynb
│   ├── NB1_1_Market_Price_Data_Ingestion.ipynb
│   ├── NB2_Event_Classification.ipynb
│   ├── delayed_reaction_analysis.ipynb
│   ├── NB3_Scoring_Watchlist.ipynb
│   ├── DB_Explorer.ipynb            # Query & inspect the SQLite database
│   └── parse_filing.py              # Core 8-K parser (Stage A0)
│
├── tests/
│   └── test_backfill.py             # Backfill pipeline tests
│
├── logs/
│   └── app.log                      # Centralized application logs
│
├── OPERATING_GUIDE.md               # How to run, stop, and debug the dashboard
├── LAUNCH_DASHBOARD.ipynb           # One-click notebook to launch the app
├── requirements.txt
├── .env                             # API keys (not in git)
└── README.md
```

---

## Dashboard

The Streamlit dashboard (`src/dashboard.py`) provides 6 interactive tabs:

| Tab | Purpose |
|-----|---------|
| 📊 **Overview** | Distribution of reaction classes, summary metrics |
| 🔬 **Scatter Analysis** | Interactive scatter plots of features vs returns |
| 🏆 **Watchlist** | Top-ranked delayed reaction filings |
| 🧬 **Live Analyzer** | Paste any 8-K text → real-time GPT-4o scoring |
| ⚙️ **Backfill** | Start/stop historical data processing from the UI |
| ⚖️ **Calibration** | Lift ratio tables (toggle-able via config) |

**Run it:**
```bash
./.venv/bin/streamlit run src/dashboard.py
```

---

## Data Platform

### SQLite Database (`data/not_yet_priced_in.db`)
Three normalized tables with 942 filings across 2021-2023:
- **`filings`** — Metadata, clean_text, EDGAR URLs
- **`features`** — Structural + LLM scoring (numeric density, importance, grounding rate)
- **`reactions`** — Market returns, reaction class labels, MOS scores

### ChromaDB Vector Store (`data/chroma_db/`)
925 filing embeddings using OpenAI `text-embedding-3-small`. Enables:
- Similarity search ("find filings like this one")
- RAG features for ML (% of similar filings that were delayed)

---

## ML Model

**Architecture:** Ensemble of RandomForest + GradientBoosting (soft voting)
**Train:** 2021-2022 (441 filings) | **Test:** 2023 (456 filings)

### Top Features by Importance
| Feature | Importance | Source |
|---------|-----------|--------|
| similar_avg_mos | 31.4% | ChromaDB RAG ⭐ |
| numeric_density | 17.4% | Rule-based (A2) |
| baseline_importance | 17.2% | Computed (A2) |
| similar_avg_importance | 7.5% | ChromaDB RAG ⭐ |
| forward_looking_density | 6.1% | Rule-based (A2) |
| MOS_prospective | 5.2% | Calibration (C3) |
| similar_delayed_pct | 4.7% | ChromaDB RAG ⭐ |

**Key finding:** ChromaDB RAG features (⭐) account for **43.6%** of total feature importance — validating the vector similarity approach.

**Current challenge:** Class imbalance (only 7.9% of filings are "Delayed"). The model achieves 91.7% accuracy by predicting "Not Delayed" for everything. Next steps: SMOTE oversampling, threshold tuning, and completing the 2021 backfill for more training data.

---

## 📚 Bibliography & Credits

### Data Sources
- **SEC EDGAR**: Primary source for 8-K filings and corporate disclosures.
- **Yahoo Finance (yfinance)**: Source for historical stock price data and S&P 500 benchmarks.
- **OpenAI**: Used for text embeddings (`text-embedding-3-small`) and event importance scoring (`gpt-4o`).

### References & Tools
- **Streamlit**: Dashboard framework for interactive visualization.
- **SQLite**: Relational database for structured filing and reaction data.
- **ChromaDB**: Vector database for similarity-based feature engineering.
- **Scikit-Learn**: Machine learning library used for the Random Forest and Gradient Boosting models.

---

## 🚀 How to Run (Reuse Instructions)

### 1. Setup & Installation
Requires Python 3.9+.

```bash
# Clone the repository
git clone <repo-url>
cd not-yet-priced-in

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file in the root directory (refer to `.env.example`):
```bash
echo "OPENAI_API_KEY=your_key_here" > .env
```

### 3. Launching the Dashboard
The easiest way to interact with the project is via the Streamlit dashboard:
```bash
./.venv/bin/streamlit run src/dashboard.py
```

### 4. Running the Pipeline (Optional)
To process new data or rebuild the models:
```bash
# Run historical backfill
./.venv/bin/python src/backfill.py --years 2022

# Rebuild SQLite database & embeddings
./.venv/bin/python src/migrate_to_sqlite.py
./.venv/bin/python src/embed_to_chroma.py

# Re-train ML models
./.venv/bin/python src/train_model.py
```

### 5. Debugging & Troubleshooting
If you encounter issues while running the dashboard:
- **ModuleNotFoundError**: Ensure you are using the virtual environment's Streamlit: `./.venv/bin/streamlit run src/dashboard.py`.
- **API Key Error**: Verify that your `.env` file is in the root directory and contains a valid `OPENAI_API_KEY`.
- **Data Not Found**: Ensure `data/not_yet_priced_in.db` and the CSV files are present.
- **Logs**: Detailed application logs (Parsing, LLM calls, Errors) are recorded in `logs/app.log`.

---

## Framing for judges

**"This is decision support, not prediction."**
MOS_prospective is a prior probability. A score of 0.80 means this filing's feature profile matches the top 20% of historical Delayed-reaction filings. An analyst still makes the final call.

**"The LLM is grounded, not hallucinating."**
Every `key_signals` entry is programmatically verified against the source text. Grounding rate is reported per filing.

**"Vector similarity validates the hypothesis."**
ChromaDB RAG features account for 43.6% of ML feature importance — filings that *look like* past delayed-reaction filings are more likely to be delayed themselves.

---

## Author

**Parul Chaudhary** · [LinkedIn](https://www.linkedin.com/in/parulchaud) · [Email](mailto:parul.jaswant@gmail.com)

Built as part of the Big Data Analytics course, M.S. in Business Analytics, Carlson School of Management, University of Minnesota.

---

## Disclosure

*This project uses AI-assisted tools including the OpenAI API for text analysis and embeddings. All LLM outputs are grounded in verbatim source text and cross-validated with rule-based and human judgments. The system is intended for research and educational purposes only and does not constitute financial advice.*

