# 📈 Not Yet Priced In — Detecting Delayed Market Reactions to SEC Filings

**When a company files an 8-K, does the market react right away — or is there a signal that's *not yet priced in*?**
An end-to-end data pipeline that ingests SEC 8-K filings and daily market data, classifies each filing event, measures the market's post-event reaction, and produces a ranked **watchlist** of names where the price may not yet reflect the news. Built for the MSBA program at the University of Minnesota (Carlson School of Management).

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10-blue" />
  <img src="https://img.shields.io/badge/pandas-Data-150458" />
  <img src="https://img.shields.io/badge/yfinance-Market%20Data-4b8bbe" />
  <img src="https://img.shields.io/badge/SEC%20EDGAR-8--K%20Filings-lightgrey" />
</p>

---

## 📌 The Idea

Markets are supposed to price in news instantly — but not all 8-K events are equal, and some reactions lag. This project builds a repeatable pipeline to (1) pull the filings and the prices, (2) classify what each filing is actually about, (3) measure how the stock moved in the days after, and (4) surface events where the reaction looks incomplete — a candidate watchlist.

---

## 🔧 Pipeline

| Notebook | What it does |
|---|---|
| `NB1.1_Market_Price_Data_Ingestion.ipynb` | Downloads daily returns for ~50 tickers via `yfinance` and computes **abnormal returns** against the S&P 500 benchmark. |
| `NB1.2_8k_Data_Ingestion.ipynb` | Pulls 8-K filing metadata from **SEC EDGAR**, downloads raw filing text, and extracts clean text from HTML. |
| `NB2_Event_Classification.ipynb` | Classifies each filing event using **rule-based + LLM** approaches. |
| `NB3_Market_Reaction.ipynb` | Calculates **post-event return windows** to quantify the market's reaction. |
| `NB4_Watchlist_Output.ipynb` | **Scores and ranks** events into a final watchlist. |

The working pipeline lives in the [`notebooks/`](notebooks). [`src/`](src) is a lightweight module scaffold set up for refactoring that notebook logic into reusable functions.

---

## 🗂️ Repository Structure

```
not-yet-priced-in/
├── README.md
├── requirements.txt
├── notebooks/          # NB1.1 → NB4 pipeline
└── src/                # edgar, prices, classifier, reactions, scoring, db, config
```

> Raw 8-K text, market data, and generated outputs are excluded from the repo. See setup below to regenerate them.

---

## ▶️ Setup

```bash
pip install -r requirements.txt
```

1. Set an `ANTHROPIC_API_KEY` environment variable (for LLM event classification).
2. Set your SEC EDGAR user-agent header (see `NB1.2` for an example).
3. Run the notebooks in order (`NB1.1` → `NB4`).

---

## 🛠️ Skills Demonstrated

`Data Ingestion (APIs)` · `SEC EDGAR` · `Financial Data (yfinance)` · `Text Extraction` · `Event Classification (rules + LLM)` · `Event-Study Analysis` · `Scoring & Ranking` · `Pipeline Design`

---

## 👤 About

Team project (MSBA, Carlson School of Management). Presented here for portfolio purposes.

Built by **Parul Chaudhary** · [LinkedIn](#) · [Email](mailto:parul.jaswant@gmail.com)
